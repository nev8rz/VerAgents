"""
Weather tool using the open-meteo API (no API key required).
- Geocodes city name -> lat/lon via open-meteo geocoding.
- Fetches current weather.
"""

from __future__ import annotations

import requests
from loguru import logger as log

from veragents.tools import ToolError, register_tool


def _geocode_city(city: str, country: str | None = None, language: str = "zh") -> tuple[float, float, str]:
    """Resolve city -> (lat, lon, resolved_name) using open-meteo geocoding."""
    attempts = []

    def _query(name: str, lang: str):
        params = {"name": name, "count": 1, "language": lang}
        if country:
            params["country"] = country
        resp = requests.get("https://geocoding-api.open-meteo.com/v1/search", params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("results") or []

    queries = [(city, language)]
    if any("\u4e00" <= ch <= "\u9fff" for ch in city):
        # If Chinese, try with/without "市"
        if not city.endswith("市"):
            queries.append((f"{city}市", language))
        else:
            queries.append((city.rstrip("市"), language))
    # English fallback
    queries.append((city, "en"))

    for name, lang in queries:
        try:
            results = _query(name, lang)
        except requests.RequestException as exc:  # network error
            log.warning("Geocoding request failed for {} lang {}: {}", name, lang, exc)
            continue
        attempts.append((name, lang, len(results)))
        if results:
            first = results[0]
            return first["latitude"], first["longitude"], first.get("name", name)

    log.error("Geocoding failed | city={} country={} attempts={}", city, country, attempts)
    raise ToolError("get_current_weather", f"City not found: {city} ({country or 'any country'})", "NotFoundError")


def _fetch_weather(lat: float, lon: float) -> dict:
    """Fetch current weather from open-meteo."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "current_weather": True,
        "timezone": "auto",
    }
    resp = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if "current_weather" not in data:
        raise ToolError("get_current_weather", "Missing current_weather in response", "ParseError")
    return data["current_weather"]


@register_tool(name="get_current_weather")
def get_current_weather(city: str, country: str | None = None) -> dict:
    """获取城市当前天气（基于 open-meteo，免密钥）。"""
    lat, lon, resolved_name = _geocode_city(city, country)
    weather = _fetch_weather(lat, lon)
    return {
        "city": resolved_name,
        "latitude": lat,
        "longitude": lon,
        "temperature": weather.get("temperature"),
        "windspeed": weather.get("windspeed"),
        "winddirection": weather.get("winddirection"),
        "weathercode": weather.get("weathercode"),
        "time": weather.get("time"),
        "source": "open-meteo",
    }
