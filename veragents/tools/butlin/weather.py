"""
Weather tool using the open-meteo API (no API key required).
- Geocodes city name -> lat/lon via open-meteo geocoding.
- Fetches current weather.
"""

from __future__ import annotations

import requests
from pydantic import BaseModel, Field

from veragents.tools import ToolError, register_tool


class WeatherParams(BaseModel):
    city: str = Field(..., description="City name, e.g., 'Beijing' or 'San Francisco'")
    country: str | None = Field(None, description="Optional country filter, e.g., 'CN' or 'US'")


def _geocode_city(city: str, country: str | None = None) -> tuple[float, float, str]:
    """Resolve city -> (lat, lon, resolved_name) using open-meteo geocoding."""
    params = {"name": city, "count": 1}
    if country:
        params["country"] = country
    resp = requests.get("https://geocoding-api.open-meteo.com/v1/search", params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results") or []
    if not results:
        raise ToolError("get_current_weather", f"City not found: {city} ({country or 'any country'})", "NotFoundError")
    first = results[0]
    return first["latitude"], first["longitude"], first.get("name", city)


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
def get_current_weather(params: WeatherParams) -> dict:
    """获取城市当前天气（基于 open-meteo，免密钥）。"""
    lat, lon, resolved_name = _geocode_city(params.city, params.country)
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
