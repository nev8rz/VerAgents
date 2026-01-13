"""
Web search tool supporting SerpAPI (SERPAPI_API_KEY) and Tavily (TVLY_API_KEY/TAVILY_API_KEY).
Chooses provider automatically if not specified.
"""

from __future__ import annotations

import os
from typing import List, Optional

import requests

from veragents.tools import ToolError, register_tool


def _serpapi_search(
    query: str,
    num: int,
    engine: str,
    gl: Optional[str],
    hl: Optional[str],
    api_key: str,
) -> dict:
    payload = {
        "q": query,
        "engine": engine,
        "api_key": api_key,
        "num": num,
    }
    if gl:
        payload["gl"] = gl
    if hl:
        payload["hl"] = hl

    try:
        resp = requests.get("https://serpapi.com/search", params=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        raise ToolError("search_web", f"SerpAPI request failed: {exc}", "NetworkError") from exc


def _tavily_search(
    query: str,
    num: int,
    api_key: str,
    search_depth: str = "basic",
) -> dict:
    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": num,
        "search_depth": search_depth,
        "include_answer": False,
    }
    try:
        resp = requests.post("https://api.tavily.com/search", json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        raise ToolError("search_web", f"Tavily request failed: {exc}", "NetworkError") from exc


def _parse_serpapi_results(data: dict, limit: int) -> List[dict]:
    organic = data.get("organic_results") or []
    results = []
    for item in organic[:limit]:
        results.append(
            {
                "title": item.get("title"),
                "link": item.get("link"),
                "snippet": item.get("snippet"),
            }
        )
    return results


def _parse_tavily_results(data: dict, limit: int) -> List[dict]:
    raw = data.get("results") or []
    results = []
    for item in raw[:limit]:
        results.append(
            {
                "title": item.get("title"),
                "link": item.get("url"),
                "snippet": item.get("content"),
            }
        )
    return results


def _choose_provider(provider: Optional[str]) -> tuple[str, str]:
    """Decide provider based on explicit param or available env keys."""
    serp_key = os.getenv("SERPAPI_API_KEY")
    tavily_key = os.getenv("TVLY_API_KEY") or os.getenv("TAVILY_API_KEY")

    if provider:
        if provider == "serpapi":
            if not serp_key:
                raise ToolError("search_web", "SERPAPI_API_KEY not set", "ConfigError")
            return "serpapi", serp_key
        if provider in {"tavily", "tvly"}:
            if not tavily_key:
                raise ToolError("search_web", "TVLY_API_KEY/TAVILY_API_KEY not set", "ConfigError")
            return "tavily", tavily_key
        raise ToolError("search_web", f"Unsupported provider: {provider}", "ConfigError")

    # Auto-pick: prefer SerpAPI if available, otherwise Tavily
    if serp_key:
        return "serpapi", serp_key
    if tavily_key:
        return "tavily", tavily_key
    raise ToolError("search_web", "No search API key configured", "ConfigError")


@register_tool(name="search_web")
def search_web(
    query: str,
    num: int = 5,
    engine: str = "google",
    gl: Optional[str] = None,
    hl: Optional[str] = None,
    provider: Optional[str] = None,  # "serpapi" | "tavily"/"tvly"
    search_depth: str = "basic",  # tavily-only
) -> dict:
    """真实 Web 搜索，支持 SerpAPI（SERPAPI_API_KEY）与 Tavily（TVLY_API_KEY/TAVILY_API_KEY）。"""
    provider_name, api_key = _choose_provider(provider)

    if provider_name == "serpapi":
        data = _serpapi_search(query, num, engine, gl, hl, api_key)
        results = _parse_serpapi_results(data, num)
    else:
        data = _tavily_search(query, num, api_key, search_depth=search_depth)
        results = _parse_tavily_results(data, num)

    return {
        "provider": provider_name,
        "query": query,
        "engine": engine if provider_name == "serpapi" else "tavily",
        "result_count": len(results),
        "results": results,
    }
