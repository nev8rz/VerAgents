"""Builtin tool collection (weather, search, etc.)."""

from .weather import get_current_weather
from .search import search_web

__all__ = ["get_current_weather", "search_web"]
