import requests
from typing import Dict
from ver_agent.tools.base import toolkit


@toolkit
class WeatherFetcher:
    """天气获取器 - 支持多个数据源"""

    def get_weather(self, location: str, method: str = 'auto') -> Dict:
        """
        获取天气信息
        :param location: 地点名称
        :param method: 'wttr', 'openmeteo', 'auto'（自动尝试）
        """
        methods = {
            'wttr': self._wttr,
            'openmeteo': self._openmeteo
        }

        if method == 'auto':
            # 自动尝试所有方法
            for method_name, method_func in methods.items():
                try:
                    result = method_func(location)
                    if 'error' not in result:
                        result['数据源'] = method_name
                        return result
                except:
                    continue
            return {'error': '所有方法均失败'}
        else:
            return methods.get(method, self._wttr)(location)

    def _wttr(self, location: str) -> Dict:
        """wttr.in 方法"""
        url = f"https://wttr.in/{location}?format=j1&lang=zh"
        response = requests.get(url, timeout=10)
        data = response.json()

        current = data['current_condition'][0]
        return {
            '地点': location,
            '温度': f"{current['temp_C']}°C",
            '体感温度': f"{current['FeelsLikeC']}°C",
            '天气描述': current['lang_zh'][0]['value'] if 'lang_zh' in current else current['weatherDesc'][0]['value'],
            '湿度': f"{current['humidity']}%",
            '风速': f"{current['windspeedKmph']} km/h"
        }

    def _openmeteo(self, location: str) -> Dict:
        """Open-Meteo 方法"""
        # 获取坐标
        geo_url = "https://nominatim.openstreetmap.org/search"
        geo_response = requests.get(
            geo_url,
            params={'q': location, 'format': 'json', 'limit': 1},
            headers={'User-Agent': 'WeatherApp/1.0'},
            timeout=10
        )
        geo_data = geo_response.json()[0]

        # 获取天气
        weather_url = "https://api.open-meteo.com/v1/forecast"
        weather_response = requests.get(
            weather_url,
            params={
                'latitude': geo_data['lat'],
                'longitude': geo_data['lon'],
                'current_weather': True
            },
            timeout=10
        )
        current = weather_response.json()['current_weather']

        return {
            '地点': location,
            '温度': f"{current['temperature']}°C",
            '风速': f"{current['windspeed']} km/h"
        }


# from ..registry import global_registry
# global_registry.register(WeatherFetcher)
