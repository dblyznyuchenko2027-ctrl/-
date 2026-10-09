"""
Модуль інтеграції із зовнішнім REST API Open-Meteo.

Відповідає за:
  - геокодування назви міста в координати (geocoding-api.open-meteo.com)
  - отримання поточної погоди за координатами (api.open-meteo.com)
  - обробку мережевих збоїв, таймаутів, кодів 4xx/5xx та несподіваного
    формату відповіді

Веб-рівень (main.py) НІЧОГО не знає про requests, HTTP-статуси зовнішнього
API чи формат JSON Open-Meteo. Він отримує від цього модуля або
WeatherResult (готовий, чистий результат), або WeatherError (зрозуміла,
класифікована помилка з кодом причини).
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import requests

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Скільки секунд чекаємо відповідь від зовнішнього сервісу, перш ніж
# вважати запит невдалим. Без цього застосунок міг би "зависнути"
# на невизначений час, якщо сервер не відповідає.
REQUEST_TIMEOUT_SECONDS = 5


class WeatherErrorReason(str, Enum):
    """Класифікація причин, з якими веб-рівень має поводитися по-різному."""

    CITY_NOT_FOUND = "city_not_found"      # місто не знайдено геокодером
    INVALID_INPUT = "invalid_input"        # порожній/некоректний ввід
    TIMEOUT = "timeout"                    # сервіс не відповів вчасно
    CONNECTION_ERROR = "connection_error"  # сервіс недоступний (мережа/DNS)
    CLIENT_ERROR = "client_error"          # 4xx від зовнішнього API
    SERVER_ERROR = "server_error"          # 5xx від зовнішнього API
    BAD_RESPONSE_FORMAT = "bad_response_format"  # JSON не той, що очікували


class WeatherError(Exception):
    """
    Класифікована помилка інтеграції.

    reason дозволяє веб-рівню вибрати правильний HTTP-статус, не знаючи
    нічого про requests чи структуру відповіді Open-Meteo.
    """

    def __init__(self, reason: WeatherErrorReason, message: str):
        self.reason = reason
        self.message = message
        super().__init__(message)


@dataclass
class WeatherResult:
    """Чистий, готовий до показу результат — без деталей HTTP/JSON."""

    city: str
    country: Optional[str]
    latitude: float
    longitude: float
    temperature_c: float
    wind_speed_kmh: float


def _geocode_city(city: str) -> tuple[float, float, str, Optional[str]]:
    """
    Перетворює назву міста на координати через геокодер Open-Meteo.

    Повертає (latitude, longitude, canonical_name, country).
    Кидає WeatherError, якщо місто не знайдено, стався мережевий збій,
    сервіс повернув 4xx/5xx, або відповідь має несподіваний формат.
    """
    try:
        response = requests.get(
            GEOCODING_URL,
            params={"name": city, "count": 1, "language": "uk", "format": "json"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.exceptions.Timeout:
        raise WeatherError(
            WeatherErrorReason.TIMEOUT,
            "Сервіс геокодування не відповів вчасно.",
        )
    except requests.exceptions.ConnectionError:
        raise WeatherError(
            WeatherErrorReason.CONNECTION_ERROR,
            "Не вдалося з'єднатися із сервісом геокодування.",
        )
    except requests.exceptions.RequestException as exc:
        raise WeatherError(
            WeatherErrorReason.CONNECTION_ERROR,
            f"Помилка запиту до сервісу геокодування: {exc}",
        )

    if 400 <= response.status_code < 500:
        raise WeatherError(
            WeatherErrorReason.CLIENT_ERROR,
            f"Сервіс геокодування відхилив запит (код {response.status_code}).",
        )
    if response.status_code >= 500:
        raise WeatherError(
            WeatherErrorReason.SERVER_ERROR,
            f"Сервіс геокодування тимчасово недоступний (код {response.status_code}).",
        )

    try:
        data = response.json()
    except ValueError:
        raise WeatherError(
            WeatherErrorReason.BAD_RESPONSE_FORMAT,
            "Сервіс геокодування повернув не-JSON відповідь.",
        )

    # Важлива особливість Open-Meteo: для ненайденого міста поля "results"
    # у відповіді взагалі немає (а не порожній список). Обидва випадки
    # трактуємо однаково — місто не знайдено.
    results = data.get("results")
    if not results:
        raise WeatherError(
            WeatherErrorReason.CITY_NOT_FOUND,
            f"Місто «{city}» не знайдено.",
        )

    first = results[0]
    try:
        latitude = float(first["latitude"])
        longitude = float(first["longitude"])
        name = str(first.get("name", city))
        country = first.get("country")
    except (KeyError, TypeError, ValueError):
        raise WeatherError(
            WeatherErrorReason.BAD_RESPONSE_FORMAT,
            "Відповідь геокодера не містить очікуваних полів.",
        )

    return latitude, longitude, name, country


def _fetch_current_weather(latitude: float, longitude: float) -> tuple[float, float]:
    """
    Отримує поточну температуру (°C) і швидкість вітру (км/год) за
    координатами. Кидає WeatherError за тими самими правилами, що й
    _geocode_city.
    """
    try:
        response = requests.get(
            FORECAST_URL,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,wind_speed_10m",
                "timezone": "auto",
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.exceptions.Timeout:
        raise WeatherError(
            WeatherErrorReason.TIMEOUT,
            "Сервіс прогнозу погоди не відповів вчасно.",
        )
    except requests.exceptions.ConnectionError:
        raise WeatherError(
            WeatherErrorReason.CONNECTION_ERROR,
            "Не вдалося з'єднатися із сервісом прогнозу погоди.",
        )
    except requests.exceptions.RequestException as exc:
        raise WeatherError(
            WeatherErrorReason.CONNECTION_ERROR,
            f"Помилка запиту до сервісу прогнозу погоди: {exc}",
        )

    if 400 <= response.status_code < 500:
        raise WeatherError(
            WeatherErrorReason.CLIENT_ERROR,
            f"Сервіс прогнозу відхилив запит (код {response.status_code}).",
        )
    if response.status_code >= 500:
        raise WeatherError(
            WeatherErrorReason.SERVER_ERROR,
            f"Сервіс прогнозу тимчасово недоступний (код {response.status_code}).",
        )

    try:
        data = response.json()
    except ValueError:
        raise WeatherError(
            WeatherErrorReason.BAD_RESPONSE_FORMAT,
            "Сервіс прогнозу повернув не-JSON відповідь.",
        )

    try:
        current = data["current"]
        temperature = float(current["temperature_2m"])
        wind_speed = float(current["wind_speed_10m"])
    except (KeyError, TypeError, ValueError):
        raise WeatherError(
            WeatherErrorReason.BAD_RESPONSE_FORMAT,
            "Відповідь сервісу прогнозу не містить очікуваних полів.",
        )

    return temperature, wind_speed


def get_weather(city: str) -> WeatherResult:
    """
    Головна публічна функція модуля: назва міста -> поточна погода.

    Це єдина функція, яку має викликати веб-рівень. Вона або повертає
    готовий WeatherResult, або кидає WeatherError з класифікованою
    причиною — веб-рівень перетворює це на HTTP-статус і повідомлення.
    """
    if not city or not city.strip():
        raise WeatherError(
            WeatherErrorReason.INVALID_INPUT,
            "Назва міста не може бути порожньою.",
        )

    city = city.strip()
    latitude, longitude, canonical_name, country = _geocode_city(city)
    temperature, wind_speed = _fetch_current_weather(latitude, longitude)

    return WeatherResult(
        city=canonical_name,
        country=country,
        latitude=latitude,
        longitude=longitude,
        temperature_c=temperature,
        wind_speed_kmh=wind_speed,
    )
