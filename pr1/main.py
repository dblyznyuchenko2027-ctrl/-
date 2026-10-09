"""
Веб-рівень застосунку.

Тут немає жодного знання про requests, HTTP-статуси Open-Meteo чи формат
його JSON — уся ця логіка захована в weather_api.py. Веб-рівень лише:
  - приймає назву міста від користувача (форма або query-параметр),
  - викликає weather_api.get_weather(city),
  - перетворює результат/помилку на HTTP-відповідь.
"""

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from weather_api import WeatherError, WeatherErrorReason, get_weather

app = FastAPI(title="Погода за назвою міста")
templates = Jinja2Templates(directory=".")

# Відповідність причини помилки інтеграції -> HTTP-статус, який побачить
# клієнт. 4xx/5xx від зовнішнього сервісу навмисно не "просвічуються"
# один в один назовні: клієнту нашого застосунку важливо, чи винен він
# сам (неправильне місто), чи тимчасово підвела зовнішня служба.
_REASON_TO_STATUS = {
    WeatherErrorReason.INVALID_INPUT: 400,
    WeatherErrorReason.CITY_NOT_FOUND: 404,
    WeatherErrorReason.CLIENT_ERROR: 502,
    WeatherErrorReason.SERVER_ERROR: 502,
    WeatherErrorReason.TIMEOUT: 504,
    WeatherErrorReason.CONNECTION_ERROR: 502,
    WeatherErrorReason.BAD_RESPONSE_FORMAT: 502,
}


def _error_response(error: WeatherError) -> JSONResponse:
    status_code = _REASON_TO_STATUS.get(error.reason, 500)
    return JSONResponse(
        status_code=status_code,
        content={"error": error.reason.value, "message": error.message},
    )


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    """Проста сторінка з формою для вводу міста."""
    return templates.TemplateResponse(
        "index.html", {"request": request, "result": None, "error": None}
    )


@app.get("/weather-page", response_class=HTMLResponse)
def weather_page(request: Request, city: str = ""):
    """Обробляє відправку форми і показує результат/помилку на тій самій сторінці."""
    if not city.strip():
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "result": None, "error": "Введіть назву міста."},
        )

    try:
        result = get_weather(city)
    except WeatherError as error:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "result": None, "error": error.message},
        )

    return templates.TemplateResponse(
        "index.html", {"request": request, "result": result, "error": None}
    )


@app.get("/api/weather")
def weather_json(city: str = Query(..., description="Назва міста, напр. Kyiv")):
    """JSON-ендпоінт: успіх -> 200 з даними, помилка -> відповідний HTTP-статус."""
    try:
        result = get_weather(city)
    except WeatherError as error:
        return _error_response(error)

    return JSONResponse(
        status_code=200,
        content={
            "city": result.city,
            "country": result.country,
            "latitude": result.latitude,
            "longitude": result.longitude,
            "temperature_c": result.temperature_c,
            "wind_speed_kmh": result.wind_speed_kmh,
        },
    )
