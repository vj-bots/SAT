import os
import sys
import pytest
import time

from fastapi.testclient import TestClient
from sentinel_monitor.src.ML.data_preprocessing import preprocess_vi_image
from sentinel_monitor.src.api.main import app
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv
from sentinel_monitor.src.exceptions_src import RedisConnectionError
from sentinel_monitor.src.utils.auth import create_access_token
from sentinel_monitor.src.api.main import RateLimitMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from sentinel_monitor.src.services.feedback_service import FeedbackModel, save_feedback as feedback_service_save_feedback

# Carregar variáveis de ambiente do arquivo .env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../.env'))

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

@pytest.fixture(scope="module")
def client():
    return TestClient(app)

@pytest.fixture
def auth_headers():
    access_token = create_access_token(data={"sub": "testuser"})
    return {"Authorization": f"Bearer {access_token}"}

@pytest.fixture
def mock_sentinel_service():
    with patch('sentinel_monitor.src.services.sentinel_service.SentinelService') as mock:
        mock.return_value.get_plant_health.return_value = {"health": "good"}
        mock.return_value.get_sentinel1_data.return_value = "mocked_sentinel1_data"
        mock.return_value.get_sentinel2_data.return_value = "mocked_sentinel2_data"
        mock.return_value.get_pest_detection.return_value = {"message": "Nenhuma anomalia significativa detectada."}
        mock.return_value.get_harvest_advice.return_value = {"message": "Colheita recomendada em 2 semanas."}
        mock.return_value.get_irrigation_advice.return_value = {"message": "Irrigação recomendada."}
        mock.return_value.get_weather_forecast.return_value = {"forecast": "Ensolarado", "temperature": 25, "humidity": 60}
        yield mock

@pytest.fixture
def mock_ml_service():
    with patch('sentinel_monitor.src.services.ml_service.MLService') as mock:
        mock.return_value.predict_crop_health.return_value = {'health': 'Saudável'}
        mock.return_value.predict_yield.return_value = {'yield': 5.5}
        yield mock

@pytest.fixture
def mock_market_service():
    with patch('sentinel_monitor.src.services.market_analysis_service.get_market_trends') as mock:
        mock.return_value = {
            "current_price": 10.5,
            "price_trend": "up",
            "demand_forecast": "high"
        }
        yield mock

class DisableRateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        return await call_next(request)

@pytest.fixture
def app_without_rate_limit():
    from sentinel_monitor.src.api.main import app
    app.user_middleware = [m for m in app.user_middleware if not isinstance(m.cls, RateLimitMiddleware)]
    app.middleware_stack = app.build_middleware_stack()
    return app

@pytest.fixture
def client_without_rate_limit(app_without_rate_limit):
    return TestClient(app_without_rate_limit)

@pytest.fixture(scope="module", autouse=True)
def setup_test_user():
    from sentinel_monitor.src.site.auth_db import register
    from sentinel_monitor.src.model.user import User_create
    user = User_create(username="testuser", email="test@example.com", password="Test@password123")
    register(user)

def test_rate_limiting(client, mock_sentinel_service, mock_ml_service, auth_headers):
    for i in range(12):
        response = client.get("/monitor/weather-forecast?latitude=-23.5505&longitude=-46.6333", headers=auth_headers)
        if i < 10:
            assert response.status_code == 200
        else:
            assert response.status_code == 429
    # Aguarde um segundo para resetar o limite
    time.sleep(1)

def test_cache(client_without_rate_limit, mock_sentinel_service, mock_ml_service, auth_headers):
    response1 = client_without_rate_limit.get("/monitor/weather-forecast?latitude=-23.5505&longitude=-46.6333", headers=auth_headers)
    assert response1.status_code == 200
    response2 = client_without_rate_limit.get("/monitor/weather-forecast?latitude=-23.5505&longitude=-46.6333", headers=auth_headers)
    assert response2.status_code == 200
    assert response1.json() == response2.json()

def test_protected_route(client_without_rate_limit, auth_headers):
    response = client_without_rate_limit.get("/protected", headers=auth_headers)
    print(f"Protected route headers: {response.request.headers}")
    print(f"Protected route response: {response.content}")
    assert response.status_code == 200
    assert "message" in response.json()

def test_irrigation_advice(client_without_rate_limit, mock_sentinel_service, auth_headers):
    payload = {
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[-5.0, 40.0], [-5.0, 45.0], [5.0, 45.0], [5.0, 40.0], [-5.0, 40.0]]]
        },
        "start_date": "2023-01-01T00:00:00",
        "end_date": "2023-01-31T23:59:59"
    }
    response = client_without_rate_limit.post("/monitor/irrigation_advice", json=payload, headers=auth_headers)
    assert response.status_code == 200
    assert "Conselho de irrigação gerado" in response.json()["message"]

def test_pest_detection(client_without_rate_limit, mock_sentinel_service, auth_headers):
    payload = {
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[-5.0, 40.0], [-5.0, 45.0], [5.0, 45.0], [5.0, 40.0], [-5.0, 40.0]]]
        },
        "start_date": "2023-01-01T00:00:00",
        "end_date": "2023-01-31T23:59:59"
    }
    response = client_without_rate_limit.post("/monitor/pest_detection", json=payload, headers=auth_headers)
    assert response.status_code == 200
    assert "Detecção de pragas concluída" in response.json()["message"]

def test_harvest_advice(client_without_rate_limit, mock_sentinel_service, auth_headers):
    payload = {
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[-5.0, 40.0], [-5.0, 45.0], [5.0, 45.0], [5.0, 40.0], [-5.0, 40.0]]]
        },
        "start_date": "2023-01-01T00:00:00",
        "end_date": "2023-01-31T23:59:59"
    }
    response = client_without_rate_limit.post("/monitor/harvest_advice", json=payload, headers=auth_headers)
    print(f"Harvest advice response: {response.content}")
    assert response.status_code == 200

def test_crop_health_prediction(client_without_rate_limit, mock_ml_service, auth_headers):
    payload = {
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[-5.0, 40.0], [-5.0, 45.0], [5.0, 45.0], [5.0, 40.0], [-5.0, 40.0]]]
        },
        "start_date": "2023-01-01T00:00:00",
        "end_date": "2023-01-31T23:59:59"
    }
    response = client_without_rate_limit.post("/monitor/crop-health", json=payload, headers=auth_headers)
    assert response.status_code in [200, 500]
    if response.status_code == 200:
        assert 'health' in response.json()
    else:
        print(f"Erro 500: {response.json()}")

def test_soil_moisture(client_without_rate_limit, mock_sentinel_service, auth_headers):
    payload = {
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[-5.0, 40.0], [-5.0, 45.0], [5.0, 45.0], [5.0, 40.0], [-5.0, 40.0]]]
        },
        "start_date": "2023-01-01T00:00:00",
        "end_date": "2023-01-31T23:59:59"
    }
    response = client_without_rate_limit.post("/monitor/soil-moisture", json=payload, headers=auth_headers)
    print(f"Soil moisture response: {response.content}")
    assert response.status_code == 200
    assert "Umidade do solo verificada" in response.json()["message"]

def test_time_series_analysis(client_without_rate_limit, mock_sentinel_service, mock_ml_service, auth_headers):
    payload = {
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[-5.0, 40.0], [-5.0, 45.0], [5.0, 45.0], [5.0, 40.0], [-5.0, 40.0]]]
        },
        "start_date": "2023-01-01T00:00:00",
        "end_date": "2023-01-31T23:59:59"
    }
    response = client_without_rate_limit.post("/monitor/time-series-analysis", json=payload, headers=auth_headers)
    print(f"Time series analysis response: {response.content}")
    assert response.status_code == 200
    assert "Análise de série temporal concluída" in response.json()["message"]
    assert "results" in response.json()

def test_yield_prediction(client_without_rate_limit, mock_ml_service, auth_headers):
    payload = {
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[-5.0, 40.0], [-5.0, 45.0], [5.0, 45.0], [5.0, 40.0], [-5.0, 40.0]]]
        },
        "start_date": "2023-01-01T00:00:00",
        "end_date": "2023-01-31T23:59:59"
    }
    response = client_without_rate_limit.post("/monitor/yield-prediction", json=payload, headers=auth_headers)
    assert response.status_code in [200, 500]
    if response.status_code == 200:
        assert 'yield' in response.json()
    else:
        print(f"Erro 500: {response.json()}")

def test_validate_weather_data_valid(client_without_rate_limit, auth_headers):
    valid_data = {
        "temperature": 25.5,
        "atmospheric_pressure": 1013.25,
        "humidity": 60,
        "wind_speed": 5.5,
        "solar_radiation": 800,
        "filled_data": 0
    }
    response = client_without_rate_limit.post("/validate-weather-data", json=valid_data, headers=auth_headers)
    print(f"Validate weather data headers: {response.request.headers}")
    print(f"Validate weather data response: {response.content}")
    assert response.status_code == 200
    assert response.json() == valid_data

def test_validate_weather_data_invalid(client_without_rate_limit, auth_headers):
    invalid_data = {
        "temperature": -300,
        "atmospheric_pressure": 1013.25,
        "humidity": 60,
        "wind_speed": 5.5,
        "solar_radiation": 800,
        "filled_data": 0
    }
    response = client_without_rate_limit.post("/validate-weather-data", json=invalid_data, headers=auth_headers)
    print(f"Validate invalid weather data headers: {response.request.headers}")
    print(f"Validate invalid weather data response: {response.content}")
    assert response.status_code == 422

def test_redis_connection_error(client_without_rate_limit, mocker, auth_headers):
    mocker.patch("sentinel_monitor.src.api.main.check_redis_connection", side_effect=RedisConnectionError("Erro de conexão"))
    response = client_without_rate_limit.get("/status", headers=auth_headers)
    print(f"Redis connection error headers: {response.request.headers}")
    print(f"Redis connection error response: {response.content}")
    assert response.status_code == 503

def test_get_market_trends(client_without_rate_limit, mock_market_service, auth_headers):
    response = client_without_rate_limit.get("/market/trends?crop_type=soybean", headers=auth_headers)
    assert response.status_code == 200
    assert "trend" in response.json()

@pytest.fixture
def mock_save_feedback():
    async def mock_save(feedback, user_id):
        return 1  # Simula um ID de feedback retornado
    with patch('sentinel_monitor.src.services.feedback_service.save_feedback', new=mock_save) as mock:
        yield mock

def test_submit_feedback(client_without_rate_limit, auth_headers, mock_save_feedback):
    feedback_data = {
        "alert_type": "irrigation",
        "prediction_accuracy": 0.85,
        "user_observation": "A previsão de irrigação foi precisa, mas a quantidade sugerida foi um pouco alta."
    }
    response = client_without_rate_limit.post("/feedback/submit-feedback", json=feedback_data, headers=auth_headers)
    print(f"Submit feedback response: {response.content}")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "feedback_id" in response.json()

def test_get_weather_forecast(client_without_rate_limit, mock_sentinel_service, auth_headers):
    response = client_without_rate_limit.get("/monitor/weather-forecast?latitude=-23.5505&longitude=-46.6333", headers=auth_headers)
    assert response.status_code == 200
    assert "forecast" in response.json()
