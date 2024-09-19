import os
import pytest
from fastapi.testclient import TestClient
from sentinel_monitor.src.ML.data_preprocessing import preprocess_vi_image
from sentinel_monitor.src.api.main import app
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv

# Carregar variáveis de ambiente do arquivo .env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../.env'))

client = TestClient(app)

@pytest.fixture
def mock_sentinel_service():
    with patch('sentinel_monitor.src.services.sentinel_service.SentinelService') as mock:
        mock.return_value.get_plant_health.return_value = {"health": "good"}
        mock.return_value.get_sentinel1_data.return_value = "mocked_sentinel1_data"
        mock.return_value.get_sentinel2_data.return_value = "mocked_sentinel2_data"
        mock.return_value.get_pest_detection.return_value = "Nenhuma anomalia significativa detectada."
        mock.return_value.get_harvest_advice.return_value = "Colheita recomendada em 2 semanas."
        yield mock

@pytest.fixture
def mock_ml_service():
    with patch('sentinel_monitor.src.services.ml_service.MLService') as mock:
        mock_instance = MagicMock()
        mock_instance.load_models = MagicMock()
        mock_instance.analyze_time_series.return_value = {"result": "analysis_data"}
        mock_instance.predict_yield.return_value = {"yield": "high"}
        mock.return_value = mock_instance
        yield mock

def test_rate_limiting(client, mock_sentinel_service, mock_ml_service):
    # Simular um usuário autenticado
    client.headers = {"Authorization": "Bearer fake_token"}

    # Fazer 11 requisições em rápida sucessão
    for i in range(11):
        response = client.get("/weather-data")
        if i < 10:
            assert response.status_code == 200
        else:
            assert response.status_code == 429  # Too Many Requests

def test_cache(client, mock_sentinel_service, mock_ml_service):
    # Simular um usuário autenticado
    client.headers = {"Authorization": "Bearer fake_token"}

    # Primeira requisição
    response1 = client.get("/weather-data")
    assert response1.status_code == 200

    # Segunda requisição (deve vir do cache)
    response2 = client.get("/weather-data")
    assert response2.status_code == 200
    assert response1.json() == response2.json()

def test_protected_route(client):
    # Tentar acessar rota protegida sem autenticação
    response = client.get("/protected")
    assert response.status_code == 401

    # Simular um usuário autenticado
    client.headers = {"Authorization": "Bearer fake_token"}
    response = client.get("/protected")
    assert response.status_code == 200
    assert "message" in response.json()

def test_irrigation_advice(mock_sentinel_service):
    mock_sentinel_service.return_value.get_irrigation_advice.return_value = "Irrigação recomendada."

    payload = {
        "coordinates": [[-5.0, 40.0], [-5.0, 45.0], [5.0, 45.0], [5.0, 40.0], [-5.0, 40.0]]
    }
    response = client.post("/monitor/irrigation_advice", json=payload)
    assert response.status_code == 200
    assert "Irrigação" in response.json()

def test_pest_detection(mock_sentinel_service):
    payload = {
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[-5.0, 40.0], [-5.0, 45.0], [5.0, 45.0], [5.0, 40.0], [-5.0, 40.0]]]
        },
        "start_date": "2023-01-01",
        "end_date": "2023-01-31"
    }
    response = client.post("/monitor/pest_detection", json=payload)
    assert response.status_code == 200
    assert "anomalia" in response.json().lower()

def test_harvest_advice(mock_sentinel_service):
    payload = {
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[-5.0, 40.0], [-5.0, 45.0], [5.0, 45.0], [5.0, 40.0], [-5.0, 40.0]]]
        },
        "start_date": "2023-01-01",
        "end_date": "2023-01-31"
    }
    response = client.post("/monitor/harvest_advice", json=payload)
    assert response.status_code == 200

def test_crop_health_prediction(self):
    X_images, _ = self.mock_data[0]
    processed_image = preprocess_vi_image(X_images[0])
    prediction = self.ml_service.predict_crop_health(processed_image)
    self.assertIn(prediction, ['Saudável', 'Não Saudável'])


def test_soil_moisture(mock_sentinel_service):
    payload = {
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[-5.0, 40.0], [-5.0, 45.0], [5.0, 45.0], [5.0, 40.0], [-5.0, 40.0]]]
        },
        "start_date": "2023-01-01",
        "end_date": "2023-01-31"
    }
    response = client.post("/monitor/soil-moisture", json=payload)
    assert response.status_code == 200
    assert "Umidade do solo verificada" in response.json()["message"]

def test_time_series_analysis(mock_sentinel_service, mock_ml_service):
    payload = {
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[-5.0, 40.0], [-5.0, 45.0], [5.0, 45.0], [5.0, 40.0], [-5.0, 40.0]]]
        },
        "start_date": "2023-01-01",
        "end_date": "2023-01-31"
    }
    response = client.post("/monitor/time-series-analysis", json=payload)
    assert response.status_code == 200
    assert "Análise de série temporal concluída" in response.json()["message"]
    assert "results" in response.json()

def test_yield_prediction(self):
    X_images, _ = self.mock_data[0]
    processed_image = preprocess_vi_image(X_images[0])
    prediction = self.ml_service.predict_yield(processed_image)
    self.assertIsInstance(prediction, float)
    self.assertGreater(prediction, 0)

def test_validate_weather_data_valid(client):
    valid_data = {
        "temperature": 25.5,
        "atmospheric_pressure": 1013.25,
        "humidity": 60,
        "wind_speed": 5.5,
        "solar_radiation": 800,
        "filled_data": 0
    }
    response = client.post("/validate-weather-data", json=valid_data)
    assert response.status_code == 200
    assert response.json() == valid_data

def test_validate_weather_data_invalid(client):
    invalid_data = {
        "temperature": -300,  # Temperatura abaixo do zero absoluto
        "atmospheric_pressure": 1013.25,
        "humidity": 60,
        "wind_speed": 5.5,
        "solar_radiation": 800,
        "filled_data": 0
    }
    response = client.post("/validate-weather-data", json=invalid_data)
    assert response.status_code == 400
    assert "A temperatura não pode ser menor que o zero absoluto" in response.json()["detail"]

def test_redis_connection_error(client, mocker):
    mocker.patch("SAT.sentinel_monitor.src.api.main.check_redis_connection", side_effect=RedisConnectionError("Erro de conexão"))
    response = client.get("/status")
    assert response.status_code == 503
    assert "Serviço temporariamente indisponível" in response.json()["message"]
