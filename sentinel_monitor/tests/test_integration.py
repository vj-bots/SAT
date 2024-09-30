import os
import io
import sys
import pytest
import asyncio
import unittest
import numpy as np

from PIL import Image
from fastapi.testclient import TestClient

# Adicione o diretório raiz do projeto ao PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from sentinel_monitor.src.ML.data_preprocessing import load_data, preprocess_vi_image, preprocess_sensor_data
from sentinel_monitor.src.ML.train_model import train_model
from sentinel_monitor.src.api.main import app
from sentinel_monitor.src.services.ml_service import MLService

client = TestClient(app)

class TestIntegration(unittest.TestCase):
    @classmethod
    async def setUpClass(cls):
        from fastapi.testclient import TestClient
        from sentinel_monitor.src.api.main import app
        cls.client = TestClient(app)
        cls.ml_service = MLService()
        
        # Criar usuário de teste
        from sentinel_monitor.src.site.auth_db import register
        from sentinel_monitor.src.model.user import User_create
        user = User_create(username="testuser", email="test@example.com", password="Test@123")
        await register(user)
        
        # Simular login e obter token real
        login_data = {"username": "testuser", "password": "Test@123"}
        response = cls.client.post("/token", data=login_data)
        if response.status_code == 200:
            cls.headers = {"Authorization": f"Bearer {response.json()['access_token']}"}
        else:
            raise Exception(f"Falha na autenticação durante a configuração do teste. Status code: {response.status_code}")

    def test_full_pipeline(self):
        # Carrega os dados
        (X_images, X_sensors), (y_irrigation, y_invasion, y_health, y_yield) = load_data()
        
        # Verifica se os dados foram carregados corretamente
        self.assertIsNotNone(X_images)
        self.assertIsNotNone(X_sensors)
        self.assertIsNotNone(y_irrigation)
        self.assertIsNotNone(y_invasion)
        self.assertIsNotNone(y_health)
        self.assertIsNotNone(y_yield)
        
        # Treina o modelo em modo de teste
        model, history = train_model(test_mode=True)
        
        # Verifica se o modelo foi treinado corretamente
        self.assertIsNotNone(model)
        self.assertIsNotNone(history)
        
        # Faz uma previsão com o modelo treinado
        sample_image = preprocess_vi_image(X_images[0:1])
        sample_sensor = preprocess_sensor_data(X_sensors[0:1])
        predictions = model.predict([sample_image, sample_sensor])
        
        # Verifica se as previsões têm o formato esperado
        self.assertEqual(len(predictions), 4)  # 4 saídas: irrigação, invasão, saúde e rendimento
        for pred in predictions:
            self.assertIsNotNone(pred)

    def test_invalid_data(self):
        # Teste com dados inválidos
        invalid_payload = {
            "geometry": "invalid_geometry",
            "start_date": "2023-01-01",
            "end_date": "2023-01-31"
        }
        response = self.client.post("/monitor/crop-health", json=invalid_payload, headers=self.headers)
        self.assertEqual(response.status_code, 400)
        self.assertIn("detail", response.json())

    def test_input_validation(self):
        # Teste de validação de entrada
        invalid_date_payload = {
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[-5.0, 40.0], [-5.0, 45.0], [5.0, 45.0], [5.0, 40.0], [-5.0, 40.0]]]
            },
            "start_date": "2023-01-32",  # Data inválida
            "end_date": "2023-01-31"
        }
        response = self.client.post("/monitor/crop-health", json=invalid_date_payload, headers=self.headers)
        self.assertEqual(response.status_code, 400)
        self.assertIn("detail", response.json())

    def test_api_full_flow(self):
        # Simular autenticação
        login_data = {
            "username": "testuser",
            "password": "testpassword"
        }
        login_response = self.client.post("/token", data=login_data)
        self.assertEqual(login_response.status_code, 200)
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Teste do fluxo completo da API
        valid_payload = {
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[-5.0, 40.0], [-5.0, 45.0], [5.0, 45.0], [5.0, 40.0], [-5.0, 40.0]]]
            },
            "start_date": "2023-01-01",
            "end_date": "2023-01-31"
        }
        
        # Teste de saúde da cultura
        response = self.client.post("/monitor/crop-health", json=valid_payload, headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertIn("health", response.json())
        
        # Teste de necessidade de irrigação
        response = self.client.post("/monitor/irrigation-need", json=valid_payload, headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertIn("irrigation_need", response.json())
        
        # Teste de detecção de pragas
        response = self.client.post("/monitor/pest-detection", json=valid_payload, headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertIn("pest_detected", response.json())
        
        # Teste de previsão de rendimento
        response = self.client.post("/monitor/yield-prediction", json=valid_payload, headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertIn("yield_prediction", response.json())
        
        # Teste de previsão do tempo
        response = self.client.get("/monitor/weather-forecast?latitude=-23.5505&longitude=-46.6333", headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertIn("forecast", response.json())
        
        # Teste de logout
        response = self.client.post("/logout", headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertIn("message", response.json())

    def test_weather_data_flow(self):
        # Teste do fluxo de dados meteorológicos
        valid_weather_data = {
            "temperature": 25.5,
            "atmospheric_pressure": 1013.25,
            "humidity": 60,
            "wind_speed": 5.5,
            "solar_radiation": 800,
            "filled_data": 0
        }
        
        # Teste de validação de dados meteorológicos
        response = self.client.post("/validate-weather-data", json=valid_weather_data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), valid_weather_data)

        # Teste de obtenção de dados meteorológicos
        response = self.client.get("/weather-data")
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json(), list)
        self.assertGreater(len(response.json()), 0)

    def test_submit_feedback(self):
        feedback_data = {
            "alert_type": "irrigation",
            "prediction_accuracy": 0.85,
            "user_observation": "A previsão de irrigação foi precisa, mas a quantidade sugerida foi um pouco alta."
        }
        response = self.client.post("/feedback/submit-feedback", json=feedback_data, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertIn("feedback_id", response.json())

    def test_get_weather_forecast(self):
        response = self.client.get("/monitor/weather-forecast?latitude=-23.5505&longitude=-46.6333")
        self.assertEqual(response.status_code, 200)
        self.assertIn("forecast", response.json())

    def test_get_market_trends(self):
        response = self.client.get("/market/trends?crop_type=soybean")
        self.assertEqual(response.status_code, 200)
        self.assertIn("current_price", response.json())
        self.assertIn("price_trend", response.json())
        self.assertIn("demand_forecast", response.json())

    def test_api_error_handling(self):
        # Configuração do token de autenticação
        login_data = {"username": "testuser", "password": "testpassword"}
        login_response = self.client.post("/token", data=login_data)
        self.assertEqual(login_response.status_code, 200)
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Teste de payload inválido para saúde da cultura
        invalid_payload = {
            "geometry": "invalid_geometry",
            "start_date": "2023-01-01",
            "end_date": "2023-01-31"
        }
        response = self.client.post("/monitor/crop-health", json=invalid_payload, headers=headers)
        self.assertEqual(response.status_code, 400)
        self.assertIn("detail", response.json())

        # Teste de datas inválidas
        invalid_date_payload = {
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[-5.0, 40.0], [-5.0, 45.0], [5.0, 45.0], [5.0, 40.0], [-5.0, 40.0]]]
            },
            "start_date": "2023-01-31",
            "end_date": "2023-01-01"
        }
        response = self.client.post("/monitor/irrigation-need", json=invalid_date_payload, headers=headers)
        self.assertEqual(response.status_code, 400)
        self.assertIn("detail", response.json())

        # Teste de dados meteorológicos inválidos
        invalid_weather_data = {
            "temperature": 1000,
            "atmospheric_pressure": -100,
            "humidity": 150,
            "wind_speed": -10,
            "solar_radiation": -500,
            "filled_data": 101
        }
        response = self.client.post("/validate-weather-data", json=invalid_weather_data, headers=headers)
        self.assertEqual(response.status_code, 400)
        self.assertIn("detail", response.json())

    def test_invalid_authentication(self):
        invalid_token = "invalid_token"
        invalid_headers = {"Authorization": f"Bearer {invalid_token}"}
        response = self.client.get("/weather-data", headers=invalid_headers)
        self.assertEqual(response.status_code, 401)
        self.assertIn("detail", response.json())

    def test_root(self):
        response = self.client.get("/")
        assert response.status_code == 200
        assert "Foguete não da ré" in response.text

    def test_health_check(self):
        response = self.client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

    def test_status_check(self):
        response = self.client.get("/status")
        assert response.status_code == 200
        assert "api_status" in response.json()
        assert "redis_status" in response.json()

    def test_invalid_token(self):
        headers = {"Authorization": "Bearer invalid_token"}
        response = self.client.get("/protected", headers=headers)
        assert response.status_code == 401

    def test_rate_limit(self):
        for _ in range(11):  # Fazer 11 requisições (limite é 10)
            response = self.client.get("/")
        assert response.status_code == 429

    def test_invalid_weather_data(self):
        invalid_data = {
            "temperature": -300,  # Temperatura inválida
            "atmospheric_pressure": 1013.25,
            "humidity": 60,
            "wind_speed": 5.5,
            "solar_radiation": 800,
            "filled_data": 0
        }
        response = self.client.post("/validate-weather-data", json=invalid_data)
        assert response.status_code == 400

    # Adicione este teste ao final do arquivo
    def test_crop_health_prediction(self):
        # Simular autenticação
        login_data = {"username": "testuser", "password": "testpassword"}
        login_response = self.client.post("/token", data=login_data)
        assert login_response.status_code == 200
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Teste de previsão de saúde da cultura
        valid_payload = {
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[-5.0, 40.0], [-5.0, 45.0], [5.0, 45.0], [5.0, 40.0], [-5.0, 40.0]]]
            },
            "start_date": "2023-01-01",
            "end_date": "2023-01-31"
        }
        response = self.client.post("/monitor/crop-health", json=valid_payload, headers=headers)
        assert response.status_code == 200
        assert "health" in response.json()

    def test_yolo_detect(self):
        # Crie uma imagem de teste
        img = Image.new('RGB', (100, 100), color = 'red')
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        response = self.client.post("/yolo/detect", files={"file": ("test.png", img_byte_arr, "image/png")})
        assert response.status_code == 200
        assert "detections" in response.json()

    def test_yolo_segment(self):
        img = Image.new('RGB', (100, 100), color = 'blue')
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        response = self.client.post("/yolo/segment", files={"file": ("test.png", img_byte_arr, "image/png")})
        assert response.status_code == 200
        assert "segmentation" in response.json()

    def test_yolo_classify(self):
        img = Image.new('RGB', (100, 100), color = 'green')
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        response = self.client.post("/yolo/classify", files={"file": ("test.png", img_byte_arr, "image/png")})
        assert response.status_code == 200
        assert "classification" in response.json()

if __name__ == '__main__':
    unittest.main()