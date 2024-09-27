import os
import sys
import unittest
import numpy as np
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
    def setUpClass(cls):
        cls.client = client
        cls.ml_service = MLService()
        cls.headers = {"Authorization": "Bearer test_token"}
        # Simular login e obter token real, se necessário
        login_data = {"username": "testuser", "password": "testpassword"}
        response = cls.client.post("/token", data=login_data)
        if response.status_code == 200:
            cls.headers["Authorization"] = f"Bearer {response.json()['access_token']}"

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
        sample_image = np.expand_dims(sample_image, axis=1)  # Adiciona uma dimensão extra
        sample_sensor = preprocess_sensor_data(X_sensors[0:1])
        predictions = model.predict([sample_image, sample_sensor])
        
        # Verifica se as previsões têm o formato esperado
        self.assertEqual(len(predictions), 4)  # 4 saídas: irrigação, invasão, saúde e rendimento
        for pred in predictions:
            self.assertEqual(pred.shape[0], 1)  # Uma previsão para cada amostra

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
        response = self.client.post("/monitor/irrigation_advice", json=valid_payload, headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertIn("message", response.json())

        # Teste de detecção de pragas
        response = self.client.post("/monitor/pest_detection", json=valid_payload, headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertIn("message", response.json())

        # Teste de previsão de rendimento
        response = self.client.post("/monitor/yield-prediction", json={"image_data": "mock_image_data"}, headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertIn("yield", response.json())

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
            "filled_data": 2
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

if __name__ == '__main__':
    unittest.main()