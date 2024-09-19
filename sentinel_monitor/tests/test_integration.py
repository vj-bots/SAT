import unittest
import numpy as np
from fastapi.testclient import TestClient
from sentinel_monitor.src.ML.data_preprocessing import load_data, preprocess_vi_image, preprocess_sensor_data
from sentinel_monitor.src.ML.train_model import train_model, create_model
from sentinel_monitor.src.api.main import app
from sentinel_monitor.src.services.ml_service import MLService

class TestIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.ml_service = MLService()

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
        response = self.client.post("/monitor/crop-health", json=invalid_payload)
        self.assertEqual(response.status_code, 422)  # Unprocessable Entity

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
        response = self.client.post("/monitor/crop-health", json=invalid_date_payload)
        self.assertEqual(response.status_code, 422)  # Unprocessable Entity

    def test_api_full_flow(self):
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
        response = self.client.post("/monitor/crop-health", json=valid_payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn("health", response.json())

        # Teste de necessidade de irrigação
        response = self.client.post("/monitor/irrigation-need", json=valid_payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn("irrigation_need", response.json())

        # Teste de detecção de pragas
        response = self.client.post("/monitor/pest-detection", json=valid_payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn("pest_detected", response.json())

        # Teste de previsão de rendimento
        response = self.client.post("/monitor/yield-prediction", json=valid_payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn("yield_prediction", response.json())

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

if __name__ == '__main__':
    unittest.main()