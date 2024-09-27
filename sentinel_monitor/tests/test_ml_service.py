import os
import sys
import time
import asyncio
import unittest
import numpy as np

from unittest.mock import MagicMock, patch

# Adicione o diretório raiz do projeto ao PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from sentinel_monitor.src.services.ml_service import MLService
from sentinel_monitor.src.ML.ml_models import train_crop_health_model, train_irrigation_model, train_pest_detection_model, train_yield_prediction_model
from SAT.sentinel_monitor.src.ML.data_preprocessing import preprocess_vi_image, preprocess_sensor_data, create_mock_data

class TestMLService(unittest.TestCase):
    async def asyncSetUp(self):
        await asyncio.sleep(0)  # Permite que o loop de eventos seja iniciado

    @classmethod
    def setUpClass(cls):
        asyncio.run(cls.asyncSetUpClass())

    @classmethod
    async def asyncSetUpClass(cls):
        with patch('SAT.sentinel_monitor.src.services.ml_service.MLService.load_models'):
            cls.ml_service = MLService()
        cls.mock_data = create_mock_data()
        
        # Criar modelos mock
        cls.ml_service.crop_health_model = MagicMock()
        cls.ml_service.irrigation_model = MagicMock()
        cls.ml_service.pest_detection_model = MagicMock()
        cls.ml_service.yield_prediction_model = MagicMock()
        
        # Configurar retornos mock para os modelos
        cls.ml_service.crop_health_model.predict.return_value = np.array([1])
        cls.ml_service.irrigation_model.predict.return_value = np.array([0.5])
        cls.ml_service.pest_detection_model.predict.return_value = np.array([0])
        cls.ml_service.yield_prediction_model.predict.return_value = np.array([100.0])

    def test_crop_health_prediction(self):
        X_images, _ = self.mock_data[0]
        processed_image = preprocess_vi_image(X_images[0])
        prediction = self.ml_service.predict_crop_health(processed_image)
        self.assertIsInstance(prediction, dict)
        self.assertIn(prediction['health'], ['Saudável', 'Não Saudável'])
        self.assertIsInstance(prediction, float)
        self.assertGreaterEqual(prediction, 0)
        _, X_sensors = self.mock_data[0]
        processed_sensor_data = preprocess_sensor_data(X_sensors[0])
        prediction = self.ml_service.predict_irrigation_need(processed_sensor_data)
        self.assertIsInstance(prediction, float)
        self.assertGreaterEqual(prediction, 0)
        processed_image = preprocess_vi_image(X_images[0])

    def test_irrigation_prediction(self):
        _, X_sensors = self.mock_data[0]
        processed_sensor_data = preprocess_sensor_data(X_sensors[0])
        prediction = self.ml_service.predict_irrigation_need(processed_sensor_data)
        self.assertIsInstance(prediction, float)
        self.assertGreaterEqual(prediction, 0)

    def test_pest_detection(self):
        X_images, _ = self.mock_data[0]
        processed_image = preprocess_vi_image(X_images[0])
        prediction = self.ml_service.detect_pests(processed_image)
        self.assertIn(prediction, ['Pragas Detectadas', 'Sem Pragas'])
        processed_image = preprocess_vi_image(X_images[0])

    async def test_yield_prediction(self):
        X_images, _ = self.mock_data[0]
        processed_image = preprocess_vi_image(X_images[0])
        prediction = await self.ml_service.predict_yield(processed_image)
        self.assertIsInstance(prediction, dict)
        self.assertIn('yield', prediction)
        self.assertIn('value', prediction)
        self.assertIsInstance(prediction['value'], float)
        self.assertGreater(prediction['value'], 0)

    def test_model_loading(self):
        self.assertIsNotNone(self.ml_service.crop_health_model)
        self.assertIsNotNone(self.ml_service.irrigation_model)
        self.assertIsNotNone(self.ml_service.pest_detection_model)
        self.assertIsNotNone(self.ml_service.yield_prediction_model)

    async def test_performance(self):
        X_images, X_sensors = self.mock_data[0]
        processed_image = preprocess_vi_image(X_images[0])
        processed_sensor_data = preprocess_sensor_data(X_sensors[0])
        start_time = time.time()
        await self.ml_service.predict(processed_image, processed_sensor_data)
        end_time = time.time()
        total_time = end_time - start_time
        self.assertLess(total_time, 1.0)  # Ajuste este valor conforme necessário

    async def test_predict_method(self):
        X_images, X_sensors = self.mock_data[0]
        processed_image = preprocess_vi_image(X_images[0])
        processed_sensor_data = preprocess_sensor_data(X_sensors[0])
        prediction = await self.ml_service.predict(processed_image, processed_sensor_data)
        self.assertIsInstance(prediction, dict)
        self.assertIn('health', prediction)
        self.assertIn('irrigation', prediction)
        self.assertIn('pest', prediction)
        self.assertIn('yield', prediction)

    def test_predict_harvest_time(self):
        X_images, _ = self.mock_data[0]
        processed_image = preprocess_vi_image(X_images)
        prediction = self.ml_service.predict_harvest_time(processed_image)
        self.assertIsInstance(prediction, str)
        self.assertIn(prediction, [
            "A colheita pode ser realizada nos próximos 7-14 dias.",
            "A colheita pode ser realizada em 2-4 semanas.",
            "A cultura ainda não está pronta para colheita. Reavalie em 4-6 semanas."
        ])

    def test_analyze_time_series(self):
        X_images, _ = self.mock_data[0]
        processed_image = preprocess_vi_image(X_images)
        analysis = self.ml_service.analyze_time_series(processed_image)
        self.assertIsInstance(analysis, dict)
        self.assertIn('mean_ndvi', analysis)
        self.assertIsInstance(analysis['mean_ndvi'], list)

if __name__ == '__main__':
    asyncio.run(unittest.main())