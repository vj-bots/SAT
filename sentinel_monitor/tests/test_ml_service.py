import unittest
import numpy as np
import time
from sentinel_monitor.src.services.ml_service import MLService
from sentinel_monitor.src.ML.ml_models import train_crop_health_model, train_irrigation_model, train_pest_detection_model, train_yield_prediction_model
from sentinel_monitor.src.ML.data_preprocessing import preprocess_vi_image, preprocess_sensor_data, create_mock_data

class TestMLService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ml_service = MLService()
        cls.mock_data = create_mock_data()
        # Treine os modelos com os dados de mock
        (X_images, X_sensors), (y_irrigation, y_invasion, y_health, y_yield) = cls.mock_data
        X_images_flat = X_images.reshape(X_images.shape[0], -1)
        cls.ml_service.crop_health_model = train_crop_health_model(X_images_flat, y_health, model_type='classifier')
        cls.ml_service.irrigation_model = train_irrigation_model(X_sensors, y_irrigation, model_type='regressor')
        cls.ml_service.pest_detection_model = train_pest_detection_model(X_images_flat, y_invasion, model_type='classifier')
        cls.ml_service.yield_prediction_model = train_yield_prediction_model(X_images_flat, y_yield, model_type='regressor')

    def test_crop_health_prediction(self):
        X_images, _ = self.mock_data[0]
        processed_image = preprocess_vi_image(X_images[0])
        prediction = self.ml_service.predict_crop_health(processed_image)
        self.assertIsInstance(prediction, dict)
        self.assertIn(prediction['health'], ['Saudável', 'Não Saudável'])

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

    def test_yield_prediction(self):
        X_images, _ = self.mock_data[0]
        processed_image = preprocess_vi_image(X_images[0])
        prediction = self.ml_service.predict_yield(processed_image)
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

    def test_performance(self):
        X_images, X_sensors = self.mock_data[0]
        processed_image = preprocess_vi_image(X_images[0])
        processed_sensor_data = preprocess_sensor_data(X_sensors[0])
        start_time = time.time()
        self.ml_service.predict(processed_image, processed_sensor_data)
        end_time = time.time()
        total_time = end_time - start_time
        self.assertLess(total_time, 1.0)  # Ajuste este valor conforme necessário

    def test_predict_method(self):
        X_images, X_sensors = self.mock_data[0]
        processed_image = preprocess_vi_image(X_images[0])
        processed_sensor_data = preprocess_sensor_data(X_sensors[0])
        prediction = self.ml_service.predict(processed_image, processed_sensor_data)
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
    unittest.main()