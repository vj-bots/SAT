import unittest
import numpy as np
from sentinel_monitor.src.ML.ml_models import (
    train_crop_health_model,
    train_irrigation_model,
    train_pest_detection_model,
    train_yield_prediction_model,
    predict_crop_health,
    predict_irrigation
)

class TestMLModels(unittest.TestCase):
    def setUp(self):
        self.X_images = np.random.rand(100, 2, 2, 4)
        self.X_sensors = np.random.rand(100, 5)
        self.y_health = np.random.randint(0, 2, 100)
        self.y_irrigation = np.random.rand(100)
        self.y_invasion = np.random.randint(0, 2, 100)
        self.y_yield = np.random.rand(100)
        self.mock_data = (np.random.rand(10, 224, 224, 3), np.random.rand(10, 5))

    def test_train_crop_health_model(self):
        X_images_flat = self.X_images.reshape(self.X_images.shape[0], -1)
        model = train_crop_health_model(X_images_flat, self.y_health, model_type='classifier')
        self.assertIsNotNone(model)

    def test_train_irrigation_model(self):
        model = train_irrigation_model(self.X_sensors, self.y_irrigation, model_type='regressor')
        self.assertIsNotNone(model)

    def test_train_pest_detection_model(self):
        X_images_flat = self.X_images.reshape(self.X_images.shape[0], -1)
        model = train_pest_detection_model(X_images_flat, self.y_invasion, model_type='classifier')
        self.assertIsNotNone(model)

    def test_train_yield_prediction_model(self):
        X_images_flat = self.X_images.reshape(self.X_images.shape[0], -1)
        model = train_yield_prediction_model(X_images_flat, self.y_yield, model_type='regressor')
        self.assertIsNotNone(model)

    def test_predict_crop_health(self):
        X_images, _ = self.mock_data[0]
        predictions = self.ml_service.predict_crop_health(X_images)
        self.assertEqual(len(predictions), len(X_images))
        for prediction in predictions:
            self.assertIn(prediction, ['Saudável', 'Não Saudável'])

    def test_predict_irrigation(self):
        model = train_irrigation_model(self.X_sensors, self.y_irrigation, model_type='regressor')
        predictions = predict_irrigation(model, self.X_sensors)
        self.assertEqual(len(predictions), len(self.y_irrigation))

if __name__ == '__main__':
    unittest.main()