import os
os.environ['TESTING'] = 'True'
import unittest
import numpy as np
from sentinel_monitor.src.ML.train_model import train_model
from sentinel_monitor.src.ML.data_preprocessing import create_mock_data
from sentinel_monitor.src.ML.ml_models import (
    train_crop_health_model,
    train_irrigation_model,
    train_pest_detection_model,
    train_yield_prediction_model
)
import tensorflow as tf

class TestTrainModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mock_data = create_mock_data()
        (cls.X_images, cls.X_sensors), (cls.y_irrigation, cls.y_invasion, cls.y_health, cls.y_yield) = cls.mock_data

    def test_train_model(self):
        model, history = train_model()
        self.assertIsNotNone(model)
        self.assertIsNotNone(history)
        self.assertTrue(isinstance(model, tf.keras.Model))
        self.assertTrue(isinstance(history, tf.keras.callbacks.History))

    def test_train_crop_health_model(self):
        X_images_flat = self.X_images.reshape(self.X_images.shape[0], -1)
        model = train_crop_health_model(X_images_flat, self.y_health, model_type='classifier')
        self.assertIsNotNone(model)
        predictions = model.predict(X_images_flat)
        self.assertEqual(len(predictions), len(self.y_health))

    def test_train_irrigation_model(self):
        model = train_irrigation_model(self.X_sensors, self.y_irrigation, model_type='regressor')
        self.assertIsNotNone(model)
        predictions = model.predict(self.X_sensors)
        self.assertEqual(len(predictions), len(self.y_irrigation))

    def test_train_pest_detection_model(self):
        X_images_flat = self.X_images.reshape(self.X_images.shape[0], -1)
        model = train_pest_detection_model(X_images_flat, self.y_invasion, model_type='classifier')
        self.assertIsNotNone(model)
        predictions = model.predict(X_images_flat)
        self.assertEqual(len(predictions), len(self.y_invasion))

    def test_train_yield_prediction_model(self):
        X_images_flat = self.X_images.reshape(self.X_images.shape[0], -1)
        model, poly_features = train_yield_prediction_model(X_images_flat, self.y_yield, model_type='regressor')
        self.assertIsNotNone(model)
        self.assertIsNotNone(poly_features)
        X_poly = poly_features.transform(X_images_flat)
        predictions = model.predict(X_poly)
        self.assertEqual(len(predictions), len(self.y_yield))

if __name__ == '__main__':
    unittest.main()