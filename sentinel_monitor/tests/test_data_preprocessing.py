import unittest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ML.data_preprocessing import (
    normalize, calculate_ndvi, calculate_evi, calculate_ndwi,
    calculate_statistics, segment_image, detect_changes,
    create_mock_data, load_data, load_sensor_data, visualize_data_distribution
)
from src.ML.satellite_utils import load_satellite_images

class TestDataPreprocessing(unittest.TestCase):
    def setUp(self):
        self.mock_data = create_mock_data()

    def test_normalize(self):
        data = np.array([1, 2, 3, 4, 5])
        normalized = normalize(data)
        self.assertTrue(np.allclose(normalized.min(), 0))
        self.assertTrue(np.allclose(normalized.max(), 1))

    def test_calculate_ndvi(self):
        nir = np.array([0.5, 0.6, 0.7])
        red = np.array([0.1, 0.2, 0.3])
        ndvi = calculate_ndvi(nir, red)
        self.assertTrue(np.all(ndvi >= -1) and np.all(ndvi <= 1))

    def test_calculate_evi(self):
        nir = np.array([0.5, 0.6, 0.7])
        red = np.array([0.1, 0.2, 0.3])
        blue = np.array([0.05, 0.1, 0.15])
        evi = calculate_evi(nir, red, blue)
        self.assertTrue(np.all(evi >= -1) and np.all(evi <= 1))

    def test_calculate_ndwi(self):
        nir = np.array([0.5, 0.6, 0.7])
        green = np.array([0.2, 0.3, 0.4])
        ndwi = calculate_ndwi(nir, green)
        self.assertTrue(np.all(ndwi >= -1) and np.all(ndwi <= 1))

    def test_calculate_statistics(self):
        data = np.array([1, 2, 3, 4, 5])
        stats = calculate_statistics(data)
        self.assertIn('mean', stats)
        self.assertIn('median', stats)
        self.assertIn('std', stats)

    def test_segment_image(self):
        image = np.random.rand(10, 10, 3)
        segmented = segment_image(image)
        self.assertEqual(segmented.shape, image.shape[:2])

    def test_detect_changes(self):
        img1 = np.random.rand(10, 10)
        img2 = np.random.rand(10, 10)
        changes = detect_changes(img1, img2)
        self.assertEqual(changes.shape, img1.shape)

    def test_create_mock_data(self):
        (X_images, X_sensors), _ = create_mock_data()
        self.assertEqual(X_images.shape[1:], (224, 224, 7))  # Ajuste para 7 canais
        self.assertEqual(X_sensors.shape[1], 6)  # 6 features para dados meteorológicos (incluindo 'Dados Preenchidos')

    def test_load_data(self):
        (X_images, X_sensors), _ = load_data()
        self.assertIsNotNone(X_images)
        self.assertIsNotNone(X_sensors)
        self.assertEqual(X_images.shape[1:], (224, 224, 7))  # Ajuste para 7 canais
        self.assertEqual(X_sensors.shape[1], 5)  # 5 features para dados meteorológicos

    def test_load_data(self):
        (X_images, X_sensors), _ = load_data()
        self.assertIsNotNone(X_images)
        self.assertIsNotNone(X_sensors)
        self.assertEqual(X_images.shape[1:], (224, 224, 7))  # Ajuste para 7 canais
        self.assertEqual(X_sensors.shape[1], 6)  # 6 features para dados meteorológicos (incluindo 'Dados Preenchidos')

    def test_load_sensor_data(self):
        sensor_data = load_sensor_data()
        self.assertIsInstance(sensor_data, pd.DataFrame)
        self.assertGreater(sensor_data.shape[0], 0)
        self.assertEqual(sensor_data.shape[1], 6)
        self.assertTrue(all(col in sensor_data.columns for col in ['Temperatura', 'Pressão Atmosférica', 'Umidade', 'Velocidade do Vento', 'Radiação Solar', 'Dados Preenchidos']))

    def test_interpolation(self):
        # Criar dados de teste com valores ausentes
        test_data = pd.DataFrame({
            'Temperatura': [20, np.nan, 22, 23, np.nan],
            'Pressão Atmosférica': [1000, 1001, np.nan, 1003, 1004],
            'Umidade': [0.5, 0.6, 0.7, np.nan, 0.8],
            'Velocidade do Vento': [5, 6, 7, 8, np.nan],
            'Radiação Solar': [100, np.nan, 120, 130, 140]
        })
        
        # Adicionar coluna 'Dados Preenchidos'
        test_data['Dados Preenchidos'] = 0
        
        # Marcar dados ausentes antes da interpolação
        test_data.loc[test_data.isna().any(axis=1), 'Dados Preenchidos'] = 1
        
        # Interpolar valores ausentes
        test_data = test_data.interpolate(method='linear', limit_direction='both')
        
        # Verificar se todos os valores NaN foram preenchidos
        self.assertFalse(test_data.isnull().any().any())
        
        # Verificar se a coluna 'Dados Preenchidos' foi atualizada corretamente
        self.assertEqual(test_data['Dados Preenchidos'].sum(), 4)  # Deve haver 4 valores preenchidos

    def test_visualize_data_distribution(self):
        # Criar dados de teste
        test_data = pd.DataFrame({
            'Temperatura': [20, 22, 23, 21, 24],
            'Pressão Atmosférica': [1000, 1001, 1003, 1002, 1004],
            'Umidade': [0.5, 0.6, 0.7, 0.6, 0.8],
            'Velocidade do Vento': [5, 6, 7, 6, 8],
            'Radiação Solar': [100, 110, 120, 115, 130],
            'Dados Preenchidos': [0, 1, 0, 1, 0]
        })

        # Chamar a função de visualização
        visualize_data_distribution(test_data)

        # Verificar se o arquivo de imagem foi criado
        self.assertTrue(os.path.exists('data_distribution.png'))

        # Remover o arquivo de imagem após o teste
        os.remove('data_distribution.png')

    def test_load_sensor_data_with_invalid_values(self):
        # Criar um arquivo CSV temporário com dados inválidos
        import tempfile
        import csv

        with tempfile.NamedTemporaryFile(mode='w', delete=False, newline='') as temp_file:
            writer = csv.writer(temp_file)
            writer.writerow(['Temperatura', 'Pressão Atmosférica', 'Umidade', 'Velocidade do Vento', 'Radiação Solar', 'Dados Preenchidos'])
            writer.writerow(['-300', '1000', '50', '5', '800', '0'])  # Temperatura inválida
            writer.writerow(['25', '-10', '50', '5', '800', '0'])  # Pressão atmosférica inválida
            temp_file_name = temp_file.name

        # Substituir temporariamente o caminho do arquivo de dados
        import SAT.sentinel_monitor.src.ML.data_preprocessing as dp
        original_data_path = dp.DATA_PATH
        dp.DATA_PATH = temp_file_name

        try:
            # Carregar os dados e verificar se os valores inválidos foram tratados
            sensor_data = load_sensor_data()
            self.assertGreater(sensor_data.shape[0], 0)
            self.assertFalse((sensor_data['Temperatura'] < -273.15).any())
            self.assertFalse((sensor_data['Pressão Atmosférica'] < 0).any())
        finally:
            # Restaurar o caminho original do arquivo de dados
            dp.DATA_PATH = original_data_path
            # Remover o arquivo temporário
            import os
            os.unlink(temp_file_name)

if __name__ == '__main__':
    unittest.main()