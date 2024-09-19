import os
import sys
import rasterio
import pandas as pd
import numpy as np
from PIL import Image
from numba import jit
import sentinelhub
from sentinelhub import SHConfig, DataCollection, SentinelHubRequest, BBox, CRS, MimeType
from .satellite_utils import load_satellite_images
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

def get_data_path(filename):
    return os.path.abspath(os.path.join(get_project_root(), 'data', filename))

def read_vi_image(file_path):
    with rasterio.open(file_path) as src:
        ndvi = src.read(1)  # Assumindo que a banda 1 seja o NDVI
        lai = src.read(2)   # Assumindo que a banda 2 seja o LAI
        fapar = src.read(3) # Banda 3 FAPAR
        ppi = src.read(4)   # Banda 4 PPI
    return ndvi, lai, fapar, ppi

def normalize(array):
    min_val = np.min(array)
    max_val = np.max(array)
    return (array - min_val) / (max_val - min_val + 1e-8)

def preprocess_vi_image(image_with_moisture):
    image = image_with_moisture[:-1].reshape(-1, 224, 224, 7)  # Assumindo imagens 224x224 com 7 canais
    moisture = image_with_moisture[-1]
    
    processed_images = []
    for img in image:
        ndvi = calculate_ndvi(img[:,:,3], img[:,:,2])  # NIR e Red
        evi = calculate_evi(img[:,:,3], img[:,:,2], img[:,:,0])  # NIR, Red e Blue
        ndwi = calculate_ndwi(img[:,:,3], img[:,:,1])  # NIR e Green
        
        processed_img = np.dstack((img, ndvi, evi, ndwi, np.full_like(ndvi, moisture)))
        processed_images.append(processed_img)
    
    return np.array(processed_images)

def preprocess_sensor_data(sensor_data):
    if sensor_data.ndim == 1:
        return sensor_data.reshape(1, -1)
    elif sensor_data.ndim == 2:
        return sensor_data
    else:
        raise ValueError("Formato de dados de sensor inválido")

def load_soil_moisture_data(file_path):
    df = pd.read_csv(file_path)
    return df['moisture'].values.reshape(-1, 1)

def load_irrigation_labels(file_path):
    df = pd.read_csv(file_path)
    return df['irrigation'].values

def load_invasion_labels(file_path):
    df = pd.read_csv(file_path)
    return df['invasion'].values

def load_health_labels(file_path):
    df = pd.read_csv(file_path)
    return df['health'].values

def load_yield_labels(file_path):
    df = pd.read_csv(file_path)
    return df['yield'].values

def load_and_preprocess_images(directory):
    images = []
    labels = []
    for label_dir in os.listdir(directory):
        label_path = os.path.join(directory, label_dir)
        if os.path.isdir(label_path):
            for image_name in os.listdir(label_path):
                image_path = os.path.join(label_path, image_name)
                processed_image = preprocess_vi_image(image_path)
                images.append(processed_image)
                labels.append(label_dir)
    return np.array(images), labels

def load_sentinel2_data(bbox, time_interval, config):
    request = SentinelHubRequest(
        data_folder="sentinel2_data",
        evalscript="""
            //VERSION=3
            function setup() {
                return {
                    input: ["B02", "B03", "B04", "B08"],
                    output: { bands: 4 }
                };
            }

            function evaluatePixel(sample) {
                return [sample.B02, sample.B03, sample.B04, sample.B08];
            }
        """,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L1C,
                time_interval=time_interval,
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        config=config
    )
    return request.get_data()

def load_modis_data(file_path):
    with rasterio.open(file_path) as src:
        return src.read()

def load_soil_data(file_path):
    return pd.read_csv(file_path)

def load_gpm_data(file_path):
    with rasterio.open(file_path) as src:
        return src.read()

def load_cropland_data(file_path):
    return pd.read_csv(file_path)

def process_sentinel2_data(data):
    blue = data[0][:,:,0]
    green = data[0][:,:,1]
    red = data[0][:,:,2]
    nir = data[0][:,:,3]

    ndvi = calculate_ndvi(nir, red)
    evi = calculate_evi(nir, red, blue)
    ndwi = calculate_ndwi(nir, green)
    
    processed_data = np.stack([blue, green, red, nir, ndvi, evi, ndwi], axis=-1)
    
    return processed_data

@jit(nopython=True)
def calculate_ndvi(nir, red):
    return (nir - red) / (nir + red + 1e-8)

@jit(nopython=True)
def calculate_evi(nir, red, blue):
    return 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))

@jit(nopython=True)
def calculate_ndwi(nir, green):
    return (green - nir) / (green + nir + 1e-8)

def calculate_statistics(array):
    if array.size == 0:
        return {
            "mean": 0,
            "median": 0,
            "std": 0,
            "min": 0,
            "max": 0
        }
    return {
        "mean": np.mean(array),
        "median": np.median(array),
        "std": np.std(array),
        "min": np.min(array),
        "max": np.max(array)
    }

def extract_features_from_sentinel2(image):
    blue, green, red, nir = image[0], image[1], image[2], image[3]
    
    ndvi = calculate_ndvi(nir, red)
    evi = calculate_evi(nir, red, blue)
    ndwi = calculate_ndwi(nir, green)
    
    features = [
        np.mean(ndvi),
        np.mean(evi),
        np.mean(ndwi),
        np.std(ndvi),
        np.std(evi),
        np.std(ndwi)
    ]
    
    return features

def segment_image(image, n_clusters=5):
    reshaped = image.reshape((-1, image.shape[-1]))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(reshaped)
    return labels.reshape(image.shape[:-1])

def detect_changes(img1, img2):
    return np.abs(img1 - img2)

def visualize_index(index, title):
    plt.figure(figsize=(10, 10))
    plt.imshow(index, cmap='RdYlGn')
    plt.colorbar(label=title)
    plt.title(title)
    plt.axis('off')
    plt.savefig(f"{title.lower().replace(' ', '_')}.png")
    plt.close()

def process_sentinel2_image(image):
    blue, green, red, nir = image[0], image[1], image[2], image[3]
    
    ndvi = calculate_ndvi(nir, red)
    evi = calculate_evi(nir, red, blue)
    ndwi = calculate_ndwi(nir, green)
    
    segmented = segment_image(np.dstack([ndvi, evi, ndwi]))
    
    visualize_index(ndvi, "NDVI")
    visualize_index(evi, "EVI")
    visualize_index(ndwi, "NDWI")
    visualize_index(segmented, "Segmented Image")
    
    return {
        "ndvi_stats": calculate_statistics(ndvi),
        "evi_stats": calculate_statistics(evi),
        "ndwi_stats": calculate_statistics(ndwi)
    }

def process_sentinel1_data(file_path):
    with rasterio.open(file_path) as src:
        img = src.read()
    
    img_normalized = normalize(img)
    
    backscatter = 10 * np.log10(img_normalized)
    
    visualize_index(backscatter[0], "Backscatter VV")
    visualize_index(backscatter[1], "Backscatter VH")

    return {
        "backscatter_vv_mean": np.mean(backscatter[0]),
        "backscatter_vh_mean": np.mean(backscatter[1])
    }

def preprocess_landsat89_data(file_path):
    with rasterio.open(file_path) as src:
        blue = src.read(2)
        green = src.read(3)
        red = src.read(4)
        nir = src.read(5)
    
    ndvi = calculate_ndvi(nir, red)
    evi = calculate_evi(nir, red, blue)
    ndwi = calculate_ndwi(nir, green)
    
    return np.stack([blue, green, red, nir, ndvi, evi, ndwi], axis=-1)

def preprocess_copernicus_land_data(file_path):
    with rasterio.open(file_path) as src:
        data = src.read(1)
    
    return normalize(data)

def save_results(results, filename):
    import json
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)

def load_and_preprocess_data(bbox, time_interval, config):
    logger.info(f"Carregando dados do Sentinel-2 para bbox: {bbox} e intervalo de tempo: {time_interval}")
    raw_data = load_sentinel2_data(bbox, time_interval, config)
    logger.info("Dados do Sentinel-2 carregados com sucesso")
    
    logger.info("Iniciando pré-processamento dos dados")
    processed_data = process_sentinel2_data(raw_data)
    logger.info("Pré-processamento concluído")
    
    return processed_data

def validate_data(data):
    if data is None or data.size == 0:
        raise ValueError("Os dados de entrada estão vazios ou são nulos.")
    
    if np.isnan(data).any():
        raise ValueError("Os dados contêm valores NaN.")
    
    if np.isinf(data).any():
        raise ValueError("Os dados contêm valores infinitos.")
    
    logger.info("Validação de dados concluída com sucesso.")

def preprocess_landsat7_data(file_path):
    with rasterio.open(file_path) as src:
        blue = src.read(1)
        green = src.read(2)
        red = src.read(3)
        nir = src.read(4)
    
    ndvi = calculate_ndvi(nir, red)
    evi = calculate_evi(nir, red, blue)
    ndwi = calculate_ndwi(nir, green)
    
    return np.stack([blue, green, red, nir, ndvi, evi, ndwi], axis=-1)

def create_mock_data():
    X_images = np.random.rand(100, 224, 224, 7)  # 7 canais (4 originais + 3 índices)
    X_sensors = np.random.rand(100, 6)  # 6 features (5 originais + 1 'Dados Preenchidos')
    y_irrigation = np.random.rand(100)
    y_invasion = np.random.randint(0, 2, 100)
    y_health = np.random.randint(0, 2, 100)
    y_yield = np.random.rand(100) * 100
    return (X_images, X_sensors), (y_irrigation, y_invasion, y_health, y_yield)

def get_data_path(filename):
    return os.path.join(get_project_root(), 'data', filename)
    
    if not os.path.exists(weather_data_path):
        logger.warning(f"O diretório {weather_data_path} não existe. Retornando dados simulados.")
        return pd.DataFrame(np.random.rand(100, 6), columns=['Temperatura', 'Pressão Atmosférica', 'Umidade', 'Velocidade do Vento', 'Radiação Solar', 'Dados Preenchidos'])
    
    for state in os.listdir(weather_data_path):
        state_path = os.path.join(weather_data_path, state)
        if os.path.isdir(state_path):
            for file in os.listdir(state_path):
                if file.endswith('.csv'):
                    file_path = os.path.join(state_path, file)
                    try:
                        df = pd.read_csv(file_path, encoding='latin1', skiprows=1)
                        logger.info(f"Colunas encontradas no arquivo {file_path}: {df.columns.tolist()}")
                        
                        df = df.iloc[:, 5:10]
                        df.columns = ['Temperatura', 'Pressão Atmosférica', 'Umidade', 'Velocidade do Vento', 'Radiação Solar']
                        
                        df = df[pd.to_numeric(df['Temperatura'], errors='coerce').notnull()]
                        df = df.astype(float)
                        
                        df['Dados Preenchidos'] = 0
                        df.loc[df.isna().any(axis=1), 'Dados Preenchidos'] = 1
                        
                        df = df.interpolate(method='linear', limit_direction='both')
                        
                        all_data.append(df)
                    except Exception as e:
                        logger.error(f"Erro ao carregar o arquivo {file_path}: {str(e)}")
    
    if not all_data:
        logger.warning("Nenhum dado meteorológico encontrado. Retornando dados simulados.")
        return pd.DataFrame(np.random.rand(100, 6), columns=['Temperatura', 'Pressão Atmosférica', 'Umidade', 'Velocidade do Vento', 'Radiação Solar', 'Dados Preenchidos'])
    
    combined_data = pd.concat(all_data, ignore_index=True)
    logger.info(f"Forma final dos dados combinados: {combined_data.shape}")
    return combined_data

def combine_image_moisture_data(images, sensor_data):
    combined_data = []
    for i, image in enumerate(images):
        if i < len(sensor_data):
            moisture = sensor_data.iloc[i]['Umidade']
            flattened_image = image.reshape(-1)
            combined = np.append(flattened_image, moisture)
            combined_data.append(combined)
    return np.array(combined_data)

def load_irrigation_data():
    irrigation_data_path = get_data_path('irrigation_data.csv')
    try:
        df = pd.read_csv(irrigation_data_path)
        return df['irrigation'].values
    except FileNotFoundError:
        logger.warning(f"Arquivo de dados de irrigação não encontrado: {irrigation_data_path}")
        return np.random.rand(100)  # Retorna dados simulados

def load_invasion_data():
    invasion_data_path = get_data_path('invasion_data.csv')
    try:
        df = pd.read_csv(invasion_data_path)
        return df['invasion'].values
    except FileNotFoundError:
        logger.warning(f"Arquivo de dados de invasão não encontrado: {invasion_data_path}")
        return np.random.randint(0, 2, 100)  # Retorna dados simulados

def load_health_data():
    health_data_path = get_data_path('health_data.csv')
    try:
        df = pd.read_csv(health_data_path)
        return df['health'].values
    except FileNotFoundError:
        logger.warning(f"Arquivo de dados de saúde não encontrado: {health_data_path}")
        return np.random.randint(0, 2, 100)  # Retorna dados simulados

def load_yield_data():
    yield_data_path = get_data_path('yield_data.csv')
    try:
        df = pd.read_csv(yield_data_path)
        return df['yield'].values
    except FileNotFoundError:
        logger.warning(f"Arquivo de dados de rendimento não encontrado: {yield_data_path}")
        return np.random.rand(100) * 100  # Retorna dados simulados

def load_sensor_data():
    weather_data_path = get_data_path('weather_data')
    all_data = []
    
    if not os.path.exists(weather_data_path):
        logger.warning(f"O diretório {weather_data_path} não existe. Retornando dados simulados.")
        return pd.DataFrame(np.random.rand(100, 6), columns=['Temperatura', 'Pressão Atmosférica', 'Umidade', 'Velocidade do Vento', 'Radiação Solar', 'Dados Preenchidos'])
    
    for state in os.listdir(weather_data_path):
        state_path = os.path.join(weather_data_path, state)
        if os.path.isdir(state_path):
            for file in os.listdir(state_path):
                if file.endswith('.csv'):
                    file_path = os.path.join(state_path, file)
                    try:
                        df = pd.read_csv(file_path, encoding='latin1', skiprows=1)
                        logger.info(f"Colunas encontradas no arquivo {file_path}: {df.columns.tolist()}")
                        
                        df = df.iloc[:, 5:10]
                        df.columns = ['Temperatura', 'Pressão Atmosférica', 'Umidade', 'Velocidade do Vento', 'Radiação Solar']
                        
                        df = df[pd.to_numeric(df['Temperatura'], errors='coerce').notnull()]
                        df = df.astype(float)
                        
                        df['Dados Preenchidos'] = 0
                        df.loc[df.isna().any(axis=1), 'Dados Preenchidos'] = 1
                        
                        df = df.interpolate(method='linear', limit_direction='both')
                        
                        all_data.append(df)
                    except Exception as e:
                        logger.error(f"Erro ao carregar o arquivo {file_path}: {str(e)}")
    
    if not all_data:
        logger.warning("Nenhum dado meteorológico encontrado. Retornando dados simulados.")
        return pd.DataFrame(np.random.rand(100, 6), columns=['Temperatura', 'Pressão Atmosférica', 'Umidade', 'Velocidade do Vento', 'Radiação Solar', 'Dados Preenchidos'])
    
    combined_data = pd.concat(all_data, ignore_index=True)
    combined_data = combined_data.dropna()  # Remove qualquer linha com NaN remanescente
    logger.info(f"Forma final dos dados combinados: {combined_data.shape}")
    visualize_data_distribution(combined_data)
    return combined_data


def visualize_data_distribution(data):
    import matplotlib
    matplotlib.use('Agg')  # Use o backend 'Agg' que não requer GUI
    import matplotlib.pyplot as plt
    
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    fig.suptitle('Distribuição dos Dados Meteorológicos')
    
    columns = ['Temperatura', 'Pressão Atmosférica', 'Umidade', 'Velocidade do Vento', 'Radiação Solar']
    
    for i, column in enumerate(columns):
        ax = axs[i // 2, i % 2]
        original_data = data[data['Dados Preenchidos'] == 0][column].dropna()
        filled_data = data[data['Dados Preenchidos'] == 1][column].dropna()
        
        if not original_data.empty:
            ax.hist(original_data, bins=50, alpha=0.5, label='Dados Originais')
        if not filled_data.empty:
            ax.hist(filled_data, bins=50, alpha=0.5, label='Dados Preenchidos')
        ax.set_title(column)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('data_distribution.png')
    plt.close(fig)

def load_data():
    try:
        # Carregando imagens de satélite
        satellite_images_path = get_data_path('satellite_images')
        logger.info(f"Tentando carregar imagens de satélite de: {satellite_images_path}")
        X_images = load_satellite_images(satellite_images_path)
        
        if X_images is None:
            logger.warning("Usando dados de imagem simulados para testes.")
            X_images = np.random.rand(100, 224, 224, 7)
        else:
            logger.info(f"Imagens de satélite carregadas com sucesso. Forma: {X_images.shape}")
        
        # Carregando dados de sensores
        X_sensors = load_sensor_data()

        # Carregando dados de rótulos
        y_irrigation = load_irrigation_data()
        y_invasion = load_invasion_data()
        y_health = load_health_data()
        y_yield = load_yield_data()

        # Garantindo que todos os dados tenham o mesmo número de amostras
        min_samples = min(len(X_images), len(X_sensors), len(y_irrigation), len(y_invasion), len(y_health), len(y_yield))
        X_images = X_images[:min_samples]
        X_sensors = X_sensors[:min_samples]
        y_irrigation = y_irrigation[:min_samples]
        y_invasion = y_invasion[:min_samples]
        y_health = y_health[:min_samples]
        y_yield = y_yield[:min_samples]

        logger.info(f"Forma dos dados de imagem: {X_images.shape}")
        logger.info(f"Forma dos dados de sensores: {X_sensors.shape}")
        logger.info("Todos os dados foram carregados com sucesso.")
        
        return (X_images, X_sensors), (y_irrigation, y_invasion, y_health, y_yield)
    
    except Exception as e:
        logger.error(f"Erro ao carregar dados: {str(e)}")
        logger.warning("Usando dados sintéticos para testes.")
        return create_mock_data()