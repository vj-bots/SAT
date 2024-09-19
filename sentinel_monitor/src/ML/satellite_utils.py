import os
import numpy as np
import rasterio
import logging

logger = logging.getLogger(__name__)

def load_satellite_images(directory):
    logger.info(f"Caminho absoluto para as imagens de satélite: {os.path.abspath(directory)}")
    if not os.path.exists(directory):
        logger.warning(f"O diretório {directory} não existe. Verifique o caminho.")
        return None

    images = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.tiff', '.tif')):
                img_path = os.path.join(root, file)
                try:
                    with rasterio.open(img_path) as src:
                        img = src.read()
                        if img.shape[0] >= 4:  # Garantir que temos pelo menos 4 bandas
                            blue, green, red, nir = img[:4]
                            ndvi = calculate_ndvi(nir, red)
                            evi = calculate_evi(nir, red, blue)
                            ndwi = calculate_ndwi(nir, green)
                            img = np.stack([blue, green, red, nir, ndvi, evi, ndwi], axis=-1)
                            img = np.transpose(img, (1, 2, 0))  # Reorganizar as dimensões
                            images.append(img)
                except Exception as e:
                    logger.error(f"Erro ao carregar a imagem {img_path}: {str(e)}")

    if not images:
        logger.warning(f"Nenhuma imagem válida encontrada em {directory}.")
        return None

    return np.array(images)

def calculate_ndvi(nir, red):
    return (nir - red) / (nir + red + 1e-8)

def calculate_evi(nir, red, blue):
    return 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))

def calculate_ndwi(nir, green):
    return (green - nir) / (green + nir + 1e-8)

satellite_images_path = os.path.join('data', '@satellite_images')
X_images = load_satellite_images(satellite_images_path)