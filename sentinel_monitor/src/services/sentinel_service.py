import numpy as np
import requests
import os
from sentinelsat import SentinelAPI
from fastapi_cache.decorator import cache
from fastapi import HTTPException
from datetime import datetime
from dotenv import load_dotenv
from sentinelhub import SHConfig
from .ml_service import MLService
from .soil_moisture_sensor_service import SoilMoistureSensor
from ..utils.logging_utils import setup_logger
from typing import Dict, Any

logger = setup_logger(__name__)

load_dotenv()

config = SHConfig()
config.instance_id = os.getenv('INSTANCE_ID')
config.sh_client_id = os.getenv('CLIENT_ID')
config.sh_client_secret = os.getenv('CLIENT_SECRET')

class SentinelService:
    def __init__(self):
        self.client_id = os.getenv('CLIENT_ID')
        self.client_secret = os.getenv('CLIENT_SECRET')
        self.instance_id = os.getenv('INSTANCE_ID')
        self.base_url = os.getenv('BASE_URL')
        self.soil_sensor = SoilMoistureSensor(analog_pin=0, digital_pin=17)
        self.sentinel_api = SentinelAPI('user', 'password', 'https://scihub.copernicus.eu/dhus')
        self.ml_service = MLService()

    def get_crop_growth_stage(self, geometry, start_date, end_date):
        stages = ["Germinação", "Desenvolvimento", "Floração", "Frutificação", "Maturação"]
        return np.random.choice(stages)

    def get_crop_type_classification(self, geometry, start_date, end_date):
        crop_types = ["Milho", "Soja", "Trigo", "Arroz", "Algodão"]
        return np.random.choice(crop_types)

    def get_crop_water_stress(self, coordinates):
        stress_levels = ["Baixo", "Médio", "Alto"]
        return {"stress_level": np.random.choice(stress_levels)}

    def get_crop_nutrient_deficiency(self, geometry, start_date, end_date):
        nutrients = ["Nitrogênio", "Fósforo", "Potássio", "Nenhuma"]
        deficiencies = np.random.choice(nutrients, size=np.random.randint(0, 3), replace=False)
        return {"deficiencies": list(deficiencies)}

    def get_field_boundary_detection(self, coordinates):
        return [[np.random.uniform(-5, 5), np.random.uniform(40, 45)] for _ in range(5)]

    def get_soil_moisture(self, geometry, start_date, end_date):
        try:
            sentinel1_data = self.get_sentinel1_data(geometry, start_date, end_date)
            # implementar logica para detecção de umidade
            return {"moisture_level": "medium", "value": 0.5}
        except Exception as e:
            logger.error(f"Erro ao obter umidade do solo: {str(e)}")
            raise

    def get_land_use_classification(self, coordinates, start_date, end_date):
        land_uses = ["Agricultura", "Floresta", "Urbano", "Água"]
        return {"classification": np.random.choice(land_uses)}

    def map_area(self, coordinates):
        return {"map": "Mapa da área simulado"}

    def get_soil_erosion_risk(self, coordinates):
        risk_levels = ["Baixo", "Médio", "Alto"]
        return {"risk_level": np.random.choice(risk_levels)}

    def get_vegetation_index_time_series(self, geometry, start_date, end_date, index_type="NDVI"):
        return [{"date": f"2023-01-{i:02d}", "value": np.random.uniform(0, 1)} for i in range(1, 32)]

    def get_crop_yield_prediction(self, geometry, start_date, end_date, crop_type=None):
        return {"predicted_yield": np.random.uniform(50, 100)}

    def get_biomass_estimation(self, geometry, start_date, end_date):
        return {"biomass": np.random.uniform(0, 1000)}

    def get_crop_stress_detection(self, coordinates, start_date, end_date):
        return {"stress_level": np.random.choice(["Baixo", "Médio", "Alto"])}

    def get_ndvi_time_series(self, geometry, start_date, end_date):
        return [{"date": f"2023-01-{i:02d}", "ndvi": np.random.uniform(0, 1)} for i in range(1, 32)]

    def download_biomass_image(self, coordinates, start_date, end_date):
        return "http://example.com/biomassa_image.png"

    def get_crop_yield_estimation(self, geometry, start_date, end_date):
        return 75.5

    def get_access_token(self):
        url = "https://services.sentinel-hub.com/oauth/token"
        payload = {
            'grant_type': 'client_credentials',
            'client_id': self.client_id,
            'client_secret': self.client_secret
        }
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        response = requests.post(url, data=payload, headers=headers)
        response.raise_for_status()
        return response.json().get('access_token')

    def _process_request(self, geometry, start_date, end_date, evalscript, data_collection):
        if os.getenv('TEST_MODE') == 'True':
            return "test_data"
        try:
            token = self.get_access_token()
            payload = {
                "input": {
                    "bounds": {
                        "geometry": geometry
                    },
                    "data": [
                        {
                            "type": data_collection,
                            "dataFilter": {
                                "timeRange": {
                                    "from": start_date,
                                    "to": end_date
                                }
                            }
                        }
                    ]
                },
                "evalscript": evalscript,
                "output": {
                    "width": 512,
                    "height": 512,
                    "responses": [
                        {
                            "identifier": "default",
                            "format": {
                                "type": "image/png"
                            }
                        }
                    ]
                }
            }
            headers = {
                'Authorization': f'Bearer {token}',
                'Content-Type': 'application/json'
            }
            response = requests.post(self.base_url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            logger.error(f"Erro na requisição HTTP: {str(e)}")
            return {"error": "Falha na requisição à API do Sentinel Hub"}
        except Exception as e:
            logger.error(f"Erro inesperado: {str(e)}")
            return {"error": "Ocorreu um erro inesperado"}

    def get_irrigation_advice(self, geometry: dict, start_date: str, end_date: str) -> Dict[str, Any]:
        try:
            analog_value, is_wet = self.soil_sensor.check_soil_moisture()
            advice = "Irrigação não necessária." if is_wet else "Irrigação recomendada"
            logger.info(f"Conselho de irrigação gerado para a geometria {geometry}")
            return {"message": advice}
        except Exception as e:
            logger.error(f"Erro ao gerar conselho de irrigação: {str(e)}")
            raise HTTPException(status_code=500, detail="Erro ao gerar conselho de irrigação")

    def detect_pests(self, coordinates, start_date, end_date):
        if os.getenv('TEST_MODE') == 'True':
            return {"message": "Nenhuma anomalia significativa detectada"}
        try:
            sentinel2_data = self.get_sentinel2_data({"type": "Polygon", "coordinates": [coordinates]}, start_date, end_date)
            pest_prediction = self.ml_service.detect_pests(sentinel2_data)
            logger.info(f"Detecção de pragas realizada para as coordenadas {coordinates}")
            return {"message": f"Probabilidade de pragas: {pest_prediction}"}
        except Exception as e:
            logger.error(f"Erro na detecção de pragas: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Erro na detecção de pragas: {str(e)}")

    async def get_weather_forecast(self, latitude: float, longitude: float) -> Dict[str, Any]:
        try:
            # Implementar pegar previsão do tempo de forma assíncrona
            return {"forecast": "Ensolarado", "temperature": 25, "humidity": 60}
        except Exception as e:
            logger.error(f"Erro ao obter previsão do tempo: {str(e)}")
            raise HTTPException(status_code=500, detail="Erro ao obter previsão do tempo")

    def get_harvest_advice(self, geometry, start_date, end_date):
        if os.getenv('TEST_MODE') == 'True':
            return "Conselho de Colheita: Colheita recomendada em 7 dias."
        try:
            sentinel2_data = self.get_sentinel2_data(geometry, start_date, end_date)
            advice = self.ml_service.predict_yield(sentinel2_data)
            logger.info(f"Conselho de colheita gerado para a geometria {geometry}")
            return f"Conselho de Colheita: {advice}"
        except Exception as e:
            logger.error(f"Erro ao gerar conselho de colheita: {str(e)}")
            return f"Erro ao gerar conselho de colheita: {str(e)}"

    def get_plant_health(self, geometry, start_date, end_date):
        if os.getenv('TEST_MODE') == 'True':
            return {"data": {"health": "Saudável"}}
        try:
            sentinel2_data = self.get_sentinel2_data(geometry, start_date, end_date)
            health = self.ml_service.predict_crop_health(sentinel2_data)
            return {"data": {"health": health}}
        except Exception as e:
            logger.error(f"Erro ao obter saúde da planta: {str(e)}")
            raise

    def get_sentinel1_data(self, geometry, start_date, end_date):
        if os.getenv('TEST_MODE') == 'True':
            return np.random.rand(10, 10, 2)  # Retorna um array 3D simulando dados Sentinel-1
        evalscript = """
        //VERSION=3
        function setup() {
            return {
                input: ["VV", "VH"],
                output: { bands: 2 }
            };
        }

        function evaluatePixel(sample) {
            return [sample.VV, sample.VH];
        }
        """
        return self._process_request(geometry, start_date, end_date, evalscript, "SENTINEL-1-GRD")

    def get_sentinel2_data(self, geometry, start_date, end_date):
        if os.getenv('TEST_MODE') == 'True':
            return np.random.rand(10, 10, 4)  # Retorna um array 3D simulando dados Sentinel-2
        evalscript = """
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
        """
        return self._process_request(geometry, start_date, end_date, evalscript, "SENTINEL-2-L2A")

    def get_sensor_data(self, geometry: dict, start_date: str, end_date: str) -> np.ndarray:
        # Implementar a lógica para obter dados do sensor
        # Por enquanto, retorne dados simulados
        return np.random.rand(10, 5)  # 10 amostras com 5 características cada