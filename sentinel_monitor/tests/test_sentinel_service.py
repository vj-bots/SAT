import pytest
from sentinel_monitor.src.services.sentinel_service import SentinelService
from unittest.mock import patch, MagicMock

@pytest.fixture
def sentinel_service():
    return SentinelService()

def test_get_irrigation_advice(sentinel_service):
    coordinates = [[-5.0, 40.0], [-5.0, 45.0], [5.0, 45.0], [5.0, 40.0], [-5.0, 40.0]]
    advice = sentinel_service.get_irrigation_advice(coordinates)
    assert isinstance(advice, str)
    assert "Irrigação" in advice

def test_get_pest_detection(sentinel_service):
    coordinates = [[-5.0, 40.0], [-5.0, 45.0], [5.0, 45.0], [5.0, 40.0], [-5.0, 40.0]]
    start_date = "2023-01-01"
    end_date = "2023-01-31"
    result = sentinel_service.get_pest_detection(coordinates, start_date, end_date)
    assert isinstance(result, str)
    assert "pragas" in result.lower() or "anomalia" in result.lower()

def test_get_weather_forecast(sentinel_service):
    coordinates = [[-5.0, 40.0], [-5.0, 45.0], [5.0, 45.0], [5.0, 40.0], [-5.0, 40.0]]
    forecast = sentinel_service.get_weather_forecast(coordinates)
    assert isinstance(forecast, dict)
    assert "temperature" in forecast
    assert "precipitation" in forecast
    assert "wind_speed" in forecast

@patch('sentinel_monitor.src.services.sentinel_service.SentinelAPI')
def test_get_sentinel2_data(mock_sentinel_api, sentinel_service):
    mock_sentinel_api.return_value.query.return_value = {"test_product": {"title": "Test Product"}}
    mock_sentinel_api.return_value.download.return_value = "test_data"
    
    geometry = {
        "type": "Polygon",
        "coordinates": [[[-5.0, 40.0], [-5.0, 45.0], [5.0, 45.0], [5.0, 40.0], [-5.0, 40.0]]]
    }
    start_date = "2023-01-01"
    end_date = "2023-01-31"
    
    result = sentinel_service.get_sentinel2_data(geometry, start_date, end_date)
    assert result == "test_data"

@patch('sentinel_monitor.src.services.sentinel_service.SentinelAPI')
def test_get_sentinel1_data(mock_sentinel_api, sentinel_service):
    mock_sentinel_api.return_value.query.return_value = {"test_product": {"title": "Test Product"}}
    mock_sentinel_api.return_value.download.return_value = "test_data"
    
    geometry = {
        "type": "Polygon",
        "coordinates": [[[-5.0, 40.0], [-5.0, 45.0], [5.0, 45.0], [5.0, 40.0], [-5.0, 40.0]]]
    }
    start_date = "2023-01-01"
    end_date = "2023-01-31"
    
    result = sentinel_service.get_sentinel1_data(geometry, start_date, end_date)
    assert result == "test_data"

def test_map_area(sentinel_service):
    coordinates = [[-5.0, 40.0], [-5.0, 45.0], [5.0, 45.0], [5.0, 40.0], [-5.0, 40.0]]
    result = sentinel_service.map_area(coordinates)
    assert isinstance(result, dict)
    assert "map" in result

def test_get_plant_health(sentinel_service):
    coordinates = [[-5.0, 40.0], [-5.0, 45.0], [5.0, 45.0], [5.0, 40.0], [-5.0, 40.0]]
    result = sentinel_service.get_plant_health(coordinates)
    assert isinstance(result, dict)
    assert "health" in result

def test_get_harvest_advice(sentinel_service):
    geometry = {
        "type": "Polygon",
        "coordinates": [[[-5.0, 40.0], [-5.0, 45.0], [5.0, 45.0], [5.0, 40.0], [-5.0, 40.0]]]
    }
    start_date = "2023-01-01"
    end_date = "2023-01-31"
    result = sentinel_service.get_harvest_advice(geometry, start_date, end_date)
    assert isinstance(result, str)
    assert "Colheita" in result

@patch('sentinel_monitor.src.services.sentinel_service.SentinelAPI')
def test_download_biomass_image(mock_sentinel_api, sentinel_service):
    mock_sentinel_api.return_value.query.return_value = {"test_product": {"title": "Test Product"}}
    mock_sentinel_api.return_value.download.return_value = "test_data"
    
    coordinates = [[-5.0, 40.0], [-5.0, 45.0], [5.0, 45.0], [5.0, 40.0], [-5.0, 40.0]]
    start_date = "2023-01-01"
    end_date = "2023-01-31"
    
    result = sentinel_service.download_biomass_image(coordinates, start_date, end_date)
    assert result is not None
    assert isinstance(result, str)
    assert "biomassa" in result.lower()

def test_get_land_use_classification(sentinel_service):
    coordinates = [[-5.0, 40.0], [-5.0, 45.0], [5.0, 45.0], [5.0, 40.0], [-5.0, 40.0]]
    start_date = "2023-01-01"
    end_date = "2023-01-31"
    result = sentinel_service.get_land_use_classification(coordinates, start_date, end_date)
    assert isinstance(result, dict)
    assert "classification" in result
    assert isinstance(result["classification"], str)

@patch('sentinel_monitor.src.services.sentinel_service.SentinelAPI')
def test_get_crop_yield_estimation(mock_sentinel_api, sentinel_service):
    mock_sentinel_api.return_value.query.return_value = {"test_product": {"title": "Test Product"}}
    mock_sentinel_api.return_value.download.return_value = "test_data"
    
    geometry = {
        "type": "Polygon",
        "coordinates": [[[-5.0, 40.0], [-5.0, 45.0], [5.0, 45.0], [5.0, 40.0], [-5.0, 40.0]]]
    }
    start_date = "2023-01-01"
    end_date = "2023-01-31"
    
    result = sentinel_service.get_crop_yield_estimation(geometry, start_date, end_date)
    assert isinstance(result, float)
    assert result > 0

def test_get_soil_moisture(sentinel_service):
    coordinates = [[-5.0, 40.0], [-5.0, 45.0], [5.0, 45.0], [5.0, 40.0], [-5.0, 40.0]]
    start_date = "2023-01-01"
    end_date = "2023-01-31"
    result = sentinel_service.get_soil_moisture(coordinates, start_date, end_date)
    assert isinstance(result, dict)
    assert "moisture" in result
    assert isinstance(result["moisture"], float)
    assert 0 <= result["moisture"] <= 100

@patch('sentinel_monitor.src.services.sentinel_service.SentinelAPI')
def test_get_ndvi_time_series(mock_sentinel_api, sentinel_service):
    mock_sentinel_api.return_value.query.return_value = {"test_product": {"title": "Test Product"}}
    mock_sentinel_api.return_value.download.return_value = "test_data"
    
    geometry = {
        "type": "Polygon",
        "coordinates": [[[-5.0, 40.0], [-5.0, 45.0], [5.0, 45.0], [5.0, 40.0], [-5.0, 40.0]]]
    }
    start_date = "2023-01-01"
    end_date = "2023-01-31"
    
    result = sentinel_service.get_ndvi_time_series(geometry, start_date, end_date)
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(item, dict) for item in result)
    assert all("date" in item and "ndvi" in item for item in result)

def test_get_crop_stress_detection(sentinel_service):
    coordinates = [[-5.0, 40.0], [-5.0, 45.0], [5.0, 45.0], [5.0, 40.0], [-5.0, 40.0]]
    start_date = "2023-01-01"
    end_date = "2023-01-31"
    result = sentinel_service.get_crop_stress_detection(coordinates, start_date, end_date)
    assert isinstance(result, dict)
    assert "stress_level" in result
    assert isinstance(result["stress_level"], str)
    assert result["stress_level"] in ["Baixo", "Médio", "Alto"]

@patch('sentinel_monitor.src.services.sentinel_service.SentinelAPI')
def test_get_crop_growth_stage(mock_sentinel_api, sentinel_service):
    mock_sentinel_api.return_value.query.return_value = {"test_product": {"title": "Test Product"}}
    mock_sentinel_api.return_value.download.return_value = "test_data"
    
    geometry = {
        "type": "Polygon",
        "coordinates": [[[-5.0, 40.0], [-5.0, 45.0], [5.0, 45.0], [5.0, 40.0], [-5.0, 40.0]]]
    }
    start_date = "2023-01-01"
    end_date = "2023-01-31"
    
    result = sentinel_service.get_crop_growth_stage(geometry, start_date, end_date)
    assert isinstance(result, str)
    assert result in ["Germinação", "Desenvolvimento", "Floração", "Frutificação", "Maturação"]

def test_get_weather_forecast(sentinel_service):
    coordinates = [[-5.0, 40.0], [-5.0, 45.0], [5.0, 45.0], [5.0, 40.0], [-5.0, 40.0]]
    forecast = sentinel_service.get_weather_forecast(coordinates)
    assert isinstance(forecast, dict)
    assert "temperature" in forecast
    assert "precipitation" in forecast
    assert "wind_speed" in forecast

@patch('sentinel_monitor.src.services.sentinel_service.SentinelAPI')
def test_get_crop_type_classification(mock_sentinel_api, sentinel_service):
    mock_sentinel_api.return_value.query.return_value = {"test_product": {"title": "Test Product"}}
    mock_sentinel_api.return_value.download.return_value = "test_data"
    
    geometry = {
        "type": "Polygon",
        "coordinates": [[[-5.0, 40.0], [-5.0, 45.0], [5.0, 45.0], [5.0, 40.0], [-5.0, 40.0]]]
    }
    start_date = "2023-01-01"
    end_date = "2023-01-31"
    
    result = sentinel_service.get_crop_type_classification(geometry, start_date, end_date)
    assert isinstance(result, str)
    assert result in ["Milho", "Soja", "Trigo", "Arroz", "Algodão"]

def test_get_soil_erosion_risk(sentinel_service):
    coordinates = [[-5.0, 40.0], [-5.0, 45.0], [5.0, 45.0], [5.0, 40.0], [-5.0, 40.0]]
    risk = sentinel_service.get_soil_erosion_risk(coordinates)
    assert isinstance(risk, dict)
    assert "risk_level" in risk
    assert risk["risk_level"] in ["Baixo", "Médio", "Alto"]

@patch('sentinel_monitor.src.services.sentinel_service.SentinelAPI')
def test_get_vegetation_index_time_series(mock_sentinel_api, sentinel_service):
    mock_sentinel_api.return_value.query.return_value = {"test_product": {"title": "Test Product"}}
    mock_sentinel_api.return_value.download.return_value = "test_data"
    
    geometry = {
        "type": "Polygon",
        "coordinates": [[[-5.0, 40.0], [-5.0, 45.0], [5.0, 45.0], [5.0, 40.0], [-5.0, 40.0]]]
    }
    start_date = "2023-01-01"
    end_date = "2023-01-31"
    index_type = "NDVI"
    
    result = sentinel_service.get_vegetation_index_time_series(geometry, start_date, end_date, index_type)
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(item, dict) for item in result)
    assert all("date" in item and "value" in item for item in result)

def test_get_crop_water_stress(sentinel_service):
    coordinates = [[-5.0, 40.0], [-5.0, 45.0], [5.0, 45.0], [5.0, 40.0], [-5.0, 40.0]]
    stress = sentinel_service.get_crop_water_stress(coordinates)
    assert isinstance(stress, dict)
    assert "stress_level" in stress
    assert stress["stress_level"] in ["Baixo", "Médio", "Alto"]

@patch('sentinel_monitor.src.services.sentinel_service.SentinelAPI')
def test_get_crop_nutrient_deficiency(mock_sentinel_api, sentinel_service):
    mock_sentinel_api.return_value.query.return_value = {"test_product": {"title": "Test Product"}}
    mock_sentinel_api.return_value.download.return_value = "test_data"
    
    geometry = {
        "type": "Polygon",
        "coordinates": [[[-5.0, 40.0], [-5.0, 45.0], [5.0, 45.0], [5.0, 40.0], [-5.0, 40.0]]]
    }
    start_date = "2023-01-01"
    end_date = "2023-01-31"
    
    result = sentinel_service.get_crop_nutrient_deficiency(geometry, start_date, end_date)
    assert isinstance(result, dict)
    assert "deficiencies" in result
    assert isinstance(result["deficiencies"], list)
    assert all(nutrient in ["Nitrogênio", "Fósforo", "Potássio", "Nenhuma"] for nutrient in result["deficiencies"])

def test_get_field_boundary_detection(sentinel_service):
    coordinates = [[-5.0, 40.0], [-5.0, 45.0], [5.0, 45.0], [5.0, 40.0], [-5.0, 40.0]]
    boundaries = sentinel_service.get_field_boundary_detection(coordinates)
    assert isinstance(boundaries, list)
    assert len(boundaries) > 0
    assert all(isinstance(coord, list) and len(coord) == 2 for coord in boundaries)

@patch('sentinel_monitor.src.services.sentinel_service.SentinelAPI')
def test_get_crop_yield_prediction(mock_sentinel_api, sentinel_service):
    mock_sentinel_api.return_value.query.return_value = {"test_product": {"title": "Test Product"}}
    mock_sentinel_api.return_value.download.return_value = "test_data"
    
    geometry = {
        "type": "Polygon",
        "coordinates": [[[-5.0, 40.0], [-5.0, 45.0], [5.0, 45.0], [5.0, 40.0], [-5.0, 40.0]]]
    }
    start_date = "2023-01-01"
    end_date = "2023-01-31"
    crop_type = "Soja"
    
    result = sentinel_service.get_crop_yield_prediction(geometry, start_date, end_date, crop_type)
    assert isinstance(result, dict)
    assert "predicted_yield" in result
    assert isinstance(result["predicted_yield"], float)
    assert result["predicted_yield"] > 0