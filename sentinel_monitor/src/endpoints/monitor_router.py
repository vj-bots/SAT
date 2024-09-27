import traceback
from datetime import datetime
from pydantic import BaseModel, Field
from ..utils.pydantic_compat import BaseModel, EmailStr, Field, field_validator
from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException, Depends
from ..utils.logging_utils import logger
from ..services.sentinel_service import SentinelService
from ..services.ml_service import MLService
from ..ML.data_preprocessing import process_sentinel2_image, process_sentinel1_data, save_results, process_sentinel2_data
from ..schemas_src import MonitorRequest, PredictionResponse, ErrorResponse
from .auth_router import get_current_user
from ..exceptions_src import SentinelAPIError, MLModelError, InvalidInputError
from ..model.user import User, User_create, User_login

router = APIRouter()

def get_sentinel_service():
    return SentinelService()

def get_ml_service():
    return MLService()

class Geometry(BaseModel):
    type: str = Field(..., pattern="^Polygon$")
    coordinates: List[List[List[float]]]

    @field_validator('type')
    @classmethod
    def validate_type(cls, v):
        if v not in ['Polygon', 'MultiPolygon']:
            raise ValueError('O tipo deve ser Polygon ou MultiPolygon')
        return v

class MonitorRequest(BaseModel):
    geometry: Geometry
    start_date: datetime
    end_date: datetime

    @field_validator('end_date')
    @classmethod
    def end_date_must_be_after_start_date(cls, v: datetime, values: Dict[str, Any]) -> datetime:
        start_date = values.data.get('start_date')
        if start_date and v <= start_date:
            raise ValueError('A data final deve ser posterior à data inicial')
        return v

@router.post("/process_sentinel2", response_model=dict, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def process_sentinel2(
    request: MonitorRequest, 
    sentinel_service: SentinelService = Depends(get_sentinel_service),
    current_user: User = Depends(get_current_user)
):
    """
    Processa dados do Sentinel-2 para a área e período especificados.

    Args:
        request (MonitorRequest): Contém a geometria e o intervalo de datas para processamento.
        sentinel_service (SentinelService): Serviço para obtenção de dados do Sentinel.
        current_user (User): Usuário autenticado.

    Returns:
        dict: Mensagem de conclusão e resultados do processamento.

    Raises:
        SentinelAPIError: Erro ao obter dados do Sentinel.
        MLModelError: Erro no processamento do modelo de machine learning.
        InvalidInputError: Dados de entrada inválidos.
        HTTPException: Erro interno do servidor.
    """
    try:
        sentinel2_data = sentinel_service.get_sentinel2_data(request.geometry, request.start_date, request.end_date)
        results = process_sentinel2_image(sentinel2_data)
        save_results(results, f"sentinel2_results_{request.start_date.date()}_{request.end_date.date()}.json")
        return {"message": "Processamento concluído", "results": results}
    except SentinelAPIError as e:
        logger.error(f"Erro na API do Sentinel: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except MLModelError as e:
        logger.error(f"Erro no modelo de ML: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except InvalidInputError as e:
        logger.error(f"Entrada inválida: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Erro inesperado: {str(e)}")
        raise HTTPException(status_code=500, detail="Ocorreu um erro inesperado")

@router.post("/process_sentinel1", response_model=dict, responses={500: {"model": ErrorResponse}})
async def process_sentinel1(
    request: MonitorRequest, 
    sentinel_service: SentinelService = Depends(get_sentinel_service),
    current_user: User = Depends(get_current_user)
):
    """
    Processa dados do Sentinel-1 para a área e período especificados.

    Args:
        request (MonitorRequest): Contém a geometria e o intervalo de datas para processamento.
        sentinel_service (SentinelService): Serviço para obtenção de dados do Sentinel.
        current_user (User): Usuário autenticado.

    Returns:
        dict: Mensagem de conclusão e resultados do processamento.

    Raises:
        SentinelAPIError: Erro ao obter dados do Sentinel.
        MLModelError: Erro no processamento do modelo de machine learning.
        InvalidInputError: Dados de entrada inválidos.
        HTTPException: Erro interno do servidor.
    """
    try:
        sentinel1_data = sentinel_service.get_sentinel1_data(request.geometry, request.start_date, request.end_date)
        results = process_sentinel1_data(sentinel1_data)
        save_results(results, f"sentinel1_results_{request.start_date}_{request.end_date}.json")
        return {"message": "Processamento concluído", "results": results}
    except SentinelAPIError as e:
        logger.error(f"Erro na API do Sentinel: {str(e)}")
        raise
    except MLModelError as e:
        logger.error(f"Erro no modelo de ML: {str(e)}")
        raise
    except InvalidInputError as e:
        logger.error(f"Entrada inválida: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Erro inesperado: {str(e)}")
        raise HTTPException(status_code=500, detail="Ocorreu um erro inesperado")

@router.post("/crop-health", response_model=PredictionResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def get_crop_health(
    request: MonitorRequest, 
    sentinel_service: SentinelService = Depends(get_sentinel_service), 
    ml_service: MLService = Depends(get_ml_service),
    current_user: User = Depends(get_current_user)
):
    """
    Analisa a saúde da cultura com base nos dados do Sentinel-2 e sensores.

    Args:
        request (MonitorRequest): Dados da requisição contendo geometria e intervalo de datas.
        sentinel_service (SentinelService): Serviço para obtenção de dados do Sentinel.
        ml_service (MLService): Serviço para processamento de machine learning.
        current_user (User): Usuário autenticado.

    Returns:
        PredictionResponse: Resultado da análise de saúde da cultura.

    Raises:
        SentinelAPIError: Erro ao obter dados do Sentinel.
        MLModelError: Erro no processamento do modelo de machine learning.
        InvalidInputError: Dados de entrada inválidos.
        HTTPException: Erro interno do servidor.
    """
    try:
        sentinel2_data = sentinel_service.get_sentinel2_data(request.geometry, request.start_date, request.end_date)
        sensor_data = sentinel_service.get_sensor_data(request.geometry, request.start_date, request.end_date)
        prediction = ml_service.predict(sentinel2_data, sensor_data)
        return PredictionResponse(**prediction)
    except SentinelAPIError as e:
        logger.error(f"Erro na API do Sentinel: {str(e)}")
        raise
    except MLModelError as e:
        logger.error(f"Erro no modelo de ML: {str(e)}")
        raise
    except InvalidInputError as e:
        logger.error(f"Entrada inválida: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Erro inesperado: {str(e)}")
        raise HTTPException(status_code=500, detail="Ocorreu um erro inesperado")

@router.post("/soil-moisture", response_model=dict, responses={500: {"model": ErrorResponse}})
async def soil_moisture(
    request: MonitorRequest, 
    sentinel_service: SentinelService = Depends(get_sentinel_service),
    current_user: User = Depends(get_current_user)
):
    """
    Verifica a umidade do solo para a área e período especificados.

    Args:
        request (MonitorRequest): Dados da requisição contendo geometria e intervalo de datas.
        sentinel_service (SentinelService): Serviço para obtenção de dados do Sentinel.
        current_user (User): Usuário autenticado.

    Returns:
        dict: Mensagem de verificação e dados de umidade do solo.

    Raises:
        SentinelAPIError: Erro ao obter dados do Sentinel.
        MLModelError: Erro no processamento do modelo de machine learning.
        InvalidInputError: Dados de entrada inválidos.
        HTTPException: Erro interno do servidor.
    """
    try:
        moisture = sentinel_service.get_soil_moisture(request.geometry, request.start_date, request.end_date)
        return {"message": "Umidade do solo verificada", "moisture": moisture}
    except SentinelAPIError as e:
        logger.error(f"Erro na API do Sentinel: {str(e)}")
        raise
    except MLModelError as e:
        logger.error(f"Erro no modelo de ML: {str(e)}")
        raise
    except InvalidInputError as e:
        logger.error(f"Entrada inválida: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Erro inesperado: {str(e)}")
        raise HTTPException(status_code=500, detail="Ocorreu um erro inesperado")

@router.post("/time-series-analysis", response_model=dict, responses={500: {"model": ErrorResponse}})
async def time_series_analysis(
    request: MonitorRequest, 
    sentinel_service: SentinelService = Depends(get_sentinel_service), 
    ml_service: MLService = Depends(get_ml_service),
    current_user: User = Depends(get_current_user)
):
    """
    Realiza análise de série temporal dos dados do Sentinel-2.

    Args:
        request (MonitorRequest): Dados da requisição contendo geometria e intervalo de datas.
        sentinel_service (SentinelService): Serviço para obtenção de dados do Sentinel.
        ml_service (MLService): Serviço para processamento de machine learning.
        current_user (User): Usuário autenticado.

    Returns:
        dict: Mensagem de conclusão e resultados da análise de série temporal.

    Raises:
        SentinelAPIError: Erro ao obter dados do Sentinel.
        MLModelError: Erro no processamento do modelo de machine learning.
        InvalidInputError: Dados de entrada inválidos.
        HTTPException: Erro interno do servidor.
    """
    try:
        sentinel2_data = sentinel_service.get_sentinel2_data(request.geometry, request.start_date, request.end_date)
        results = ml_service.analyze_time_series(sentinel2_data)
        return {"message": "Análise de série temporal concluída", "results": results}
    except SentinelAPIError as e:
        logger.error(f"Erro na API do Sentinel: {str(e)}")
        raise
    except MLModelError as e:
        logger.error(f"Erro no modelo de ML: {str(e)}")
        raise
    except InvalidInputError as e:
        logger.error(f"Entrada inválida: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Erro inesperado: {str(e)}")
        raise HTTPException(status_code=500, detail="Ocorreu um erro inesperado")

@router.post("/yield-prediction", response_model=dict, responses={500: {"model": ErrorResponse}})
async def yield_prediction(
    request: MonitorRequest, 
    sentinel_service: SentinelService = Depends(get_sentinel_service), 
    ml_service: MLService = Depends(get_ml_service),
    current_user: User = Depends(get_current_user)
):
    """
    Realiza previsão de rendimento com base nos dados do Sentinel-2.

    Args:
        request (MonitorRequest): Dados da requisição contendo geometria e intervalo de datas.
        sentinel_service (SentinelService): Serviço para obtenção de dados do Sentinel.
        ml_service (MLService): Serviço para processamento de machine learning.
        current_user (User): Usuário autenticado.

    Returns:
        dict: Mensagem de conclusão e previsão de rendimento.

    Raises:
        SentinelAPIError: Erro ao obter dados do Sentinel.
        MLModelError: Erro no processamento do modelo de machine learning.
        InvalidInputError: Dados de entrada inválidos.
        HTTPException: Erro interno do servidor.
    """
    try:
        sentinel2_data = await sentinel_service.get_sentinel2_data(request.geometry, request.start_date, request.end_date)
        prediction = await ml_service.predict_yield(sentinel2_data)
        return {"message": "Previsão de rendimento concluída", "prediction": prediction}
    except SentinelAPIError as e:
        logger.error(f"Erro na API do Sentinel: {str(e)}")
        raise
    except MLModelError as e:
        logger.error(f"Erro no modelo de ML: {str(e)}")
        raise
    except InvalidInputError as e:
        logger.error(f"Entrada inválida: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Erro inesperado: {str(e)}")
        raise HTTPException(status_code=500, detail="Ocorreu um erro inesperado")

@router.get("/weather-forecast")
async def get_weather_forecast(
    latitude: float, 
    longitude: float,
    sentinel_service: SentinelService = Depends(get_sentinel_service)
):
    try:
        forecast = await sentinel_service.get_weather_forecast(latitude, longitude)
        return {"forecast": forecast}
    except Exception as e:
        logger.error(f"Erro ao obter previsão do tempo: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro ao obter previsão do tempo")
    
@router.post("/irrigation_advice", response_model=dict)
async def get_irrigation_advice(
    request: MonitorRequest, 
    sentinel_service: SentinelService = Depends(get_sentinel_service),
    current_user: User = Depends(get_current_user)
):
    try:
        advice = sentinel_service.get_irrigation_advice(request.geometry, request.start_date, request.end_date)
        return {"message": "Conselho de irrigação gerado", "advice": advice}
    except Exception as e:
        logger.error(f"Erro ao gerar conselho de irrigação: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro ao gerar conselho de irrigação")

@router.post("/pest_detection", response_model=dict)
async def detect_pests(
    request: MonitorRequest, 
    sentinel_service: SentinelService = Depends(get_sentinel_service),
    current_user: User = Depends(get_current_user)
):
    try:
        pests = sentinel_service.detect_pests(request.geometry, request.start_date, request.end_date)
        return {"message": "Detecção de pragas concluída", "pests": pests}
    except Exception as e:
        logger.error(f"Erro na detecção de pragas: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro na detecção de pragas")

@router.post("/harvest_advice", response_model=dict)
async def get_harvest_advice(
    request: MonitorRequest, 
    sentinel_service: SentinelService = Depends(get_sentinel_service),
    current_user: User = Depends(get_current_user)
):
    try:
        advice = sentinel_service.get_harvest_advice(request.geometry, request.start_date, request.end_date)
        return {"message": "Conselho de colheita gerado", "advice": advice}
    except Exception as e:
        logger.error(f"Erro ao gerar conselho de colheita: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro ao gerar conselho de colheita")