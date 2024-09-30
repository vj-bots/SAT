import os
import sys
import time
import uuid
import logging
import aioredis
import traceback
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Depends, status, Query, UploadFile, File
from fastapi.openapi.utils import get_openapi
from sentinel_monitor.src.services import ml_service
from ..services.feedback_service import FeedbackModel
from sentinel_monitor.src.endpoints import auth_router, feedback_router, monitor_router, crop_biomass_router, land_use_router
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache
from fastapi.security import OAuth2PasswordBearer, OAuth2AuthorizationCodeBearer, OAuth2PasswordRequestForm
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from redis import asyncio as aioredis
from ..utils.pydantic_compat import BaseModel, EmailStr, Field, field_validator
from typing import List
from sentinel_monitor.src.api.exceptions_api import RedisConnectionError
from ..utils.logging_utils import setup_logger
from ..ML.data_preprocessing import load_sensor_data
from sentinel_monitor.src.schemas_src import MonitoringData
from sentinel_monitor.src.endpoints.auth_router import Token, get_current_user
from sentinel_monitor.src.model.user import User
from starlette.middleware.base import BaseHTTPMiddleware
from sentinel_monitor.src.services.sentinel_service import SentinelService
from sentinel_monitor.src.services.ml_service import MLService
from sentinel_monitor.src.site.auth_db import authenticate_user, create_users_table
from sentinel_monitor.src.services import feedback_service
from sentinel_monitor.src.services.feedback_service import save_feedback as feedback_service_save_feedback
from sentinel_monitor.src.services.yolo_service import yolo_service
import numpy as np
import cv2

# Configuração inicial
load_dotenv()
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root_dir)

# Configuração de logging
logger = setup_logger(__name__)
logger.setLevel(os.getenv('LOG_LEVEL', 'INFO'))
logger.info("PYTHONPATH: %s", sys.path)

# Configurações de autenticação
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Criação da tabela de usuários
async def setup():
    await create_users_table()

# Inicialização do FastAPI
app = FastAPI(title="Sentinel Monitor API", version="1.0.0", description="API para monitoramento agrícola usando dados do Sentinel.")

# Executar a criação da tabela de usuários na inicialização
@app.on_event("startup")
async def startup_event():
    await setup()

# Configuração do OpenAPI
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Sentinel Monitor API",
        version="1.0.0",
        description="API para monitoramento agrícola usando dados do Sentinel",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Funções auxiliares
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Configuração do FastAPI
app = FastAPI(
    title="Sentinel Monitor API",
    description="API para monitoramento de culturas usando dados do Sentinel e aprendizado de máquina",
    version="1.0.0",
    openapi_tags=[
        {"name": "auth", "description": "Operações de autenticação"},
        {"name": "monitor", "description": "Operações de monitoramento de culturas"},
        {"name": "feedback", "description": "Operações de feedback do usuário"},
        {"name": "crop-biomass", "description": "Operações relacionadas à biomassa das culturas"},
        {"name": "land-use", "description": "Operações relacionadas ao uso da terra"},
        {"name": "weather", "description": "Operações relacionadas a dados meteorológicos"},
        {"name": "health", "description": "Verificação de saúde da API"},
    ]
)

# Middleware de limite de taxa
class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, rate_limit: int = 10, time_window: int = 60):
        super().__init__(app)
        self.rate_limit = rate_limit
        self.time_window = time_window
        self.request_counts = {}

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        current_time = time.time()

        if client_ip not in self.request_counts:
            self.request_counts[client_ip] = []

        self.request_counts[client_ip] = [t for t in self.request_counts[client_ip] if current_time - t < self.time_window]

        if len(self.request_counts[client_ip]) >= self.rate_limit:
            return JSONResponse(status_code=429, content={"detail": "Too many requests"})

        self.request_counts[client_ip].append(current_time)
        response = await call_next(request)
        return response

app.add_middleware(RateLimitMiddleware, rate_limit=10, time_window=60)

# Carregamento de dados do sensor
sensor_data = load_sensor_data()

# Configuração do limitador
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Configuração do CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclusão de routers
app.include_router(auth_router)
app.include_router(monitor_router, prefix="/monitor")
app.include_router(feedback_router, prefix="/feedback")
app.include_router(crop_biomass_router)
app.include_router(land_use_router)

# Debug: listar todas as rotas registradas
for route in app.routes:
    print(f"Route: {route.path}, methods: {route.methods}")

# Modelo de dados meteorológicos
class WeatherData(BaseModel):
    temperature: float
    atmospheric_pressure: float
    humidity: float
    wind_speed: float
    solar_radiation: float
    filled_data: int

    @field_validator('temperature')
    @classmethod
    def validate_temperature(cls, v):
        if v < -273.15:
            raise ValueError("A temperatura não pode ser menor que o zero absoluto.")
        return v
    
    @field_validator('humidity')
    @classmethod
    def validate_humidity(cls, v):
        if v < 0 or v > 100:
            raise ValueError("A umidade não pode ser menor que 0 ou maior que 100.")
        return v
    
    @field_validator('wind_speed')
    @classmethod
    def validate_wind_speed(cls, v):
        if v < 0:
            raise ValueError("A velocidade do vento não pode ser menor que 0.")
        return v
    
    @field_validator('solar_radiation')
    @classmethod
    def validate_solar_radiation(cls, v):
        if v < 0:
            raise ValueError("A radiação solar não pode ser menor que 0.")
        return v
    
    @field_validator('atmospheric_pressure')
    @classmethod
    def validate_atmospheric_pressure(cls, v):
        if v < 0:
            raise ValueError("A pressão atmosférica não pode ser menor que 0.")
        return v
    
    @field_validator('filled_data')
    @classmethod
    def validate_filled_data(cls, v):
        if v < 0 or v > 100:
            raise ValueError("A quantidade de dados preenchidos não pode ser menor que 0 ou maior que 100.")
        return v

# Funções auxiliares
async def check_redis_connection():
    try:
        redis = aioredis.from_url(os.getenv("REDIS_URL", "redis://localhost"), encoding="utf8", decode_responses=True)
        await redis.ping()
        return True
    except Exception as e:
        logger.error("Erro ao conectar com Redis: %s", str(e))
        raise RedisConnectionError(f"Falha na conexão com o Redis: {str(e)}")
    
async def save_feedback(feedback: FeedbackModel, user_id: int):
    return await feedback_service_save_feedback(feedback, user_id)

# Rotas da API
@app.post("/feedback/submit-feedback")
async def submit_feedback(feedback: FeedbackModel, current_user: User = Depends(get_current_user)):
    try:
        feedback_id = await save_feedback(feedback, current_user.id)
        return {"message": "Feedback recebido com sucesso", "feedback_id": feedback_id}
    except Exception as e:
        logger.error(f"Erro ao processar feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao processar feedback: {str(e)}")

@app.get("/weather-data", response_model=List[WeatherData], tags=["weather"])
@limiter.limit("10/minute")
@cache(expire=3600)
async def get_weather_data(
    request: Request,
    current_user: User = Depends(get_current_user),
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100)
):
    """
    Retorna dados meteorológicos para o usuário autenticado.

    Args:
        request (Request): Objeto de requisição do FastAPI.
        current_user (User): Usuário autenticado.
        skip (int): Número de registros a serem pulados (para paginação).
        limit (int): Número máximo de registros a serem retornados.

    Returns:
        List[WeatherData]: Lista de dados meteorológicos.

    Raises:
        HTTPException: Se ocorrer um erro ao carregar os dados meteorológicos.

    Note:
        - Os dados são armazenados em cache por 1 hora para melhorar o desempenho.
        - A rota é limitada a 10 requisições por minuto por usuário.
    """
    try:
        sensor_data = load_sensor_data()
        weather_data = [WeatherData(**row) for row in sensor_data]
        logger.info("Dados meteorológicos carregados com sucesso", 
                    user_id=current_user.id, 
                    skip=skip, 
                    limit=limit)
        return weather_data[skip : skip + limit]
    except ValueError as e:
        logger.error("Erro de validação nos dados meteorológicos", 
                     error=str(e), 
                     user_id=current_user.id)
        raise HTTPException(status_code=400, detail=f"Erro de validação: {str(e)}")
    except Exception as e:
        logger.error("Erro ao carregar dados meteorológicos", 
                     error=str(e), 
                     user_id=current_user.id)
        raise HTTPException(status_code=500, detail="Erro interno ao carregar dados meteorológicos")

@app.get("/")
def root():
    return "Foguete não da ré e satélite da a volta, praga nois esmaga e colheita nois prevê"

@app.get("/status", tags=["status"])
async def status_check():
    """
    Verifica o status do servidor e dos serviços conectados.

    Returns:
        dict: Um dicionário contendo o status da API e do Redis.
            - api_status (str): O status atual da API ('online' ou 'offline').
            - redis_status (str): O status da conexão com o Redis ('OK' ou 'ERROR').

    Note:
        Esta rota é útil para monitoramento e verificações de integridade do sistema.
    """
    redis_status = "OK" if await check_redis_connection() else "ERROR"
    return {"api_status": "online", "redis_status": redis_status}

@app.get("/health", tags=["health"])
async def health_check():
    """
    Verifica o status de saúde da API.
    """
    return {"status": "healthy"}

@app.get("/protected", tags=["auth"])
async def protected_route(current_user: dict = Depends(get_current_user)):
    """
    Rota protegida que exige autenticação.
    """
    return {"message": f"Olá, {current_user['username']}! Você acessou uma rota protegida."}

@app.post("/logout", tags=["auth"])
async def logout(current_user: User = Depends(get_current_user)):
    """
    Realiza o logout do usuário autenticado.
    """
    return {"message": f"Usuário {current_user.username} deslogado com sucesso!"}

@app.post("/validate-weather-data", response_model=WeatherData, tags=["weather"])
async def validate_weather_data(data: WeatherData):
    """
    Valida os dados meteorológicos fornecidos.

    Args:
        data (WeatherData): Dados meteorológicos a serem validados.

    Returns:
        WeatherData: Dados meteorológicos validados.

    Raises:
        HTTPException: Se os dados fornecidos forem inválidos.
    """
    try:
        logger.info("Iniciando validação de dados meteorológicos")
        validated_data = WeatherData(**data.model_dump())
        logger.info("Dados meteorológicos validados com sucesso")
        return validated_data
    except ValueError as e:
        logger.error(f"Erro na validação de dados meteorológicos: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credenciais inválidas",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/monitor/weather-forecast")
async def get_weather_forecast(latitude: float, longitude: float):
    # Implementação da previsão do tempo
    return {"forecast": "Ensolarado"}

@app.get("/market/trends")
async def get_market_trends(crop_type: str):
    # Implementação das tendências de mercado
    return {"trend": "Em alta"}

@app.post("/monitor/crop-health")
async def crop_health(data: MonitoringData, current_user: User = Depends(get_current_user)):
    try:
        sentinel_data = SentinelService.get_sentinel2_data(data.geometry, data.start_date, data.end_date)
        health_prediction = ml_service.predict_crop_health(sentinel_data)
        return {"health": health_prediction}
    except Exception as e:
        logger.error(f"Erro na previsão de saúde da cultura: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro na previsão de saúde da cultura: {str(e)}")

@app.middleware("http")
async def error_handling_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        logger.error(f"Erro não tratado: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Ocorreu um erro interno no servidor."}
        )

redis = aioredis.from_url("redis://localhost")

@app.on_event("startup")
async def startup_event():
    app.state.redis = redis

@app.on_event("shutdown")
async def shutdown_event():
    await app.state.redis.close()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response: {response.status_code}")
    return response

@app.post("/yolo/detect")
async def detect_objects(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    results = yolo_service.detect_objects(img)
    return {"detections": results}

@app.post("/yolo/segment")
async def segment_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    results = yolo_service.segment_image(img)
    return {"segmentation": results}

@app.post("/yolo/classify")
async def classify_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    results = yolo_service.classify_image(img)
    return {"classification": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)