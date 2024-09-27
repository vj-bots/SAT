import os
import sys
import time
import uuid
import aioredis
import traceback


from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Depends, status, Query

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
from sentinel_monitor.src.endpoints.auth_router import get_current_user
from sentinel_monitor.src.model.user import User
from starlette.middleware.base import BaseHTTPMiddleware
from sentinel_monitor.src.services.sentinel_service import SentinelService
from sentinel_monitor.src.services.ml_service import MLService

from sentinel_monitor.src.services import feedback_service
from sentinel_monitor.src.services.feedback_service import save_feedback as feedback_service_save_feedback

load_dotenv()

# Adiciona o diretório raiz ao PYTHONPATH
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root_dir)

from sentinel_monitor.src.endpoints import auth_router, feedback_router, monitor_router, crop_biomass_router, land_use_router

logger = setup_logger(__name__)
logger.setLevel(os.getenv('LOG_LEVEL', 'INFO'))
logger.info("PYTHONPATH: %s", sys.path)

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

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

sensor_data = load_sensor_data()

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

app.include_router(auth_router)
app.include_router(monitor_router, prefix="/monitor")
app.include_router(feedback_router, prefix="/feedback")
app.include_router(crop_biomass_router)
app.include_router(land_use_router)

# Adiciona um print de debug para listar todas as rotas registradas
for route in app.routes:
    print(f"Route: {route.path}, methods: {route.methods}")

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

@app.post("/feedback/submit-feedback")
async def submit_feedback(feedback: FeedbackModel, current_user: User = Depends(get_current_user)):
    try:
        feedback_id = await save_feedback(feedback, current_user.id)
        return {"message": "Feedback recebido com sucesso", "feedback_id": feedback_id}
    except Exception as e:
        logger.error(f"Erro ao processar feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao processar feedback: {str(e)}")

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error("HTTP Exception", extra={"status_code": exc.status_code, "detail": exc.detail})
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled Exception: %s", str(exc), extra={"traceback": traceback.format_exc()})
    return JSONResponse(
        status_code=500,
        content={"message": "Ocorreu um erro interno no servidor. Por favor, tente novamente mais tarde."},
    )

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

@app.on_event("startup")
async def startup():
    try:
        redis = await aioredis.from_url("redis://localhost", encoding="utf8", decode_responses=True)
        await redis.ping()
        FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")
        logger.info("Conexão com Redis estabelecida com sucesso")
    except RedisConnectionError as e:
        logger.warning("Redis não está disponível. O cache será desabilitado: %s", str(e))
    except Exception as e:
        logger.error("Erro inesperado durante a inicialização: %s", str(e))
        raise

app.add_middleware(
    TrustedHostMiddleware, allowed_hosts=["*"]
)

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

@app.exception_handler(RedisConnectionError)
async def redis_connection_error_handler(request: Request, exc: RedisConnectionError):
    logger.error("Erro de conexão com Redis: %s", str(exc))
    return JSONResponse(
        status_code=503,
        content={"message": "Serviço temporariamente indisponível. Por favor, tente novamente mais tarde."},
    )

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    if form_data.username == "testuser" and form_data.password == "testpassword":
        return {"access_token": "test_token", "token_type": "bearer"}
    raise HTTPException(status_code=400, detail="Credenciais inválidas")

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)