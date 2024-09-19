import os
import sys
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Depends, status, Query
from .endpoints.auth_router import get_current_user, User
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache
from fastapi.security import OAuth2PasswordBearer, OAuth2AuthorizationCodeBearer
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from redis import asyncio as aioredis
from pydantic import BaseModel, validator
from typing import List
from .exceptions import RedisConnectionError
from .utils.logging import setup_logger
from ..ML.data_preprocessing import load_sensor_data

load_dotenv()

# Adiciona o diretório raiz ao `PYTHONPATH`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from sentinel_monitor.src.endpoints import auth_router, feedback_router, monitor_router, crop_biomass_router, land_use_router

logger = setup_logger(__name__)
logger.setLevel(os.getenv('LOG_LEVEL', 'INFO'))
logger.info("PYTHONPATH: %s", sys.path)

SECRET_KEY = os.getenv("SECRET_KEY", "sua_chave_secreta_aqui")
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

    @validator('temperature')
    def validate_temperature(cls, v):
        if v < -273.15:
            raise ValueError("A temperatura não pode ser menor que o zero absoluto.")
        return v
    
    @validator('humidity')
    def validate_humidity(cls, v):
        if v < 0 or v > 100:
            raise ValueError("A umidade não pode ser menor que 0 ou maior que 100.")
        return v
    
    @validator('wind_speed')
    def validate_wind_speed(cls, v):
        if v < 0:
            raise ValueError("A velocidade do vento não pode ser menor que 0.")
        return v
    
    @validator('solar_radiation')
    def validate_solar_radiation(cls, v):
        if v < 0:
            raise ValueError("A radiação solar não pode ser menor que 0.")
        return v
    
    @validator('atmospheric_pressure')
    def validate_atmospheric_pressure(cls, v):
        if v < 0:
            raise ValueError("A pressão atmosférica não pode ser menor que 0.")
        return v
    
    @validator('filled_data')
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
        logger.error("Erro ao conectar com Redis", error=str(e))
        raise RedisConnectionError(f"Falha na conexão com o Redis: {str(e)}")

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error("HTTP Exception", status_code=exc.status_code, detail=exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail},
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled Exception", error=str(exc), traceback=traceback.format_exc())
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
        redis = aioredis.from_url(os.getenv("REDIS_URL", "redis://localhost"), encoding="utf8", decode_responses=True)
        await redis.ping()
        FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")
        logger.info("Conexão com Redis estabelecida com sucesso")
    except RedisConnectionError as e:
        logger.warning("Redis não está disponível. O cache será desabilitado.", error=str(e))
    except Exception as e:
        logger.error("Erro inesperado durante a inicialização", error=str(e))
        raise

# Configuração do middleware de host confiável
app.add_middleware(
    TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1"]
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
async def protected_route(current_user: User = Depends(get_current_user)):
    """
    Rota protegida que exige autenticação.
    """
    return {"message": f"Olá, {current_user.username}! Você acessou uma rota protegida."}

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
        return data
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.exception_handler(RedisConnectionError)
async def redis_connection_error_handler(request: Request, exc: RedisConnectionError):
    logger.error("Erro de conexão com Redis", error=str(exc))
    return JSONResponse(
        status_code=503,
        content={"message": "Serviço temporariamente indisponível. Por favor, tente novamente mais tarde."},
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)