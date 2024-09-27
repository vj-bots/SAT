import aiohttp
import os
from dotenv import load_dotenv

load_dotenv()

async def get_market_trends(crop_type: str):
    MARKET_API_KEY = os.getenv("MARKET_API_KEY")
    if not MARKET_API_KEY:
        raise ValueError("MARKET_API_KEY não encontrada no arquivo .env")

    async with aiohttp.ClientSession() as session:
        url = f"https://api.marketdata.com/crops/{crop_type}?api_key={MARKET_API_KEY}"
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                return {
                    "current_price": data["current_price"],
                    "price_trend": data["price_trend"],
                    "demand_forecast": data["demand_forecast"]
                }
            else:
                raise Exception("Falha ao obter dados de mercado")