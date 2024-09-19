from fastapi import APIRouter
from sentinel_monitor.src.services.sentinel_service import SentinelService

router = APIRouter()

@router.post("/download/land-use")
async def download_land_use(geometry: dict, start_date: str, end_date: str):
    sentinel_service = SentinelService()
    sentinel_service.download_land_use_image(geometry, start_date, end_date)
    return {"message": "Land Use data downloaded successfully"}