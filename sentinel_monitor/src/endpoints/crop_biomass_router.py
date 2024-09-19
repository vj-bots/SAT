from fastapi import APIRouter, Depends
from ..services.sentinel_service import SentinelService

router = APIRouter()

@router.post("/download/crop-biomass")
async def download_crop_biomass(geometry: dict, start_date: str, end_date: str):
    sentinel_service = SentinelService()
    sentinel_service.download_crop_biomass_image(geometry, start_date, end_date)
    return {"message": "Crop Biomass data downloaded successfully"}