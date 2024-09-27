from pydantic import BaseModel, Field
from typing import List, Union, Dict
from datetime import date

class Coordinates(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)

class Geometry(BaseModel):
    type: str = Field(..., pattern="^Polygon$")
    coordinates: List[List[List[float]]]

class DateRange(BaseModel):
    start_date: date
    end_date: date

class MonitorRequest(BaseModel):
    geometry: Geometry
    date_range: DateRange

class PredictionResponse(BaseModel):
    health: str
    irrigation: float
    pest: str
    yield_prediction: dict

class ErrorResponse(BaseModel):
    detail: str

class MonitoringData(BaseModel):
    geometry: Dict
    start_date: str
    end_date: str