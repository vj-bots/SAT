from pydantic import BaseModel, Field
from typing import List, Union
from datetime import date

class Coordinates(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)

class Geometry(BaseModel):
    type: str = Field(..., regex="^Polygon$")
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