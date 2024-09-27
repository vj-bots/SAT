from unittest.mock import MagicMock, patch
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import JSONResponse
from sentinel_monitor.src.services.ml_service import MLService
from pydantic import BaseModel
from ..services.feedback_service import FeedbackModel, save_feedback, update_model_with_feedback
from ..utils.auth import verify_token
from .auth_router import get_current_user
from ..model.user import User

router = APIRouter()
ml_service = None

def get_ml_service():
    global ml_service
    if ml_service is None:
        ml_service = MLService()
    return ml_service

class FeedbackRequest(BaseModel):
    label: int

@router.post("/")
async def feedback(image: UploadFile = File(...), request: FeedbackRequest = FeedbackRequest(label=0)):
    image_path = f"feedback/{image.filename}"
    with open(image_path, "wb") as f:
        f.write(image.file.read())
    get_ml_service().retrain(image_path, request.label)
    return {"status": "feedback recebido"}

@router.post("/feedback")
async def process_feedback(image_path: str):
    try:
        irrigation, invasion, health = get_ml_service().predict(image_path)
        return {"irrigation": float(irrigation), "invasion": float(invasion), "health": int(health)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/submit-feedback")
async def submit_feedback(feedback: FeedbackModel, current_user: User = Depends(get_current_user)):
    try:
        feedback_id = await save_feedback(feedback, current_user.id)
        await update_model_with_feedback(feedback)
        return {"message": "Feedback recebido com sucesso", "feedback_id": feedback_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar feedback: {str(e)}")