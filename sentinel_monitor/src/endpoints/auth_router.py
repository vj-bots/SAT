from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from ..utils.auth import create_access_token, verify_token, token_required
from ..model.user import User, User_create, User_login
from ..site import auth_db as auth
from datetime import timedelta
from sentinel_monitor.src.site import auth_db

router = APIRouter()

ACCESS_TOKEN_EXPIRE_MINUTES = 30

class Token(BaseModel):
    access_token: str
    token_type: str

@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await auth.authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credenciais inválidas",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

async def get_current_user(token: str = Depends(verify_token)):
    user_data = await auth_db.get_user(username=token)
    if not user_data:
        raise HTTPException(status_code=401, detail="Usuário não encontrado")
    return User(id=user_data[0], username=user_data[1], email=user_data[2], full_name=user_data[1], disabled=False)

class RegisterRequest(BaseModel):
    username: str
    email: EmailStr
    password: str

@router.post("/register")
async def register_router(request: RegisterRequest):
    user = User_create(username=request.username, email=request.email, password=request.password)
    return await auth.register(user)

class LoginRequest(BaseModel):
    email: str
    password: str

@router.post("/login")
async def login_router(request: LoginRequest):
    user = User_login(email=request.email, password=request.password)
    return await auth.login(user)

@router.post("/teste")
@token_required
async def teste(token: str) -> str:
    return f'teste autenticado para o usuário: {token}'