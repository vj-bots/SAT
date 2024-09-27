import re
import sqlite3

from fastapi import HTTPException
from passlib.context import CryptContext
from ..utils.pydantic_compat import BaseModel, EmailStr, Field, field_validator

from ..services.connection_service import get_connection  # Alterado para importação relativa

DATABASE_URL = "test.db"

# Modelos Pydantic
class User_create(BaseModel):
    username: str = Field(..., min_length=3, max_length=20, pattern=r"^[a-zA-Z0-9_]*$")
    email: EmailStr
    password: str
    password: str

    @field_validator('username')
    @classmethod
    def check_username(cls, value):
        if not re.match(r"^[a-zA-Z0-9_]*$", value):
            raise ValueError("Usuário precisa ser alfanumérico e conter caracteres especiais. Exemplo: 'joao_123'")
        return value

    @field_validator('password')
    @classmethod
    def check_password(cls, value):
        if not re.match(r"^(?=.*[A-Za-z])(?=.*\d)(?=.*[@$!%*#?&])[A-Za-z\d@$!%*#?&]{8,}$", value):
            raise ValueError('A senha necessita ao menos 8 dígitos, incluindo letras, números e caracteres especiais.')
        return value

class User_login(BaseModel):
    email: EmailStr
    password: str

    @field_validator('password')
    @classmethod
    def check_password(cls, value):
        if not re.match(r"^(?=.*[A-Za-z])(?=.*\d)(?=.*[@$!%*#?&])[A-Za-z\d@$!%*#?&]{8,}$", value):
            raise ValueError('Senha Incorreta!')
        return value

class User(BaseModel):
    id: int
    username: str
    email: str
    full_name: str
    disabled: bool = False
