from pydantic import BaseModel, Field, EmailStr

try:
    from pydantic import field_validator
except ImportError:
    from pydantic import validator as field_validator

__all__ = ['BaseModel', 'Field', 'field_validator', 'EmailStr']