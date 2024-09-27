from fastapi import HTTPException
from sentinel_monitor.src.api.exceptions_api import RedisConnectionError

class SentinelAPIError(HTTPException):
    def __init__(self, status_code: int, detail: str):
        super().__init__(status_code, detail)

class MLModelError(HTTPException):
    def __init__(self, status_code: int, detail: str):
        super().__init__(status_code, detail)

class InvalidInputError(HTTPException):
    def __init__(self, status_code: int, detail: str):
        super().__init__(status_code, detail)