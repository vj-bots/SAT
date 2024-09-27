from ..utils.auth import verify_token

async def get_current_user(token: str):
    return await verify_token(token)