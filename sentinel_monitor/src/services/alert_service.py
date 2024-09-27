from ..models.alert import AlertModel
from ..database.database import database

async def create_alert(alert: AlertModel):
    query = """
    INSERT INTO alerts (user_id, type, message, severity, created_at)
    VALUES (:user_id, :type, :message, :severity, CURRENT_TIMESTAMP)
    RETURNING id
    """
    values = {**alert.dict()}
    return await database.execute(query=query, values=values)

async def get_alerts_for_user(user_id: int):
    query = "SELECT * FROM alerts WHERE user_id = :user_id ORDER BY created_at DESC"
    return await database.fetch_all(query=query, values={"user_id": user_id})