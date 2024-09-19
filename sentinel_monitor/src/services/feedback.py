import sqlite3
from fastapi import HTTPException

DATABASE_URL = "test.db"


def get_connection():
    conn = sqlite3.connect(DATABASE_URL, check_same_thread=False)
    return conn

def store_feedback(image_id: int, correct_classification: str):
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO feedback (image_id, correct_classification) VALUES (?, ?)",
            (image_id, correct_classification))
        conn.commit()
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail="Erro ao armazenar feedback") from e
    finally:
        cursor.close()
        conn.close()
    return {"message": "Feedback armazenado com sucesso!"}


def get_feedback(image_id: int):
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM feedback WHERE image_id = ?",
                       (image_id, ))
        feedback = cursor.fetchone()
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail="Erro ao obter feedback") from e
    finally:
        cursor.close()
        conn.close()
    if feedback:
        return {"image_id": feedback[0], "correct_classification": feedback[1]}
    else:
        return None


def get_all_feedback_ids():
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT image_id FROM feedback")
        feedback_ids = cursor.fetchall()
        return [id[0] for id in feedback_ids]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Erro ao obter todos os IDs de feedback") from e
    finally:
        cursor.close()
        conn.close()
