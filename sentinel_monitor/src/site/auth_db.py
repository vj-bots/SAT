import sqlite3

from fastapi import HTTPException
from passlib.context import CryptContext

from ..services.connection import get_connection

import string
import random

DATABASE_URL = "test.db"

# Contexto do Passlib para hash de senha
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# Funções auxiliares para hash e verificação de senha
def get_password_hash(password):
    return pwd_context.hash(password)


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


# Register Function
async def register(user):
    hashed_password = get_password_hash(user.password)
    conn = get_connection()
    try:
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM users WHERE email = ?', (user.email, ))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="Email já existe")

        cursor.execute('SELECT * FROM users WHERE username = ?',
                       (user.username, ))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="Username já existe")

        cursor.execute(
            'INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
            (user.username, user.email, hashed_password))
        conn.commit()

        cursor.execute('SELECT * FROM users WHERE email = ?', (user.email, ))
        db_user = cursor.fetchone()
        cursor.close()

        if db_user:
            return {"message": "Usuário registrado!"}
        else:
            raise HTTPException(
                status_code=500,
                detail="Falha ao registrar usuário no banco de dados.")
    except sqlite3.IntegrityError as e:
        raise HTTPException(
            status_code=400,
            detail="Erro de integridade do banco de dados") from e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Falha ao registrar usuário: {str(e)}") from e
    finally:
        conn.close()


# Login Function
async def login(user):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT * FROM users WHERE email = ?', (user.email, ))
        db_user = cursor.fetchone()
        if db_user:
            if verify_password(user.password, db_user[3]):
                occurrences = db_user[4] if len(db_user) > 4 else 0
                return {
                    "message": "Login com sucesso!",
                    "name": db_user[1],
                    "id_cliente": db_user[0],
                    "occurrences": occurrences
                }
            else:
                raise HTTPException(status_code=400,
                                    detail="Credenciais Inválidas.")
        else:
            raise HTTPException(status_code=400,
                                detail="Usuário não encontrado.")
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Erro ao logar: {str(e)}") from e
    finally:
        cursor.close()
        conn.close()


#  Generate Token Function
async def generate_token(user):
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute('SELECT * FROM users WHERE email = ? LIMIT 1;',
                       (user.email, ))
        db_user = cursor.fetchone()

        if db_user:
            token = ''.join(
                random.choice(string.ascii_uppercase + string.digits)
                for _ in range(5))
            cursor.execute('INSERT INTO tokens (email, token) VALUES (?, ?)',
                           (db_user[2], token))

            conn.commit()

            return token

    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Erro ao gerar token: {str(e)}") from e

    finally:
        cursor.close()
        conn.close()


#  Validate Token Function
async def validate_token(user):
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute('SELECT * FROM users WHERE email = ? LIMIT 1;',
                       (user.email, ))
        db_user = cursor.fetchone()

        if db_user:
            cursor.execute(
                'SELECT * FROM tokens WHERE email = ? AND token = ? AND ja_usado = 0 LIMIT 1;',
                (user.email, user.token))

            is_valid = cursor.fetchone()

            if is_valid:
                cursor.execute('UPDATE tokens SET ja_usado = 1 WHERE email = ?',
                               (db_user[2],))
    
                conn.commit()

                return {'message': 'Sucesso. Token Válido!'}

    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Erro ao validar o token: {str(e)}") from e

    finally:
        cursor.close()
        conn.close()


#  Validate Token Function
async def forgot_pass(user):
    conn = get_connection()
    cursor = conn.cursor()
    hashed_password = get_password_hash(user.senha)

    cursor.execute('SELECT * FROM users WHERE email = ?', (user.email, ))
    
    if not cursor.fetchone():
        raise HTTPException(status_code=400, detail="Email não existe.")

    cursor.execute(
        'UPDATE users SET password = ? WHERE email = ?',
        (hashed_password, user.email, ))
    conn.commit()

    cursor.execute('SELECT * FROM users WHERE email = ?', (user.email, ))
    db_user = cursor.fetchone()
    cursor.close()

    if db_user:
        return {"message": "Senha atualizada!"}
    else:
        raise HTTPException(
            status_code=500,
            detail="Falha ao atualizar a senha do usuário no banco de dados.")
