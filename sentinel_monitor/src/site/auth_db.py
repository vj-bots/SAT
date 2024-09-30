import sqlite3

import aiosqlite
from fastapi import HTTPException
from passlib.context import CryptContext

from ..services.connection_service import get_connection

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
    async with aiosqlite.connect(DATABASE_URL) as conn:
        try:
            await conn.execute('SELECT * FROM users WHERE email = ?', (user.email,))
            if await conn.fetchone():
                raise HTTPException(status_code=400, detail="Email já existe")

            await conn.execute('SELECT * FROM users WHERE username = ?', (user.username,))
            if await conn.fetchone():
                raise HTTPException(status_code=400, detail="Username já existe")

            await conn.execute(
                'INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
                (user.username, user.email, hashed_password))
            await conn.commit()

            await conn.execute('SELECT * FROM users WHERE email = ?', (user.email,))
            db_user = await conn.fetchone()

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

async def get_user(username: str):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT id, username, email FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        if user:
            return user
        else:
            return None
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao buscar usuário: {str(e)}") from e
    finally:
        cursor.close()
        conn.close()

async def create_users_table():
    conn = await get_connection()
    cursor = await conn.cursor()
    await cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT, 
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    full_name TEXT,
    disabled BOOLEAN DEFAULT 0
    )
    ''')
    await conn.commit()
    await conn.close()

async def authenticate_user(username: str, password: str):
    conn = await get_connection()
    cursor = await conn.cursor()
    try:
        await cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = await cursor.fetchone()
        if user and verify_password(password, user[3]):
            return {"id": user[0], "username": user[1], "email": user[2]}
        return None
    except Exception as e:
        print(f"Erro ao autenticar usuário: {str(e)}")
        return None
    finally:
        await cursor.close()
        await conn.close()