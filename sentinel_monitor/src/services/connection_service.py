import aiosqlite

DATABASE_URL = "test.db"

async def get_connection():
    """
    Establishes an asynchronous connection to the SQLite database.

    Returns:
        aiosqlite.Connection: An asynchronous connection object to the database.
    """
    return await aiosqlite.connect(DATABASE_URL)