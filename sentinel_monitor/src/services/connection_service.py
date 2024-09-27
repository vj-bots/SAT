import sqlite3

def get_connection():
    """
    Establishes a connection to the SQLite database.

    Returns:
        sqlite3.Connection: A connection object to the database.
    """
    conn = sqlite3.connect("test.db")
    return conn
    # Criar BD
