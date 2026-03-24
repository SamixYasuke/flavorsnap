import os
from typing import Optional, Tuple


def get_database_url() -> Optional[str]:
    url = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_DSN")
    if url and url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    return url


def get_connection():
    url = get_database_url()
    if not url:
        return None
    try:
        import psycopg2
        return psycopg2.connect(url)
    except Exception:
        return None


def is_db_available() -> bool:
    return get_connection() is not None


def execute_sql(sql: str, params: Optional[Tuple] = None):
    conn = get_connection()
    if not conn:
        return None
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, params or ())
                try:
                    return cur.fetchall()
                except Exception:
                    return None
    finally:
        try:
            conn.close()
        except Exception:
            pass
