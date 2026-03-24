import os
from pathlib import Path
from typing import List
from db_config import get_connection


def applied_versions(conn) -> List[str]:
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations (version TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW())"
            )
            cur.execute("SELECT version FROM schema_migrations ORDER BY version")
            rows = cur.fetchall() or []
            return [r[0] for r in rows]


def apply_migration(conn, version: str, sql_path: Path):
    with open(sql_path, "r", encoding="utf-8") as f:
        sql = f.read()
    with conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            cur.execute("INSERT INTO schema_migrations (version) VALUES (%s) ON CONFLICT DO NOTHING", (version,))


def main():
    conn = get_connection()
    if not conn:
        print("Database not configured")
        return
    try:
        already = set(applied_versions(conn))
        migrations_dir = Path(__file__).parent / "migrations"
        migrations = sorted([p for p in migrations_dir.glob("*.sql") if p.name.endswith(".sql") and not p.name.endswith("_down.sql")])
        for path in migrations:
            version = path.stem
            if version in already:
                continue
            apply_migration(conn, version, path)
            print(f"Applied {version}")
    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
