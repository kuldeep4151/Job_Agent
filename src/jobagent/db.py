"""SQLite engine (WAL mode, so the dashboard can read while a run writes)."""

import os
from pathlib import Path

from sqlalchemy import JSON, event, inspect, text
from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel, create_engine

from jobagent import settings

# JOBAGENT_DB points a trial run at a scratch database instead of the real one.
DB_PATH = Path(os.environ.get("JOBAGENT_DB", settings.DATA_DIR / "jobs.db"))

_engine: Engine | None = None


def add_missing_columns(engine: Engine) -> list[str]:
    """Bring an existing database up to date with new nullable model fields (create_all only makes tables)."""
    inspector = inspect(engine)
    added = []
    with engine.begin() as connection:
        for table in SQLModel.metadata.sorted_tables:
            if not inspector.has_table(table.name):
                continue
            existing = {column["name"] for column in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in existing:
                    continue
                if not column.nullable or isinstance(column.type, JSON):
                    raise RuntimeError(
                        f"{table.name}.{column.name} is non-nullable or JSON; add it with a manual migration"
                    )
                ddl_type = column.type.compile(engine.dialect)
                connection.execute(text(f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {ddl_type}'))
                added.append(f"{table.name}.{column.name}")
    return added


def configure(path: Path) -> Engine:
    """Point the app (or a test) at a database file and create/upgrade the tables."""
    global _engine
    engine = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False})

    @event.listens_for(engine, "connect")
    def _pragmas(dbapi_connection, _record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.close()

    SQLModel.metadata.create_all(engine)
    add_missing_columns(engine)
    _engine = engine
    return engine


def engine() -> Engine:
    return _engine or configure(DB_PATH)


def session() -> Session:
    return Session(engine(), expire_on_commit=False)
