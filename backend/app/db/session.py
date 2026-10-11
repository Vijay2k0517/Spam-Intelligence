"""SQLAlchemy engine and session management for SQLite."""

from pathlib import Path
from typing import Generator, Optional
from sqlalchemy import create_engine, Engine, event
from sqlalchemy.orm import sessionmaker, Session

from app.core.config import settings
from app.core.logging import get_logger
from app.db.base import Base

logger = get_logger(__name__)


def create_sqlite_engine(db_path_or_url: Optional[str] = None) -> Engine:
    """Create SQLAlchemy engine configured for SQLite with WAL mode and foreign key enforcement."""
    if db_path_or_url is None:
        db_file = settings.database_file
        db_file.parent.mkdir(parents=True, exist_ok=True)
        url = settings.database_url
    elif db_path_or_url.startswith("sqlite"):
        url = db_path_or_url
    else:
        file_path = Path(db_path_or_url)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite:///{file_path.resolve().as_posix()}"

    eng = create_engine(
        url,
        connect_args={"check_same_thread": False},
        echo=False,
    )

    # Enable SQLite foreign key constraints
    @event.listens_for(eng, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    return eng


# Application-level engine and sessionmaker
engine = create_sqlite_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db(target_engine: Optional[Engine] = None) -> None:
    """Create all database tables if they do not exist."""
    eng = target_engine or engine
    logger.info("Initializing SQLite database tables...")
    Base.metadata.create_all(bind=eng)
    logger.info("Database tables initialized successfully.")


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a database session and ensuring proper cleanup."""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
