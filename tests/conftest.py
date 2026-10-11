"""Shared fixtures and paths for unit tests.

NOTE ON RESEARCH INTEGRITY:
These fixtures contain minimal synthetic examples designed solely to validate
software components (parsers, extractors, pipelines) and are NEVER presented
as real benchmark results.
"""

from pathlib import Path
import sys
import pytest

# Ensure backend package is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
FIXTURES_DIR = ROOT_DIR / "tests" / "fixtures"

for p in [str(BACKEND_DIR), str(ROOT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture(scope="session")
def test_fixtures_paths():
    """Return dictionary of static test fixture filepaths."""
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = FIXTURES_DIR / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    latin1_file = FIXTURES_DIR / "latin1_sample.tsv"
    if not latin1_file.exists():
        content = "spam\tWin £1000 cash! Café reward\nham\tRegular meeting\n"
        latin1_file.write_bytes(content.encode("latin-1"))

    return {
        "fixtures_dir": FIXTURES_DIR,
        "out_dir": out_dir,
        "tsv": FIXTURES_DIR / "synthetic_sms.tsv",
        "csv": FIXTURES_DIR / "synthetic_sms.csv",
        "malformed": FIXTURES_DIR / "malformed.tsv",
        "latin1": latin1_file,
    }


@pytest.fixture
def db_session(tmp_path):
    """Provide an isolated, temporary SQLite database session for unit tests."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.db.base import Base

    test_db_path = tmp_path / "test_app.db"
    test_engine = create_engine(
        f"sqlite:///{test_db_path.as_posix()}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=test_engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        test_engine.dispose()


@pytest.fixture
def client(db_session):
    """Provide FastAPI TestClient with database session overridden by test database."""
    from fastapi.testclient import TestClient
    from app.db.session import get_db
    from app.main import app

    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()

