"""FastAPI dependency injection providers."""

from typing import Generator
from fastapi import Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.services.classification_service import ClassificationService
from app.services.experiment_service import ExperimentService
from app.services.history_service import HistoryService
from app.services.model_registry_service import ModelRegistryService


# Singleton service instances for stateless services
_registry_service = ModelRegistryService()
_experiment_service = ExperimentService()


def get_model_registry_service() -> ModelRegistryService:
    """Provide ModelRegistryService instance."""
    return _registry_service


def get_experiment_service() -> ExperimentService:
    """Provide ExperimentService instance."""
    return _experiment_service


def get_history_service(db: Session = Depends(get_db)) -> HistoryService:
    """Provide HistoryService instance with scoped database session."""
    return HistoryService(db=db)


def get_classification_service(
    db: Session = Depends(get_db),
    registry: ModelRegistryService = Depends(get_model_registry_service),
) -> ClassificationService:
    """Provide ClassificationService instance with database session and registry."""
    return ClassificationService(db=db, registry_service=registry)
