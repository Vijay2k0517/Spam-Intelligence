"""Version 1 API router combining health, classification, models, experiments, and history."""

from fastapi import APIRouter

from app.api.v1.classify import router as classify_router
from app.api.v1.experiments import router as experiments_router
from app.api.v1.health import router as health_router
from app.api.v1.history import router as history_router
from app.api.v1.models import router as models_router

api_v1_router = APIRouter()

api_v1_router.include_router(health_router)
api_v1_router.include_router(classify_router)
api_v1_router.include_router(models_router)
api_v1_router.include_router(experiments_router)
api_v1_router.include_router(history_router)
