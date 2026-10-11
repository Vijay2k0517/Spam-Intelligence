"""Main FastAPI application entrypoint with lifespan events, CORS, and centralized routing."""

from contextlib import asynccontextmanager
from typing import Any, Dict
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.router import api_v1_router
from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.db.session import engine, init_db
from app.ml.inference import get_pipeline
from app.schemas.common import ErrorDetail, ErrorResponse

logger = get_logger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context managing startup initialization and graceful shutdown."""
    # 1. Startup phase
    configure_logging(settings.log_level)
    logger.info("Starting %s v%s...", settings.app_name, settings.app_version)
    logger.info("API prefix: %s | Database: %s", settings.api_prefix, settings.database_file)

    # Initialize SQLite database schema
    try:
        init_db(engine)
        logger.info("SQLite database tables initialized successfully.")
    except Exception as e:
        logger.error("Failed to initialize database tables: %s", e, exc_info=True)

    # Verify and warm up default model artifact
    default_model = settings.default_model
    try:
        logger.info("Preloading default model artifact '%s' into in-memory cache...", default_model)
        get_pipeline(default_model)
        logger.info("Default model '%s' loaded and verified successfully.", default_model)
    except Exception as e:
        logger.warning(
            "Default model '%s' could not be preloaded at startup: %s. "
            "The API remains live, but predictions using this model will return 503 until resolved.",
            default_model,
            e,
        )

    yield

    # 2. Shutdown phase
    logger.info("Shutting down %s...", settings.app_name)
    engine.dispose()
    logger.info("Database engine connections closed.")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Production-quality REST API for real-time SMS spam classification "
        "using Naïve Bayes and Support Vector Machine classifiers across textual, "
        "handcrafted metadata, and hybrid feature spaces."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Cross-Origin Resource Sharing (CORS) Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register versioned API router
app.include_router(api_v1_router, prefix=settings.api_prefix)


@app.get(
    "/",
    summary="API Root Information",
    description="Returns high-level API metadata and documentation links.",
    tags=["Root"],
)
def root_endpoint() -> Dict[str, Any]:
    """Root endpoint welcoming clients and directing to documentation."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "api_v1": f"{settings.api_prefix}",
        "docs_url": "/docs",
        "openapi_url": "/openapi.json",
        "default_model": settings.default_model,
    }


# Standardized Exception Handlers


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic request validation failures with clean error structure."""
    details = [
        ErrorDetail(
            code="validation_error",
            field=" -> ".join(str(loc) for loc in err.get("loc", [])),
            message=err.get("msg", "Validation error"),
        )
        for err in exc.errors()
    ]
    logger.warning("Request validation error on %s: %s", request.url.path, details)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error="Validation Error",
            message="The request payload failed validation criteria.",
            details=details,
        ).model_dump(mode="json"),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle explicit HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=f"HTTP {exc.status_code}",
            message=str(exc.detail),
            details=None,
        ).model_dump(mode="json"),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Catch-all handler preventing internal path and traceback exposure to clients."""
    logger.error("Unhandled server exception on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal Server Error",
            message="An unexpected server error occurred. Please contact the administrator.",
            details=None,
        ).model_dump(mode="json"),
    )
