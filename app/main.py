from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import health, runs
from app.db import init_db
from app.exceptions import ApiError, api_error_handler


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Paper Trading Agent", version="0.1.0", lifespan=lifespan)
    app.add_exception_handler(ApiError, api_error_handler)
    app.include_router(health.router)
    app.include_router(runs.router)
    return app


app = create_app()
