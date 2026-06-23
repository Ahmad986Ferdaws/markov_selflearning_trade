from collections.abc import Generator

from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db import get_db


def get_settings_dep() -> Settings:
    return get_settings()


def get_db_dep() -> Generator[Session, None, None]:
    yield from get_db()
