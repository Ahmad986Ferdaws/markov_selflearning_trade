"""Pins for the two small infrastructure modules: app/exceptions.py and app/db.py."""

from __future__ import annotations

import json

import pytest
from sqlalchemy import inspect
from sqlalchemy.orm import Session

from app import db as db_module
from app.config import Settings
from app.exceptions import ApiError, api_error_handler


# --- ApiError ------------------------------------------------------------------------
def test_api_error_defaults_to_400_and_keeps_its_message():
    exc = ApiError("boom")
    assert (exc.message, exc.status_code) == ("boom", 400)
    assert str(exc) == "boom"
    assert exc.args == ("boom",)


async def test_api_error_handler_returns_the_status_and_a_detail_body():
    response = await api_error_handler(None, ApiError("not here", 418))
    assert response.status_code == 418
    assert json.loads(response.body) == {"detail": "not here"}
    assert response.media_type == "application/json"
