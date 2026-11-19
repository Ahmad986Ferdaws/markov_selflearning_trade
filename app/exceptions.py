"""The one exception type the web surface turns into a JSON error body."""

from fastapi import Request
from fastapi.responses import JSONResponse


class ApiError(Exception):
    """A client-facing error: ``message`` becomes ``{"detail": message}`` with
    ``status_code`` (default 400)."""

    def __init__(self, message: str, status_code: int = 400):
        # Pass the message up so str(exc), tracebacks, and log lines show it
        # (previously they showed an empty string).
        super().__init__(message)
        self.message = message
        self.status_code = status_code


async def api_error_handler(_request: Request, exc: ApiError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})
