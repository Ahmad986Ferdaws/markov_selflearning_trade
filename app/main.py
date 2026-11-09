from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.api.routes import health, runs
from app.db import init_db
from app.exceptions import ApiError, api_error_handler


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    init_db()
    yield


DESCRIPTION = """\
The quarantined, simulation-only legacy intraday API: paper runs that poll a
price, apply the hardcoded risk guards, and fill against a simulated ledger.
No broker, wallet, key or on-chain code is reachable from here.

The LLM agent is CLI-only and is refused on this surface (a `strategy: agent`
run is a 400 unless `ALLOW_LEGACY_AGENT_API=true` is set deliberately for
local experimentation). The current product is `evaluate-cli`, the daily,
leakage-checked evaluation harness; this web app is kept for the legacy
`/runs` endpoints and is not where new work lands.
"""


def create_app() -> FastAPI:
    app = FastAPI(
        title="Paper Trading Agent",
        summary="Quarantined, simulation-only legacy intraday API (the LLM agent is CLI-only).",
        description=DESCRIPTION,
        version=__version__,
        license_info={
            "name": "MIT",
            "url": "https://github.com/Ahmad986Ferdaws/markov_selflearning_trade/blob/main/LICENSE",
        },
        lifespan=lifespan,
    )
    app.add_exception_handler(ApiError, api_error_handler)
    app.include_router(health.router)
    app.include_router(runs.router)
    return app


app = create_app()
