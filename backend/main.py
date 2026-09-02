"""FastAPI application entry point.

Serves the REST API under /api and mounts the Plotly Dash dashboard at /.
Run with: python backend/main.py
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
from pydantic import BaseModel

from backend.config import settings
from backend.database import init_db
from backend.routers import experiments, metrics, runs
from backend.scheduler import start_scheduler

_scheduler = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _scheduler
    init_db()
    _scheduler = start_scheduler()
    yield
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)


app = FastAPI(title="Fine-tune Tracker", version="0.1.0", lifespan=lifespan)

app.include_router(experiments.router)
app.include_router(runs.router)
app.include_router(metrics.router)


class HealthOut(BaseModel):
    status: str


@app.get("/api/health", tags=["health"], response_model=HealthOut)
def health() -> HealthOut:
    return HealthOut(status="ok")


# Mount the Dash dashboard at the root. Imported lazily so the API can be
# tested without pulling in the full dashboard stack.
def _mount_dashboard() -> None:
    from dashboard.app import create_dash_app

    dash_app = create_dash_app()
    app.mount("/", WSGIMiddleware(dash_app.server))


_mount_dashboard()


def main() -> None:
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
