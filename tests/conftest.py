"""Shared pytest fixtures: isolated DB + FastAPI TestClient, no global state."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.database import get_session
from backend.models import Base
from backend.routers import experiments, metrics, runs


@pytest.fixture()
def session_factory():
    """A fresh in-memory SQLite database per test."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    yield factory
    engine.dispose()


@pytest.fixture()
def app(session_factory) -> FastAPI:
    """A FastAPI app wired to the test DB, without the Dash mount or scheduler."""
    application = FastAPI()
    application.include_router(experiments.router)
    application.include_router(runs.router)
    application.include_router(metrics.router)

    def _override_session():
        with session_factory() as session:
            yield session

    application.dependency_overrides[get_session] = _override_session
    return application


@pytest.fixture()
def client(app) -> TestClient:
    return TestClient(app)


@pytest.fixture()
def experiment(client) -> dict:
    resp = client.post("/api/experiments", json={"name": "exp-1"})
    assert resp.status_code == 201
    return resp.json()
