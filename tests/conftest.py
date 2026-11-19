import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.model import train_model


@pytest.fixture(scope="session")
def trained():
    """Train once for the whole test run (training is the slow part)."""
    return train_model()


@pytest.fixture
def model(trained):
    return trained[0]


@pytest.fixture
def client():
    return TestClient(app)
