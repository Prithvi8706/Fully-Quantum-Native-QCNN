"""Shared fixtures for the freeze and protocol test suites."""
import pytest

from QCNN import freeze


@pytest.fixture(scope="session")
def headline_model():
    """The frozen headline model. Read-only: never mutate its parameters."""
    return freeze.build_headline_model()


@pytest.fixture(scope="session")
def archived_params(headline_model):
    return freeze.load_archived_params(headline_model)


@pytest.fixture(scope="session")
def regression_inputs():
    return freeze.fixed_regression_inputs()
