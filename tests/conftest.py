"""Shared fixtures: the PJRT CPU plugin and a fresh disk cache."""

import os
from collections.abc import Iterator
from pathlib import Path

import cyxla
import pytest

TESTS_DIRECTORY = Path(__file__).parent
"""The directory that holds the `unit` and `integration` directories."""


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Marks each test as `unit` when it is under `tests/unit`, and `integration` otherwise.

    The code blocks of the README and of the documentation count as integration tests.
    """
    for item in items:
        is_unit = (TESTS_DIRECTORY / "unit") in item.path.parents
        item.add_marker(pytest.mark.unit if is_unit else pytest.mark.integration)


@pytest.fixture(scope="session", autouse=True)
def runtime(tmp_path_factory: pytest.TempPathFactory) -> Iterator[cyxla.RuntimeInfo]:
    """Loads the installed CPU plugin, with a fresh disk cache."""
    previous = os.environ.get(cyxla.CACHE_ENVIRONMENT_VARIABLE)
    os.environ[cyxla.CACHE_ENVIRONMENT_VARIABLE] = str(tmp_path_factory.mktemp("cache"))
    yield cyxla.initialize(platform="cpu")
    if previous is None:
        del os.environ[cyxla.CACHE_ENVIRONMENT_VARIABLE]
    else:
        os.environ[cyxla.CACHE_ENVIRONMENT_VARIABLE] = previous
