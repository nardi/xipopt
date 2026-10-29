"""Shared fixtures: the PJRT CPU plugin and a fresh disk cache."""

import os
from collections.abc import Iterator

import cyxla
import pytest


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
