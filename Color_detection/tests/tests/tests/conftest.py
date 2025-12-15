# tests/conftest.py
import pytest

@pytest.fixture(scope="session")
def tf():
    import tensorflow as tf
    return tf
