# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_synthesis

import pytest
import requests
from requests.adapters import HTTPAdapter

from coreason_synthesis.utils.http import create_retry_session


def test_create_session_defaults() -> None:
    """Test creating session with defaults."""
    session = create_retry_session()

    # Check retry adapter is mounted
    assert "https://" in session.adapters
    assert "http://" in session.adapters

    adapter = session.adapters["https://"]
    assert isinstance(adapter, HTTPAdapter)
    assert adapter.max_retries.total == 3 # type: ignore

    # Check no auth header by default
    assert "Authorization" not in session.headers


def test_create_session_custom_retries() -> None:
    """Test creating session with custom retry counts."""
    # Zero retries
    session_zero = create_retry_session(max_retries=0)
    adapter_zero = session_zero.adapters["https://"]
    assert adapter_zero.max_retries.total == 0 # type: ignore

    # Many retries
    session_many = create_retry_session(max_retries=10)
    adapter_many = session_many.adapters["https://"]
    assert adapter_many.max_retries.total == 10 # type: ignore


def test_create_session_with_api_key() -> None:
    """Test creating session with API key."""
    key = "secret-token"
    session = create_retry_session(api_key=key)

    assert "Authorization" in session.headers
    assert session.headers["Authorization"] == f"Bearer {key}"


def test_create_session_adapter_methods() -> None:
    """Test that adapter handles standard methods."""
    session = create_retry_session()
    adapter = session.adapters["https://"]
    allowed = adapter.max_retries.allowed_methods # type: ignore
    assert "POST" in allowed
    assert "GET" in allowed
    assert "PUT" in allowed
    assert "DELETE" in allowed
