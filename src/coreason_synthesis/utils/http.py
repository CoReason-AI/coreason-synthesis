# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_synthesis

from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry


def create_retry_session(
    api_key: Optional[str] = None,
    max_retries: int = 3,
) -> requests.Session:
    """
    Creates a requests.Session with retry logic and optional authentication.

    Args:
        api_key: Optional Bearer token for Authorization header.
        max_retries: Number of retries for the adapter.

    Returns:
        Configured requests.Session object.
    """
    # Configure Retry Strategy
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=1,  # Exponential backoff: 1s, 2s, 4s...
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST", "GET", "PUT", "DELETE"],  # Expanded allowed methods for generality
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)

    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    if api_key:
        session.headers.update({"Authorization": f"Bearer {api_key}"})

    return session
