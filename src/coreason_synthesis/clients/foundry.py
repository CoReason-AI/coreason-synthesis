# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_synthesis

from typing import List, Optional

import requests

from coreason_synthesis.models import SyntheticTestCase
from coreason_synthesis.utils.http import create_retry_session


class FoundryClient:
    """
    Client for pushing synthetic test cases to Coreason Foundry.
    Handles authentication and retries.
    """

    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: int = 30, max_retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = create_retry_session(api_key=api_key, max_retries=max_retries)

    def push_cases(self, cases: List[SyntheticTestCase]) -> int:
        """
        Pushes a list of synthetic test cases to the Foundry API.

        Args:
            cases: List of SyntheticTestCase objects to push.

        Returns:
            The number of cases successfully pushed (as reported by the API or the list length).
        """
        if not cases:
            return 0

        # Serialize cases to list of dicts
        # model_dump(mode='json') handles UUIDs and Enums correctly for JSON serialization
        payload = [case.model_dump(mode="json") for case in cases]

        try:
            # Endpoint: /api/v1/test-cases
            url = f"{self.base_url}/api/v1/test-cases"
            response = self.session.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()

            # Assuming API returns a JSON with count or we just trust successful 2xx implies all were received.
            # If the API returns detailed status, we might parse it.
            # For now, we assume standard behavior: 200 OK means batch accepted.
            return len(cases)

        except requests.RequestException as e:
            # Propagate exception
            raise e
