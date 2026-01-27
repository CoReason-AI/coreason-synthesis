# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_synthesis

"""
Foundry client module.

This module provides the client interface for interacting with the Coreason
Foundry service, primarily for pushing generated test cases to the staging area.
"""

from typing import List, Optional

from coreason_identity.models import UserContext

from coreason_synthesis.models import SyntheticTestCase
from coreason_synthesis.utils.http import create_retry_session


class FoundryClient:
    """Client for pushing synthetic test cases to Coreason Foundry.

    Handles authentication, serialization, and retries when communicating
    with the Foundry API.
    """

    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: int = 30, max_retries: int = 3):
        """Initializes the FoundryClient.

        Args:
            base_url: The base URL of the Foundry service.
            api_key: Optional API key for authentication.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retries for failed requests.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = create_retry_session(api_key=api_key, max_retries=max_retries)

    def push_cases(self, cases: List[SyntheticTestCase], user_context: Optional[UserContext] = None) -> int:
        """Pushes a list of synthetic test cases to the Foundry API.

        Args:
            cases: List of SyntheticTestCase objects to push.
            user_context: Optional user context for identity propagation.

        Returns:
            The number of cases successfully pushed.

        Raises:
            requests.RequestException: If the API request fails after retries.
        """
        if not cases:
            return 0

        # Serialize cases to list of dicts
        # model_dump(mode='json') handles UUIDs and Enums correctly for JSON serialization
        payload = [case.model_dump(mode="json") for case in cases]

        # Prepare headers for identity propagation
        headers = {}
        if user_context:
            # Safely access downstream_token as it might not be in all UserContext versions
            token = getattr(user_context, "downstream_token", None)
            if token:
                headers["Authorization"] = f"Bearer {token}"
            headers["X-CoReason-On-Behalf-Of"] = user_context.sub

        # Endpoint: /api/v1/test-cases
        url = f"{self.base_url}/api/v1/test-cases"
        response = self.session.post(url, json=payload, headers=headers, timeout=self.timeout)
        response.raise_for_status()

        # Assuming API returns a JSON with count or we just trust successful 2xx implies all were received.
        # If the API returns detailed status, we might parse it.
        # For now, we assume standard behavior: 200 OK means batch accepted.
        return len(cases)
