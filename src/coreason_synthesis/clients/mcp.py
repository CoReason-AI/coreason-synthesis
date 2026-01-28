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
MCP client module.

This module provides the client interface for interacting with the Model
Context Protocol (MCP) service to search and retrieve documents.
"""

from typing import List, Optional

import httpx
from coreason_identity.models import UserContext

from coreason_synthesis.interfaces import MCPClient
from coreason_synthesis.models import Document


class HttpMCPClient(MCPClient):
    """Concrete implementation of the MCP Client using HTTP requests.

    Connects to an MCP-compliant REST API to perform vector-based searches.
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        """Initializes the HttpMCPClient.

        Args:
            base_url: The base URL of the MCP service.
            api_key: Optional API key for authentication.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retries for failed requests.
            client: Optional httpx.AsyncClient to use for requests.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        # If a client is provided, we use it. Otherwise, we create one internally.
        # Note: In a proper async context, we should manage the lifecycle of the internal client.
        self._internal_client = client is None
        self._client = client or httpx.AsyncClient(headers=headers, timeout=timeout)

    async def close(self) -> None:
        """Closes the underlying HTTP client if it was created internally."""
        if self._internal_client:
            await self._client.aclose()

    async def search(self, query_vector: List[float], user_context: UserContext, limit: int) -> List[Document]:
        """Searches the MCP for relevant documents.

        Args:
            query_vector: The embedding vector to search with.
            user_context: Context for RBAC and access control.
            limit: Maximum number of documents to retrieve.

        Returns:
            List of retrieved Document objects.

        Raises:
            httpx.HTTPError: If the API request fails.
        """
        payload = {"vector": query_vector, "context": user_context.model_dump(mode="json"), "limit": limit}

        try:
            response = await self._client.post(f"{self.base_url}/search", json=payload)
            response.raise_for_status()

            data = response.json()
            documents = []
            for item in data.get("results", []):
                # Use unpacking to leverage Pydantic validation (raises ValidationError if invalid)
                documents.append(Document(**item))
            return documents

        except httpx.HTTPError as e:
            # Propagate exception or handle it. For now, propagate so caller knows it failed.
            raise e
