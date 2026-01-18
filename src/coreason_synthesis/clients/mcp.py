# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_synthesis

from typing import Any, Dict, List, Optional

import requests

from coreason_synthesis.interfaces import MCPClient
from coreason_synthesis.models import Document
from coreason_synthesis.utils.http import create_retry_session


class HttpMCPClient(MCPClient):
    """
    Concrete implementation of the MCP Client using requests.
    Handles rate limiting and retries.
    """

    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: int = 30, max_retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = create_retry_session(api_key=api_key, max_retries=max_retries)

    def search(self, query_vector: List[float], user_context: Dict[str, Any], limit: int) -> List[Document]:
        """
        Searches the MCP for relevant documents.
        """
        payload = {"vector": query_vector, "context": user_context, "limit": limit}

        try:
            response = self.session.post(f"{self.base_url}/search", json=payload, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()
            documents = []
            for item in data.get("results", []):
                # Use unpacking to leverage Pydantic validation (raises ValidationError if invalid)
                documents.append(Document(**item))
            return documents

        except requests.RequestException as e:
            # Propagate exception or handle it. For now, propagate so caller knows it failed.
            raise e
