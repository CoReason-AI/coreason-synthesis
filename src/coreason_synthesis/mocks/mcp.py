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
Mock MCP client for testing.
"""

from typing import Any, Dict, List, Optional

from coreason_synthesis.interfaces import MCPClient
from coreason_synthesis.models import Document


class MockMCPClient(MCPClient):
    """Mock MCP Client for testing."""

    def __init__(self, documents: Optional[List[Document]] = None):
        """Initializes the mock MCP client.

        Args:
            documents: List of pre-seeded documents to return in search.
        """
        self.documents = documents or []
        self.last_query_vector: List[float] = []
        self.last_user_context: Dict[str, Any] = {}
        self.last_limit = 0

    def search(self, query_vector: List[float], user_context: Dict[str, Any], limit: int) -> List[Document]:
        """Simulates a search by returning pre-loaded documents.

        Args:
            query_vector: The query vector (stored for verification).
            user_context: User context (stored for verification).
            limit: Limit (stored and applied).

        Returns:
            A slice of the pre-loaded documents.
        """
        self.last_query_vector = query_vector
        self.last_user_context = user_context
        self.last_limit = limit
        # Return all docs (filtering logic is in Forager, usually MCP does vector search too)
        # For test, we just return the pre-seeded docs limited by input or available
        return self.documents[:limit]
