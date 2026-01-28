# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_synthesis

import json

import httpx
import pytest
import respx

from coreason_identity.models import UserContext
from coreason_synthesis.clients.mcp import HttpMCPClient


class TestHttpMCPClientComplex:
    @pytest.fixture
    def client(self) -> HttpMCPClient:
        return HttpMCPClient(base_url="http://test.mcp", api_key="secret")

    @respx.mock  # type: ignore[misc]
    @pytest.mark.asyncio
    async def test_search_json_decode_error(self, client: HttpMCPClient) -> None:
        """Server returns 200 OK but body is not JSON."""
        respx.post("http://test.mcp/search").mock(return_value=httpx.Response(200, text="Not JSON"))
        user_context = UserContext(sub="test_user", email="test@example.com")

        with pytest.raises(json.JSONDecodeError):
            await client.search([0.1], user_context, 10)

    @respx.mock  # type: ignore[misc]
    @pytest.mark.asyncio
    async def test_search_validation_error(self, client: HttpMCPClient) -> None:
        """Server returns JSON that doesn't match Document model."""
        # Missing 'content'
        invalid_docs = {"results": [{"source_urn": "u1"}]}
        respx.post("http://test.mcp/search").mock(return_value=httpx.Response(200, json=invalid_docs))
        user_context = UserContext(sub="test_user", email="test@example.com")

        # Pydantic validation error should be raised
        with pytest.raises(Exception) as exc:  # Catching broad exception to include ValidationError
            await client.search([0.1], user_context, 10)
        assert "validation error" in str(exc.value).lower()
