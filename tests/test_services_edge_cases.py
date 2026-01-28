# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_synthesis

import httpx
import pytest
import respx
from pydantic import ValidationError

from coreason_identity.models import UserContext
from coreason_synthesis.clients.mcp import HttpMCPClient


class TestHttpMCPClientEdgeCases:
    @pytest.fixture
    def client(self) -> HttpMCPClient:
        return HttpMCPClient(base_url="http://test.mcp")

    @respx.mock  # type: ignore[misc]
    @pytest.mark.asyncio
    async def test_search_empty_results(self, client: HttpMCPClient) -> None:
        respx.post("http://test.mcp/search").mock(return_value=httpx.Response(200, json={"results": []}))
        user_context = UserContext(sub="test_user", email="test@example.com")
        docs = await client.search([0.1], user_context, 10)
        assert docs == []

    @respx.mock  # type: ignore[misc]
    @pytest.mark.asyncio
    async def test_search_malformed_json_response(self, client: HttpMCPClient) -> None:
        # Server returns valid JSON but missing 'results' key
        respx.post("http://test.mcp/search").mock(return_value=httpx.Response(200, json={"data": []}))
        # Should return empty list (get('results', []) defaults to [])
        user_context = UserContext(sub="test_user", email="test@example.com")
        docs = await client.search([0.1], user_context, 10)
        assert docs == []

    @respx.mock  # type: ignore[misc]
    @pytest.mark.asyncio
    async def test_search_invalid_document_structure(self, client: HttpMCPClient) -> None:
        # 'content' is missing in one item
        bad_data = {"results": [{"source_urn": "u1"}]}
        respx.post("http://test.mcp/search").mock(return_value=httpx.Response(200, json=bad_data))

        # Catch specific ValidationError
        user_context = UserContext(sub="test_user", email="test@example.com")
        with pytest.raises(ValidationError):
            await client.search([0.1], user_context, 10)

    @respx.mock  # type: ignore[misc]
    @pytest.mark.asyncio
    async def test_search_unauthorized(self, client: HttpMCPClient) -> None:
        respx.post("http://test.mcp/search").mock(return_value=httpx.Response(401))

        user_context = UserContext(sub="test_user", email="test@example.com")
        with pytest.raises(httpx.HTTPStatusError) as exc:
            await client.search([0.1], user_context, 10)
        assert exc.value.response.status_code == 401

    @respx.mock  # type: ignore[misc]
    @pytest.mark.asyncio
    async def test_search_timeout(self, client: HttpMCPClient) -> None:
        # Simulate timeout
        respx.post("http://test.mcp/search").mock(side_effect=httpx.TimeoutException("Timeout"))

        user_context = UserContext(sub="test_user", email="test@example.com")
        with pytest.raises(httpx.TimeoutException):
            await client.search([0.1], user_context, 10)
