# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_synthesis

from unittest.mock import MagicMock, patch

import pytest
import requests
from pydantic import ValidationError
from requests.exceptions import JSONDecodeError, RequestException

from coreason_synthesis.clients.foundry import FoundryClient
from coreason_synthesis.clients.mcp import HttpMCPClient
from coreason_synthesis.models import ProvenanceType, SyntheticTestCase


class TestHttpMCPClientComplex:
    """Complex scenarios for HttpMCPClient."""

    @pytest.fixture
    def client(self) -> HttpMCPClient:
        return HttpMCPClient(base_url="http://mock-mcp", api_key="test-key")

    @patch("requests.Session.post")
    def test_search_json_decode_error(self, mock_post: MagicMock, client: HttpMCPClient) -> None:
        """Test handling of non-JSON response (e.g. 502 Bad Gateway HTML)."""
        mock_response = MagicMock()
        mock_response.status_code = 200  # Mimic an API that returns 200 OK but sends HTML or garbage
        mock_response.json.side_effect = JSONDecodeError("Expecting value", "doc", 0)
        mock_post.return_value = mock_response

        # Depending on implementation, this might raise JSONDecodeError or generic Exception
        # The current implementation calls response.json() inside try block but catches RequestException
        # JSONDecodeError is NOT a subclass of RequestException in older requests, but might be in newer.
        # Actually in requests < 2.27 it wasn't. Let's check.
        # If it propagates, we expect JSONDecodeError.

        with pytest.raises(JSONDecodeError):
            client.search(query_vector=[0.1], user_context={}, limit=1)

    @patch("requests.Session.post")
    def test_search_validation_error(self, mock_post: MagicMock, client: HttpMCPClient) -> None:
        """Test handling of valid JSON that fails Pydantic validation."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        # Return results missing required 'content' field
        mock_response.json.return_value = {"results": [{"source_urn": "urn:1"}]}
        mock_post.return_value = mock_response

        with pytest.raises(ValidationError):
            client.search(query_vector=[0.1], user_context={}, limit=1)

    def test_base_url_stripping(self) -> None:
        """Test that trailing slashes are stripped from base_url."""
        client = HttpMCPClient(base_url="http://api.example.com/")
        assert client.base_url == "http://api.example.com"

        client2 = HttpMCPClient(base_url="http://api.example.com")
        assert client2.base_url == "http://api.example.com"


class TestFoundryClientComplex:
    """Complex scenarios for FoundryClient."""

    @pytest.fixture
    def client(self) -> FoundryClient:
        return FoundryClient(base_url="http://foundry", api_key="key")

    def test_push_cases_empty_list(self, client: FoundryClient) -> None:
        """Test pushing an empty list of cases."""
        # Should return 0 and make no network calls
        with patch("requests.Session.post") as mock_post:
            count = client.push_cases([])
            assert count == 0
            mock_post.assert_not_called()

    @patch("requests.Session.post")
    def test_push_cases_serialization(self, mock_post: MagicMock, client: FoundryClient) -> None:
        """Test that complex fields (UUID, Enum) are serialized correctly."""
        # Create a case with UUID and Enum
        case = SyntheticTestCase(
            verbatim_context="ctx",
            synthetic_question="q",
            golden_chain_of_thought="cot",
            expected_json={"a": 1},
            provenance=ProvenanceType.SYNTHETIC_PERTURBED,  # Enum
            source_urn="urn:1",
            complexity=5.0,
            ambiguity=0.0,
            diversity=0.1,
            validity_confidence=0.9,
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        client.push_cases([case])

        # Verify payload
        args, kwargs = mock_post.call_args
        payload = kwargs["json"]
        assert len(payload) == 1
        item = payload[0]

        # Provenance should be string value, not Enum object
        assert item["provenance"] == "SYNTHETIC_PERTURBED"
        # Other fields
        assert item["complexity"] == 5.0

    @patch("requests.Session.post")
    def test_push_cases_server_error(self, mock_post: MagicMock, client: FoundryClient) -> None:
        """Test handling of 500 error from server."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.HTTPError("500 Server Error")
        mock_post.return_value = mock_response

        case = SyntheticTestCase(
            verbatim_context="ctx",
            synthetic_question="q",
            golden_chain_of_thought="cot",
            expected_json={},
            provenance=ProvenanceType.VERBATIM_SOURCE,
            source_urn="urn",
            complexity=1,
            ambiguity=0.0,
            diversity=0,
            validity_confidence=1,
        )

        with pytest.raises(RequestException):
            client.push_cases([case])
