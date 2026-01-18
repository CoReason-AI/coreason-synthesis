# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the License).
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_synthesis

from unittest.mock import MagicMock, patch

import pytest
import requests
from pydantic import BaseModel, ValidationError
from requests import RequestException

from coreason_synthesis.clients.mcp import HttpMCPClient
from coreason_synthesis.mocks.teacher import MockTeacher


class RandomModel(BaseModel):
    field: str


def test_mock_teacher_unknown_model_error() -> None:
    """Test that MockTeacher raises NotImplementedError for unknown models."""
    teacher = MockTeacher()
    with pytest.raises(
        NotImplementedError, match="MockTeacher.generate_structured does not know how to mock RandomModel"
    ):
        teacher.generate_structured("prompt", RandomModel)


def test_mock_teacher_synthesis_template_partial() -> None:
    """Test the exception block in MockTeacher by passing a model that matches name but not fields."""

    # Define a model with matching name but strict required fields that differ from default mock
    class SynthesisTemplate(BaseModel):
        required_field_not_in_mock: str

    teacher = MockTeacher()
    # This should hit the except Exception block and then fall through to NotImplementedError
    # or handle it gracefully if we change the implementation.
    # Current implementation: pass in except, then raise NotImplementedError at end.

    with pytest.raises(NotImplementedError):
        teacher.generate_structured("prompt", SynthesisTemplate)


class TestHttpMCPClientEdgeCases:
    """Edge case tests for HttpMCPClient."""

    @pytest.fixture
    def client(self) -> HttpMCPClient:
        return HttpMCPClient(base_url="http://mock-mcp", api_key="test-key")

    @patch("requests.Session.post")
    def test_search_empty_results(self, mock_post: MagicMock, client: HttpMCPClient) -> None:
        """Test search returning 200 OK but empty results list."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": []}
        mock_post.return_value = mock_response

        docs = client.search([0.1], {}, 10)
        assert docs == []

    @patch("requests.Session.post")
    def test_search_malformed_json_response(self, mock_post: MagicMock, client: HttpMCPClient) -> None:
        """Test response missing the 'results' key."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"not_results": []}
        mock_post.return_value = mock_response

        # Depending on implementation, this might return empty list (if .get("results") used)
        # or raise error. The implementation uses .get("results", []), so it should be empty list.
        docs = client.search([0.1], {}, 10)
        assert docs == []

    @patch("requests.Session.post")
    def test_search_invalid_document_structure(self, mock_post: MagicMock, client: HttpMCPClient) -> None:
        """Test response with items missing required fields."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        # Missing 'content' and 'source_urn'
        mock_response.json.return_value = {"results": [{"foo": "bar"}]}
        mock_post.return_value = mock_response

        # Pydantic validation error should be raised
        with pytest.raises(ValidationError):
            client.search([0.1], {}, 10)

    @patch("requests.Session.post")
    def test_search_unauthorized(self, mock_post: MagicMock, client: HttpMCPClient) -> None:
        """Test 401 Unauthorized response."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = requests.HTTPError("401 Client Error")
        mock_post.return_value = mock_response

        with pytest.raises(RequestException):
            client.search([0.1], {}, 10)

    @patch("requests.Session.post")
    def test_search_timeout(self, mock_post: MagicMock, client: HttpMCPClient) -> None:
        """Test network timeout."""
        mock_post.side_effect = requests.Timeout("Connection timed out")

        with pytest.raises(requests.Timeout):
            client.search([0.1], {}, 10)
