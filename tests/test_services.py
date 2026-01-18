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
from pydantic import BaseModel
from requests import RequestException

from coreason_synthesis.analyzer import TemplateAnalysis
from coreason_synthesis.appraiser import AppraisalAnalysis
from coreason_synthesis.clients.mcp import HttpMCPClient
from coreason_synthesis.compositor import GenerationOutput
from coreason_synthesis.mocks.embedding import DummyEmbeddingService
from coreason_synthesis.mocks.mcp import MockMCPClient
from coreason_synthesis.mocks.teacher import MockTeacher
from coreason_synthesis.models import Document, SynthesisTemplate


def test_dummy_embedding_service() -> None:
    """Test DummyEmbeddingService determinism and shape."""
    service = DummyEmbeddingService(dimension=10)
    vec1 = service.embed("hello")
    vec2 = service.embed("hello")
    vec3 = service.embed("world")

    assert len(vec1) == 10
    assert vec1 == vec2
    assert vec1 != vec3


def test_mock_mcp_client() -> None:
    """Test MockMCPClient behavior."""
    docs = [Document(content="test", source_urn="urn:1")]
    client = MockMCPClient(documents=docs)

    results = client.search([0.1], {}, 10)
    assert len(results) == 1
    assert results[0].content == "test"
    assert client.last_query_vector == [0.1]


def test_mock_teacher_generate_default() -> None:
    """Test that MockTeacher.generate returns default mock response."""
    teacher = MockTeacher()
    assert teacher.generate("some prompt") == "Mock generated response"


def test_mock_teacher_generate_structure() -> None:
    """Test that MockTeacher.generate returns structure-specific response."""
    teacher = MockTeacher()
    response = teacher.generate("Describe structure")
    assert "Structure: Question + JSON Output" in response
    assert "Complexity: Requires multi-hop reasoning" in response


class TestMockTeacherStructured:
    class GenericModel(BaseModel):
        field: str

    def test_unknown_model_raises(self) -> None:
        teacher = MockTeacher()
        with pytest.raises(NotImplementedError):
            teacher.generate_structured("prompt", self.GenericModel)

    def test_synthesis_template(self) -> None:
        teacher = MockTeacher()
        result = teacher.generate_structured("prompt", SynthesisTemplate)
        assert isinstance(result, SynthesisTemplate)
        assert result.structure == "Question + JSON Output"

    def test_template_analysis(self) -> None:
        teacher = MockTeacher()
        result = teacher.generate_structured("prompt", TemplateAnalysis)
        assert isinstance(result, TemplateAnalysis)
        assert result.structure == "Question + JSON Output"

    def test_generation_output(self) -> None:
        teacher = MockTeacher()
        result = teacher.generate_structured("prompt", GenerationOutput)
        assert isinstance(result, GenerationOutput)
        assert result.synthetic_question == "Synthetic question?"

    def test_appraisal_analysis(self) -> None:
        teacher = MockTeacher()
        result = teacher.generate_structured("prompt", AppraisalAnalysis)
        assert isinstance(result, AppraisalAnalysis)
        assert result.complexity_score == 5.0


class TestHttpMCPClient:
    """Tests for HttpMCPClient."""

    @pytest.fixture
    def client(self) -> HttpMCPClient:
        return HttpMCPClient(base_url="http://mock-mcp", api_key="test-key")

    @patch("requests.Session.post")
    def test_search_success(self, mock_post: MagicMock, client: HttpMCPClient) -> None:
        """Test successful search request."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "content": "Doc 1 content",
                    "source_urn": "urn:doc:1",
                    "metadata": {"title": "Doc 1"},
                }
            ]
        }
        mock_post.return_value = mock_response

        docs = client.search(query_vector=[0.1, 0.2], user_context={"user": "test"}, limit=10)

        assert len(docs) == 1
        assert docs[0].content == "Doc 1 content"
        assert docs[0].source_urn == "urn:doc:1"
        assert docs[0].metadata["title"] == "Doc 1"

        # Verify headers
        assert client.session.headers["Authorization"] == "Bearer test-key"

        # Verify payload
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert kwargs["json"]["vector"] == [0.1, 0.2]
        assert kwargs["json"]["limit"] == 10

    @patch("requests.Session.post")
    def test_search_failure(self, mock_post: MagicMock, client: HttpMCPClient) -> None:
        """Test search failure handling."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.HTTPError("500 Error")
        mock_post.return_value = mock_response

        with pytest.raises(RequestException):
            client.search(query_vector=[0.1, 0.2], user_context={"user": "test"}, limit=10)

    @patch("requests.Session.post")
    def test_search_retry_logic(self, mock_post: MagicMock, client: HttpMCPClient) -> None:
        """
        Test logic for retries using the adapter.
        Since we are mocking session.post, the adapter logic isn't triggered directly.
        We can only verify the adapter is mounted.
        """
        assert "https://" in client.session.adapters
        assert "http://" in client.session.adapters

        adapter = client.session.adapters["https://"]
        assert adapter.max_retries.total == 3  # type: ignore

    def test_init_no_api_key(self) -> None:
        """Test initialization without API key."""
        client = HttpMCPClient(base_url="http://mock-mcp")
        assert "Authorization" not in client.session.headers
