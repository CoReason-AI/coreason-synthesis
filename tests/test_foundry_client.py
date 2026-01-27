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
from coreason_identity.models import UserContext
from requests import RequestException

from coreason_synthesis.clients.foundry import FoundryClient
from coreason_synthesis.models import ProvenanceType, SyntheticTestCase


class TestFoundryClient:
    @pytest.fixture
    def client(self) -> FoundryClient:
        return FoundryClient(base_url="http://mock-foundry", api_key="test-key")

    @pytest.fixture
    def sample_case(self) -> SyntheticTestCase:
        return SyntheticTestCase(
            verbatim_context="Context",
            synthetic_question="Question?",
            golden_chain_of_thought="Reasoning",
            expected_json={"answer": "yes"},
            provenance=ProvenanceType.VERBATIM_SOURCE,
            source_urn="urn:test",
            modifications=[],
            complexity=5.0,
            diversity=0.5,
            validity_confidence=0.9,
        )

    @patch("requests.Session.post")
    def test_push_cases_success(
        self, mock_post: MagicMock, client: FoundryClient, sample_case: SyntheticTestCase
    ) -> None:
        """Test successful push."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        count = client.push_cases([sample_case])

        assert count == 1

        # Verify call
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        # kwargs['url'] might be passed as positional arg or kwarg depending on requests implementation
        # but typically keyword if we called it with url=...
        # Wait, in implementation I called: self.session.post(url, json=payload, ...)
        # So url is the first positional argument.
        assert args[0] == "http://mock-foundry/api/v1/test-cases"
        assert len(kwargs["json"]) == 1
        assert kwargs["json"][0]["verbatim_context"] == "Context"
        # Check that provenance enum is serialized to string
        assert kwargs["json"][0]["provenance"] == "VERBATIM_SOURCE"

        # Verify Auth
        assert client.session.headers["Authorization"] == "Bearer test-key"

    def test_push_empty_list(self, client: FoundryClient) -> None:
        assert client.push_cases([]) == 0

    @patch("requests.Session.post")
    def test_push_failure(self, mock_post: MagicMock, client: FoundryClient, sample_case: SyntheticTestCase) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.HTTPError("500 Error")
        mock_post.return_value = mock_response

        with pytest.raises(RequestException):
            client.push_cases([sample_case])

    def test_init_without_key(self) -> None:
        client = FoundryClient(base_url="http://mock")
        assert "Authorization" not in client.session.headers

    @patch("requests.Session.post")
    def test_push_large_batch(
        self, mock_post: MagicMock, client: FoundryClient, sample_case: SyntheticTestCase
    ) -> None:
        """Test pushing a large batch of cases."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        # Create 100 cases
        cases = [sample_case for _ in range(100)]
        count = client.push_cases(cases)

        assert count == 100
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        assert len(kwargs["json"]) == 100

    @patch("requests.Session.post")
    def test_push_special_characters(
        self, mock_post: MagicMock, client: FoundryClient, sample_case: SyntheticTestCase
    ) -> None:
        """Test pushing cases with unicode/special characters."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        # Create case with special chars
        special_case = sample_case.model_copy()
        special_case.verbatim_context = "Unicode content: 💊 ⚡ テスト"
        special_case.synthetic_question = "Question with emoji 🚀?"

        count = client.push_cases([special_case])

        assert count == 1
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        payload = kwargs["json"][0]
        assert payload["verbatim_context"] == "Unicode content: 💊 ⚡ テスト"
        assert payload["synthetic_question"] == "Question with emoji 🚀?"

    @patch("requests.Session.post")
    def test_push_cases_with_identity(
        self, mock_post: MagicMock, client: FoundryClient, sample_case: SyntheticTestCase
    ) -> None:
        """Test push with identity propagation."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        # Use MagicMock to simulate UserContext with downstream_token
        # as the current installed version of coreason-identity might verify fields
        user_context = MagicMock(spec=UserContext)
        user_context.sub = "user123"
        user_context.downstream_token = "token456"

        count = client.push_cases([sample_case], user_context=user_context)

        assert count == 1

        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        headers = kwargs["headers"]
        assert headers["Authorization"] == "Bearer token456"
        assert headers["X-CoReason-On-Behalf-Of"] == "user123"
