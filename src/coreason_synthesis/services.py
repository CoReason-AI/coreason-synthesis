# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_synthesis

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, TypeVar

import numpy as np
import requests
from pydantic import BaseModel
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from .interfaces import TeacherModel
from .models import Document, SyntheticTestCase

T = TypeVar("T", bound=BaseModel)


class EmbeddingService(ABC):
    """Abstract interface for embedding generation."""

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """
        Generates a vector embedding for the given text.

        Args:
            text: The input text string.

        Returns:
            A list of floats representing the embedding vector.
        """
        pass


class DummyEmbeddingService(EmbeddingService):
    """Deterministic mock embedding service for testing."""

    def __init__(self, dimension: int = 1536):
        self.dimension = dimension

    def embed(self, text: str) -> List[float]:
        """
        Returns a deterministic pseudo-random vector based on text length.
        This ensures the same text always gets the same vector in tests.
        """
        # Use a seed based on text content for determinism
        seed = sum(ord(c) for c in text)
        rng = np.random.default_rng(seed)
        # Explicitly cast to List[float] for mypy
        vector: List[float] = rng.random(self.dimension).tolist()
        return vector


class MCPClient(ABC):
    """Abstract interface for the Model Context Protocol (MCP) client."""

    @abstractmethod
    def search(self, query_vector: List[float], user_context: Dict[str, Any], limit: int) -> List[Document]:
        """
        Searches the MCP for relevant documents using a vector query.

        Args:
            query_vector: The embedding vector to search with.
            user_context: Context for RBAC (e.g., auth token).
            limit: Maximum number of documents to retrieve.

        Returns:
            List of retrieved Documents.
        """
        pass


class MockMCPClient(MCPClient):
    """Mock MCP Client for testing."""

    def __init__(self, documents: Optional[List[Document]] = None):
        self.documents = documents or []
        self.last_query_vector: List[float] = []
        self.last_user_context: Dict[str, Any] = {}
        self.last_limit = 0

    def search(self, query_vector: List[float], user_context: Dict[str, Any], limit: int) -> List[Document]:
        self.last_query_vector = query_vector
        self.last_user_context = user_context
        self.last_limit = limit
        # Return all docs (filtering logic is in Forager, usually MCP does vector search too)
        # For test, we just return the pre-seeded docs limited by input or available
        return self.documents[:limit]


class HttpMCPClient(MCPClient):
    """
    Concrete implementation of the MCP Client using requests.
    Handles rate limiting and retries.
    """

    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: int = 30, max_retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

        # Configure Retry Strategy
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,  # Exponential backoff: 1s, 2s, 4s...
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"],  # Retry on POST as search is idempotent-ish
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)

        self.session = requests.Session()
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})

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


class MockTeacher(TeacherModel):
    """Deterministic mock teacher model for testing."""

    def generate(self, prompt: str, context: Optional[str] = None) -> str:
        """
        Returns a mock response based on the prompt content.
        """
        if "structure" in prompt.lower():
            return (
                "Structure: Question + JSON Output\n"
                "Complexity: Requires multi-hop reasoning\n"
                "Domain: Oncology / Inclusion Criteria"
            )
        return "Mock generated response"

    def generate_structured(self, prompt: str, response_model: Type[T], context: Optional[str] = None) -> T:
        """
        Returns a mock structured response based on the prompt content and response model.
        """
        # We need to construct a dummy instance of T.
        # This is tricky without knowing the exact structure of T, but for our tests we know what T will be.
        # However, to be generic in the mock, we can try to instantiate it with default values or known test values.

        # Check if the response_model is one we expect
        if "SynthesisTemplate" in response_model.__name__ or "TemplateAnalysis" in response_model.__name__:
            # We can return a dict compatible with the expected fields, validated by the model
            # Note: SynthesisTemplate requires embedding_centroid, but TemplateAnalysis (local model) might not.
            # We'll assume the caller uses a model compatible with these fields.
            try:
                # Attempt to instantiate with test data
                return response_model(
                    structure="Question + JSON Output",
                    complexity_description="Requires multi-hop reasoning",
                    domain="Oncology / Inclusion Criteria",
                    embedding_centroid=[0.1, 0.2, 0.3],  # Dummy centroid if needed
                )
            except Exception:  # pragma: no cover
                # If T doesn't match above, try to construct with defaults if possible, or raise
                pass
        elif "GenerationOutput" in response_model.__name__:
            try:
                return response_model(
                    synthetic_question="Synthetic question?",
                    golden_chain_of_thought="Step 1. Step 2.",
                    expected_json={"result": "value"},
                )
            except Exception:  # pragma: no cover
                pass
        elif "AppraisalAnalysis" in response_model.__name__:
            try:
                return response_model(
                    complexity_score=5.0,
                    ambiguity_score=2.0,
                    validity_confidence=0.9,
                )
            except Exception:  # pragma: no cover
                pass

        # If we can't determine what to return, we might need a more sophisticated mock or hardcode for specific tests.
        # For now, let's try to construct it with dummy data if it's a simple model, or raise NotImplementedError
        # allowing tests to patch it if needed.
        raise NotImplementedError(
            f"MockTeacher.generate_structured does not know how to mock {response_model.__name__}"
        )


class FoundryClient:
    """
    Client for pushing synthetic test cases to Coreason Foundry.
    Handles authentication and retries.
    """

    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: int = 30, max_retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

        # Configure Retry Strategy
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)

        self.session = requests.Session()
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def push_cases(self, cases: List[SyntheticTestCase]) -> int:
        """
        Pushes a list of synthetic test cases to the Foundry API.

        Args:
            cases: List of SyntheticTestCase objects to push.

        Returns:
            The number of cases successfully pushed (as reported by the API or the list length).
        """
        if not cases:
            return 0

        # Serialize cases to list of dicts
        # model_dump(mode='json') handles UUIDs and Enums correctly for JSON serialization
        payload = [case.model_dump(mode="json") for case in cases]

        try:
            # Endpoint: /api/v1/test-cases
            url = f"{self.base_url}/api/v1/test-cases"
            response = self.session.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()

            # Assuming API returns a JSON with count or we just trust successful 2xx implies all were received.
            # If the API returns detailed status, we might parse it.
            # For now, we assume standard behavior: 200 OK means batch accepted.
            return len(cases)

        except requests.RequestException as e:
            # Propagate exception
            raise e
