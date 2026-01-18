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
from pydantic import BaseModel

from .interfaces import TeacherModel
from .models import Document

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
            except Exception:
                # If T doesn't match above, try to construct with defaults if possible, or raise
                pass

        # If we can't determine what to return, we might need a more sophisticated mock or hardcode for specific tests.
        # For now, let's try to construct it with dummy data if it's a simple model, or raise NotImplementedError
        # allowing tests to patch it if needed.
        raise NotImplementedError(
            f"MockTeacher.generate_structured does not know how to mock {response_model.__name__}"
        )
