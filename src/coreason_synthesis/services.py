from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from .interfaces import TeacherModel


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
