# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_synthesis

from typing import List

import numpy as np

from coreason_synthesis.interfaces import EmbeddingService


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
