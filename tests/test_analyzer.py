import uuid
from typing import Optional

import numpy as np
import pytest

from coreason_synthesis.analyzer import PatternAnalyzerImpl
from coreason_synthesis.models import SeedCase, SynthesisTemplate
from coreason_synthesis.services import DummyEmbeddingService, MockTeacher


@pytest.fixture  # type: ignore[misc]
def analyzer() -> PatternAnalyzerImpl:
    teacher = MockTeacher()
    embedder = DummyEmbeddingService(dimension=4)  # Small dimension for easy checking
    return PatternAnalyzerImpl(teacher, embedder)


def test_centroid_calculation(analyzer: PatternAnalyzerImpl) -> None:
    """Test that the centroid is correctly calculated as the mean of embeddings."""
    seed1 = SeedCase(id=uuid.uuid4(), context="A", question="Q1", expected_output="A1")
    seed2 = SeedCase(id=uuid.uuid4(), context="B", question="Q2", expected_output="A2")

    vec1 = analyzer.embedder.embed("A")
    vec2 = analyzer.embedder.embed("B")
    expected_centroid = np.mean([vec1, vec2], axis=0).tolist()

    template = analyzer.analyze([seed1, seed2])

    assert template.embedding_centroid is not None
    assert np.allclose(template.embedding_centroid, expected_centroid)


def test_analyze_flow_mock_teacher(analyzer: PatternAnalyzerImpl) -> None:
    """Test that the analyzer correctly parses the MockTeacher output."""
    seed = SeedCase(id=uuid.uuid4(), context="Context", question="Question", expected_output="Output")

    template = analyzer.analyze([seed])

    assert isinstance(template, SynthesisTemplate)
    # Checks against strings defined in MockTeacher
    assert template.structure == "Question + JSON Output"
    assert template.complexity_description == "Requires multi-hop reasoning"
    assert template.domain == "Oncology / Inclusion Criteria"


def test_analyze_flow_mock_teacher_fallback(analyzer: PatternAnalyzerImpl) -> None:
    """Test the fallback logic when the teacher returns an unexpected response."""

    # Custom MockTeacher for this test
    class BadTeacher(MockTeacher):
        def generate(self, prompt: str, context: Optional[str] = None) -> str:
            return "Unexpected format"

    embedder = DummyEmbeddingService(dimension=4)
    bad_analyzer = PatternAnalyzerImpl(BadTeacher(), embedder)

    seed = SeedCase(id=uuid.uuid4(), context="Context", question="Question", expected_output="Output")

    template = bad_analyzer.analyze([seed])

    assert template.structure == "Unknown Structure"
    assert template.complexity_description == "Unknown Complexity"
    assert template.domain == "Unknown Domain"


def test_empty_seeds_error(analyzer: PatternAnalyzerImpl) -> None:
    """Test that analyzing an empty list raises ValueError."""
    with pytest.raises(ValueError, match="Seed list cannot be empty"):
        analyzer.analyze([])


def test_mock_teacher_fallback() -> None:
    """Ensure MockTeacher returns fallback string for unknown prompts."""
    teacher = MockTeacher()
    assert teacher.generate("unknown prompt") == "Mock generated response"
