import uuid
from typing import List, Optional

import numpy as np
import pytest

from coreason_synthesis.analyzer import PatternAnalyzerImpl
from coreason_synthesis.models import SeedCase, SynthesisTemplate
from coreason_synthesis.services import DummyEmbeddingService, EmbeddingService, MockTeacher


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


def test_analyze_single_seed(analyzer: PatternAnalyzerImpl) -> None:
    """Test that for a single seed, the centroid matches the seed's embedding exactly."""
    seed = SeedCase(id=uuid.uuid4(), context="Single", question="Q", expected_output="A")
    expected_vec = analyzer.embedder.embed("Single")

    template = analyzer.analyze([seed])

    assert template.embedding_centroid is not None
    assert np.allclose(template.embedding_centroid, expected_vec)


def test_analyze_partial_teacher_response() -> None:
    """Test handling when teacher returns partial information."""

    class PartialTeacher(MockTeacher):
        def generate(self, prompt: str, context: Optional[str] = None) -> str:
            return "Structure: Custom Structure"

    embedder = DummyEmbeddingService(dimension=4)
    analyzer = PatternAnalyzerImpl(PartialTeacher(), embedder)
    seed = SeedCase(id=uuid.uuid4(), context="C", question="Q", expected_output="A")

    template = analyzer.analyze([seed])

    assert template.structure == "Custom Structure"
    assert template.complexity_description == "Unknown Complexity"
    assert template.domain == "Unknown Domain"


def test_analyze_empty_teacher_response() -> None:
    """Test handling when teacher returns an empty string."""

    class EmptyTeacher(MockTeacher):
        def generate(self, prompt: str, context: Optional[str] = None) -> str:
            return ""

    embedder = DummyEmbeddingService(dimension=4)
    analyzer = PatternAnalyzerImpl(EmptyTeacher(), embedder)
    seed = SeedCase(id=uuid.uuid4(), context="C", question="Q", expected_output="A")

    template = analyzer.analyze([seed])

    assert template.structure == "Unknown Structure"
    assert template.complexity_description == "Unknown Complexity"
    assert template.domain == "Unknown Domain"


def test_analyze_large_batch(analyzer: PatternAnalyzerImpl) -> None:
    """Test stability with a larger batch of seeds."""
    seeds = []
    for i in range(50):
        seeds.append(SeedCase(id=uuid.uuid4(), context=f"Context {i}", question=f"Q{i}", expected_output=f"A{i}"))

    # Just ensure it runs without error and returns a valid template
    template = analyzer.analyze(seeds)
    assert isinstance(template, SynthesisTemplate)
    assert template.embedding_centroid is not None
    assert len(template.embedding_centroid) == 4


def test_inconsistent_embedding_dimensions() -> None:
    """Test behavior when embedding service returns inconsistent dimensions."""

    class BadEmbedder(EmbeddingService):
        def __init__(self) -> None:
            self.call_count = 0

        def embed(self, text: str) -> List[float]:
            self.call_count += 1
            if self.call_count == 1:
                return [1.0, 2.0]
            else:
                return [1.0, 2.0, 3.0]  # Dimension mismatch

    teacher = MockTeacher()
    embedder = BadEmbedder()
    analyzer = PatternAnalyzerImpl(teacher, embedder)

    seed1 = SeedCase(id=uuid.uuid4(), context="A", question="Q", expected_output="A")
    seed2 = SeedCase(id=uuid.uuid4(), context="B", question="Q", expected_output="A")

    # Numpy should raise a ValueError when trying to meanragged arrays or incompatible shapes
    # If it constructs an object array (because of ragged nested lists), mean might fail or return something weird.
    # We expect a crash or an error here, which is better than silent failure.
    with pytest.raises(ValueError):
        # Depending on numpy version, creating the array might fail, or the mean might fail.
        # np.mean on ragged list might raise or warn.
        # Let's see what happens.
        analyzer.analyze([seed1, seed2])
