import uuid
from typing import Optional, Type, TypeVar

import numpy as np
import pytest
from pydantic import BaseModel

from coreason_synthesis.analyzer import PatternAnalyzerImpl, TemplateAnalysis
from coreason_synthesis.models import SeedCase, SynthesisTemplate
from coreason_synthesis.services import DummyEmbeddingService, MockTeacher

T = TypeVar("T", bound=BaseModel)


@pytest.fixture
def analyzer() -> PatternAnalyzerImpl:
    # Update MockTeacher to handle TemplateAnalysis specifically if needed,
    # but the default implementation in services.py checks for "TemplateAnalysis" in name.
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
    """Test that the analyzer correctly parses the MockTeacher output via generate_structured."""
    seed = SeedCase(id=uuid.uuid4(), context="Context", question="Question", expected_output="Output")

    template = analyzer.analyze([seed])

    assert isinstance(template, SynthesisTemplate)
    # Checks against values returned by MockTeacher.generate_structured for TemplateAnalysis
    assert template.structure == "Question + JSON Output"
    assert template.complexity_description == "Requires multi-hop reasoning"
    assert template.domain == "Oncology / Inclusion Criteria"


def test_empty_seeds_error(analyzer: PatternAnalyzerImpl) -> None:
    """Test that analyzing an empty list raises ValueError."""
    with pytest.raises(ValueError, match="Seed list cannot be empty"):
        analyzer.analyze([])


def test_analyze_single_seed(analyzer: PatternAnalyzerImpl) -> None:
    """Test that for a single seed, the centroid matches the seed's embedding exactly."""
    seed = SeedCase(id=uuid.uuid4(), context="Single", question="Q", expected_output="A")
    expected_vec = analyzer.embedder.embed("Single")

    template = analyzer.analyze([seed])

    assert template.embedding_centroid is not None
    assert np.allclose(template.embedding_centroid, expected_vec)


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


def test_custom_teacher_structured_response() -> None:
    """Test with a custom mock teacher returning specific values."""

    class CustomTeacher(MockTeacher):
        def generate_structured(self, prompt: str, response_model: Type[T], context: Optional[str] = None) -> T:
            if response_model == TemplateAnalysis:
                return TemplateAnalysis(
                    structure="Custom Structure",
                    complexity_description="High",
                    domain="Finance",
                )  # type: ignore
            return super().generate_structured(prompt, response_model, context)

    embedder = DummyEmbeddingService(dimension=4)
    custom_analyzer = PatternAnalyzerImpl(CustomTeacher(), embedder)
    seed = SeedCase(id=uuid.uuid4(), context="C", question="Q", expected_output="A")

    template = custom_analyzer.analyze([seed])

    assert template.structure == "Custom Structure"
    assert template.complexity_description == "High"
    assert template.domain == "Finance"
