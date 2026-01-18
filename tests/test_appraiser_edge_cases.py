# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_synthesis

from typing import List, Optional, Type, TypeVar

import pytest
from pydantic import BaseModel

from coreason_synthesis.appraiser import AppraisalAnalysis, AppraiserImpl
from coreason_synthesis.models import (
    ProvenanceType,
    SynthesisTemplate,
    SyntheticTestCase,
)
from coreason_synthesis.services import DummyEmbeddingService, MockTeacher

T = TypeVar("T", bound=BaseModel)


class MockJudge(MockTeacher):
    """Mock Teacher that returns specific appraisal scores."""

    def __init__(self, complexity: float = 5.0, ambiguity: float = 2.0, validity: float = 0.9) -> None:
        self.complexity = complexity
        self.ambiguity = ambiguity
        self.validity = validity

    def generate_structured(self, prompt: str, response_model: Type[T], context: Optional[str] = None) -> T:
        if response_model == AppraisalAnalysis:
            return AppraisalAnalysis(
                complexity_score=self.complexity,
                ambiguity_score=self.ambiguity,
                validity_confidence=self.validity,
            )  # type: ignore
        return super().generate_structured(prompt, response_model, context)


@pytest.fixture
def base_case() -> SyntheticTestCase:
    return SyntheticTestCase(
        verbatim_context="Test context",
        synthetic_question="Test question?",
        golden_chain_of_thought="Test logic",
        expected_json={"answer": "yes"},
        provenance=ProvenanceType.VERBATIM_SOURCE,
        source_urn="urn:test",
        complexity=0.0,
        diversity=0.0,
        validity_confidence=0.0,
    )


@pytest.fixture
def template() -> SynthesisTemplate:
    return SynthesisTemplate(
        structure="QA",
        complexity_description="Medium",
        domain="General",
        embedding_centroid=[1.0, 0.0, 0.0],  # Simple unit vector along X
    )


def test_teacher_failure_propagation(base_case: SyntheticTestCase, template: SynthesisTemplate) -> None:
    """Test that if the teacher fails (raises exception), it propagates up."""

    class FailingTeacher(MockTeacher):
        def generate_structured(self, prompt: str, response_model: Type[T], context: Optional[str] = None) -> T:
            raise RuntimeError("Teacher model failure")

    appraiser = AppraiserImpl(FailingTeacher(), DummyEmbeddingService())

    with pytest.raises(RuntimeError, match="Teacher model failure"):
        appraiser.appraise([base_case], template)


def test_boundary_scores(base_case: SyntheticTestCase, template: SynthesisTemplate) -> None:
    """Test handling of boundary scores (0.0 and 10.0)."""
    # Teacher returns 0.0 complexity
    teacher_low = MockJudge(complexity=0.0, validity=1.0)
    appraiser_low = AppraiserImpl(teacher_low, DummyEmbeddingService(dimension=3))
    results_low = appraiser_low.appraise([base_case], template)
    assert results_low[0].complexity == 0.0

    # Teacher returns 10.0 complexity
    teacher_high = MockJudge(complexity=10.0, validity=1.0)
    appraiser_high = AppraiserImpl(teacher_high, DummyEmbeddingService(dimension=3))
    results_high = appraiser_high.appraise([base_case], template)
    assert results_high[0].complexity == 10.0


def test_nan_embeddings(base_case: SyntheticTestCase, template: SynthesisTemplate) -> None:
    """Test handling of NaN values in embeddings."""

    class NanEmbedder(DummyEmbeddingService):
        def embed(self, text: str) -> List[float]:
            return [float("nan"), 0.0, 0.0]

    appraiser = AppraiserImpl(MockJudge(), NanEmbedder())
    # Should not crash, diversity likely becomes nan or 0 depending on numpy handling
    # Numpy linalg norm of nan is nan.
    # Dot product with nan is nan.
    # We clip diversity to [0,1].
    # But comparison `case_norm > 0` with nan? nan > 0 is False.
    # So it skips calculation? Let's verify.
    # Wait, nan > 0 is False?
    # Python: float('nan') > 0 is False.
    # So `case_norm > 0` check protects us if norm is nan!
    # Result diversity should be 0.0 (default initialization).

    results = appraiser.appraise([base_case], template)
    assert results[0].diversity == 0.0
