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
from coreason_synthesis.mocks.embedding import DummyEmbeddingService
from coreason_synthesis.mocks.teacher import MockTeacher
from coreason_synthesis.models import (
    ProvenanceType,
    SynthesisTemplate,
    SyntheticTestCase,
)

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
        ambiguity=0.0,
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


def test_appraise_updates_metrics(base_case: SyntheticTestCase, template: SynthesisTemplate) -> None:
    """Test that appraise updates complexity, diversity, and validity."""
    teacher = MockJudge(complexity=8.5, validity=0.95)

    # Setup embedder to return a vector orthogonal to template centroid [1, 0, 0]
    # If case vector is [0, 1, 0], dot product is 0, similarity is 0.
    # Diversity = 1 - 0 = 1.0 (Max diversity)
    class OrthogonalEmbedder(DummyEmbeddingService):
        def embed(self, text: str) -> List[float]:
            return [0.0, 1.0, 0.0]

    appraiser = AppraiserImpl(teacher, OrthogonalEmbedder())

    results = appraiser.appraise([base_case], template)

    assert len(results) == 1
    assert results[0].complexity == 8.5
    assert results[0].ambiguity == 2.0  # Default from MockJudge
    assert results[0].validity_confidence == 0.95
    assert results[0].diversity == 1.0  # 1 - 0


def test_appraise_diversity_calculation(base_case: SyntheticTestCase, template: SynthesisTemplate) -> None:
    """Test diversity calculation with parallel vectors."""
    teacher = MockJudge()

    # Case vector is identical to centroid [1, 0, 0]
    # Dot product = 1, Sim = 1
    # Diversity = 1 - 1 = 0.0
    class ParallelEmbedder(DummyEmbeddingService):
        def embed(self, text: str) -> List[float]:
            return [1.0, 0.0, 0.0]

    appraiser = AppraiserImpl(teacher, ParallelEmbedder())

    results = appraiser.appraise([base_case], template)

    # Use approx because float math
    assert results[0].diversity == pytest.approx(0.0)


def test_appraise_filtering(base_case: SyntheticTestCase, template: SynthesisTemplate) -> None:
    """Test that cases below min_validity_score are discarded."""
    # Teacher returns low validity
    teacher = MockJudge(validity=0.5)
    # Ensure embedder dimension matches template centroid (3)
    embedder = DummyEmbeddingService(dimension=3)
    appraiser = AppraiserImpl(teacher, embedder)

    results = appraiser.appraise([base_case], template, min_validity_score=0.8)

    assert len(results) == 0


def test_appraise_sorting(template: SynthesisTemplate) -> None:
    """Test sorting of cases."""

    class DynamicJudge(MockTeacher):
        def __init__(self) -> None:
            self.call_count = 0

        def generate_structured(self, prompt: str, response_model: Type[T], context: Optional[str] = None) -> T:
            self.call_count += 1
            # Case 1 (processed first): High complexity
            if self.call_count == 1:
                return AppraisalAnalysis(complexity_score=9.0, ambiguity_score=0, validity_confidence=1.0)  # type: ignore
            # Case 2: Low complexity
            return AppraisalAnalysis(complexity_score=2.0, ambiguity_score=0, validity_confidence=1.0)  # type: ignore

    # Ensure embedder dimension matches template centroid (3)
    appraiser = AppraiserImpl(DynamicJudge(), DummyEmbeddingService(dimension=3))

    case1 = SyntheticTestCase(
        verbatim_context="C1",
        synthetic_question="Q1",
        golden_chain_of_thought="L1",
        expected_json={},
        provenance=ProvenanceType.VERBATIM_SOURCE,
        source_urn="u1",
        complexity=0,
        ambiguity=0,
        diversity=0,
        validity_confidence=0,
    )
    case2 = SyntheticTestCase(
        verbatim_context="C2",
        synthetic_question="Q2",
        golden_chain_of_thought="L2",
        expected_json={},
        provenance=ProvenanceType.VERBATIM_SOURCE,
        source_urn="u2",
        complexity=0,
        ambiguity=0,
        diversity=0,
        validity_confidence=0,
    )

    # Sort by complexity descending
    results = appraiser.appraise([case1, case2], template, sort_by="complexity_desc")

    assert len(results) == 2
    # case1 (processed 1st) got score 9.0
    # case2 (processed 2nd) got score 2.0
    assert results[0].complexity == 9.0
    assert results[1].complexity == 2.0


def test_missing_centroid(base_case: SyntheticTestCase) -> None:
    """Test diversity is 0 if template has no centroid."""
    template_no_centroid = SynthesisTemplate(
        structure="S", complexity_description="C", domain="D", embedding_centroid=None
    )
    teacher = MockJudge()
    appraiser = AppraiserImpl(teacher, DummyEmbeddingService(dimension=3))

    results = appraiser.appraise([base_case], template_no_centroid)

    assert results[0].diversity == 0.0


def test_sorting_variants(template: SynthesisTemplate) -> None:
    """Test all sorting options."""

    class FixedJudge(MockTeacher):
        def generate_structured(self, prompt: str, response_model: Type[T], context: Optional[str] = None) -> T:
            # Return scores based on context to distinguish cases
            if "C1" in prompt:
                return AppraisalAnalysis(complexity_score=1.0, ambiguity_score=0, validity_confidence=0.9)  # type: ignore
            return AppraisalAnalysis(complexity_score=10.0, ambiguity_score=0, validity_confidence=0.95)  # type: ignore

    # We need to control diversity too.
    # C1 -> Embed=[1,0,0] -> Sim=1 -> Div=0
    # C2 -> Embed=[0,1,0] -> Sim=0 -> Div=1
    class ControlledEmbedder(DummyEmbeddingService):
        def embed(self, text: str) -> List[float]:
            if text == "C1":
                return [1.0, 0.0, 0.0]
            return [0.0, 1.0, 0.0]

    appraiser = AppraiserImpl(FixedJudge(), ControlledEmbedder())

    case1 = SyntheticTestCase(
        verbatim_context="C1",
        synthetic_question="Q",
        golden_chain_of_thought="L",
        expected_json={},
        provenance=ProvenanceType.VERBATIM_SOURCE,
        source_urn="u",
        complexity=0,
        ambiguity=0,
        diversity=0,
        validity_confidence=0,
    )
    case2 = SyntheticTestCase(
        verbatim_context="C2",
        synthetic_question="Q",
        golden_chain_of_thought="L",
        expected_json={},
        provenance=ProvenanceType.VERBATIM_SOURCE,
        source_urn="u",
        complexity=0,
        ambiguity=0,
        diversity=0,
        validity_confidence=0,
    )

    # C1: Comp=1, Div=0, Val=0.9
    # C2: Comp=10, Div=1, Val=0.95

    # Complexity ASC
    res = appraiser.appraise([case1, case2], template, sort_by="complexity_asc")
    assert res[0].verbatim_context == "C1"

    # Diversity DESC
    res = appraiser.appraise([case1, case2], template, sort_by="diversity_desc")
    assert res[0].verbatim_context == "C2"

    # Diversity ASC
    res = appraiser.appraise([case1, case2], template, sort_by="diversity_asc")
    assert res[0].verbatim_context == "C1"

    # Validity DESC
    res = appraiser.appraise([case1, case2], template, sort_by="validity_desc")
    assert res[0].verbatim_context == "C2"

    # Validity ASC
    res = appraiser.appraise([case1, case2], template, sort_by="validity_asc")
    assert res[0].verbatim_context == "C1"

    # Default (fallback)
    res = appraiser.appraise([case1, case2], template, sort_by="unknown")
    assert res[0].verbatim_context == "C2"  # fallback to complexity desc


def test_empty_case_list(template: SynthesisTemplate) -> None:
    """Test that appraising empty list returns empty list."""
    appraiser = AppraiserImpl(MockJudge(), DummyEmbeddingService())
    assert appraiser.appraise([], template) == []


def test_dimension_mismatch(template: SynthesisTemplate, base_case: SyntheticTestCase) -> None:
    """Test that dimension mismatch is handled gracefully (no diversity score)."""
    # Template is 3D (from fixture)
    # Service default is 1536D
    appraiser = AppraiserImpl(MockJudge(), DummyEmbeddingService())

    results = appraiser.appraise([base_case], template)
    assert len(results) == 1
    assert results[0].diversity == 0.0


def test_zero_vector_edge_case(template: SynthesisTemplate, base_case: SyntheticTestCase) -> None:
    """Test handling of zero vectors (division by zero protection)."""

    class ZeroEmbedder(DummyEmbeddingService):
        def embed(self, text: str) -> List[float]:
            return [0.0, 0.0, 0.0]

    appraiser = AppraiserImpl(MockJudge(), ZeroEmbedder())
    results = appraiser.appraise([base_case], template)
    assert results[0].diversity == 0.0
