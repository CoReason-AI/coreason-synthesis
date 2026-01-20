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
from unittest.mock import AsyncMock

import pytest

from coreason_synthesis.appraiser import AppraisalAnalysis, AppraiserImpl
from coreason_synthesis.interfaces import EmbeddingService, TeacherModel
from coreason_synthesis.models import (
    ProvenanceType,
    SynthesisTemplate,
    SyntheticTestCase,
)


@pytest.fixture
def mock_teacher() -> AsyncMock:
    return AsyncMock(spec=TeacherModel)


@pytest.fixture
def mock_embedder() -> AsyncMock:
    mock = AsyncMock(spec=EmbeddingService)
    # Default dummy
    mock.embed.return_value = [1.0, 0.0]
    return mock


@pytest.fixture
def appraiser(mock_teacher: AsyncMock, mock_embedder: AsyncMock) -> AppraiserImpl:
    return AppraiserImpl(teacher=mock_teacher, embedder=mock_embedder)


@pytest.fixture
def template() -> SynthesisTemplate:
    return SynthesisTemplate(
        structure="Q",
        complexity_description="M",
        domain="D",
        embedding_centroid=[1.0, 0.0],
    )


@pytest.fixture
def cases() -> List[SyntheticTestCase]:
    return [
        SyntheticTestCase(
            verbatim_context="C1",
            synthetic_question="Q1",
            golden_chain_of_thought="R1",
            expected_json={},
            provenance=ProvenanceType.VERBATIM_SOURCE,
            source_urn="u1",
            complexity=0.0,
            diversity=0.0,
            validity_confidence=0.0,
        )
    ]


@pytest.mark.asyncio
async def test_appraise_updates_metrics(
    appraiser: AppraiserImpl,
    mock_teacher: AsyncMock,
    mock_embedder: AsyncMock,
    template: SynthesisTemplate,
    cases: List[SyntheticTestCase],
) -> None:
    # Setup
    mock_teacher.generate_structured.return_value = AppraisalAnalysis(
        complexity_score=8.0, ambiguity_score=5.0, validity_confidence=0.9
    )
    # Centroid=[1,0], Case=[1,0] -> Sim=1.0 -> Diversity=0.0
    mock_embedder.embed.return_value = [1.0, 0.0]

    # Act
    results = await appraiser.appraise(cases, template)

    # Assert
    assert len(results) == 1
    c = results[0]
    assert c.complexity == 8.0
    assert c.validity_confidence == 0.9
    assert c.diversity == 0.0  # (1 - 1.0)
    mock_embedder.embed.assert_awaited_once()
    mock_teacher.generate_structured.assert_awaited_once()


@pytest.mark.asyncio
async def test_appraise_diversity_calculation(
    appraiser: AppraiserImpl,
    mock_embedder: AsyncMock,
    template: SynthesisTemplate,
    cases: List[SyntheticTestCase],
) -> None:
    # Centroid=[1,0]
    # Case=[0,1] -> Sim=0.0 -> Diversity=1.0
    mock_embedder.embed.return_value = [0.0, 1.0]

    # Mock teacher minimal response
    appraiser.teacher.generate_structured.return_value = AppraisalAnalysis(  # type: ignore
        complexity_score=5, ambiguity_score=5, validity_confidence=1.0
    )

    results = await appraiser.appraise(cases, template)
    assert results[0].diversity == 1.0


@pytest.mark.asyncio
async def test_appraise_filtering(
    appraiser: AppraiserImpl,
    mock_teacher: AsyncMock,
    template: SynthesisTemplate,
    cases: List[SyntheticTestCase],
) -> None:
    # Validity 0.5 < min 0.8
    mock_teacher.generate_structured.return_value = AppraisalAnalysis(
        complexity_score=5, ambiguity_score=5, validity_confidence=0.5
    )

    results = await appraiser.appraise(cases, template, min_validity_score=0.8)
    assert len(results) == 0


@pytest.mark.asyncio
async def test_appraise_sorting(
    appraiser: AppraiserImpl,
    mock_teacher: AsyncMock,
    template: SynthesisTemplate,
) -> None:
    c1 = SyntheticTestCase(
        verbatim_context="C1",
        synthetic_question="Q1",
        golden_chain_of_thought="R1",
        expected_json={},
        provenance=ProvenanceType.VERBATIM_SOURCE,
        source_urn="u1",
        complexity=0.0,
        diversity=0.0,
        validity_confidence=0.0,
    )
    c2 = c1.model_copy()
    c3 = c1.model_copy()

    cases = [c1, c2, c3]

    # Mock return values for each call
    # Order of iteration is c1, c2, c3
    # We want final complexities: 2, 8, 5
    mock_teacher.generate_structured.side_effect = [
        AppraisalAnalysis(complexity_score=2, ambiguity_score=0, validity_confidence=1),
        AppraisalAnalysis(complexity_score=8, ambiguity_score=0, validity_confidence=1),
        AppraisalAnalysis(complexity_score=5, ambiguity_score=0, validity_confidence=1),
    ]

    results = await appraiser.appraise(cases, template, sort_by="complexity_desc")

    assert len(results) == 3
    assert results[0].complexity == 8
    assert results[1].complexity == 5
    assert results[2].complexity == 2


@pytest.mark.asyncio
async def test_missing_centroid(
    appraiser: AppraiserImpl,
    mock_embedder: AsyncMock,
    cases: List[SyntheticTestCase],
) -> None:
    t_no_centroid = SynthesisTemplate(structure="", complexity_description="", domain="", embedding_centroid=[])
    appraiser.teacher.generate_structured.return_value = AppraisalAnalysis(  # type: ignore
        complexity_score=5, ambiguity_score=5, validity_confidence=1
    )

    results = await appraiser.appraise(cases, t_no_centroid)
    # Embedder not called
    mock_embedder.embed.assert_not_called()
    assert results[0].diversity == 0.0


@pytest.mark.asyncio
async def test_sorting_variants(
    appraiser: AppraiserImpl,
    mock_teacher: AsyncMock,
    template: SynthesisTemplate,
    cases: List[SyntheticTestCase],
) -> None:
    # Just to cover other sort keys
    mock_teacher.generate_structured.return_value = AppraisalAnalysis(
        complexity_score=5, ambiguity_score=5, validity_confidence=1
    )
    # We can't easily affect diversity results independently without complex side_effects on embedder,
    # but we can verify the function runs.
    await appraiser.appraise(cases, template, sort_by="validity_asc")
    await appraiser.appraise(cases, template, sort_by="diversity_desc")


@pytest.mark.asyncio
async def test_empty_case_list(appraiser: AppraiserImpl, template: SynthesisTemplate) -> None:
    results = await appraiser.appraise([], template)
    assert results == []


@pytest.mark.asyncio
async def test_dimension_mismatch(
    appraiser: AppraiserImpl,
    mock_embedder: AsyncMock,
    template: SynthesisTemplate,
    cases: List[SyntheticTestCase],
) -> None:
    # Template has 2D centroid
    # Embedder returns 3D -> Mismatch
    mock_embedder.embed.return_value = [1.0, 1.0, 1.0]
    appraiser.teacher.generate_structured.return_value = AppraisalAnalysis(  # type: ignore
        complexity_score=5, ambiguity_score=5, validity_confidence=1
    )

    results = await appraiser.appraise(cases, template)
    # Should catch mismatch and default diversity to 0
    assert results[0].diversity == 0.0


@pytest.mark.asyncio
async def test_zero_vector_edge_case(
    appraiser: AppraiserImpl,
    mock_embedder: AsyncMock,
    template: SynthesisTemplate,
    cases: List[SyntheticTestCase],
) -> None:
    mock_embedder.embed.return_value = [0.0, 0.0]
    appraiser.teacher.generate_structured.return_value = AppraisalAnalysis(  # type: ignore
        complexity_score=5, ambiguity_score=5, validity_confidence=1
    )
    results = await appraiser.appraise(cases, template)
    assert results[0].diversity == 0.0
