# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_synthesis

import math
from typing import cast
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
    return AsyncMock(spec=EmbeddingService)


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
def cases() -> list[SyntheticTestCase]:
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
async def test_teacher_failure_propagation(
    appraiser: AppraiserImpl,
    mock_teacher: AsyncMock,
    template: SynthesisTemplate,
    cases: list[SyntheticTestCase],
) -> None:
    mock_teacher.generate_structured.side_effect = ValueError("Teacher Failed")

    with pytest.raises(ValueError, match="Teacher Failed"):
        await appraiser.appraise(cases, template)


@pytest.mark.asyncio
async def test_boundary_scores(
    appraiser: AppraiserImpl,
    mock_teacher: AsyncMock,
    mock_embedder: AsyncMock,
    template: SynthesisTemplate,
    cases: list[SyntheticTestCase],
) -> None:
    """Test min/max score inputs."""
    mock_embedder.embed.return_value = [1.0, 0.0]
    # Max possible scores
    # Cast appraiser.teacher to AsyncMock to satisfy mypy when setting return_value
    cast(AsyncMock, appraiser.teacher).generate_structured.return_value = AppraisalAnalysis(
        complexity_score=10.0, ambiguity_score=10.0, validity_confidence=1.0
    )

    results = await appraiser.appraise(cases, template)
    assert results[0].complexity == 10.0
    assert results[0].validity_confidence == 1.0

    # Min possible scores
    cast(AsyncMock, appraiser.teacher).generate_structured.return_value = AppraisalAnalysis(
        complexity_score=0.0, ambiguity_score=0.0, validity_confidence=0.0
    )

    # With validity 0, it should be filtered out (default threshold 0.8)
    results = await appraiser.appraise(cases, template, min_validity_score=0.1)
    assert len(results) == 0


@pytest.mark.asyncio
async def test_nan_embeddings(
    appraiser: AppraiserImpl,
    mock_embedder: AsyncMock,
    template: SynthesisTemplate,
    cases: list[SyntheticTestCase],
) -> None:
    """Ensure NaNs in embeddings don't crash the calculation."""
    mock_embedder.embed.return_value = [float("nan"), 0.0]
    cast(AsyncMock, appraiser.teacher).generate_structured.return_value = AppraisalAnalysis(
        complexity_score=5, ambiguity_score=5, validity_confidence=1
    )

    results = await appraiser.appraise(cases, template)

    # Diversity should likely be 0 or NaN. Our logic:
    # dot prod with NaN -> NaN
    # norm -> NaN
    # We didn't explicitly handle NaN in the code, but numpy usually propagates it.
    # Casting to float might raise ValueError or result in nan.
    # If it is nan, it passes through.
    assert math.isnan(results[0].diversity) or results[0].diversity == 0.0
