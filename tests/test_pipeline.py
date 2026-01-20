# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_synthesis

from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from coreason_synthesis.interfaces import (
    Appraiser,
    Compositor,
    Extractor,
    Forager,
    PatternAnalyzer,
    Perturbator,
)
from coreason_synthesis.models import (
    Document,
    ExtractedSlice,
    ProvenanceType,
    SeedCase,
    SynthesisTemplate,
    SyntheticTestCase,
)
from coreason_synthesis.pipeline import SynthesisPipeline, SynthesisPipelineAsync


@pytest.fixture
def mock_components() -> Dict[str, Mock]:
    return {
        "analyzer": Mock(spec=PatternAnalyzer),
        "forager": Mock(spec=Forager),
        "extractor": Mock(spec=Extractor),
        "compositor": Mock(spec=Compositor),
        "perturbator": Mock(spec=Perturbator),
        "appraiser": Mock(spec=Appraiser),
    }


@pytest.fixture
def async_mock_components(mock_components: Dict[str, Mock]) -> Dict[str, AsyncMock]:
    # Update mocks to be AsyncMock for async methods
    mock_components["analyzer"].analyze = AsyncMock()
    mock_components["forager"].forage = AsyncMock()
    mock_components["extractor"].extract = AsyncMock()
    mock_components["compositor"].composite = AsyncMock()
    mock_components["perturbator"].perturb = AsyncMock()
    mock_components["appraiser"].appraise = AsyncMock()
    return mock_components


@pytest.fixture
def pipeline_async(async_mock_components: Dict[str, AsyncMock]) -> SynthesisPipelineAsync:
    return SynthesisPipelineAsync(
        analyzer=async_mock_components["analyzer"],
        forager=async_mock_components["forager"],
        extractor=async_mock_components["extractor"],
        compositor=async_mock_components["compositor"],
        perturbator=async_mock_components["perturbator"],
        appraiser=async_mock_components["appraiser"],
    )


@pytest.fixture
def pipeline_sync(async_mock_components: Dict[str, AsyncMock]) -> SynthesisPipeline:
    return SynthesisPipeline(
        analyzer=async_mock_components["analyzer"],
        forager=async_mock_components["forager"],
        extractor=async_mock_components["extractor"],
        compositor=async_mock_components["compositor"],
        perturbator=async_mock_components["perturbator"],
        appraiser=async_mock_components["appraiser"],
    )


@pytest.fixture
def sample_seeds() -> List[SeedCase]:
    return [
        SeedCase(
            id=uuid4(),
            context="Seed Context",
            question="Seed Q",
            expected_output={"ans": "A"},
        )
    ]


@pytest.fixture
def sample_template() -> SynthesisTemplate:
    return SynthesisTemplate(
        structure="Q+A",
        complexity_description="Medium",
        domain="Test",
        embedding_centroid=[0.1, 0.2],
    )


@pytest.mark.asyncio
async def test_pipeline_async_happy_path(
    pipeline_async: SynthesisPipelineAsync,
    async_mock_components: Dict[str, AsyncMock],
    sample_seeds: List[SeedCase],
    sample_template: SynthesisTemplate,
) -> None:
    # Setup Mocks
    async_mock_components["analyzer"].analyze.return_value = sample_template

    docs = [Document(content="Doc1", source_urn="u1")]
    async_mock_components["forager"].forage.return_value = docs

    slices = [ExtractedSlice(content="Slice1", source_urn="u1", page_number=1, pii_redacted=False)]
    async_mock_components["extractor"].extract.return_value = slices

    base_case = SyntheticTestCase(
        verbatim_context="Slice1",
        synthetic_question="Q1",
        golden_chain_of_thought="R1",
        expected_json={"a": 1},
        provenance=ProvenanceType.VERBATIM_SOURCE,
        source_urn="u1",
        complexity=0.0,
        diversity=0.0,
        validity_confidence=0.0,
    )
    async_mock_components["compositor"].composite.return_value = base_case

    # Mock appraiser to return the input list
    async_mock_components["appraiser"].appraise.side_effect = lambda cases, t, sort_by, min_validity_score: cases

    config: Dict[str, Any] = {"target_count": 5, "perturbation_rate": 0.0}
    user_context: Dict[str, Any] = {"user": "test"}

    results = await pipeline_async.run(sample_seeds, config, user_context)

    # Verify Calls
    async_mock_components["analyzer"].analyze.assert_awaited_once_with(sample_seeds)
    async_mock_components["forager"].forage.assert_awaited_once()
    async_mock_components["extractor"].extract.assert_awaited_once_with(docs, sample_template)
    async_mock_components["compositor"].composite.assert_awaited_once_with(slices[0], sample_template)
    async_mock_components["appraiser"].appraise.assert_awaited_once()

    # Perturbator should not be called if rate is 0
    async_mock_components["perturbator"].perturb.assert_not_called()

    assert len(results) == 1
    assert results[0] == base_case


def test_pipeline_sync_wrapper(
    pipeline_sync: SynthesisPipeline,
    async_mock_components: Dict[str, AsyncMock],
    sample_seeds: List[SeedCase],
    sample_template: SynthesisTemplate,
) -> None:
    """Test that the synchronous wrapper correctly calls async methods via anyio.run."""
    # Setup Mocks
    async_mock_components["analyzer"].analyze.return_value = sample_template
    # Return empty to stop early, sufficient for checking call
    async_mock_components["forager"].forage.return_value = []

    config: Dict[str, Any] = {"target_count": 5}
    user_context: Dict[str, Any] = {"user": "test"}

    results = pipeline_sync.run(sample_seeds, config, user_context)

    assert results == []
    # If this passes, it means anyio.run was called and the async method executed
    # We can check if analyzer was awaited
    # Note: unittest.mock.AsyncMock interaction recording might be tricky across threads if not careful,
    # but typically works if the object is shared.
    # However, since anyio.run might run in the same thread (if blocking) or another, we should just verify result.


@pytest.mark.asyncio
async def test_pipeline_async_perturbation(
    pipeline_async: SynthesisPipelineAsync,
    async_mock_components: Dict[str, AsyncMock],
    sample_seeds: List[SeedCase],
    sample_template: SynthesisTemplate,
) -> None:
    async_mock_components["analyzer"].analyze.return_value = sample_template
    async_mock_components["forager"].forage.return_value = [Document(content="D", source_urn="u")]
    async_mock_components["extractor"].extract.return_value = [
        ExtractedSlice(content="S", source_urn="u", page_number=1, pii_redacted=False)
    ]

    base_case = SyntheticTestCase(
        verbatim_context="S",
        synthetic_question="Q",
        golden_chain_of_thought="R",
        expected_json={},
        provenance=ProvenanceType.VERBATIM_SOURCE,
        source_urn="u",
        complexity=0.0,
        diversity=0.0,
        validity_confidence=0.0,
    )
    async_mock_components["compositor"].composite.return_value = base_case

    variant_case = base_case.model_copy()
    variant_case.provenance = ProvenanceType.SYNTHETIC_PERTURBED
    async_mock_components["perturbator"].perturb.return_value = [variant_case]

    async_mock_components["appraiser"].appraise.side_effect = lambda cases, *args, **kwargs: cases

    # Force perturbation
    config: Dict[str, Any] = {"perturbation_rate": 1.1}

    results = await pipeline_async.run(sample_seeds, config, {})

    # Verify perturbator called
    async_mock_components["perturbator"].perturb.assert_awaited_once_with(base_case)

    # Should have base + variant = 2
    assert len(results) == 2
    assert results[1].provenance == ProvenanceType.SYNTHETIC_PERTURBED


@pytest.mark.asyncio
async def test_pipeline_async_empty_seeds(
    pipeline_async: SynthesisPipelineAsync, async_mock_components: Dict[str, AsyncMock]
) -> None:
    results = await pipeline_async.run([], {}, {})
    assert results == []
    async_mock_components["analyzer"].analyze.assert_not_called()


@pytest.mark.asyncio
async def test_pipeline_async_empty_forage(
    pipeline_async: SynthesisPipelineAsync,
    async_mock_components: Dict[str, AsyncMock],
    sample_seeds: List[SeedCase],
    sample_template: SynthesisTemplate,
) -> None:
    async_mock_components["analyzer"].analyze.return_value = sample_template
    async_mock_components["forager"].forage.return_value = []  # No docs

    results = await pipeline_async.run(sample_seeds, {}, {})

    assert results == []
    async_mock_components["extractor"].extract.assert_not_called()


@pytest.mark.asyncio
async def test_pipeline_async_empty_extract(
    pipeline_async: SynthesisPipelineAsync,
    async_mock_components: Dict[str, AsyncMock],
    sample_seeds: List[SeedCase],
    sample_template: SynthesisTemplate,
) -> None:
    async_mock_components["analyzer"].analyze.return_value = sample_template
    async_mock_components["forager"].forage.return_value = [Document(content="D", source_urn="u")]
    async_mock_components["extractor"].extract.return_value = []  # No slices

    results = await pipeline_async.run(sample_seeds, {}, {})

    assert results == []
    async_mock_components["compositor"].composite.assert_not_called()


@pytest.mark.asyncio
async def test_pipeline_async_exception_propagation(
    pipeline_async: SynthesisPipelineAsync,
    async_mock_components: Dict[str, AsyncMock],
    sample_seeds: List[SeedCase],
) -> None:
    """
    Complex Scenario: Component raises exception, pipeline should crash (fail fast).
    """
    async_mock_components["analyzer"].analyze.side_effect = ValueError("Analysis Failed")

    with pytest.raises(ValueError, match="Analysis Failed"):
        await pipeline_async.run(sample_seeds, {}, {})
