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
from unittest.mock import Mock, patch
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
from coreason_synthesis.pipeline import SynthesisPipeline


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
def pipeline(mock_components: Dict[str, Mock]) -> SynthesisPipeline:
    return SynthesisPipeline(
        analyzer=mock_components["analyzer"],
        forager=mock_components["forager"],
        extractor=mock_components["extractor"],
        compositor=mock_components["compositor"],
        perturbator=mock_components["perturbator"],
        appraiser=mock_components["appraiser"],
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


def test_pipeline_happy_path(
    pipeline: SynthesisPipeline,
    mock_components: Dict[str, Mock],
    sample_seeds: List[SeedCase],
    sample_template: SynthesisTemplate,
) -> None:
    # Setup Mocks
    mock_components["analyzer"].analyze.return_value = sample_template

    docs = [Document(content="Doc1", source_urn="u1")]
    mock_components["forager"].forage.return_value = docs

    slices = [ExtractedSlice(content="Slice1", source_urn="u1", page_number=1, pii_redacted=False)]
    mock_components["extractor"].extract.return_value = slices

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
    mock_components["compositor"].composite.return_value = base_case

    # Mock appraiser to return the input list
    mock_components["appraiser"].appraise.side_effect = lambda cases, t, sort_by, min_validity_score: cases

    config: Dict[str, Any] = {"target_count": 5, "perturbation_rate": 0.0}
    user_context: Dict[str, Any] = {"user": "test"}

    results = pipeline.run(sample_seeds, config, user_context)

    # Verify Calls
    mock_components["analyzer"].analyze.assert_called_once_with(sample_seeds)
    mock_components["forager"].forage.assert_called_once()
    mock_components["extractor"].extract.assert_called_once_with(docs, sample_template)
    mock_components["compositor"].composite.assert_called_once_with(slices[0], sample_template)
    mock_components["appraiser"].appraise.assert_called_once()

    # Perturbator should not be called if rate is 0
    mock_components["perturbator"].perturb.assert_not_called()

    assert len(results) == 1
    assert results[0] == base_case


def test_pipeline_perturbation(
    pipeline: SynthesisPipeline,
    mock_components: Dict[str, Mock],
    sample_seeds: List[SeedCase],
    sample_template: SynthesisTemplate,
) -> None:
    mock_components["analyzer"].analyze.return_value = sample_template
    mock_components["forager"].forage.return_value = [Document(content="D", source_urn="u")]
    mock_components["extractor"].extract.return_value = [
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
    mock_components["compositor"].composite.return_value = base_case

    variant_case = base_case.model_copy()
    variant_case.provenance = ProvenanceType.SYNTHETIC_PERTURBED
    mock_components["perturbator"].perturb.return_value = [variant_case]

    mock_components["appraiser"].appraise.side_effect = lambda cases, *args, **kwargs: cases

    # Force perturbation
    # Since we can't easily mock random.random in the imported module without patching,
    # we can set rate to 1.1 (always true)
    config: Dict[str, Any] = {"perturbation_rate": 1.1}

    results = pipeline.run(sample_seeds, config, {})

    # Verify perturbator called
    mock_components["perturbator"].perturb.assert_called_once_with(base_case)

    # Should have base + variant = 2
    assert len(results) == 2
    assert results[1].provenance == ProvenanceType.SYNTHETIC_PERTURBED


def test_pipeline_empty_seeds(pipeline: SynthesisPipeline, mock_components: Dict[str, Mock]) -> None:
    results = pipeline.run([], {}, {})
    assert results == []
    mock_components["analyzer"].analyze.assert_not_called()


def test_pipeline_empty_forage(
    pipeline: SynthesisPipeline,
    mock_components: Dict[str, Mock],
    sample_seeds: List[SeedCase],
    sample_template: SynthesisTemplate,
) -> None:
    mock_components["analyzer"].analyze.return_value = sample_template
    mock_components["forager"].forage.return_value = []  # No docs

    results = pipeline.run(sample_seeds, {}, {})

    assert results == []
    mock_components["extractor"].extract.assert_not_called()


def test_pipeline_empty_extract(
    pipeline: SynthesisPipeline,
    mock_components: Dict[str, Mock],
    sample_seeds: List[SeedCase],
    sample_template: SynthesisTemplate,
) -> None:
    mock_components["analyzer"].analyze.return_value = sample_template
    mock_components["forager"].forage.return_value = [Document(content="D", source_urn="u")]
    mock_components["extractor"].extract.return_value = []  # No slices

    results = pipeline.run(sample_seeds, {}, {})

    assert results == []
    mock_components["compositor"].composite.assert_not_called()


def test_pipeline_all_filtered_by_appraiser(
    pipeline: SynthesisPipeline,
    mock_components: Dict[str, Mock],
    sample_seeds: List[SeedCase],
    sample_template: SynthesisTemplate,
) -> None:
    """
    Complex Scenario: Pipeline runs fully, but appraiser filters everything out.
    """
    mock_components["analyzer"].analyze.return_value = sample_template
    mock_components["forager"].forage.return_value = [Document(content="D", source_urn="u")]
    mock_components["extractor"].extract.return_value = [
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
    mock_components["compositor"].composite.return_value = base_case

    # Appraiser returns empty list
    mock_components["appraiser"].appraise.return_value = []

    results = pipeline.run(sample_seeds, {}, {})

    assert results == []
    mock_components["appraiser"].appraise.assert_called_once()


def test_pipeline_perturbation_bad_luck(
    pipeline: SynthesisPipeline,
    mock_components: Dict[str, Mock],
    sample_seeds: List[SeedCase],
    sample_template: SynthesisTemplate,
) -> None:
    """
    Edge Case: Perturbation rate > 0, but random roll fails (simulated by patch).
    """
    mock_components["analyzer"].analyze.return_value = sample_template
    mock_components["forager"].forage.return_value = [Document(content="D", source_urn="u")]
    mock_components["extractor"].extract.return_value = [
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
    mock_components["compositor"].composite.return_value = base_case

    # Pass through appraiser
    mock_components["appraiser"].appraise.side_effect = lambda cases, *args, **kwargs: cases

    config = {"perturbation_rate": 0.5}

    # Patch random.random to return 0.6 (fail condition > 0.5)
    with patch("random.random", return_value=0.6):
        results = pipeline.run(sample_seeds, config, {})

    # Perturbator NOT called
    mock_components["perturbator"].perturb.assert_not_called()

    # Only base case returned
    assert len(results) == 1
    assert results[0].provenance == ProvenanceType.VERBATIM_SOURCE


def test_pipeline_exception_propagation(
    pipeline: SynthesisPipeline,
    mock_components: Dict[str, Mock],
    sample_seeds: List[SeedCase],
) -> None:
    """
    Complex Scenario: Component raises exception, pipeline should crash (fail fast).
    """
    mock_components["analyzer"].analyze.side_effect = ValueError("Analysis Failed")

    with pytest.raises(ValueError, match="Analysis Failed"):
        pipeline.run(sample_seeds, {}, {})


def test_pipeline_config_defaults(
    pipeline: SynthesisPipeline,
    mock_components: Dict[str, Mock],
    sample_seeds: List[SeedCase],
    sample_template: SynthesisTemplate,
) -> None:
    """
    Edge Case: Minimal config provided, verify defaults passed to components.
    """
    mock_components["analyzer"].analyze.return_value = sample_template
    mock_components["forager"].forage.return_value = [Document(content="D", source_urn="u")]
    mock_components["extractor"].extract.return_value = [
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
    mock_components["compositor"].composite.return_value = base_case

    # Empty config
    pipeline.run(sample_seeds, {}, {})

    # Verify defaults
    # Forager default limit 10
    mock_components["forager"].forage.assert_called_with(sample_template, {}, limit=10)

    # Appraiser default sort="complexity_desc", min_validity=0.8
    mock_components["appraiser"].appraise.assert_called_with(
        [base_case], sample_template, sort_by="complexity_desc", min_validity_score=0.8
    )
