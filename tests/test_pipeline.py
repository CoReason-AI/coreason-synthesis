from typing import Any, Dict, List
from unittest.mock import Mock, MagicMock
from uuid import uuid4

import pytest

from coreason_synthesis.models import (
    SeedCase,
    SynthesisTemplate,
    SyntheticTestCase,
    Document,
    ExtractedSlice,
    ProvenanceType,
)
from coreason_synthesis.pipeline import SynthesisPipeline
from coreason_synthesis.interfaces import (
    PatternAnalyzer,
    Forager,
    Extractor,
    Compositor,
    Perturbator,
    Appraiser,
)


@pytest.fixture
def mock_components():
    return {
        "analyzer": Mock(spec=PatternAnalyzer),
        "forager": Mock(spec=Forager),
        "extractor": Mock(spec=Extractor),
        "compositor": Mock(spec=Compositor),
        "perturbator": Mock(spec=Perturbator),
        "appraiser": Mock(spec=Appraiser),
    }


@pytest.fixture
def pipeline(mock_components):
    return SynthesisPipeline(
        analyzer=mock_components["analyzer"],
        forager=mock_components["forager"],
        extractor=mock_components["extractor"],
        compositor=mock_components["compositor"],
        perturbator=mock_components["perturbator"],
        appraiser=mock_components["appraiser"],
    )


@pytest.fixture
def sample_seeds():
    return [
        SeedCase(
            id=uuid4(),
            context="Seed Context",
            question="Seed Q",
            expected_output={"ans": "A"},
        )
    ]


@pytest.fixture
def sample_template():
    return SynthesisTemplate(
        structure="Q+A",
        complexity_description="Medium",
        domain="Test",
        embedding_centroid=[0.1, 0.2],
    )


def test_pipeline_happy_path(pipeline, mock_components, sample_seeds, sample_template):
    # Setup Mocks
    mock_components["analyzer"].analyze.return_value = sample_template

    docs = [Document(content="Doc1", source_urn="u1")]
    mock_components["forager"].forage.return_value = docs

    slices = [ExtractedSlice(content="Slice1", source_urn="u1")]
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

    config = {"target_count": 5, "perturbation_rate": 0.0}
    user_context = {"user": "test"}

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


def test_pipeline_perturbation(pipeline, mock_components, sample_seeds, sample_template):
    mock_components["analyzer"].analyze.return_value = sample_template
    mock_components["forager"].forage.return_value = [Document(content="D", source_urn="u")]
    mock_components["extractor"].extract.return_value = [ExtractedSlice(content="S", source_urn="u")]

    base_case = SyntheticTestCase(
        verbatim_context="S", synthetic_question="Q", golden_chain_of_thought="R",
        expected_json={}, provenance=ProvenanceType.VERBATIM_SOURCE, source_urn="u",
        complexity=0.0, diversity=0.0, validity_confidence=0.0
    )
    mock_components["compositor"].composite.return_value = base_case

    variant_case = base_case.model_copy()
    variant_case.provenance = ProvenanceType.SYNTHETIC_PERTURBED
    mock_components["perturbator"].perturb.return_value = [variant_case]

    mock_components["appraiser"].appraise.side_effect = lambda cases, *args, **kwargs: cases

    # Force perturbation
    # Since we can't easily mock random.random in the imported module without patching,
    # we can set rate to 1.1 (always true)
    config = {"perturbation_rate": 1.1}

    results = pipeline.run(sample_seeds, config, {})

    # Verify perturbator called
    mock_components["perturbator"].perturb.assert_called_once_with(base_case)

    # Should have base + variant = 2
    assert len(results) == 2
    assert results[1].provenance == ProvenanceType.SYNTHETIC_PERTURBED


def test_pipeline_empty_seeds(pipeline, mock_components):
    results = pipeline.run([], {}, {})
    assert results == []
    mock_components["analyzer"].analyze.assert_not_called()


def test_pipeline_empty_forage(pipeline, mock_components, sample_seeds, sample_template):
    mock_components["analyzer"].analyze.return_value = sample_template
    mock_components["forager"].forage.return_value = [] # No docs

    results = pipeline.run(sample_seeds, {}, {})

    assert results == []
    mock_components["extractor"].extract.assert_not_called()


def test_pipeline_empty_extract(pipeline, mock_components, sample_seeds, sample_template):
    mock_components["analyzer"].analyze.return_value = sample_template
    mock_components["forager"].forage.return_value = [Document(content="D", source_urn="u")]
    mock_components["extractor"].extract.return_value = [] # No slices

    results = pipeline.run(sample_seeds, {}, {})

    assert results == []
    mock_components["compositor"].composite.assert_not_called()
