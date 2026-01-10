# tests/test_models.py
from uuid import uuid4

import pytest
from pydantic import ValidationError

from coreason_synthesis.models import (
    Diff,
    ProvenanceType,
    SeedCase,
    SyntheticJob,
    SyntheticTestCase,
)


def test_seed_case_creation() -> None:
    seed_id = uuid4()
    seed = SeedCase(
        id=seed_id,
        context="This is a context.",
        question="What is this?",
        expected_output={"answer": "A context"},
    )
    assert seed.id == seed_id
    assert seed.context == "This is a context."
    assert seed.metadata == {}


def test_synthetic_job_creation() -> None:
    job_id = uuid4()
    seed_ids = [uuid4(), uuid4()]
    job = SyntheticJob(
        id=job_id,
        seed_ids=seed_ids,
        config={"perturbation_rate": 0.2, "target_count": 50},
    )
    assert job.id == job_id
    assert job.seed_ids == seed_ids
    assert job.config["target_count"] == 50


def test_synthetic_test_case_valid() -> None:
    tc = SyntheticTestCase(
        verbatim_context="Real data",
        synthetic_question="Question?",
        golden_chain_of_thought="Step 1...",
        expected_json={"result": "OK"},
        provenance=ProvenanceType.VERBATIM_SOURCE,
        source_urn="mcp://doc1",
        complexity=5.5,
        diversity=0.8,
        validity_confidence=0.9,
    )
    assert tc.complexity == 5.5
    assert tc.provenance == "VERBATIM_SOURCE"


def test_synthetic_test_case_modifications() -> None:
    # Test with Diff object
    diff_obj = Diff(description="Changed value", original="50mg", new="5000mg")
    tc1 = SyntheticTestCase(
        verbatim_context="ctx",
        synthetic_question="q",
        golden_chain_of_thought="cot",
        expected_json={},
        provenance=ProvenanceType.SYNTHETIC_PERTURBED,
        source_urn="urn",
        modifications=[diff_obj],
        complexity=5,
        diversity=0.5,
        validity_confidence=0.9,
    )
    assert isinstance(tc1.modifications[0], Diff)
    assert tc1.modifications[0].description == "Changed value"

    # Test with string
    tc2 = SyntheticTestCase(
        verbatim_context="ctx",
        synthetic_question="q",
        golden_chain_of_thought="cot",
        expected_json={},
        provenance=ProvenanceType.SYNTHETIC_PERTURBED,
        source_urn="urn",
        modifications=["Simple string diff"],
        complexity=5,
        diversity=0.5,
        validity_confidence=0.9,
    )
    assert tc2.modifications[0] == "Simple string diff"


def test_synthetic_test_case_validation_error() -> None:
    with pytest.raises(ValidationError):
        SyntheticTestCase(
            verbatim_context="ctx",
            synthetic_question="q",
            golden_chain_of_thought="cot",
            expected_json={},
            provenance=ProvenanceType.VERBATIM_SOURCE,
            source_urn="urn",
            complexity=11.0,  # Invalid: > 10
            diversity=0.5,
            validity_confidence=0.9,
        )

    with pytest.raises(ValidationError):
        SyntheticTestCase(
            verbatim_context="ctx",
            synthetic_question="q",
            golden_chain_of_thought="cot",
            expected_json={},
            provenance=ProvenanceType.VERBATIM_SOURCE,
            source_urn="urn",
            complexity=5.0,
            diversity=1.1,  # Invalid: > 1
            validity_confidence=0.9,
        )
