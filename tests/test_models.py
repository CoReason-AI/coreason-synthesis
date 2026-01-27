# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_synthesis


import pytest
from pydantic import BaseModel, ValidationError

from coreason_synthesis.models import Diff, ProvenanceType, SyntheticTestCase


class MockModel(BaseModel):
    pass


def test_synthetic_test_case_init() -> None:
    case = SyntheticTestCase(
        verbatim_context="ctx",
        synthetic_question="q",
        golden_chain_of_thought="r",
        expected_json={"a": 1},
        provenance=ProvenanceType.VERBATIM_SOURCE,
        source_urn="urn:1",
        modifications=[],
        complexity=1.0,
        diversity=0.5,
        validity_confidence=0.9,
    )
    assert case.verbatim_context == "ctx"
    assert case.complexity == 1.0


def test_synthetic_test_case_modifications() -> None:
    diff = Diff(description="change", original="old", new="new")
    # This was previously testing string support too, but we removed it.
    case = SyntheticTestCase(
        verbatim_context="ctx",
        synthetic_question="q",
        golden_chain_of_thought="r",
        expected_json={},
        provenance=ProvenanceType.SYNTHETIC_PERTURBED,
        source_urn="urn:1",
        modifications=[diff],
        complexity=1.0,
        diversity=0.5,
        validity_confidence=0.0,
    )
    assert len(case.modifications) == 1
    assert isinstance(case.modifications[0], Diff)
    assert case.modifications[0].description == "change"


def test_synthetic_test_case_json_serialization() -> None:
    diff = Diff(description="Simple diff")
    case = SyntheticTestCase(
        verbatim_context="ctx",
        synthetic_question="q",
        golden_chain_of_thought="r",
        expected_json={},
        provenance=ProvenanceType.SYNTHETIC_PERTURBED,
        source_urn="urn:1",
        modifications=[diff],
        complexity=1.0,
        diversity=0.5,
        validity_confidence=0.0,
    )
    json_str = case.model_dump_json()
    assert "Simple diff" in json_str


def test_synthetic_test_case_invalid_modification_type() -> None:
    # Now that we enforce List[Diff], passing strings should fail validation
    with pytest.raises(ValidationError):  # validation error
        SyntheticTestCase(
            verbatim_context="ctx",
            synthetic_question="q",
            golden_chain_of_thought="r",
            expected_json={},
            provenance=ProvenanceType.SYNTHETIC_PERTURBED,
            source_urn="urn:1",
            modifications=["Just a string"],  # type: ignore
            complexity=1.0,
            diversity=0.5,
            validity_confidence=0.0,
        )

def test_synthetic_test_case_ownership() -> None:
    case = SyntheticTestCase(
        verbatim_context="ctx",
        synthetic_question="q",
        golden_chain_of_thought="r",
        expected_json={"a": 1},
        provenance=ProvenanceType.VERBATIM_SOURCE,
        source_urn="urn:1",
        modifications=[],
        complexity=1.0,
        diversity=0.5,
        validity_confidence=0.9,
        created_by="user123",
        tenant_id="tenant456"
    )
    assert case.created_by == "user123"
    assert case.tenant_id == "tenant456"

    # Test defaults
    case_default = SyntheticTestCase(
        verbatim_context="ctx",
        synthetic_question="q",
        golden_chain_of_thought="r",
        expected_json={"a": 1},
        provenance=ProvenanceType.VERBATIM_SOURCE,
        source_urn="urn:1",
        complexity=1.0,
        diversity=0.5,
        validity_confidence=0.9,
    )
    assert case_default.created_by is None
    assert case_default.tenant_id is None
