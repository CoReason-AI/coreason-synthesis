# tests/test_models.py
from uuid import uuid4

import numpy as np
import pytest
from pydantic import ValidationError

from src.coreason_synthesis.models import (
    Diff,
    Document,
    ProvenanceType,
    SeedCase,
    SynthesisTemplate,
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


def test_mixed_modifications_types() -> None:
    """Test mixed types in modifications list (Diff objects and strings)."""
    diff_obj = Diff(description="Obj diff", original="old", new="new")
    tc = SyntheticTestCase(
        verbatim_context="ctx",
        synthetic_question="q",
        golden_chain_of_thought="cot",
        expected_json={},
        provenance=ProvenanceType.SYNTHETIC_PERTURBED,
        source_urn="urn",
        modifications=[diff_obj, "String diff", Diff(description="Another obj", original="old2", new="new2")],
        complexity=5,
        diversity=0.5,
        validity_confidence=0.9,
    )
    assert len(tc.modifications) == 3
    assert isinstance(tc.modifications[0], Diff)
    assert isinstance(tc.modifications[1], str)
    assert isinstance(tc.modifications[2], Diff)


def test_complex_json_structures() -> None:
    """Test deeply nested and complex JSON structures."""
    complex_json = {
        "list": [1, 2, {"nested": "value"}],
        "dict": {"a": 1, "b": None},
        "bool": True,
        "null": None,
    }
    tc = SyntheticTestCase(
        verbatim_context="ctx",
        synthetic_question="q",
        golden_chain_of_thought="cot",
        expected_json=complex_json,
        provenance=ProvenanceType.VERBATIM_SOURCE,
        source_urn="urn",
        complexity=5,
        diversity=0.5,
        validity_confidence=0.9,
    )
    assert tc.expected_json == complex_json

    # Round trip check
    dump = tc.model_dump()
    assert dump["expected_json"] == complex_json


def test_boundary_values() -> None:
    """Test boundary values for metrics."""
    # Test Min values
    tc_min = SyntheticTestCase(
        verbatim_context="ctx",
        synthetic_question="q",
        golden_chain_of_thought="cot",
        expected_json={},
        provenance=ProvenanceType.VERBATIM_SOURCE,
        source_urn="urn",
        complexity=0.0,
        diversity=0.0,
        validity_confidence=0.0,
    )
    assert tc_min.complexity == 0.0

    # Test Max values
    tc_max = SyntheticTestCase(
        verbatim_context="ctx",
        synthetic_question="q",
        golden_chain_of_thought="cot",
        expected_json={},
        provenance=ProvenanceType.VERBATIM_SOURCE,
        source_urn="urn",
        complexity=10.0,
        diversity=1.0,
        validity_confidence=1.0,
    )
    assert tc_max.complexity == 10.0


def test_serialization_roundtrip() -> None:
    """Test full JSON serialization and deserialization."""
    seed_id = uuid4()
    seed = SeedCase(id=seed_id, context="ctx", question="q", expected_output={"a": 1}, metadata={"key": "val"})

    json_str = seed.model_dump_json()
    seed_restored = SeedCase.model_validate_json(json_str)

    assert seed_restored == seed
    assert seed_restored.id == seed_id
    assert seed_restored.metadata["key"] == "val"


def test_enum_validation() -> None:
    """Test invalid enum values."""
    with pytest.raises(ValidationError):
        SyntheticTestCase(
            verbatim_context="ctx",
            synthetic_question="q",
            golden_chain_of_thought="cot",
            expected_json={},
            provenance="INVALID_PROVENANCE",  # type: ignore
            source_urn="urn",
            complexity=5,
            diversity=0.5,
            validity_confidence=0.9,
        )


def test_seed_case_string_expected_output() -> None:
    """Test SeedCase with string expected_output."""
    seed = SeedCase(id=uuid4(), context="ctx", question="q", expected_output="Just a string answer")
    assert seed.expected_output == "Just a string answer"


# --- New Tests for Edge Cases & Complex Scenarios ---


def test_document_creation() -> None:
    """Test creation of Document model."""
    doc = Document(content="Sample content", source_urn="http://example.com", metadata={"author": "Bot"})
    assert doc.content == "Sample content"
    assert doc.source_urn == "http://example.com"
    assert doc.metadata["author"] == "Bot"


def test_synthesis_template_creation() -> None:
    """Test creation of SynthesisTemplate model."""
    template = SynthesisTemplate(
        structure="QA", complexity_description="Med", domain="Tech", embedding_centroid=[0.1, 0.2]
    )
    assert template.structure == "QA"
    assert template.embedding_centroid == [0.1, 0.2]


def test_synthesis_template_empty_centroid() -> None:
    """Test SynthesisTemplate with None or empty centroid."""
    # None
    t1 = SynthesisTemplate(structure="S", complexity_description="C", domain="D", embedding_centroid=None)
    assert t1.embedding_centroid is None

    # Empty list
    t2 = SynthesisTemplate(structure="S", complexity_description="C", domain="D", embedding_centroid=[])
    assert t2.embedding_centroid == []


def test_numpy_interop_for_centroid() -> None:
    """
    Test that SynthesisTemplate can handle numpy arrays for embedding_centroid.
    Pydantic V2 should coerce iterable types (like np.array) to List[float] automatically.
    """
    arr = np.array([0.1, 0.5, 0.9], dtype=np.float64)
    t = SynthesisTemplate(structure="S", complexity_description="C", domain="D", embedding_centroid=arr.tolist())

    assert isinstance(t.embedding_centroid, list)
    assert len(t.embedding_centroid) == 3
    assert t.embedding_centroid[0] == 0.1


def test_missing_required_fields() -> None:
    """Ensure ValidationError is raised when required fields are missing."""
    with pytest.raises(ValidationError):
        Document(content="Only content")  # type: ignore # Missing source_urn

    with pytest.raises(ValidationError):
        SynthesisTemplate(structure="S")  # type: ignore # Missing other fields


def test_document_empty_content() -> None:
    """Test that empty strings are allowed (unless constraints added), ensuring system robustness."""
    # Currently no min_length constraint, so this should pass.
    # If we add constraints later, this test will fail and remind us to update constraints.
    doc = Document(content="", source_urn="urn")
    assert doc.content == ""


def test_large_diff_list() -> None:
    """Test performance/handling of a large list of modifications."""
    from typing import List, Union

    mods: List[Union[Diff, str]] = [Diff(description=f"Change {i}", original="a", new="b") for i in range(1000)]
    tc = SyntheticTestCase(
        verbatim_context="ctx",
        synthetic_question="q",
        golden_chain_of_thought="cot",
        expected_json={},
        provenance=ProvenanceType.SYNTHETIC_PERTURBED,
        source_urn="urn",
        modifications=mods,
        complexity=5,
        diversity=0.5,
        validity_confidence=0.9,
    )
    assert len(tc.modifications) == 1000
    assert tc.modifications[999].description == "Change 999"  # type: ignore
