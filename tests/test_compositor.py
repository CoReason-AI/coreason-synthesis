from typing import Optional, Type, TypeVar
from unittest.mock import Mock

import pytest
from pydantic import BaseModel

from coreason_synthesis.compositor import CompositorImpl, GenerationOutput
from coreason_synthesis.interfaces import TeacherModel
from coreason_synthesis.models import (
    ExtractedSlice,
    ProvenanceType,
    SynthesisTemplate,
    SyntheticTestCase,
)

T = TypeVar("T", bound=BaseModel)


class MockCompositorTeacher(TeacherModel):
    """
    Mock Teacher specifically for testing Compositor.
    It returns a GenerationOutput with predictable values.
    """

    def generate(self, prompt: str, context: Optional[str] = None) -> str:
        return "Mock String Response"

    def generate_structured(self, prompt: str, response_model: Type[T], context: Optional[str] = None) -> T:
        if response_model == GenerationOutput:
            # Return a valid GenerationOutput
            return GenerationOutput(
                synthetic_question="What is the mock question?",
                golden_chain_of_thought="Step 1: Mock reasoning.",
                expected_json={"result": "mock_value"},
            )  # type: ignore
        raise NotImplementedError(f"Unexpected response model: {response_model}")


@pytest.fixture
def mock_teacher() -> MockCompositorTeacher:
    return MockCompositorTeacher()


@pytest.fixture
def compositor(mock_teacher: MockCompositorTeacher) -> CompositorImpl:
    return CompositorImpl(teacher=mock_teacher)


@pytest.fixture
def sample_template() -> SynthesisTemplate:
    return SynthesisTemplate(
        structure="Question + JSON",
        complexity_description="High",
        domain="Test Domain",
        embedding_centroid=[0.1, 0.2, 0.3],
    )


@pytest.fixture
def sample_slice() -> ExtractedSlice:
    return ExtractedSlice(
        content="This is the verbatim context.",
        source_urn="doc_123",
        page_number=1,
        pii_redacted=False,
    )


def test_composite_happy_path(
    compositor: CompositorImpl, sample_slice: ExtractedSlice, sample_template: SynthesisTemplate
) -> None:
    """
    Test the standard flow of generating a test case.
    """
    result = compositor.composite(sample_slice, sample_template)

    assert isinstance(result, SyntheticTestCase)
    assert result.verbatim_context == sample_slice.content
    assert result.synthetic_question == "What is the mock question?"
    assert result.golden_chain_of_thought == "Step 1: Mock reasoning."
    assert result.expected_json == {"result": "mock_value"}
    assert result.provenance == ProvenanceType.VERBATIM_SOURCE
    assert result.source_urn == sample_slice.source_urn


def test_composite_lineage(
    compositor: CompositorImpl, sample_slice: ExtractedSlice, sample_template: SynthesisTemplate
) -> None:
    """
    Verify that source_urn is correctly propagated.
    """
    sample_slice.source_urn = "mcp://custom/urn"
    result = compositor.composite(sample_slice, sample_template)

    assert result.source_urn == "mcp://custom/urn"
    assert result.verbatim_context == sample_slice.content


def test_composite_metrics_init(
    compositor: CompositorImpl, sample_slice: ExtractedSlice, sample_template: SynthesisTemplate
) -> None:
    """
    Verify that metrics are initialized to 0.0.
    """
    result = compositor.composite(sample_slice, sample_template)

    assert result.complexity == 0.0
    assert result.diversity == 0.0
    assert result.validity_confidence == 0.0


def test_composite_prompt_structure(
    mock_teacher: MockCompositorTeacher, sample_slice: ExtractedSlice, sample_template: SynthesisTemplate
) -> None:
    """
    Verify that the prompt is constructed correctly.
    """
    # Wrap the teacher with a real Mock to inspect calls
    spy_teacher = Mock(wraps=mock_teacher)
    compositor = CompositorImpl(teacher=spy_teacher)

    compositor.composite(sample_slice, sample_template)

    spy_teacher.generate_structured.assert_called_once()
    call_args = spy_teacher.generate_structured.call_args
    prompt = call_args.kwargs["prompt"]

    assert "Test Domain" in prompt
    assert "Question + JSON" in prompt
    assert "High" in prompt
    assert sample_slice.content in prompt


def test_composite_teacher_failure(
    compositor: CompositorImpl, sample_slice: ExtractedSlice, sample_template: SynthesisTemplate
) -> None:
    """
    Verify that exceptions raised by the TeacherModel are propagated correctly.
    """
    # Override the teacher to raise an exception
    compositor.teacher.generate_structured = Mock(side_effect=RuntimeError("Teacher failed"))  # type: ignore

    with pytest.raises(RuntimeError, match="Teacher failed"):
        compositor.composite(sample_slice, sample_template)


def test_composite_complex_expected_json(
    compositor: CompositorImpl, sample_slice: ExtractedSlice, sample_template: SynthesisTemplate
) -> None:
    """
    Verify that deeply nested JSON objects in the expected output are preserved.
    """
    complex_json = {
        "nested": {"level1": {"level2": [1, 2, 3]}},
        "mixed": [
            {"id": 1, "val": "a"},
            {"id": 2, "val": "b"},
        ],
        "null_val": None,
    }

    # Setup mock to return complex JSON
    mock_output = GenerationOutput(
        synthetic_question="q",
        golden_chain_of_thought="r",
        expected_json=complex_json,
    )
    compositor.teacher.generate_structured = Mock(return_value=mock_output)  # type: ignore

    result = compositor.composite(sample_slice, sample_template)

    assert result.expected_json == complex_json


def test_composite_special_chars_context(
    compositor: CompositorImpl, sample_slice: ExtractedSlice, sample_template: SynthesisTemplate
) -> None:
    """
    Verify that the system handles context with special characters without breaking.
    """
    # Context with quotes, newlines, unicode emojis, and tabs
    special_context = 'Context with "quotes", \nnewlines, \t tabs, and emojis 🧪🚀.'
    sample_slice.content = special_context

    # We assume the mock teacher behaves normally (ignoring context content for logic, but we check if it crashes)
    # The real test is that prompt construction doesn't fail and it passes data through.

    # Spy to check prompt
    spy_teacher = Mock(wraps=compositor.teacher)
    compositor.teacher = spy_teacher

    result = compositor.composite(sample_slice, sample_template)

    assert result.verbatim_context == special_context

    # Verify prompt contains the special context exactly
    call_args = spy_teacher.generate_structured.call_args
    prompt = call_args.kwargs["prompt"]
    assert special_context in prompt


def test_composite_empty_fields_from_teacher(
    compositor: CompositorImpl, sample_slice: ExtractedSlice, sample_template: SynthesisTemplate
) -> None:
    """
    Verify behavior when the Teacher returns empty strings for required fields.
    """
    mock_output = GenerationOutput(
        synthetic_question="",  # Empty question
        golden_chain_of_thought="",  # Empty reasoning
        expected_json={},  # Empty dict
    )
    compositor.teacher.generate_structured = Mock(return_value=mock_output)  # type: ignore

    result = compositor.composite(sample_slice, sample_template)

    assert result.synthetic_question == ""
    assert result.golden_chain_of_thought == ""
    assert result.expected_json == {}
