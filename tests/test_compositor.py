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
