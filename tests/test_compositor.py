# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_synthesis

from unittest.mock import AsyncMock

import pytest

from coreason_synthesis.compositor import CompositorImpl, GenerationOutput
from coreason_synthesis.interfaces import TeacherModel
from coreason_synthesis.models import (
    ExtractedSlice,
    ProvenanceType,
    SynthesisTemplate,
)


@pytest.fixture
def mock_teacher() -> AsyncMock:
    return AsyncMock(spec=TeacherModel)


@pytest.fixture
def compositor(mock_teacher: AsyncMock) -> CompositorImpl:
    return CompositorImpl(teacher=mock_teacher)


@pytest.fixture
def sample_slice() -> ExtractedSlice:
    return ExtractedSlice(
        content="Verbatim Context",
        source_urn="urn:1",
        page_number=1,
        pii_redacted=False,
    )


@pytest.fixture
def sample_template() -> SynthesisTemplate:
    return SynthesisTemplate(
        structure="Q",
        complexity_description="M",
        domain="D",
        embedding_centroid=[],
    )


@pytest.mark.asyncio
async def test_composite_happy_path(
    compositor: CompositorImpl,
    mock_teacher: AsyncMock,
    sample_slice: ExtractedSlice,
    sample_template: SynthesisTemplate,
) -> None:
    # Setup
    mock_teacher.generate_structured.return_value = GenerationOutput(
        synthetic_question="Q?",
        golden_chain_of_thought="Reasoning",
        expected_json={"ans": 42},
    )

    # Act
    result = await compositor.composite(sample_slice, sample_template)

    # Assert
    assert result.synthetic_question == "Q?"
    assert result.expected_json == {"ans": 42}
    assert result.verbatim_context == "Verbatim Context"
    assert result.provenance == ProvenanceType.VERBATIM_SOURCE

    mock_teacher.generate_structured.assert_awaited_once()


@pytest.mark.asyncio
async def test_composite_lineage(
    compositor: CompositorImpl,
    mock_teacher: AsyncMock,
    sample_slice: ExtractedSlice,
    sample_template: SynthesisTemplate,
) -> None:
    mock_teacher.generate_structured.return_value = GenerationOutput(
        synthetic_question="Q", golden_chain_of_thought="R", expected_json={}
    )

    result = await compositor.composite(sample_slice, sample_template)

    assert result.source_urn == sample_slice.source_urn


@pytest.mark.asyncio
async def test_composite_metrics_init(
    compositor: CompositorImpl,
    mock_teacher: AsyncMock,
    sample_slice: ExtractedSlice,
    sample_template: SynthesisTemplate,
) -> None:
    mock_teacher.generate_structured.return_value = GenerationOutput(
        synthetic_question="Q", golden_chain_of_thought="R", expected_json={}
    )

    result = await compositor.composite(sample_slice, sample_template)

    assert result.complexity == 0.0
    assert result.diversity == 0.0
    assert result.validity_confidence == 0.0


@pytest.mark.asyncio
async def test_composite_prompt_structure(
    compositor: CompositorImpl,
    mock_teacher: AsyncMock,
    sample_slice: ExtractedSlice,
    sample_template: SynthesisTemplate,
) -> None:
    mock_teacher.generate_structured.return_value = GenerationOutput(
        synthetic_question="Q", golden_chain_of_thought="R", expected_json={}
    )

    await compositor.composite(sample_slice, sample_template)

    args, kwargs = mock_teacher.generate_structured.call_args
    prompt = kwargs.get("prompt") or args[0]
    context_arg = kwargs.get("context")

    assert "Context:\nVerbatim Context" in prompt
    assert sample_template.domain in prompt
    assert context_arg == "Verbatim Context"


@pytest.mark.asyncio
async def test_composite_teacher_failure(
    compositor: CompositorImpl,
    mock_teacher: AsyncMock,
    sample_slice: ExtractedSlice,
    sample_template: SynthesisTemplate,
) -> None:
    mock_teacher.generate_structured.side_effect = Exception("Model Error")

    with pytest.raises(Exception, match="Model Error"):
        await compositor.composite(sample_slice, sample_template)


@pytest.mark.asyncio
async def test_composite_complex_expected_json(
    compositor: CompositorImpl,
    mock_teacher: AsyncMock,
    sample_slice: ExtractedSlice,
    sample_template: SynthesisTemplate,
) -> None:
    complex_json = {"nested": [{"id": 1}, {"id": 2}]}
    mock_teacher.generate_structured.return_value = GenerationOutput(
        synthetic_question="Q",
        golden_chain_of_thought="R",
        expected_json=complex_json,
    )

    result = await compositor.composite(sample_slice, sample_template)
    assert result.expected_json == complex_json


@pytest.mark.asyncio
async def test_composite_special_chars_context(
    compositor: CompositorImpl,
    mock_teacher: AsyncMock,
    sample_slice: ExtractedSlice,
    sample_template: SynthesisTemplate,
) -> None:
    sample_slice.content = "Special chars: \u2022 \n \t"
    mock_teacher.generate_structured.return_value = GenerationOutput(
        synthetic_question="Q", golden_chain_of_thought="R", expected_json={}
    )

    await compositor.composite(sample_slice, sample_template)
    # Just ensure no crash
    mock_teacher.generate_structured.assert_awaited_once()


@pytest.mark.asyncio
async def test_composite_empty_fields_from_teacher(
    compositor: CompositorImpl,
    mock_teacher: AsyncMock,
    sample_slice: ExtractedSlice,
    sample_template: SynthesisTemplate,
) -> None:
    # Model might return empty strings
    mock_teacher.generate_structured.return_value = GenerationOutput(
        synthetic_question="", golden_chain_of_thought="", expected_json={}
    )

    result = await compositor.composite(sample_slice, sample_template)
    assert result.synthetic_question == ""
