# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_synthesis

"""
Tests for the mocks module to ensure they behave as expected and for code coverage.
"""

import pytest
from pydantic import BaseModel

from coreason_synthesis.appraiser import AppraisalAnalysis
from coreason_synthesis.compositor import GenerationOutput
from coreason_synthesis.mocks.embedding import DummyEmbeddingService
from coreason_synthesis.mocks.mcp import MockMCPClient
from coreason_synthesis.mocks.teacher import MockTeacher
from coreason_synthesis.models import Document, SynthesisTemplate


class SampleModel(BaseModel):
    name: str = "default"


@pytest.mark.asyncio
async def test_dummy_embedding_service() -> None:
    service = DummyEmbeddingService(dimension=3)
    vec1 = await service.embed("test")
    vec2 = await service.embed("test")
    vec3 = await service.embed("other")

    assert len(vec1) == 3
    assert vec1 == vec2  # Deterministic
    assert vec1 != vec3  # Different inputs


@pytest.mark.asyncio
async def test_mock_mcp_client() -> None:
    docs = [Document(content="A", source_urn="1")]
    client = MockMCPClient(documents=docs)

    results = await client.search([0.1], {}, limit=1)
    assert len(results) == 1
    assert results[0].content == "A"

    # Check stored state
    assert client.last_limit == 1
    assert client.last_query_vector == [0.1]


@pytest.mark.asyncio
async def test_mock_teacher_generate() -> None:
    teacher = MockTeacher()

    resp1 = await teacher.generate("prompt about structure")
    assert "Structure:" in resp1

    resp2 = await teacher.generate("random prompt")
    assert resp2 == "Mock generated response"


@pytest.mark.asyncio
async def test_mock_teacher_generate_structured() -> None:
    teacher = MockTeacher()

    # Test known models

    # Test SynthesisTemplate
    tmpl = await teacher.generate_structured("p", SynthesisTemplate)
    assert isinstance(tmpl, SynthesisTemplate)
    assert tmpl.structure == "Question + JSON Output"

    # Test GenerationOutput
    gen = await teacher.generate_structured("p", GenerationOutput)
    assert isinstance(gen, GenerationOutput)

    # Test AppraisalAnalysis
    appr = await teacher.generate_structured("p", AppraisalAnalysis)
    assert isinstance(appr, AppraisalAnalysis)

    # Test unknown model -> NotImplementedError
    class UnknownModel(BaseModel):
        pass

    with pytest.raises(NotImplementedError):
        await teacher.generate_structured("p", UnknownModel)
