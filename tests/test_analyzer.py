# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_synthesis

from typing import List
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from coreason_synthesis.analyzer import PatternAnalyzerImpl, TemplateAnalysis
from coreason_synthesis.interfaces import EmbeddingService, TeacherModel
from coreason_synthesis.models import SeedCase, SynthesisTemplate


@pytest.fixture
def mock_teacher() -> AsyncMock:
    mock = AsyncMock(spec=TeacherModel)
    return mock


@pytest.fixture
def mock_embedder() -> AsyncMock:
    mock = AsyncMock(spec=EmbeddingService)
    # Default behavior: return a dummy vector
    mock.embed.return_value = [0.1, 0.2, 0.3]
    return mock


@pytest.fixture
def analyzer(mock_teacher: AsyncMock, mock_embedder: AsyncMock) -> PatternAnalyzerImpl:
    return PatternAnalyzerImpl(teacher=mock_teacher, embedder=mock_embedder)


@pytest.fixture
def sample_seeds() -> List[SeedCase]:
    return [
        SeedCase(
            id=uuid4(),
            context="Ctx1",
            question="Q1",
            expected_output={"ans": "A"},
        ),
        SeedCase(
            id=uuid4(),
            context="Ctx2",
            question="Q2",
            expected_output={"ans": "B"},
        ),
    ]


@pytest.mark.asyncio
async def test_analyze_flow_mock_teacher(
    analyzer: PatternAnalyzerImpl,
    mock_teacher: AsyncMock,
    mock_embedder: AsyncMock,
    sample_seeds: List[SeedCase],
) -> None:
    # Setup
    mock_teacher.generate_structured.return_value = TemplateAnalysis(
        structure="Q+A", complexity_description="Hard", domain="TestDomain"
    )

    # Act
    result = await analyzer.analyze(sample_seeds)

    # Assert
    assert isinstance(result, SynthesisTemplate)
    assert result.structure == "Q+A"
    assert result.complexity_description == "Hard"
    assert result.domain == "TestDomain"
    assert len(result.embedding_centroid or []) == 3  # Based on dummy vector

    # Verify calls
    assert mock_embedder.embed.call_count == 2
    mock_teacher.generate_structured.assert_awaited_once()


@pytest.mark.asyncio
async def test_empty_seeds_error(analyzer: PatternAnalyzerImpl) -> None:
    with pytest.raises(ValueError, match="Seed list cannot be empty"):
        await analyzer.analyze([])


@pytest.mark.asyncio
async def test_analyze_single_seed(
    analyzer: PatternAnalyzerImpl,
    mock_embedder: AsyncMock,
    sample_seeds: List[SeedCase],
) -> None:
    single_seed = [sample_seeds[0]]
    mock_embedder.embed.return_value = [1.0, 1.0, 1.0]

    result = await analyzer.analyze(single_seed)

    assert result.embedding_centroid == [1.0, 1.0, 1.0]
    assert mock_embedder.embed.call_count == 1


@pytest.mark.asyncio
async def test_centroid_calculation(
    analyzer: PatternAnalyzerImpl,
    mock_embedder: AsyncMock,
    sample_seeds: List[SeedCase],
) -> None:
    # Seed 1 -> [0, 0, 0], Seed 2 -> [2, 2, 2] => Centroid [1, 1, 1]
    mock_embedder.embed.side_effect = [[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]

    result = await analyzer.analyze(sample_seeds)

    assert result.embedding_centroid == [1.0, 1.0, 1.0]


@pytest.mark.asyncio
async def test_analyze_large_batch(
    analyzer: PatternAnalyzerImpl,
    mock_embedder: AsyncMock,
) -> None:
    # Create 100 seeds
    seeds = [SeedCase(id=uuid4(), context="C", question="Q", expected_output={}) for _ in range(100)]
    mock_embedder.embed.return_value = [1.0]

    result = await analyzer.analyze(seeds)

    assert mock_embedder.embed.call_count == 100
    assert result.embedding_centroid == [1.0]


@pytest.mark.asyncio
async def test_custom_teacher_structured_response(
    analyzer: PatternAnalyzerImpl,
    mock_teacher: AsyncMock,
    sample_seeds: List[SeedCase],
) -> None:
    # Ensure the model passed to generate_structured is correct
    await analyzer.analyze(sample_seeds)

    args, kwargs = mock_teacher.generate_structured.call_args
    assert kwargs.get("response_model") == TemplateAnalysis or args[1] == TemplateAnalysis


@pytest.mark.asyncio
async def test_teacher_failure_propagation(
    analyzer: PatternAnalyzerImpl,
    mock_teacher: AsyncMock,
    sample_seeds: List[SeedCase],
) -> None:
    mock_teacher.generate_structured.side_effect = RuntimeError("LLM Failure")

    with pytest.raises(RuntimeError, match="LLM Failure"):
        await analyzer.analyze(sample_seeds)


@pytest.mark.asyncio
async def test_embedding_service_failure(
    analyzer: PatternAnalyzerImpl,
    mock_embedder: AsyncMock,
    sample_seeds: List[SeedCase],
) -> None:
    mock_embedder.embed.side_effect = ConnectionError("Embedder Down")

    with pytest.raises(ConnectionError, match="Embedder Down"):
        await analyzer.analyze(sample_seeds)


@pytest.mark.asyncio
async def test_zero_vector_embeddings(
    analyzer: PatternAnalyzerImpl,
    mock_embedder: AsyncMock,
    sample_seeds: List[SeedCase],
) -> None:
    mock_embedder.embed.return_value = [0.0, 0.0]
    result = await analyzer.analyze(sample_seeds)
    assert result.embedding_centroid == [0.0, 0.0]


@pytest.mark.asyncio
async def test_mixed_domain_seeds(
    analyzer: PatternAnalyzerImpl,
    mock_teacher: AsyncMock,
    sample_seeds: List[SeedCase],
) -> None:
    # Just verifying that the prompt construction handles multiple seeds
    # We can inspect the prompt string in call_args
    await analyzer.analyze(sample_seeds)

    call_args = mock_teacher.generate_structured.call_args
    prompt = call_args[0][0]

    assert "Seed 1:" in prompt
    assert "Seed 2:" in prompt
