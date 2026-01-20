# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_synthesis

from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from coreason_synthesis.appraiser import AppraisalAnalysis, AppraiserImpl
from coreason_synthesis.forager import ForagerImpl
from coreason_synthesis.interfaces import EmbeddingService, MCPClient, TeacherModel
from coreason_synthesis.models import (
    Document,
    ProvenanceType,
    SynthesisTemplate,
    SyntheticTestCase,
)


class TestCoverageGapFill:
    @pytest.fixture
    def mock_teacher(self) -> AsyncMock:
        return AsyncMock(spec=TeacherModel)

    @pytest.fixture
    def mock_embedder(self) -> AsyncMock:
        return AsyncMock(spec=EmbeddingService)

    @pytest.fixture
    def mock_mcp(self) -> AsyncMock:
        return AsyncMock(spec=MCPClient)

    @pytest.fixture
    def appraiser(self, mock_teacher: AsyncMock, mock_embedder: AsyncMock) -> AppraiserImpl:
        return AppraiserImpl(teacher=mock_teacher, embedder=mock_embedder)

    @pytest.fixture
    def forager(self, mock_mcp: AsyncMock, mock_embedder: AsyncMock) -> ForagerImpl:
        return ForagerImpl(mcp_client=mock_mcp, embedder=mock_embedder)

    @pytest.fixture
    def template(self) -> SynthesisTemplate:
        return SynthesisTemplate(
            structure="Q",
            complexity_description="M",
            domain="D",
            embedding_centroid=[1.0, 0.0],
        )

    @pytest.mark.asyncio
    async def test_appraiser_sort_missing_branches(self, appraiser: AppraiserImpl, template: SynthesisTemplate) -> None:
        """Test sorting branches not covered by existing tests."""
        c1 = SyntheticTestCase(
            verbatim_context="C1",
            synthetic_question="Q1",
            golden_chain_of_thought="R1",
            expected_json={},
            provenance=ProvenanceType.VERBATIM_SOURCE,
            source_urn="u1",
            complexity=1.0,
            diversity=0.1,
            validity_confidence=0.1,
            modifications=[],
        )
        c2 = SyntheticTestCase(
            verbatim_context="C2",
            synthetic_question="Q2",
            golden_chain_of_thought="R2",
            expected_json={},
            provenance=ProvenanceType.VERBATIM_SOURCE,
            source_urn="u2",
            complexity=10.0,
            diversity=0.9,
            validity_confidence=0.9,
            modifications=[],
        )

        # Setup mocks to return consistent values so re-calculation doesn't change them
        # Teacher: returns same complexity/validity as input
        # Embedder: returns vectors yielding same diversity
        # c1: div=0.1 (sim=0.9). c2: div=0.9 (sim=0.1).
        # Centroid=[1,0]. c1=[0.9, ...], c2=[0.1, ...]

        async def mock_generate_structured_side_effect(
            prompt: str, response_model: Any, context: Any = None
        ) -> AppraisalAnalysis:
            # Check context to determine which case
            if "C1" in prompt:
                return AppraisalAnalysis(complexity_score=1.0, ambiguity_score=0, validity_confidence=0.1)
            else:
                return AppraisalAnalysis(complexity_score=10.0, ambiguity_score=0, validity_confidence=0.9)

        cast(AsyncMock, appraiser.teacher).generate_structured.side_effect = mock_generate_structured_side_effect

        async def mock_embed_side_effect(text: str) -> list[float]:
            if text == "C1":
                return [0.9, 0.435]  # ~norm 1
            else:
                return [0.1, 0.995]  # ~norm 1

        cast(AsyncMock, appraiser.embedder).embed.side_effect = mock_embed_side_effect

        cases = [c1, c2]

        # 1. diversity_asc (c1 < c2)
        res = await appraiser.appraise(cases, template, sort_by="diversity_asc", min_validity_score=0.0)
        assert res[0].verbatim_context == "C1"
        assert res[1].verbatim_context == "C2"

        # 2. validity_desc (c2 > c1)
        # Re-set side effects? No, logic is stateless based on input content
        res = await appraiser.appraise(cases, template, sort_by="validity_desc", min_validity_score=0.0)
        assert res[0].verbatim_context == "C2"
        assert res[1].verbatim_context == "C1"

        # 3. Unknown sort key (fallback -> complexity_desc -> c2 > c1)
        res = await appraiser.appraise(cases, template, sort_by="unknown_key", min_validity_score=0.0)
        assert res[0].verbatim_context == "C2"
        assert res[1].verbatim_context == "C1"

    @pytest.mark.asyncio
    async def test_forager_mmr_zero_vectors_multi_step(
        self, forager: ForagerImpl, mock_embedder: AsyncMock, template: SynthesisTemplate
    ) -> None:
        """Test MMR calculation with zero vectors and limit >= 2 to hit sim_ij=0 branch."""
        docs = [
            Document(content="Z1", source_urn="z1"),
            Document(content="Z2", source_urn="z2"),
        ]
        # Return zero vectors
        mock_embedder.embed.return_value = [0.0, 0.0]

        # We call _apply_mmr directly to avoid mocking MCP search
        # limit=2 forces at least 2 iterations.
        # Iter 1: Pick one (doesn't matter, both score 0 or -inf).
        # Iter 2: Calculate sim against selected. norm_j (selected) is 0.
        # This triggers `if norm_i == 0 or norm_j == 0: sim_ij = 0.0`

        res = await forager._apply_mmr([1.0, 0.0], docs, limit=2)
        assert len(res) == 2
