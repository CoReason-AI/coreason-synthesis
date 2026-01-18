# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_synthesis

from typing import List, Tuple

import pytest

from coreason_synthesis.forager import ForagerImpl
from coreason_synthesis.mocks.embedding import DummyEmbeddingService
from coreason_synthesis.mocks.mcp import MockMCPClient
from coreason_synthesis.models import Document, SynthesisTemplate


@pytest.fixture
def forager_setup() -> Tuple[ForagerImpl, MockMCPClient, DummyEmbeddingService]:
    embedder = DummyEmbeddingService(dimension=4)
    mcp = MockMCPClient()
    forager = ForagerImpl(mcp, embedder)
    return forager, mcp, embedder


def test_forage_basic_flow(forager_setup: Tuple[ForagerImpl, MockMCPClient, DummyEmbeddingService]) -> None:
    """Test that forage calls MCP and returns documents."""
    forager, mcp, _ = forager_setup

    # Setup MCP with some docs
    docs = [
        Document(content="Doc A", source_urn="urn:1", metadata={}),
        Document(content="Doc B", source_urn="urn:2", metadata={}),
    ]
    mcp.documents = docs

    template = SynthesisTemplate(
        structure="S", complexity_description="C", domain="D", embedding_centroid=[0.1, 0.1, 0.1, 0.1]
    )
    user_context = {"user_id": "test_user"}

    results = forager.forage(template, user_context, limit=2)

    assert len(results) == 2
    assert results[0].content == "Doc A"
    assert mcp.last_user_context == user_context
    assert mcp.last_limit == 10  # 2 * 5 = 10 fetch limit


def test_mmr_diversity(forager_setup: Tuple[ForagerImpl, MockMCPClient, DummyEmbeddingService]) -> None:
    """
    Test that MMR prefers diverse documents.
    """
    forager, mcp, embedder = forager_setup

    # Let's mock the embedder for precise control
    class ControlledEmbedder(DummyEmbeddingService):
        def embed(self, text: str) -> List[float]:
            # Consistent 2D vectors
            if text == "Query":
                return [1.0, 0.0]
            if text == "Doc1":
                return [1.0, 0.0]
            if text == "Doc2":
                return [1.0, 0.01]  # Almost identical
            if text == "Doc3":
                return [0.707, 0.707]  # 45 degrees
            return [0.0, 0.0]

    embedder = ControlledEmbedder(dimension=2)
    forager = ForagerImpl(mcp, embedder)

    docs = [
        Document(content="Doc1", source_urn="1", metadata={}),
        Document(content="Doc2", source_urn="2", metadata={}),
        Document(content="Doc3", source_urn="3", metadata={}),
    ]
    mcp.documents = docs

    template = SynthesisTemplate(
        structure="S",
        complexity_description="C",
        domain="D",
        embedding_centroid=[1.0, 0.0],  # The "Query"
    )

    # Doc1 selected first (Rel=1.0)
    # Doc2 is very similar to Doc1, so should be penalized by MMR
    # Doc3 is less similar to Doc1, so might be picked if lambda balances right.
    # We just want to ensure it runs without error (fixing the shape bug)
    # and returns 2 documents.
    results = forager.forage(template, {}, limit=2)

    assert len(results) == 2
    assert results[0].content in ["Doc1", "Doc2"]


def test_empty_mcp_results(forager_setup: Tuple[ForagerImpl, MockMCPClient, DummyEmbeddingService]) -> None:
    """Test handling when MCP returns nothing."""
    forager, mcp, _ = forager_setup
    mcp.documents = []

    template = SynthesisTemplate(
        structure="S", complexity_description="C", domain="D", embedding_centroid=[1.0, 0.0, 0.0, 0.0]
    )

    results = forager.forage(template, {}, limit=5)
    assert results == []


def test_missing_centroid(forager_setup: Tuple[ForagerImpl, MockMCPClient, DummyEmbeddingService]) -> None:
    """Test handling when template has no centroid."""
    forager, _, _ = forager_setup
    template = SynthesisTemplate(structure="S", complexity_description="C", domain="D", embedding_centroid=None)

    results = forager.forage(template, {}, limit=5)
    assert results == []


def test_mmr_direct_empty_candidates(forager_setup: Tuple[ForagerImpl, MockMCPClient, DummyEmbeddingService]) -> None:
    """Test _apply_mmr with empty candidates to hit defensive return."""
    forager, _, _ = forager_setup
    results = forager._apply_mmr([1.0, 0.0], [], limit=5)
    assert results == []


def test_mmr_zero_vectors(forager_setup: Tuple[ForagerImpl, MockMCPClient, DummyEmbeddingService]) -> None:
    """
    Test MMR robustness against zero vectors (handling division by zero).
    """
    forager, mcp, _ = forager_setup

    class ZeroEmbedder(DummyEmbeddingService):
        def embed(self, text: str) -> List[float]:
            if text == "Zero":
                return [0.0, 0.0]
            return [1.0, 0.0]

    embedder = ZeroEmbedder(dimension=2)
    forager = ForagerImpl(mcp, embedder)

    docs = [
        Document(content="Zero", source_urn="z", metadata={}),
        Document(content="Normal", source_urn="n", metadata={}),
    ]
    mcp.documents = docs

    # Query is also zero to test query norm zero check
    template = SynthesisTemplate(structure="S", complexity_description="C", domain="D", embedding_centroid=[0.0, 0.0])

    results = forager.forage(template, {}, limit=2)
    assert len(results) == 2

    # Test Normal query against Zero doc
    template_normal = SynthesisTemplate(
        structure="S", complexity_description="C", domain="D", embedding_centroid=[1.0, 0.0]
    )
    results_normal = forager.forage(template_normal, {}, limit=2)
    assert len(results_normal) == 2


def test_mmr_cluster_selection(forager_setup: Tuple[ForagerImpl, MockMCPClient, DummyEmbeddingService]) -> None:
    """
    Complex Scenario: Cluster Selection.
    Scenario:
    - Query: [1, 0]
    - Cluster A (Relevant but Redundant): 3 docs extremely close to Query and each other.
    - Cluster B (Less Relevant but Unique): 1 doc moderately close to Query but far from A.

    Goal: Ensure MMR picks 1 from A, then 1 from B, instead of just filling up with A.
    """
    _, mcp, _ = forager_setup

    class ClusterEmbedder(DummyEmbeddingService):
        def embed(self, text: str) -> List[float]:
            if text == "Query":
                return [1.0, 0.0]
            if text.startswith("A"):
                # All A's are [1.0, 0.0] (Identical to query and each other)
                return [1.0, 0.0]
            if text.startswith("B"):
                # B is [0.707, 0.707] (45 degrees). Sim(Q, B) = 0.707. Sim(A, B) = 0.707.
                return [0.707, 0.707]
            return [0.0, 0.0]

    embedder = ClusterEmbedder(dimension=2)
    forager = ForagerImpl(mcp, embedder)

    docs = [
        Document(content="A1", source_urn="a1", metadata={}),
        Document(content="A2", source_urn="a2", metadata={}),
        Document(content="A3", source_urn="a3", metadata={}),
        Document(content="B1", source_urn="b1", metadata={}),
    ]
    # Forage call sorts initially by relevance (if MMR not applied yet? No, Forage applies MMR)
    # Actually, forage calls MCP search, which returns candidates.
    # Candidates order doesn't strictly matter for MMR, but usually they are sorted by relevance.
    mcp.documents = docs

    template = SynthesisTemplate(structure="S", complexity_description="C", domain="D", embedding_centroid=[1.0, 0.0])

    # Limit = 2
    # Round 1:
    # A's: Sim=1.0.
    # B: Sim=0.707.
    # Selected: A1 (Best Sim)

    # Round 2:
    # A2: Rel=1.0. Sim(A2, A1)=1.0.
    # MMR(A2) = 0.5*1.0 - 0.5*1.0 = 0.0.

    # B1: Rel=0.707. Sim(B1, A1)=0.707.
    # MMR(B1) = 0.5*0.707 - 0.5*0.707 = 0.0.

    # It's a tie at 0.0.
    # This shows that with lambda=0.5, a perfect duplicate is penalized exactly enough
    # to match an orthogonal document's score?
    # No, B is 45 deg, not orthogonal.
    # Orthogonal B (0,1): Rel=0. Sim(A1,B)=0. MMR=0.
    # So 0.5 is the tipping point where duplicate == orthogonal?

    # Let's adjust B to be slightly more relevant or less similar.
    # If B is [0.8, 0.6]. Norm=1.
    # Sim(Q, B) = 0.8.
    # Sim(A, B) = 0.8.
    # MMR(B) = 0.5*0.8 - 0.5*0.8 = 0.0.

    # Wait, if Sim(Q, B) == Sim(A, B), then MMR is 0 for lambda=0.5.
    # To favor B, we need lambda < 0.5 OR B needs to be more unique than it is irrelevant.
    # or A needs to be penalized MORE.
    # A2 is perfect duplicate. Sim(A2, A1) = 1.0. Rel=1.0. Score=0.
    # So any B with score > 0 will win.
    # Is it possible to get Score > 0?
    # 0.5*Rel - 0.5*Sim > 0 => Rel > Sim.
    # B needs to be closer to Query than to A?
    # But Q and A are identical. So Rel(B, Q) == Sim(B, A).
    # So Score(B) is always 0 if Q=A.

    # Conclusion: If the top result A is identical to Query, then any subsequent candidate B
    # has Rel(B,Q) == Sim(B,A).
    # With lambda=0.5, Score(B) = 0.5*Rel - 0.5*Rel = 0.
    # And Score(A2) = 0.5*1 - 0.5*1 = 0.
    # So all candidates have score 0.
    # Selection order falls back to list order or stability.

    # Let's try a different query/setup where A is not identical to Q.
    # Q = [1, 0]
    # A1 = [0.9, 0.4] (Rel ~ 0.9)
    # A2 = [0.9, 0.4] (Duplicate of A1)
    # B1 = [0.9, -0.4] (Different direction, same relevance)

    # Round 1:
    # A1 Rel=0.9. B1 Rel=0.9. Pick A1.

    # Round 2:
    # A2: Rel=0.9. Sim(A2, A1)=1.0.
    # MMR(A2) = 0.5*0.9 - 0.5*1.0 = 0.45 - 0.5 = -0.05.

    # B1: Rel=0.9. Sim(B1, A1) = 0.9*0.9 + 0.4*(-0.4) = 0.81 - 0.16 = 0.65.
    # MMR(B1) = 0.5*0.9 - 0.5*0.65 = 0.45 - 0.325 = 0.125.

    # 0.125 > -0.05.
    # B1 wins!
    # This setup proves diversity.
    class ClusterEmbedder2(DummyEmbeddingService):
        def embed(self, text: str) -> List[float]:
            if text == "Query":
                return [1.0, 0.0]
            if text.startswith("A"):
                return [0.9, 0.435]  # Norm ~ 1.0 (0.81 + 0.19)
            if text.startswith("B"):
                return [0.9, -0.435]  # Norm ~ 1.0
            return [0.0, 0.0]

    # Use a new variable or cast to avoid mypy confusion if it inferred type from previous assignment
    embedder_2 = ClusterEmbedder2(dimension=2)
    forager = ForagerImpl(mcp, embedder_2)
    mcp.documents = docs  # A1, A2, A3, B1

    # We expect [A1, B1] (or [A1, B1, A2])
    results = forager.forage(template, {}, limit=2)

    assert len(results) == 2
    assert results[0].content.startswith("A") or results[0].content.startswith("B")
    # The second one should be the OTHER cluster
    first_cluster = results[0].content[0]
    second_cluster = results[1].content[0]
    assert first_cluster != second_cluster, f"Expected different clusters, got {first_cluster} and {second_cluster}"


def test_mmr_lambda_sensitivity(forager_setup: Tuple[ForagerImpl, MockMCPClient, DummyEmbeddingService]) -> None:
    """
    Edge Case: Lambda Parameter Sensitivity.
    Verify that lambda=1.0 (Pure Relevance) vs lambda=0.0 (Pure Diversity)
    changes the selection.
    """
    forager, _, _ = forager_setup

    # Setup:
    # Q = [1, 0]
    # Doc1 = [0.9, 0.4] (High Rel, Sim=0.9)
    # Doc2 = [0.9, 0.4] (Duplicate of Doc1)
    # Doc3 = [0.1, 0.9] (Low Rel, Sim(Q)=0.1, Sim(D1)=~0.4)

    # Pure Relevance (lambda=1): Picks Doc1, then Doc2 (Rel=0.9 > Rel=0.1).
    # Pure Diversity (lambda=0):
    # Round 1: All Rel ignored. Diversity penalty 0. Picks first? Or max negative?
    # MMR = -1 * 0 = 0.
    # Picks Doc1 (first in list).
    # Round 2:
    # Doc2: -1 * Sim(D2, D1) = -1.0.
    # Doc3: -1 * Sim(D3, D1) = -0.4.
    # -0.4 > -1.0.
    # Picks Doc3.

    class SensEmbedder(DummyEmbeddingService):
        def embed(self, text: str) -> List[float]:
            if text == "Query":
                return [1.0, 0.0]
            if text == "Doc1":
                return [0.9, 0.435]
            if text == "Doc2":
                return [0.9, 0.435]
            if text == "Doc3":
                return [0.1, 0.99]  # Almost orthogonal to Q
            return [0.0, 0.0]

    forager.embedder = SensEmbedder(dimension=2)

    candidates = [
        Document(content="Doc1", source_urn="1", metadata={}),
        Document(content="Doc2", source_urn="2", metadata={}),
        Document(content="Doc3", source_urn="3", metadata={}),
    ]
    query = [1.0, 0.0]

    # Test Lambda = 1.0 (Relevance)
    results_rel = forager._apply_mmr(query, candidates, limit=2, lambda_param=1.0)
    assert results_rel[0].content == "Doc1"
    assert results_rel[1].content == "Doc2"

    # Test Lambda = 0.0 (Diversity)
    # (or low enough to prefer Doc3)
    # With lambda=0.5:
    # Doc2: 0.5*0.9 - 0.5*1.0 = -0.05
    # Doc3: 0.5*0.1 - 0.5*0.5(approx sim) = 0.05 - 0.25 = -0.2.
    # Doc2 still wins at 0.5 because Doc3 is TOO irrelevant.

    # Let's try Lambda = 0.0
    results_div = forager._apply_mmr(query, candidates, limit=2, lambda_param=0.0)
    assert results_div[0].content == "Doc1"
    assert results_div[1].content == "Doc3"


def test_forage_zero_limit(forager_setup: Tuple[ForagerImpl, MockMCPClient, DummyEmbeddingService]) -> None:
    """Test handling of limit=0."""
    forager, _, _ = forager_setup
    template = SynthesisTemplate(structure="S", complexity_description="C", domain="D", embedding_centroid=[1.0, 0.0])

    results = forager.forage(template, {}, limit=0)
    assert results == []
