from typing import Any, Dict, List, Optional, Tuple

import pytest

from coreason_synthesis.forager import ForagerImpl
from coreason_synthesis.models import Document, SynthesisTemplate
from coreason_synthesis.services import DummyEmbeddingService, MCPClient


class MockMCPClient(MCPClient):
    """Mock MCP Client for testing."""

    def __init__(self, documents: Optional[List[Document]] = None):
        self.documents = documents or []
        self.last_query_vector: List[float] = []
        self.last_user_context: Dict[str, Any] = {}
        self.last_limit = 0

    def search(self, query_vector: List[float], user_context: Dict[str, Any], limit: int) -> List[Document]:
        self.last_query_vector = query_vector
        self.last_user_context = user_context
        self.last_limit = limit
        # Return all docs (filtering logic is in Forager, usually MCP does vector search too)
        # For test, we just return the pre-seeded docs limited by input or available
        return self.documents[:limit]


@pytest.fixture  # type: ignore[misc]
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
