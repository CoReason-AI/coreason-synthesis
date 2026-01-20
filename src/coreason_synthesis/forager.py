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
Foraging module.

This module implements the retrieval logic for finding relevant documents
from the MCP (Model Context Protocol) knowledge base, ensuring diversity
via Maximal Marginal Relevance (MMR).
"""

from typing import Any, Dict, List, cast

import anyio
import numpy as np

from .interfaces import EmbeddingService, Forager, MCPClient
from .models import Document, SynthesisTemplate


class ForagerImpl(Forager):
    """Concrete implementation of the Forager.

    Retrieves documents from MCP and enforces diversity using MMR to prevent
    duplicates and ensure a broad coverage of the domain.
    """

    def __init__(self, mcp_client: MCPClient, embedder: EmbeddingService):
        """Initializes the Forager.

        Args:
            mcp_client: Client for the Model Context Protocol.
            embedder: Service for calculating embeddings (used in MMR).
        """
        self.mcp_client = mcp_client
        self.embedder = embedder

    async def forage(
        self, template: SynthesisTemplate, user_context: Dict[str, Any], limit: int = 10
    ) -> List[Document]:
        """Retrieves documents based on the synthesis template's centroid.

        Applies Maximal Marginal Relevance (MMR) to ensure diversity.

        Args:
            template: The synthesis template containing the vector centroid.
            user_context: Context for RBAC (e.g., auth token, user ID).
            limit: Maximum number of documents to retrieve. Defaults to 10.

        Returns:
            List of retrieved and diversified Documents.
        """
        if not template.embedding_centroid:
            # Fallback if no centroid is present (should not happen in normal flow)
            # We cannot search without a centroid in this architecture
            return []

        # 1. Fetch Candidates from MCP
        # We fetch more than 'limit' to allow for filtering/re-ranking
        # Fetching 5x the limit is a common heuristic
        fetch_limit = limit * 5
        candidates = await self.mcp_client.search(template.embedding_centroid, user_context, fetch_limit)

        if not candidates:
            return []

        # 2. Apply MMR for Diversity
        # MMR calculation is CPU intensive, so we offload it to a thread
        selected_docs = await self._apply_mmr(template.embedding_centroid, candidates, limit)

        return selected_docs

    async def _apply_mmr(
        self, query_vector: List[float], candidates: List[Document], limit: int, lambda_param: float = 0.5
    ) -> List[Document]:
        """Applies Maximal Marginal Relevance (MMR) ranking.

        MMR = ArgMax [ lambda * Sim(Di, Q) - (1-lambda) * max(Sim(Di, Dj)) ]
        where Q is query, Di is candidate, Dj is already selected.

        Args:
            query_vector: The centroid vector.
            candidates: List of candidate documents.
            limit: Number of documents to select.
            lambda_param: Trade-off between relevance (1.0) and diversity (0.0).
                Defaults to 0.5.

        Returns:
            List of selected Documents.
        """
        if not candidates:
            return []

        # Pre-calculate embeddings for all candidates
        # This might involve I/O if the embedder calls an external service
        candidate_embeddings = []
        for doc in candidates:
            emb = np.array(await self.embedder.embed(doc.content))
            candidate_embeddings.append(emb)

        # The actual MMR calculation is purely CPU bound.
        # We can run it in a thread if the number of candidates is large.
        # For now, let's wrap the numpy heavy part.

        return cast(
            List[Document],
            await anyio.to_thread.run_sync(
                self._calculate_mmr_sync,
                query_vector,
                candidates,
                candidate_embeddings,
                limit,
                lambda_param,
            ),
        )

    def _calculate_mmr_sync(
        self,
        query_vector: List[float],
        candidates: List[Document],
        candidate_embeddings: List[np.ndarray],
        limit: int,
        lambda_param: float,
    ) -> List[Document]:
        """Synchronous part of MMR calculation."""
        # Convert query to numpy array
        query_np = np.array(query_vector)
        query_norm = np.linalg.norm(query_np)

        # Calculate Similarity(Candidate, Query)
        # Cosine similarity: (A . B) / (|A| * |B|)
        sim_query = []
        for emb in candidate_embeddings:
            norm = np.linalg.norm(emb)
            if norm == 0 or query_norm == 0:
                sim = 0.0
            else:
                sim = np.dot(emb, query_np) / (norm * query_norm)
            sim_query.append(sim)

        selected_indices: List[int] = []
        candidate_indices = set(range(len(candidates)))

        # Iteratively select the best candidate
        for _ in range(min(limit, len(candidates))):
            best_mmr = -float("inf")
            best_idx = -1

            for idx in candidate_indices:
                # Sim(Di, Q)
                relevance = sim_query[idx]

                # max(Sim(Di, Dj)) for Dj in selected
                if not selected_indices:
                    diversity_penalty = 0.0
                else:
                    similarities_to_selected = []
                    emb_i = candidate_embeddings[idx]
                    norm_i = np.linalg.norm(emb_i)

                    for sel_idx in selected_indices:
                        emb_j = candidate_embeddings[sel_idx]
                        norm_j = np.linalg.norm(emb_j)

                        if norm_i == 0 or norm_j == 0:
                            sim_ij = 0.0
                        else:
                            sim_ij = np.dot(emb_i, emb_j) / (norm_i * norm_j)
                        similarities_to_selected.append(sim_ij)

                    diversity_penalty = max(similarities_to_selected)

                # MMR Score
                mmr_score = (lambda_param * relevance) - ((1 - lambda_param) * diversity_penalty)

                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = idx

            # best_idx is guaranteed to be found because candidate_indices is never empty
            # in this loop (loop runs min(limit, len) times).
            selected_indices.append(best_idx)
            candidate_indices.remove(best_idx)

        return [candidates[i] for i in selected_indices]
