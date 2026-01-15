# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_synthesis

from typing import Any, Dict, List

import numpy as np

from .interfaces import Forager
from .models import Document, SynthesisTemplate
from .services import EmbeddingService, MCPClient


class ForagerImpl(Forager):
    """
    Concrete implementation of the Forager.
    Retrieves documents from MCP and enforces diversity using MMR.
    """

    def __init__(self, mcp_client: MCPClient, embedder: EmbeddingService):
        self.mcp_client = mcp_client
        self.embedder = embedder

    def forage(self, template: SynthesisTemplate, user_context: Dict[str, Any], limit: int = 10) -> List[Document]:
        """
        Retrieves documents based on the synthesis template's centroid.
        Applies Maximal Marginal Relevance (MMR) to ensure diversity.
        """
        if not template.embedding_centroid:
            # Fallback if no centroid is present (should not happen in normal flow)
            # We cannot search without a centroid in this architecture
            return []

        # 1. Fetch Candidates from MCP
        # We fetch more than 'limit' to allow for filtering/re-ranking
        # Fetching 5x the limit is a common heuristic
        fetch_limit = limit * 5
        candidates = self.mcp_client.search(template.embedding_centroid, user_context, fetch_limit)

        if not candidates:
            return []

        # 2. Apply MMR for Diversity
        selected_docs = self._apply_mmr(template.embedding_centroid, candidates, limit)

        return selected_docs

    def _apply_mmr(
        self, query_vector: List[float], candidates: List[Document], limit: int, lambda_param: float = 0.5
    ) -> List[Document]:
        """
        Applies Maximal Marginal Relevance (MMR) ranking.

        MMR = ArgMax [ lambda * Sim(Di, Q) - (1-lambda) * max(Sim(Di, Dj)) ]
        where Q is query, Di is candidate, Dj is already selected.

        Args:
            query_vector: The centroid vector.
            candidates: List of candidate documents.
            limit: Number of documents to select.
            lambda_param: Trade-off between relevance (1.0) and diversity (0.0).

        Returns:
            List of selected Documents.
        """
        if not candidates:
            return []

        # Convert query to numpy array
        query_np = np.array(query_vector)
        query_norm = np.linalg.norm(query_np)

        # Pre-calculate embeddings for all candidates
        # Note: In a production system, MCP might return embeddings to avoid re-embedding.
        # Here we use the embedder service.
        candidate_embeddings = []
        for doc in candidates:
            emb = np.array(self.embedder.embed(doc.content))
            candidate_embeddings.append(emb)

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
