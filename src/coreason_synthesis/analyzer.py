from typing import List

import numpy as np
from pydantic import BaseModel, Field

from .interfaces import PatternAnalyzer, TeacherModel
from .models import SeedCase, SynthesisTemplate
from .services import EmbeddingService


class TemplateAnalysis(BaseModel):
    """Internal model for the structured output of the TeacherModel analysis."""

    structure: str = Field(..., description="Description of the question/output structure")
    complexity_description: str = Field(..., description="Description of the complexity")
    domain: str = Field(..., description="The identified domain of the seeds")


class PatternAnalyzerImpl(PatternAnalyzer):
    """
    Concrete implementation of the PatternAnalyzer.
    Uses an EmbeddingService to calculate centroids and a TeacherModel to extract templates.
    """

    def __init__(self, teacher: TeacherModel, embedder: EmbeddingService):
        self.teacher = teacher
        self.embedder = embedder

    def analyze(self, seeds: List[SeedCase]) -> SynthesisTemplate:
        """
        Analyzes seed cases to extract a synthesis template and vector centroid.
        """
        if not seeds:
            raise ValueError("Seed list cannot be empty.")

        # 1. Calculate Centroid
        embeddings = []
        for seed in seeds:
            # Embed the context of the seed (most relevant for retrieval)
            vector = self.embedder.embed(seed.context)
            embeddings.append(vector)

        # Calculate mean vector (centroid)
        centroid = np.mean(embeddings, axis=0).tolist()

        # 2. Extract Template via Teacher
        # Construct a prompt for the teacher
        seed_summaries = "\n".join(
            [f"Seed {i + 1}: {seed.question} -> {seed.expected_output}" for i, seed in enumerate(seeds)]
        )

        prompt = (
            f"Analyze the following {len(seeds)} seed examples and extract the testing pattern.\n"
            f"Examples:\n{seed_summaries}\n\n"
            "Identify:\n"
            "1. The Structure (e.g., Question + JSON)\n"
            "2. The Complexity (e.g., Simple lookup vs Reasoning)\n"
            "3. The Domain (e.g., Clinical Trials, Finance)\n"
        )

        # Use generate_structured to get a typed response
        analysis: TemplateAnalysis = self.teacher.generate_structured(prompt, TemplateAnalysis)

        return SynthesisTemplate(
            structure=analysis.structure,
            complexity_description=analysis.complexity_description,
            domain=analysis.domain,
            embedding_centroid=centroid,
        )
