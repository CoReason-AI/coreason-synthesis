from typing import List

import numpy as np

from .interfaces import PatternAnalyzer, TeacherModel
from .models import SeedCase, SynthesisTemplate
from .services import EmbeddingService


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
            "Provide the output as structured text."
        )

        analysis_text = self.teacher.generate(prompt)

        # Simple parsing logic for the mock/MVP (in production, we might ask for JSON)
        # For now, we'll map the raw text or use defaults if parsing fails
        # Assuming the MockTeacher returns a specific format we control.

        # Default fallback values
        structure = "Unknown Structure"
        complexity = "Unknown Complexity"
        domain = "Unknown Domain"

        # Naive parsing based on MockTeacher's expected output
        lines = analysis_text.split("\n")
        for line in lines:
            if line.startswith("Structure:"):
                structure = line.replace("Structure:", "").strip()
            elif line.startswith("Complexity:"):
                complexity = line.replace("Complexity:", "").strip()
            elif line.startswith("Domain:"):
                domain = line.replace("Domain:", "").strip()

        return SynthesisTemplate(
            structure=structure, complexity_description=complexity, domain=domain, embedding_centroid=centroid
        )
