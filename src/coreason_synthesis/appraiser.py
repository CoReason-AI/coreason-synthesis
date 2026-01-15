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

import numpy as np
from pydantic import BaseModel, Field

from .interfaces import Appraiser, TeacherModel
from .models import SynthesisTemplate, SyntheticTestCase
from .services import EmbeddingService


class AppraisalAnalysis(BaseModel):
    """Internal model for the Teacher's appraisal output."""

    complexity_score: float = Field(..., ge=0, le=10, description="Estimated logical steps required (0-10)")
    ambiguity_score: float = Field(..., ge=0, le=10, description="How implicit is the answer? (0-10)")
    validity_confidence: float = Field(..., ge=0, le=1, description="Self-consistency score (0-1)")


class AppraiserImpl(Appraiser):
    """
    Concrete implementation of the Appraiser.
    Scores cases using Teacher Model (Complexity, Ambiguity, Validity)
    and Embedding Service (Diversity).
    """

    def __init__(self, teacher: TeacherModel, embedder: EmbeddingService):
        self.teacher = teacher
        self.embedder = embedder

    def appraise(
        self,
        cases: List[SyntheticTestCase],
        template: SynthesisTemplate,
        sort_by: str = "complexity_desc",
        min_validity_score: float = 0.8,
    ) -> List[SyntheticTestCase]:
        """
        Scores and filters test cases.
        """
        appraised_cases: List[SyntheticTestCase] = []

        # Pre-calculate centroid norm if available for diversity
        centroid_np = None
        centroid_norm = 0.0
        if template.embedding_centroid:
            centroid_np = np.array(template.embedding_centroid)
            centroid_norm = np.linalg.norm(centroid_np)

        for case in cases:
            # 1. Calculate Diversity (Distance from Seed Centroid)
            diversity_score = 0.0
            if centroid_np is not None:
                # Embed the verbatim context (The "Real Data")
                case_vec = self.embedder.embed(case.verbatim_context)
                case_np = np.array(case_vec)

                # Check dimensions to prevent crash
                if case_np.shape != centroid_np.shape:
                     # Log warning or handle? For now, we'll just skip diversity calc (default 0)
                     # ideally we should raise or log, but to match interface we proceed safely
                     pass
                else:
                    case_norm = np.linalg.norm(case_np)

                    if centroid_norm > 0 and case_norm > 0:
                        cosine_sim = np.dot(case_np, centroid_np) / (case_norm * centroid_norm)
                        # Diversity = 1 - Cosine Similarity (Distance)
                        # Clip to [0, 1] range to match requirement
                        diversity_score = float(max(0.0, min(1.0, 1.0 - cosine_sim)))

            # 2. Calculate Complexity, Ambiguity, Validity via Teacher
            prompt = self._construct_prompt(case, template)
            analysis: AppraisalAnalysis = self.teacher.generate_structured(prompt, AppraisalAnalysis)

            # 3. Update Case Metrics
            # Use model_copy to create a new instance with updated metrics
            updated_case = case.model_copy(update={
                "complexity": analysis.complexity_score,
                # Note: Ambiguity score is calculated but not stored in SyntheticTestCase
                # as per current models.py schema. We use it for internal logic if needed
                # or if the model changes later.
                "diversity": diversity_score,
                "validity_confidence": analysis.validity_confidence
            })

            # 4. Filter by Validity
            if updated_case.validity_confidence >= min_validity_score:
                appraised_cases.append(updated_case)

        # 5. Sort Cases
        return self._sort_cases(appraised_cases, sort_by)

    def _construct_prompt(self, case: SyntheticTestCase, template: SynthesisTemplate) -> str:
        """Constructs the prompt for the Teacher Model (Judge)."""
        return (
            f"You are an expert Judge for Synthetic Data Quality in the domain: {template.domain}.\n"
            f"Evaluate the following test case against the intended pattern.\n\n"
            f"Intended Structure: {template.structure}\n"
            f"Intended Complexity: {template.complexity_description}\n\n"
            f"Test Case:\n"
            f"Context: {case.verbatim_context}\n"
            f"Question: {case.synthetic_question}\n"
            f"Reasoning: {case.golden_chain_of_thought}\n"
            f"Expected Output: {case.expected_json}\n\n"
            "Please score the case on:\n"
            "1. Complexity (0-10): Logical steps required.\n"
            "2. Ambiguity (0-10): Implicitness of the answer (Higher = Harder).\n"
            "3. Validity Confidence (0-1): Does the Expected Output logically follow from the Context?\n"
        )

    def _sort_cases(self, cases: List[SyntheticTestCase], sort_by: str) -> List[SyntheticTestCase]:
        """Sorts cases based on the requested metric."""
        if sort_by == "complexity_desc":
            return sorted(cases, key=lambda c: c.complexity, reverse=True)
        elif sort_by == "complexity_asc":
            return sorted(cases, key=lambda c: c.complexity)
        elif sort_by == "diversity_desc":
            return sorted(cases, key=lambda c: c.diversity, reverse=True)
        elif sort_by == "diversity_asc":
            return sorted(cases, key=lambda c: c.diversity)
        elif sort_by == "validity_desc":
            return sorted(cases, key=lambda c: c.validity_confidence, reverse=True)
        elif sort_by == "validity_asc":
            return sorted(cases, key=lambda c: c.validity_confidence)

        # Default fallback (preserve order or by complexity desc)
        return sorted(cases, key=lambda c: c.complexity, reverse=True)
