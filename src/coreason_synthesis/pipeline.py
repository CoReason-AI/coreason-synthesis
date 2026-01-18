# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_synthesis

import random
from typing import Any, Dict, List

from .interfaces import (
    Appraiser,
    Compositor,
    Extractor,
    Forager,
    PatternAnalyzer,
    Perturbator,
)
from .models import SeedCase, SyntheticTestCase


class SynthesisPipeline:
    """
    Orchestrates the Pattern-Forage-Fabricate-Rank Loop for synthetic data generation.
    """

    def __init__(
        self,
        analyzer: PatternAnalyzer,
        forager: Forager,
        extractor: Extractor,
        compositor: Compositor,
        perturbator: Perturbator,
        appraiser: Appraiser,
    ):
        self.analyzer = analyzer
        self.forager = forager
        self.extractor = extractor
        self.compositor = compositor
        self.perturbator = perturbator
        self.appraiser = appraiser

    def run(
        self, seeds: List[SeedCase], config: Dict[str, Any], user_context: Dict[str, Any]
    ) -> List[SyntheticTestCase]:
        """
        Executes the full synthesis pipeline.

        Args:
            seeds: List of user-provided seed examples.
            config: Configuration dictionary (e.g., perturbation_rate, sort_by).
            user_context: Context for RBAC and identity.

        Returns:
            A list of appraised and ranked SyntheticTestCase objects.
        """
        if not seeds:
            return []

        # 1. Analyze Pattern
        template = self.analyzer.analyze(seeds)

        # 2. Forage for Documents
        # Default limit to 10 if not specified in config
        limit = config.get("target_count", 10)
        # We might want to forage a bit more than target count to have room for extraction filtering
        forage_limit = max(limit, 10)
        documents = self.forager.forage(template, user_context, limit=forage_limit)

        if not documents:
            return []

        # 3. Extract Slices
        slices = self.extractor.extract(documents, template)

        if not slices:
            return []

        # 4. Composite & Perturb (Fabricate)
        generated_cases: List[SyntheticTestCase] = []
        perturbation_rate = config.get("perturbation_rate", 0.0)

        for context_slice in slices:
            # Generate the base case (Verbatim)
            base_case = self.compositor.composite(context_slice, template)
            generated_cases.append(base_case)

            # Apply perturbation if lucky
            if perturbation_rate > 0 and random.random() < perturbation_rate:
                # perturb returns a list of variants (usually including original or just variants?
                # Interface says "Returns: A list containing the original and/or perturbed cases."
                # But typically PerturbatorImpl returns JUST the variants based on the code I read.
                # Let's check PerturbatorImpl: it returns only variants.
                # So we append them to the list.
                variants = self.perturbator.perturb(base_case)
                generated_cases.extend(variants)

        if not generated_cases:
            return []  # pragma: no cover

        # 5. Appraise and Rank
        sort_by = config.get("sort_by", "complexity_desc")
        min_validity = config.get("min_validity_score", 0.8)

        final_cases = self.appraiser.appraise(
            generated_cases, template, sort_by=sort_by, min_validity_score=min_validity
        )

        return final_cases
