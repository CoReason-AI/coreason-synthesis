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
Pipeline orchestration module.

This module contains the `SynthesisPipeline` class, which connects all
components (Analyzer, Forager, Extractor, Compositor, Perturbator, Appraiser)
to execute the full synthetic data generation workflow.
"""

import random
from typing import Any, Dict, List, Optional, Self, cast

import anyio
import httpx

from .interfaces import (
    Appraiser,
    Compositor,
    Extractor,
    Forager,
    PatternAnalyzer,
    Perturbator,
)
from .models import SeedCase, SyntheticTestCase


class SynthesisPipelineAsync:
    """Orchestrates the Pattern-Forage-Fabricate-Rank Loop for synthetic data generation.

    This class serves as the main entry point for running synthesis jobs.
    It manages the flow of data between the various specialized components.
    """

    def __init__(
        self,
        analyzer: PatternAnalyzer,
        forager: Forager,
        extractor: Extractor,
        compositor: Compositor,
        perturbator: Perturbator,
        appraiser: Appraiser,
        client: Optional[httpx.AsyncClient] = None,
    ):
        """Initializes the synthesis pipeline with required components.

        Args:
            analyzer: Component to analyze seed patterns.
            forager: Component to retrieve documents.
            extractor: Component to mine text slices.
            compositor: Component to generate test cases.
            perturbator: Component to apply adversarial mutations.
            appraiser: Component to score and rank cases.
            client: Optional external httpx client for resource sharing.
        """
        self.analyzer = analyzer
        self.forager = forager
        self.extractor = extractor
        self.compositor = compositor
        self.perturbator = perturbator
        self.appraiser = appraiser

        # In a real-world scenario, we might want to pass this client to the components
        # if they were initialized inside here or if we had a way to inject it later.
        # Given the current structure where components are passed in, we assume they are already configured.
        # However, to strictly follow the pattern "ServiceAsync handles resources", we keep track of it.
        self._internal_client = client is None
        self._client = client or httpx.AsyncClient()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._internal_client:
            await self._client.aclose()
        # Potentially close other resources if components exposed a close method

    async def run(
        self, seeds: List[SeedCase], config: Dict[str, Any], user_context: Dict[str, Any]
    ) -> List[SyntheticTestCase]:
        """Executes the full synthesis pipeline.

        1. Analyzes seeds to create a template.
        2. Forages for relevant documents.
        3. Extracts and sanitizes text slices.
        4. Composites base test cases (Verbatim).
        5. Perturbs cases (Adversarial) based on configuration.
        6. Appraises and ranks the final output.

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
        template = await self.analyzer.analyze(seeds)

        # 2. Forage for Documents
        # Default limit to 10 if not specified in config
        limit = config.get("target_count", 10)
        # We might want to forage a bit more than target count to have room for extraction filtering
        forage_limit = max(limit, 10)
        documents = await self.forager.forage(template, user_context, limit=forage_limit)

        if not documents:
            return []

        # 3. Extract Slices
        slices = await self.extractor.extract(documents, template)

        if not slices:
            return []

        # 4. Composite & Perturb (Fabricate)
        generated_cases: List[SyntheticTestCase] = []
        perturbation_rate = config.get("perturbation_rate", 0.0)

        for context_slice in slices:
            # Generate the base case (Verbatim)
            base_case = await self.compositor.composite(context_slice, template)
            generated_cases.append(base_case)

            # Apply perturbation if lucky
            if perturbation_rate > 0 and random.random() < perturbation_rate:
                variants = await self.perturbator.perturb(base_case)
                generated_cases.extend(variants)

        if not generated_cases:
            return []  # pragma: no cover

        # 5. Appraise and Rank
        sort_by = config.get("sort_by", "complexity_desc")
        min_validity = config.get("min_validity_score", 0.8)

        final_cases = await self.appraiser.appraise(
            generated_cases, template, sort_by=sort_by, min_validity_score=min_validity
        )

        return final_cases


class SynthesisPipeline:
    """Synchronous Facade for SynthesisPipelineAsync.

    Wraps the core async logic to provide a blocking interface.
    """

    def __init__(
        self,
        analyzer: PatternAnalyzer,
        forager: Forager,
        extractor: Extractor,
        compositor: Compositor,
        perturbator: Perturbator,
        appraiser: Appraiser,
        client: Optional[httpx.AsyncClient] = None,
    ):
        self._async = SynthesisPipelineAsync(
            analyzer=analyzer,
            forager=forager,
            extractor=extractor,
            compositor=compositor,
            perturbator=perturbator,
            appraiser=appraiser,
            client=client,
        )

    def __enter__(self) -> Self:
        # We don't necessarily need to start a loop here if we use anyio.run in methods
        # But if we want to support resource management (like open/close client),
        # we should probably manage that scope.
        # However, `anyio.run` starts a new loop.
        # The prompt pattern suggests:
        # def __enter__(self): return self
        # def __exit__(self, ...): anyio.run(self._async.__aexit__, ...)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        anyio.run(self._async.__aexit__, exc_type, exc_val, exc_tb)

    def run(
        self, seeds: List[SeedCase], config: Dict[str, Any], user_context: Dict[str, Any]
    ) -> List[SyntheticTestCase]:
        """Executes the full synthesis pipeline synchronously.

        Delegates to SynthesisPipelineAsync.run via anyio.run.
        """
        # Mypy inference on anyio.run might be incomplete, cast result if needed
        # but SyntheticPipelineAsync.run returns List[SyntheticTestCase]
        return cast(List[SyntheticTestCase], anyio.run(self._async.run, seeds, config, user_context))
