# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_synthesis

import argparse
import json
import os
import sys
from typing import List
from uuid import uuid4

from coreason_synthesis.analyzer import PatternAnalyzerImpl
from coreason_synthesis.appraiser import AppraiserImpl
from coreason_synthesis.clients.foundry import FoundryClient
from coreason_synthesis.clients.mcp import HttpMCPClient
from coreason_synthesis.clients.openai import OpenAITeacher
from coreason_synthesis.compositor import CompositorImpl
from coreason_synthesis.extractor import ExtractorImpl
from coreason_synthesis.forager import ForagerImpl
from coreason_synthesis.interfaces import EmbeddingService, MCPClient, TeacherModel
from coreason_synthesis.mocks.embedding import DummyEmbeddingService
from coreason_synthesis.mocks.teacher import MockTeacher
from coreason_synthesis.models import SeedCase, SyntheticTestCase
from coreason_synthesis.perturbator import PerturbatorImpl
from coreason_synthesis.pipeline import SynthesisPipeline
from coreason_synthesis.utils.logger import logger


def load_seeds(filepath: str) -> List[SeedCase]:
    """Load seeds from a JSON file."""
    try:
        with open(filepath, "r") as f:
            data = json.load(f)

        seeds = []
        for item in data:
            # Ensure ID is present or generate one
            if "id" not in item:
                item["id"] = str(uuid4())
            seeds.append(SeedCase(**item))
        return seeds
    except Exception as e:
        logger.error(f"Failed to load seeds from {filepath}: {e}")
        sys.exit(1)


def save_output(cases: List[SyntheticTestCase], filepath: str) -> None:
    """Save generated cases to a JSON file."""
    try:
        output_data = [case.model_dump(mode="json") for case in cases]
        with open(filepath, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Saved {len(cases)} cases to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save output to {filepath}: {e}")


def get_embedding_service() -> EmbeddingService:
    """Factory for embedding service. Currently returns a dummy/mock service."""
    # In a real implementation, this would connect to an OpenAI or similar embedding provider.
    # For now, we use the DummyEmbeddingService as per available components.
    # If a concrete OpenAIEmbeddingService existed, we'd use it if configured.
    return DummyEmbeddingService()


def main() -> None:
    parser = argparse.ArgumentParser(description="Coreason Synthesis CLI")
    parser.add_argument("--seeds", required=True, help="Path to JSON file containing seed cases")
    parser.add_argument("--output", default="output.json", help="Path to save generated test cases")
    parser.add_argument("--target-count", type=int, default=10, help="Number of cases to generate")
    parser.add_argument("--perturbation-rate", type=float, default=0.2, help="Probability of perturbation (0-1)")
    parser.add_argument("--min-validity", type=float, default=0.8, help="Minimum validity score to keep")
    parser.add_argument("--mcp-url", default=os.getenv("COREASON_MCP_URL"), help="MCP Server URL")
    parser.add_argument("--foundry-url", default=os.getenv("COREASON_FOUNDRY_URL"), help="Foundry Server URL")
    parser.add_argument("--openai-key", default=os.getenv("OPENAI_API_KEY"), help="OpenAI API Key")
    parser.add_argument("--dry-run", action="store_true", help="Use mocks instead of real services")
    parser.add_argument("--push-to-foundry", action="store_true", help="Push results to Foundry")

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.seeds):
        logger.error(f"Seeds file not found: {args.seeds}")
        sys.exit(1)

    # 1. Initialize Components
    embedder = get_embedding_service()

    # Define variables with interface types to satisfy mypy
    teacher: TeacherModel
    mcp_client: MCPClient

    if args.dry_run:
        logger.info("Running in DRY RUN mode (Mocks)")
        # We need a MockMCPClient, but it's abstract in interfaces and implemented in mocks/mcp.py?
        # Let's check imports.
        # It seems MockMCPClient is not readily imported or defined in common locations?
        # Based on file list: src/coreason_synthesis/mocks/mcp.py exists.
        from coreason_synthesis.mocks.mcp import MockMCPClient

        teacher = MockTeacher()
        mcp_client = MockMCPClient()
    else:
        # Check required envs
        if not args.openai_key:
            logger.error("OpenAI API Key required for live run (set OPENAI_API_KEY or use --dry-run)")
            sys.exit(1)
        if not args.mcp_url:
            logger.error("MCP URL required for live run (set COREASON_MCP_URL or use --dry-run)")
            sys.exit(1)

        teacher = OpenAITeacher(api_key=args.openai_key)
        mcp_client = HttpMCPClient(base_url=args.mcp_url, api_key=os.getenv("COREASON_MCP_API_KEY"))

    # Instantiate Core Components
    analyzer = PatternAnalyzerImpl(teacher, embedder)
    forager = ForagerImpl(mcp_client, embedder)
    extractor = ExtractorImpl()
    compositor = CompositorImpl(teacher)
    perturbator = PerturbatorImpl()
    appraiser = AppraiserImpl(teacher, embedder)

    pipeline = SynthesisPipeline(analyzer, forager, extractor, compositor, perturbator, appraiser)

    # 2. Load Seeds
    seeds = load_seeds(args.seeds)
    logger.info(f"Loaded {len(seeds)} seeds.")

    # 3. Run Pipeline
    config = {
        "target_count": args.target_count,
        "perturbation_rate": args.perturbation_rate,
        "min_validity_score": args.min_validity,
        "sort_by": "complexity_desc",
    }
    user_context = {"user_id": "cli-user", "roles": ["sre"]}  # Default context

    logger.info("Starting synthesis pipeline...")
    results = pipeline.run(seeds, config, user_context)
    logger.info(f"Generated {len(results)} valid test cases.")

    # 4. Save Output
    save_output(results, args.output)

    # 5. Push to Foundry (Optional)
    if args.push_to_foundry and not args.dry_run:
        if not args.foundry_url:
            logger.error("Foundry URL required to push (set COREASON_FOUNDRY_URL)")
        else:
            foundry = FoundryClient(base_url=args.foundry_url, api_key=os.getenv("COREASON_FOUNDRY_API_KEY"))
            try:
                count = foundry.push_cases(results)
                logger.info(f"Successfully pushed {count} cases to Foundry.")
            except Exception as e:
                logger.error(f"Failed to push to Foundry: {e}")


if __name__ == "__main__":
    main()
