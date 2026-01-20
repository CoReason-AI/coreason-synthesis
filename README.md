# coreason-synthesis

Grounded Synthetic Data Generation (SDG) for the CoReason-AI platform.

[![License](https://img.shields.io/badge/license-Prosperity%203.0-blue)](https://prosperitylicense.com/versions/3.0.0)
[![Build Status](https://github.com/CoReason-AI/coreason_synthesis/actions/workflows/main.yml/badge.svg)](https://github.com/CoReason-AI/coreason_synthesis/actions)
[![Code Style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Documentation](https://img.shields.io/badge/docs-Product%20Requirements-blue)](docs/product_requirements.md)

## Overview

**coreason-synthesis** is the "Amplifier" of the CoReason platform. It solves the "Cold Start Problem" of evaluation by manufacturing high-quality, domain-specific Benchmark Evaluation Corpora (BEC) from a small set of user-provided examples.

Unlike standard GenAI approaches that rely on hallucination, this library implements a **Grounded Synthesis Pipeline**:

1.  **Learns** the testing pattern from user-provided Seeds.
2.  **Forages** for real, semantically similar documents via MCP.
3.  **Extracts** verbatim text slices (The "Real Data").
4.  **Composites** synthetic questions around that data (The "Fake Scenario").
5.  **Appraises** and ranks the results by complexity and diversity.

The output is a rigorous, stratified test suite that validates the agent against *actual* enterprise data variances, not idealized synthetic text.

## Features

*   **Pattern-Forage-Fabricate-Rank Loop**: A complete pipeline for generating high-quality test data.
*   **Few-Shot Intent Inference**: Infers testing intent from a few examples.
*   **Verbatim Defense**: Uses pixel-perfect copies of real data (preserving errors/formatting) as context.
*   **Lineage Transparency**: Distinguishes between "Verbatim/Real" and "Adversarial/Perturbed" data.
*   **Quality Ranking**: Appraises and ranks cases by complexity, ambiguity, diversity, and validity.
*   **Safety & Privacy**: Includes PII Sanitization filters.

For detailed requirements and specifications, see [docs/product_requirements.md](docs/product_requirements.md).

## Installation

```bash
pip install coreason-synthesis
```

## Usage

Here is a concise example of how to initialize and use the library (using built-in mocks for demonstration):

```python
import uuid
from coreason_synthesis.pipeline import SynthesisPipeline
from coreason_synthesis.analyzer import PatternAnalyzerImpl
from coreason_synthesis.forager import ForagerImpl
from coreason_synthesis.extractor import ExtractorImpl
from coreason_synthesis.compositor import CompositorImpl
from coreason_synthesis.perturbator import PerturbatorImpl
from coreason_synthesis.appraiser import AppraiserImpl
from coreason_synthesis.models import SeedCase, Document

# Import mocks for demonstration (replace with real implementations in prod)
from coreason_synthesis.mocks.teacher import MockTeacher
from coreason_synthesis.mocks.embedding import DummyEmbeddingService
from coreason_synthesis.mocks.mcp import MockMCPClient

# 1. Initialize Dependencies
teacher = MockTeacher()
embedder = DummyEmbeddingService()
mcp_client = MockMCPClient(
    documents=[
        Document(
            content="Standard Dose: 50mg. Included: Adults.",
            source_urn="doc:1"
        )
    ]
)

# 2. Initialize Components
analyzer = PatternAnalyzerImpl(teacher, embedder)
forager = ForagerImpl(mcp_client, embedder)
extractor = ExtractorImpl()
compositor = CompositorImpl(teacher)
perturbator = PerturbatorImpl()
appraiser = AppraiserImpl(teacher, embedder)

# 3. Assemble Pipeline
pipeline = SynthesisPipeline(
    analyzer=analyzer,
    forager=forager,
    extractor=extractor,
    compositor=compositor,
    perturbator=perturbator,
    appraiser=appraiser
)

# 4. Define Seeds
seeds = [
    SeedCase(
        id=uuid.uuid4(),
        question="Calculate BSA for 180cm, 80kg patient",
        expected_output={"bsa": 2.0},
        context="Formula: sqrt((height*weight)/3600)"
    )
]

# 5. Run Synthesis
config = {
    "target_count": 5,
    "perturbation_rate": 0.5,
    "sort_by": "complexity_desc"
}
user_context = {"user_id": "demo-user"}

results = pipeline.run(seeds, config, user_context)

# 6. Use Results
for case in results:
    print(f"Generated Case ({case.provenance}): {case.synthetic_question}")
```
