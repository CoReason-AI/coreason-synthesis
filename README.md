# coreason-synthesis

[![License: Prosperity 3.0](https://img.shields.io/badge/License-Prosperity%203.0-blue?style=flat&label=License&color=blue)](LICENSE)
![Build Status](https://github.com/CoReason-AI/coreason-synthesis/actions/workflows/build.yml/badge.svg)
[![Code Style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](pyproject.toml)

**The Test Data Factory for Grounded Synthetic Data Generation.**

`coreason-synthesis` acts as the "Amplifier" for the CoReason platform. It addresses the "Cold Start Problem" in evaluation by manufacturing high-quality, domain-specific Benchmark Evaluation Corpora (BEC) from a small set of user-provided examples. By implementing a **Pattern-Forage-Fabricate-Rank Loop**, it ensures that synthetic data is grounded in reality, not hallucinated.

## Features

*   **Pattern Analyzer:** Infers testing intent and structure from few-shot seed examples.
*   **Forager:** Retrieves real, domain-specific documents (via MCP) using vector similarity and diversity enforcement.
*   **Extractor:** Mines verbatim text slices (tables, paragraphs) while sanitizing PII.
*   **Compositor:** Generates "Golden Chain-of-Thought" and synthetic questions around real data using high-reasoning teacher models.
*   **Perturbator:** Creates "Hard Negatives" and edge cases through value swapping, negation, and noise injection.
*   **Appraiser:** Ranks generated cases by complexity, ambiguity, and diversity to ensure only high-value tests are kept.

## Installation

```bash
pip install coreason-synthesis
```

## Usage

Here is a quick example of how to initialize the pipeline and generate synthetic test cases.

```python
import uuid
from coreason_synthesis import SynthesisPipeline, SeedCase

# 1. Define your Seed Case (The "Few-Shot" Example)
seed = SeedCase(
    id=uuid.uuid4(),
    context="The patient was administered 50mg of Drug X...",
    question="What was the dosage of Drug X?",
    expected_output="50mg",
    metadata={"domain": "medical"}
)

# 2. Initialize the Pipeline (assuming default mocked components for demo)
# In production, you would inject real clients for MCP, Embedding, etc.
pipeline = SynthesisPipeline(...)

# 3. Run the Synthesis Job
# This will Pattern -> Forage -> Extract -> Composite -> Perturb -> Appraise
test_cases = pipeline.run(
    seeds=[seed],
    config={"target_count": 5, "perturbation_rate": 0.2},
    user_context={"user_id": "demo_user"}
)

# 4. Use the Results
for case in test_cases:
    print(f"Question: {case.synthetic_question}")
    print(f"Context: {case.verbatim_context[:50]}...")
    print(f"Score: {case.complexity}")
```

## License

This project is licensed under the **Prosperity Public License 3.0**. See the [LICENSE](LICENSE) file for details.
