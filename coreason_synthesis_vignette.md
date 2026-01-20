# The Architecture and Utility of coreason-synthesis

### 1. The Philosophy (The Why)

In the rapidly evolving landscape of GenAI, the "Cold Start Problem" of evaluation is a critical bottleneck. How do you rigorously test an agent before it sees production traffic? Standard approaches often rely on "pure hallucination"—asking an LLM to imagine test cases. This results in idealized, sterile scenarios that fail to capture the messy reality of enterprise data.

`coreason-synthesis` rejects this approach. Its core philosophy is: **"The Scenario is Fake, The Data is Real."**

Instead of asking a model to invent a document, this package acts as a **Test Data Factory**. It takes a small handful of human-curated "Seed Cases" and amplifies them into thousands of grounded, high-quality test assets. It achieves this by *foraging* for real documents in your enterprise knowledge base that match the semantic pattern of your seeds. It then extracts verbatim slices of truth and wraps them in synthetic questions.

The result is a Benchmark Evaluation Corpus (BEC) that validates your agent against actual data variances—OCR errors, complex tables, and domain-specific nuances—rather than the smooth, predictable output of a language model.

### 2. Under the Hood (The Dependencies & Logic)

The package architecture is built on a lightweight but powerful stack designed to support the **"Pattern-Forage-Fabricate-Rank"** loop.

*   **`pydantic`**: The backbone of the system. It enforces rigorous data schemas (`SyntheticTestCase`, `SeedCase`), ensuring that every generated asset is structured, validatable, and machine-readable. This transforms the output from "text blobs" into reliable integration artifacts.
*   **`numpy`**: The mathematical engine behind the "Brain" and the "Crawler." It is used to calculate the **Vector Centroid** of the user's seed examples, defining the "search neighborhood." Crucially, it powers the **Maximal Marginal Relevance (MMR)** algorithm in the Forager, which mathematically enforces diversity in retrieval. This prevents the common failure mode where a synthesizer returns 50 variations of the exact same document.
*   **`requests`**: Handles the communication with the Model Context Protocol (MCP) server, allowing the package to reach out and touch the real enterprise data layer.

**The Logic Flow:**
1.  **Pattern Analyzer:** Digs into the provided seeds to infer the *testing intent* (e.g., "reasoning over tabular data").
2.  **Forager:** Uses the vector centroid to find semantically similar but distinct documents from the real world.
3.  **Extractor & Compositor:** Mines specific text slices and wraps them in a synthetic "User Prompt" and "Golden Answer."
4.  **Appraiser:** The final gatekeeper. It scores every generated case on **Complexity** (logical depth) and **Validity** (self-consistency), ranking them so that only the highest-value cases make it to the test suite.

### 3. In Practice (The How)

The following examples demonstrate the "Happy Path" of amplifying a single seed case into a rigorous test suite.

#### A. Defining the Seed
First, we capture the human intent. A `SeedCase` is a single, perfect example of what we want to test. In this case, we are testing the agent's ability to extract specific inclusion criteria from clinical protocols.

```python
from uuid import uuid4
from coreason_synthesis.models import SeedCase

# The "Golden Example" provided by a Subject Matter Expert
seed = SeedCase(
    id=uuid4(),
    context="Patients must be between 18 and 65 years of age to be included.",
    question="What is the maximum age for inclusion?",
    expected_output={"max_age": 65, "unit": "years"},
    metadata={"domain": "clinical_trials", "complexity": "simple_lookup"}
)
```

#### B. The Synthesis Loop
The `SynthesisPipeline` orchestrates the entire process. It analyzes the seed, finds 50 real documents that "look like" the seed, and manufactures new test cases.

```python
from coreason_synthesis.pipeline import SynthesisPipeline

# Assume components (analyzer, forager, etc.) are injected via dependency injection
pipeline = SynthesisPipeline(
    analyzer=analyzer,
    forager=forager,
    extractor=extractor,
    compositor=compositor,
    perturbator=perturbator,
    appraiser=appraiser
)

# Configuration for the job
config = {
    "target_count": 50,          # We want 50 new test cases
    "perturbation_rate": 0.2,    # 20% of cases should include "hard negatives"
    "sort_by": "complexity_desc" # Prioritize complex reasoning tasks
}

# Execute the factory
generated_cases = pipeline.run(
    seeds=[seed],
    config=config,
    user_context={"user_id": "sre_team_alpha"}
)
```

#### C. Consuming the Output
The output is a ranked list of `SyntheticTestCase` objects. Each case carries a "Provenance Badge" indicating whether it is a verbatim slice of reality or a perturbed "trap" for the agent.

```python
for case in generated_cases[:3]:
    print(f"--- Case ID: {case.source_urn} ---")
    print(f"Type: {case.provenance}")
    print(f"Complexity Score: {case.complexity}/10")
    print(f"Question: {case.synthetic_question}")
    print(f"Golden Answer: {case.golden_chain_of_thought}")

    # If this is a perturbed case (e.g., values were swapped to test safety),
    # we can inspect the modifications.
    if case.provenance == "SYNTHETIC_PERTURBED":
        for mod in case.modifications:
            print(f"   [DIFF] {mod.description}")
```
