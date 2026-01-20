# Product Requirements Document: coreason-synthesis

Domain: Grounded Synthetic Data Generation (SDG)
Core Philosophy: "The Scenario is Fake, The Data is Real, The Value is Ranked."
Architectural Role: The Test Data Factory
Dependencies: coreason-mcp (Source), coreason-foundry (UI/Staging), coreason-assay (Consumer)

---

## 1. Executive Summary

coreason-synthesis is the "Amplifier" of the CoReason platform. It solves the "Cold Start Problem" of evaluation by manufacturing high-quality, domain-specific Benchmark Evaluation Corpora (BEC) from a small set of user-provided examples.

It rejects the "pure hallucination" approach of standard GenAI. Instead, it implements a **Grounded Synthesis Pipeline**:

1. **Learns** the testing pattern from user-provided Seeds.
2. **Forages** for real, semantically similar documents via MCP.
3. **Extracts** verbatim text slices (The "Real Data").
4. **Composites** synthetic questions around that data (The "Fake Scenario").
5. **Appraises** and ranks the results by complexity and diversity.

The output is a rigorous, stratified test suite that validates the agent against *actual* enterprise data variances, not idealized synthetic text.

## 2. Functional Philosophy

The agent must implement the **Pattern-Forage-Fabricate-Rank Loop**:

1. **Few-Shot Intent Inference:** The system does not just copy the seed; it infers the *testing intent* (e.g., "The user is testing the agent's ability to cross-reference footnotes").
2. **Verbatim Defense:** The "Context" provided to the agent must be a pixel-perfect copy of real MCP data (preserving OCR errors and bad formatting). We do not rewrite the truth; we only wrap it in a question.
3. **Lineage Transparency:** Every generated case must carry a "Provenance Badge." Users must instantly distinguish between "Real/Verbatim" data and "Adversarial/Perturbed" data.
4. **Quality over Quantity:** We generate $N$ cases but only show the top $K$. The system uses an **Appraiser** to score and sort cases, hiding low-quality or repetitive generations.

---

## 3. Core Functional Requirements (Component Level)

### 3.1 The Pattern Analyzer (The Brain)

**Concept:** A pre-processing engine that deconstructs the User's Seeds.

* **Input:** List[SeedCase] (1 to $N$ examples provided by SRE).
* **Vector Centroid:** Calculates the average embedding vector of the seeds to define the "Search Neighborhood."
* **Template Extraction:** Uses a Teacher Model to derive the **Synthesis Template**:
    * *Structure:* "Question + JSON Output."
    * *Complexity:* "Requires multi-hop reasoning."
    * *Domain:* "Oncology / Inclusion Criteria."

### 3.2 The Forager (The Crawler)

**Concept:** The retrieval engine that finds raw material.

* **Mechanism:** Queries coreason-mcp using the Vector Centroid.
* **Diversity Enforcement:** Implements **Maximal Marginal Relevance (MMR)** or Clustering to ensure the retrieved documents are distinct (e.g., "Don't bring me 50 versions of the same protocol; bring me 50 *different* protocols").
* **Scope:** Must respect the User's RBAC (Role-Based Access Control) via coreason-identity. It cannot forage documents the SRE is not allowed to see.

### 3.3 The Extractor (The Miner)

**Concept:** Targeted mining of text slices.

* **Function:** Identifies and copies text chunks (paragraphs, tables) that match the *structural pattern* of the seeds.
* **PII Sanitization (Critical GxP Feature):** Before any text enters the generation pipeline, it must pass through a De-Identification filter to mask real patient names or MRNs found in the source docs, replacing them with generic placeholders (e.g., [PATIENT_NAME]).
* **Traceability:** Logs the source_urn and page_number.

### 3.4 The Compositor (The Generator)

**Concept:** The engine that wraps real data in synthetic interactions.

* **Teacher Model:** Uses a high-reasoning model (e.g., o1/Opus) to:
    1. **Read** the Verbatim Slice.
    2. **Generate** a User Prompt based on the *Synthesis Template*.
    3. **Generate** the "Golden Chain-of-Thought" (The Answer Key).
* **Output:** A DraftTestCase tagged as provenance="VERBATIM_SOURCE".

### 3.5 The Perturbator (The Red Team)

**Concept:** Creates "Hard Negatives" and "Edge Cases."

* **Configuration:** Configurable "Mutation Rate" (e.g., Apply to 20% of cases).
* **Strategies:**
    * **Value Swap:** Change "50mg" to "5000mg" (Safety Test).
    * **Negation:** Change "Included" to "Excluded" (Logic Test).
    * **Noise Injection:** Insert irrelevant text to test robustness.
* **Lineage Update:** If applied, updates tag to provenance="SYNTHETIC_PERTURBED" and logs the diff.

### 3.6 The Appraiser (The Judge)

**Concept:** A scoring engine that ranks quality *before* human review.

* **Metrics:**
    1. **Complexity Score (0-10):** Estimated logical steps required to solve.
    2. **Ambiguity Score (0-10):** How implicit is the answer? (Higher = Harder).
    3. **Diversity Score (0-1):** Distance from the Seed's centroid.
    4. **Validity Confidence (0-1):** Self-consistency check (Does the Answer actually match the Context?).
* **Action:** Discards cases with Validity < 0.8. Sorts the rest based on user preference.

---

## 4. Integration Requirements (The Ecosystem)

* **coreason-mcp:**
    * **Read-Only:** The synthesis package must have a dedicated, read-only service account to query knowledge bases.
* **coreason-foundry:**
    * **UI Elements:** Needs to render the "Review Queue" with:
        * Badges: [VERBATIM] (Green), [PERTURBED] (Orange).
        * Sort Controls: "Sort by Complexity," "Sort by Diversity."
        * Diff View: For Perturbed cases, show Original vs. Mutated text side-by-side.
* **coreason-assay:**
    * The destination for approved cases.

---

## 5. User Stories (Behavioral Expectations)

### Story A: The "Exam Creator" (Ranked by Complexity)

Trigger: SRE wants to test the agent's ability to handle complex Math.
Seeds: SRE uploads 1 example of a Body Surface Area (BSA) calculation.
Generation: System finds 100 other dosing tables via MCP.
Appraisal: The Appraiser scores them. Simple lookups get a Complexity of 2.0. Multi-variable calculations get 9.0.
Review: SRE sorts by "Complexity (Desc)." They approve the top 20 hardest math problems.

### Story B: The "Safety Stress Test" (Perturbed Data)

Trigger: SRE wants to ensure the agent catches "Protocol Deviations."
Action: SRE enables "Perturbation Mode: High."
Execution: The Perturbator takes real inclusion criteria and flips the ages (e.g., changing "18-65" to "12-65").
Review: SRE filters the list by "Perturbed Only." They verify that the "Golden Answer" for these cases correctly expects a "REFUSAL" response from the agent.

### Story C: The "Reality Check" (Verbatim Only)

Trigger: SRE suspects the agent fails on bad OCR scanning.
Action: SRE filters by "Verbatim Only."
Review: SRE looks for cases with low Validity scores (where the Teacher model struggled). They find a PDF with garbled text. They approve this case to ensure the production agent handles "Unreadable Input" errors gracefully.

---

## 6. Data Schema & Object Model

### SyntheticJob

* id: UUID
* seed_ids: List[UUID]
* config: { perturbation_rate: 0.2, target_count: 50 }

### SyntheticTestCase (The Artifact)

* **Content:**
    * verbatim_context: String (The Real Data)
    * synthetic_question: String (The User Proxy)
    * golden_chain_of_thought: String (The Teacher's Logic)
    * expected_json: JSON
* **Provenance:**
    * type: ENUM("VERBATIM_SOURCE", "SYNTHETIC_PERTURBED")
    * source_urn: mcp://sharepoint/protocol_123.pdf
    * modifications: List[Diff] (e.g., ["Changed 'Male' to 'Female'"])
* **Metrics:**
    * complexity: Float (0-10)
    * diversity: Float (0-1)
    * validity_confidence: Float (0-1)

---

## 7. Implementation Checklist for the Coding Agent

1. **Class Structure:** Create abstract base classes for Forager, Extractor, Compositor, Perturbator, and Appraiser.
2. **MCP Client:** Implement a robust client that handles Rate Limiting when fetching 50+ documents.
3. **PII Filter:** Integrate a regex/NER-based scrubber in the Extractor class.
4. **Teacher Integration:** Ensure the Compositor can target a different model ID (e.g., o1) than the standard runtime.
5. **Foundry Bridge:** Implement the API calls to push SyntheticTestCase objects into the Foundry SQL staging table.
