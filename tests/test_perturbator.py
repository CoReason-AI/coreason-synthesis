# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the License).
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_synthesis

import pytest

from coreason_synthesis.models import Diff, ProvenanceType, SyntheticTestCase
from coreason_synthesis.perturbator import PerturbatorImpl


@pytest.fixture
def perturbator() -> PerturbatorImpl:
    return PerturbatorImpl()


@pytest.fixture
def base_case() -> SyntheticTestCase:
    return SyntheticTestCase(
        verbatim_context="Patient receives 50mg of Aspirin.",
        synthetic_question="What is the dose?",
        golden_chain_of_thought="The text says 50mg.",
        expected_json={"dose": "50mg"},
        provenance=ProvenanceType.VERBATIM_SOURCE,
        source_urn="urn:test",
        modifications=[],
        complexity=1.0,
        ambiguity=0.0,
        diversity=0.0,
        validity_confidence=1.0,
    )


def test_numeric_swap(perturbator: PerturbatorImpl, base_case: SyntheticTestCase) -> None:
    """Test that numeric values are swapped (multiplied by 100)."""
    variants = perturbator.perturb(base_case)

    # Check if we got at least one variant (should get numeric swap)
    assert len(variants) >= 1

    # Find the numeric variant
    # We must check isinstance before accessing .description because modification can be str or Diff
    numeric_variant = next(
        (v for v in variants if any(isinstance(m, Diff) and "Numeric" in m.description for m in v.modifications)),
        None,
    )
    assert numeric_variant is not None
    # We asserted not None, but type checker might need help for field access if Optional
    assert numeric_variant.verbatim_context is not None

    # 50 * 100 = 5000
    assert "5000" in numeric_variant.verbatim_context
    assert "50mg" not in numeric_variant.verbatim_context
    assert numeric_variant.provenance == ProvenanceType.SYNTHETIC_PERTURBED

    mod = numeric_variant.modifications[0]
    assert isinstance(mod, Diff)
    assert mod.original == "50"
    assert mod.new == "5000"


def test_negation_swap(perturbator: PerturbatorImpl) -> None:
    """Test that negation keywords are swapped (Title Case)."""
    case = SyntheticTestCase(
        verbatim_context="Criteria: Patients are Included if healthy.",
        synthetic_question="Is patient included?",
        golden_chain_of_thought="Yes.",
        expected_json={"included": True},
        provenance=ProvenanceType.VERBATIM_SOURCE,
        source_urn="urn:test",
        modifications=[],
        complexity=1.0,
        ambiguity=0.0,
        diversity=0.0,
        validity_confidence=1.0,
    )

    variants = perturbator.perturb(case)
    negation_variant = next(
        (v for v in variants if any(isinstance(m, Diff) and "Negation" in m.description for m in v.modifications)),
        None,
    )

    assert negation_variant is not None
    assert negation_variant.verbatim_context is not None
    # Included (Title Case) -> Excluded (Title Case)
    assert "Excluded" in negation_variant.verbatim_context
    assert "Included" not in negation_variant.verbatim_context

    mod = negation_variant.modifications[0]
    assert isinstance(mod, Diff)
    assert mod.original == "Included"
    assert mod.new == "Excluded"


def test_negation_swap_lowercase(perturbator: PerturbatorImpl) -> None:
    """Test that negation keywords are swapped (lowercase)."""
    case = SyntheticTestCase(
        verbatim_context="Criteria: patients are included if healthy.",
        synthetic_question="Is patient included?",
        golden_chain_of_thought="Yes.",
        expected_json={"included": True},
        provenance=ProvenanceType.VERBATIM_SOURCE,
        source_urn="urn:test",
        modifications=[],
        complexity=1.0,
        ambiguity=0.0,
        diversity=0.0,
        validity_confidence=1.0,
    )

    variants = perturbator.perturb(case)
    negation_variant = next(
        (v for v in variants if any(isinstance(m, Diff) and "Negation" in m.description for m in v.modifications)),
        None,
    )

    assert negation_variant is not None
    assert negation_variant.verbatim_context is not None
    # included (lower) -> excluded (lower)
    assert "excluded" in negation_variant.verbatim_context
    assert "included" not in negation_variant.verbatim_context

    mod = negation_variant.modifications[0]
    assert isinstance(mod, Diff)
    assert mod.original == "included"
    assert mod.new == "excluded"


def test_noise_injection(perturbator: PerturbatorImpl) -> None:
    """Test that noise is injected deterministically."""
    case = SyntheticTestCase(
        verbatim_context="Clean context.",
        synthetic_question="Q",
        golden_chain_of_thought="A",
        expected_json={},
        provenance=ProvenanceType.VERBATIM_SOURCE,
        source_urn="urn:test",
        modifications=[],
        complexity=1.0,
        ambiguity=0.0,
        diversity=0.0,
        validity_confidence=1.0,
    )

    variants = perturbator.perturb(case)

    # Expect noise injection variant
    noise_variant = next(
        (v for v in variants if any(isinstance(m, Diff) and "Noise" in m.description for m in v.modifications)),
        None,
    )
    assert noise_variant is not None
    assert noise_variant.verbatim_context is not None

    # Check if any noise phrase is present
    found_noise = False
    for phrase in perturbator.NOISE_PHRASES:
        if phrase in noise_variant.verbatim_context:
            found_noise = True
            break
    assert found_noise

    # Determinism check: running again on same input should produce identical output
    variants2 = perturbator.perturb(case)
    noise_variant2 = next(
        (v for v in variants2 if any(isinstance(m, Diff) and "Noise" in m.description for m in v.modifications)),
        None,
    )
    assert noise_variant2 is not None
    assert noise_variant2.verbatim_context is not None
    assert noise_variant.verbatim_context == noise_variant2.verbatim_context


def test_no_perturbations_possible_but_noise(perturbator: PerturbatorImpl) -> None:
    """
    Test case with no numbers and no keywords.
    Should still produce a noise injection variant.
    """
    case = SyntheticTestCase(
        verbatim_context="The sky is blue.",
        synthetic_question="Color?",
        golden_chain_of_thought="Blue.",
        expected_json={"color": "blue"},
        provenance=ProvenanceType.VERBATIM_SOURCE,
        source_urn="urn:test",
        modifications=[],
        complexity=1.0,
        ambiguity=0.0,
        diversity=0.0,
        validity_confidence=1.0,
    )

    variants = perturbator.perturb(case)
    # No numeric, no negation -> only noise
    assert len(variants) == 1
    mod = variants[0].modifications[0]
    assert isinstance(mod, Diff)
    assert "Noise" in mod.description


def test_deep_copy_independence(perturbator: PerturbatorImpl, base_case: SyntheticTestCase) -> None:
    """Ensure original case is not modified."""
    original_context = base_case.verbatim_context
    original_provenance = base_case.provenance

    variants = perturbator.perturb(base_case)

    assert base_case.verbatim_context == original_context
    assert base_case.provenance == original_provenance
    assert len(base_case.modifications) == 0

    assert len(variants) > 0
    assert variants[0].verbatim_context != original_context
    assert variants[0].provenance == ProvenanceType.SYNTHETIC_PERTURBED


def test_multiple_strategies(perturbator: PerturbatorImpl) -> None:
    """Test that all strategies can apply to the same input, creating separate variants."""
    case = SyntheticTestCase(
        verbatim_context="Include 50 patients.",
        synthetic_question="Count?",
        golden_chain_of_thought="50.",
        expected_json={"count": 50},
        provenance=ProvenanceType.VERBATIM_SOURCE,
        source_urn="urn:test",
        modifications=[],
        complexity=1.0,
        ambiguity=0.0,
        diversity=0.0,
        validity_confidence=1.0,
    )

    variants = perturbator.perturb(case)

    # Should produce 3 variants: Numeric, Negation, Noise
    assert len(variants) == 3

    descriptions = set()
    for v in variants:
        mod = v.modifications[0]
        if isinstance(mod, Diff):
            descriptions.add(mod.description)

    assert any("Numeric" in d for d in descriptions)
    assert any("Negation" in d for d in descriptions)
    assert any("Noise" in d for d in descriptions)


def test_decimal_scaling(perturbator: PerturbatorImpl) -> None:
    """Test handling of decimal numbers."""
    case = SyntheticTestCase(
        verbatim_context="Dose is 0.5mg.",
        synthetic_question="Dose?",
        golden_chain_of_thought="0.5.",
        expected_json={"dose": 0.5},
        provenance=ProvenanceType.VERBATIM_SOURCE,
        source_urn="urn:test",
        modifications=[],
        complexity=1.0,
        ambiguity=0.0,
        diversity=0.0,
        validity_confidence=1.0,
    )

    variants = perturbator.perturb(case)
    # Numeric variant should exist
    numeric_v = next(
        v for v in variants if isinstance(v.modifications[0], Diff) and "Numeric" in v.modifications[0].description
    )
    # 0.5 * 100 = 50.0 -> "50"
    assert "Dose is 50mg." in numeric_v.verbatim_context


# --- Edge Case & Complex Scenario Tests ---


def test_multiple_numbers(perturbator: PerturbatorImpl) -> None:
    """
    Edge Case: Context has multiple numbers.
    Expectation: Only the first number is perturbed (based on current 'first match' logic).
    """
    case = SyntheticTestCase(
        verbatim_context="First dose 50mg, second dose 100mg.",
        synthetic_question="Doses?",
        golden_chain_of_thought="50 and 100.",
        expected_json={"doses": [50, 100]},
        provenance=ProvenanceType.VERBATIM_SOURCE,
        source_urn="urn:test",
        modifications=[],
        complexity=1.0,
        ambiguity=0.0,
        diversity=0.0,
        validity_confidence=1.0,
    )

    variants = perturbator.perturb(case)
    variant = next(
        v for v in variants if isinstance(v.modifications[0], Diff) and "Numeric" in v.modifications[0].description
    )

    # First number (50) should be 5000
    assert "First dose 5000mg" in variant.verbatim_context
    # Second number (100) should remain 100
    assert "second dose 100mg" in variant.verbatim_context


def test_word_boundary_safety(perturbator: PerturbatorImpl) -> None:
    """
    Edge Case: Partial word matches.
    Expectation: 'conclude' should not trigger 'include' logic. 'inclusive' should not trigger 'include'.
    """
    case = SyntheticTestCase(
        verbatim_context="We conclude that the inclusive policy is good.",
        synthetic_question="What policy?",
        golden_chain_of_thought="Inclusive.",
        expected_json={"policy": "inclusive"},
        provenance=ProvenanceType.VERBATIM_SOURCE,
        source_urn="urn:test",
        modifications=[],
        complexity=1.0,
        ambiguity=0.0,
        diversity=0.0,
        validity_confidence=1.0,
    )

    variants = perturbator.perturb(case)
    # Neither "conclude" nor "inclusive" are in the swap list.
    # Should only get Noise variant
    assert len(variants) == 1
    mod = variants[0].modifications[0]
    assert isinstance(mod, Diff)
    assert "Noise" in mod.description


def test_all_caps_handling(perturbator: PerturbatorImpl) -> None:
    """
    Edge Case: All-caps keywords.
    Expectation: Current simple logic sees "INCLUDE" (isupper=True) -> returns "Exclude" (Title Case).
    This test documents valid behavior, even if imperfect.
    """
    case = SyntheticTestCase(
        verbatim_context="PATIENTS MUST BE INCLUDED.",
        synthetic_question="Status?",
        golden_chain_of_thought="Included.",
        expected_json={"status": "INCLUDED"},
        provenance=ProvenanceType.VERBATIM_SOURCE,
        source_urn="urn:test",
        modifications=[],
        complexity=1.0,
        ambiguity=0.0,
        diversity=0.0,
        validity_confidence=1.0,
    )

    variants = perturbator.perturb(case)
    variant = next(
        v for v in variants if isinstance(v.modifications[0], Diff) and "Negation" in v.modifications[0].description
    )

    # "INCLUDED" -> isupper() is True -> replacement.capitalize() -> "Excluded"
    assert "PATIENTS MUST BE Excluded." in variant.verbatim_context


def test_formatted_number(perturbator: PerturbatorImpl) -> None:
    """
    Edge Case: Number with comma '1,000'.
    Expectation: Regex catches '1', stops at comma. Swaps '1' -> '100'. Result '100,000'.
    """
    case = SyntheticTestCase(
        verbatim_context="Cost is 1,000 dollars.",
        synthetic_question="Cost?",
        golden_chain_of_thought="1000.",
        expected_json={"cost": 1000},
        provenance=ProvenanceType.VERBATIM_SOURCE,
        source_urn="urn:test",
        modifications=[],
        complexity=1.0,
        ambiguity=0.0,
        diversity=0.0,
        validity_confidence=1.0,
    )

    variants = perturbator.perturb(case)
    variant = next(
        v for v in variants if isinstance(v.modifications[0], Diff) and "Numeric" in v.modifications[0].description
    )

    # "1" matches. 1*100 = 100. replaced "1" with "100".
    # "1,000" becomes "100,000".
    assert "Cost is 100,000 dollars." in variant.verbatim_context


def test_chained_perturbation(perturbator: PerturbatorImpl) -> None:
    """
    Complex Scenario: Re-perturbing a perturbed case.
    Expectation: History is preserved, new diff added.
    """
    # 1. Create a perturbed case (simulated output from first pass)
    initial_diff = Diff(description="First Mod", original="A", new="B")
    perturbed_case = SyntheticTestCase(
        verbatim_context="Include 50 patients.",
        synthetic_question="Q",
        golden_chain_of_thought="A",
        expected_json={},
        provenance=ProvenanceType.SYNTHETIC_PERTURBED,
        source_urn="urn:test",
        modifications=[initial_diff],
        complexity=0.0,
        ambiguity=0.0,
        diversity=0.0,
        validity_confidence=0.0,
    )

    # 2. Perturb again
    variants = perturbator.perturb(perturbed_case)

    # Should find 3 variants (Numeric, Negation, Noise)
    assert len(variants) == 3

    # Check numeric variant
    num_var = next(
        v for v in variants if isinstance(v.modifications[-1], Diff) and "Numeric" in v.modifications[-1].description
    )

    # Should have 2 modifications now
    assert len(num_var.modifications) == 2
    assert isinstance(num_var.modifications[0], Diff)
    assert num_var.modifications[0].description == "First Mod"
    assert isinstance(num_var.modifications[1], Diff)
    assert "Numeric" in num_var.modifications[1].description

    # Check that it started from the *current* context of input
    assert "Include 5000 patients." in num_var.verbatim_context


def test_noise_injection_empty_text(perturbator: PerturbatorImpl) -> None:
    """Test noise injection on empty text returns None/empty variants."""
    case = SyntheticTestCase(
        verbatim_context="",
        synthetic_question="Q",
        golden_chain_of_thought="A",
        expected_json={},
        provenance=ProvenanceType.VERBATIM_SOURCE,
        source_urn="urn",
        modifications=[],
        complexity=0,
        ambiguity=0,
        diversity=0,
        validity_confidence=0,
    )

    variants = perturbator.perturb(case)
    # Should be empty because no numbers, no keywords, and empty text for noise
    assert len(variants) == 0
