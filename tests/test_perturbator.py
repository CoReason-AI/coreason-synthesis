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
        diversity=0.0,
        validity_confidence=1.0,
    )


def test_numeric_swap(perturbator: PerturbatorImpl, base_case: SyntheticTestCase) -> None:
    """Test that numeric values are swapped (multiplied by 100)."""
    variants = perturbator.perturb(base_case)

    # Check if we got at least one variant (should get numeric swap)
    assert len(variants) >= 1

    # Find the numeric variant
    numeric_variant = next(
        (v for v in variants if any(isinstance(m, Diff) and "Numeric" in m.description for m in v.modifications)), None
    )
    assert numeric_variant is not None

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
        diversity=0.0,
        validity_confidence=1.0,
    )

    variants = perturbator.perturb(case)
    negation_variant = next(
        (v for v in variants if any(isinstance(m, Diff) and "Negation" in m.description for m in v.modifications)), None
    )

    assert negation_variant is not None
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
        diversity=0.0,
        validity_confidence=1.0,
    )

    variants = perturbator.perturb(case)
    negation_variant = next(
        (v for v in variants if any(isinstance(m, Diff) and "Negation" in m.description for m in v.modifications)), None
    )

    assert negation_variant is not None
    # included (lower) -> excluded (lower)
    assert "excluded" in negation_variant.verbatim_context
    assert "included" not in negation_variant.verbatim_context

    mod = negation_variant.modifications[0]
    assert isinstance(mod, Diff)
    assert mod.original == "included"
    assert mod.new == "excluded"


def test_no_perturbations_possible(perturbator: PerturbatorImpl) -> None:
    """Test case with no numbers and no keywords."""
    case = SyntheticTestCase(
        verbatim_context="The sky is blue.",
        synthetic_question="Color?",
        golden_chain_of_thought="Blue.",
        expected_json={"color": "blue"},
        provenance=ProvenanceType.VERBATIM_SOURCE,
        source_urn="urn:test",
        modifications=[],
        complexity=1.0,
        diversity=0.0,
        validity_confidence=1.0,
    )

    variants = perturbator.perturb(case)
    assert len(variants) == 0


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
    """Test that both strategies can apply to the same input, creating separate variants."""
    case = SyntheticTestCase(
        verbatim_context="Include 50 patients.",
        synthetic_question="Count?",
        golden_chain_of_thought="50.",
        expected_json={"count": 50},
        provenance=ProvenanceType.VERBATIM_SOURCE,
        source_urn="urn:test",
        modifications=[],
        complexity=1.0,
        diversity=0.0,
        validity_confidence=1.0,
    )

    variants = perturbator.perturb(case)

    # Should produce 2 variants: one for 'Include'->'Exclude', one for '50'->'5000'
    assert len(variants) == 2

    variant_texts = [v.verbatim_context for v in variants]
    assert "Exclude 50 patients." in variant_texts
    assert "Include 5000 patients." in variant_texts


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
        diversity=0.0,
        validity_confidence=1.0,
    )

    variants = perturbator.perturb(case)
    # 0.5 * 100 = 50.0 -> "50"
    variant = variants[0]
    # My impl uses rstrip('0').rstrip('.') so 50.0 -> 50.
    assert "Dose is 50mg." in variant.verbatim_context
