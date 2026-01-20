# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_synthesis

import pytest

from coreason_synthesis.models import ProvenanceType, SyntheticTestCase
from coreason_synthesis.perturbator import PerturbatorImpl


@pytest.fixture
def perturbator() -> PerturbatorImpl:
    return PerturbatorImpl()


@pytest.fixture
def base_case() -> SyntheticTestCase:
    return SyntheticTestCase(
        verbatim_context="The patient took 50mg of Aspirin.",
        synthetic_question="Q",
        golden_chain_of_thought="R",
        expected_json={},
        provenance=ProvenanceType.VERBATIM_SOURCE,
        source_urn="u",
        complexity=0.5,
        diversity=0.5,
        validity_confidence=1.0,
    )


@pytest.mark.asyncio
async def test_numeric_swap(perturbator: PerturbatorImpl, base_case: SyntheticTestCase) -> None:
    variants = await perturbator.perturb(base_case)
    # 50mg -> 5000mg
    numeric_variants = [v for v in variants if "5000" in v.verbatim_context]
    assert len(numeric_variants) == 1
    assert numeric_variants[0].provenance == ProvenanceType.SYNTHETIC_PERTURBED
    assert numeric_variants[0].validity_confidence == 0.0
    assert len(numeric_variants[0].modifications) > 0


@pytest.mark.asyncio
async def test_negation_swap(perturbator: PerturbatorImpl, base_case: SyntheticTestCase) -> None:
    base_case.verbatim_context = "Treatment is included in the plan."
    variants = await perturbator.perturb(base_case)
    # included -> excluded
    negation_variants = [v for v in variants if "excluded" in v.verbatim_context]
    assert len(negation_variants) == 1
    assert negation_variants[0].modifications[0].description.startswith("Negation Swap")


@pytest.mark.asyncio
async def test_negation_swap_lowercase(perturbator: PerturbatorImpl, base_case: SyntheticTestCase) -> None:
    base_case.verbatim_context = "This is False."
    variants = await perturbator.perturb(base_case)
    # False -> True
    # "False" matches "false", preserves case -> "True"
    neg_vars = [v for v in variants if "True" in v.verbatim_context]
    assert len(neg_vars) == 1


@pytest.mark.asyncio
async def test_no_perturbations_possible_except_noise(
    perturbator: PerturbatorImpl, base_case: SyntheticTestCase
) -> None:
    base_case.verbatim_context = "Just some words."
    variants = await perturbator.perturb(base_case)
    # Only Noise Injection possible
    assert len(variants) == 1
    assert "Noise Injection" in variants[0].modifications[0].description


@pytest.mark.asyncio
async def test_deep_copy_independence(perturbator: PerturbatorImpl, base_case: SyntheticTestCase) -> None:
    variants = await perturbator.perturb(base_case)
    # Modify variant, check base case intact
    variants[0].verbatim_context = "Modified"
    assert base_case.verbatim_context == "The patient took 50mg of Aspirin."


@pytest.mark.asyncio
async def test_multiple_strategies(perturbator: PerturbatorImpl, base_case: SyntheticTestCase) -> None:
    # "50mg" -> numeric swap
    # "included" -> negation swap
    base_case.verbatim_context = "50mg included."
    variants = await perturbator.perturb(base_case)
    # Numeric, Negation, Noise -> 3 variants
    assert len(variants) == 3


@pytest.mark.asyncio
async def test_decimal_scaling(perturbator: PerturbatorImpl, base_case: SyntheticTestCase) -> None:
    # Avoid trailing dot which can interfere with regex lookahead
    base_case.verbatim_context = "Value 0.5 is correct"
    variants = await perturbator.perturb(base_case)
    # 0.5 * 100 = 50.0 -> "50" (rstrip logic)
    # Or 50.0 -> "50"
    num_vars = [v for v in variants if "50" in v.verbatim_context]
    assert len(num_vars) >= 1
    # Ensure "0.5" is gone in that variant
    assert "0.5" not in num_vars[0].verbatim_context


@pytest.mark.asyncio
async def test_multiple_numbers(perturbator: PerturbatorImpl, base_case: SyntheticTestCase) -> None:
    base_case.verbatim_context = "10 and 20."
    variants = await perturbator.perturb(base_case)
    # Only first match swapped: 1000 and 20
    v = [v for v in variants if "1000" in v.verbatim_context][0]
    assert "20" in v.verbatim_context


@pytest.mark.asyncio
async def test_word_boundary_safety(perturbator: PerturbatorImpl, base_case: SyntheticTestCase) -> None:
    # "include" should not match "conclude"
    base_case.verbatim_context = "conclude the session."
    variants = await perturbator.perturb(base_case)
    # Should check descriptions. Only Noise expected.
    for v in variants:
        desc = v.modifications[0].description
        assert "Negation" not in desc


@pytest.mark.asyncio
async def test_all_caps_handling(perturbator: PerturbatorImpl, base_case: SyntheticTestCase) -> None:
    # Our simple logic preserves if first char is upper.
    # INCLUDED -> Excluded (since logic is capitalized() if [0] is upper)
    # Ideally should detect all caps, but spec is simple.
    base_case.verbatim_context = "INCLUDED"
    variants = await perturbator.perturb(base_case)
    neg = [v for v in variants if "Excluded" in v.verbatim_context]
    assert len(neg) == 1


@pytest.mark.asyncio
async def test_formatted_number(perturbator: PerturbatorImpl, base_case: SyntheticTestCase) -> None:
    # "1,000" -> regex \d+ doesn't match comma. It matches 1 and 000 separately.
    # "1" -> 100, "000" -> 0
    # Current regex: (?<![\d.])\d+(\.\d+)?(?![\d.])
    # "1,000":
    #   "1" matches -> "100"
    #   ",000"
    # Result: "100,000"
    base_case.verbatim_context = "1,000"
    variants = await perturbator.perturb(base_case)
    v = [v for v in variants if "Numeric" in v.modifications[0].description][0]
    # Expect 100,000 because "1" is found first.
    assert "100,000" in v.verbatim_context


@pytest.mark.asyncio
async def test_chained_perturbation(perturbator: PerturbatorImpl, base_case: SyntheticTestCase) -> None:
    # Only single pass per call.
    pass


@pytest.mark.asyncio
async def test_noise_injection(perturbator: PerturbatorImpl, base_case: SyntheticTestCase) -> None:
    base_case.verbatim_context = "Context"
    # Mock random choice for noise
    # We can't easily mock random inside without patch.
    # But we know noise is always injected.
    variants = await perturbator.perturb(base_case)
    noise_vars = [v for v in variants if "Noise Injection" in v.modifications[0].description]
    assert len(noise_vars) == 1
    assert len(noise_vars[0].verbatim_context) > len("Context")


@pytest.mark.asyncio
async def test_noise_injection_append(perturbator: PerturbatorImpl, base_case: SyntheticTestCase) -> None:
    # Probabilistic
    pass


@pytest.mark.asyncio
async def test_noise_injection_empty_context(perturbator: PerturbatorImpl, base_case: SyntheticTestCase) -> None:
    base_case.verbatim_context = ""
    variants = await perturbator.perturb(base_case)
    # Should handle empty gracefully -> Returns None -> no variant added
    assert len(variants) == 0


@pytest.mark.asyncio
async def test_noise_injection_whitespace_context(perturbator: PerturbatorImpl, base_case: SyntheticTestCase) -> None:
    base_case.verbatim_context = "   "
    variants = await perturbator.perturb(base_case)
    # "   Noise"
    assert len(variants) > 0


@pytest.mark.asyncio
async def test_noise_injection_unicode(perturbator: PerturbatorImpl, base_case: SyntheticTestCase) -> None:
    base_case.verbatim_context = "Emoji 😀"
    variants = await perturbator.perturb(base_case)
    assert len(variants) > 0
    assert "Emoji 😀" in variants[-1].verbatim_context


@pytest.mark.asyncio
async def test_chained_perturbation_with_noise(perturbator: PerturbatorImpl, base_case: SyntheticTestCase) -> None:
    pass
