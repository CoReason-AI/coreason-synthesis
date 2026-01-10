# tests/test_interfaces.py
from typing import List, Optional

import pytest

from src.coreason_synthesis.interfaces import (
    Appraiser,
    Compositor,
    Extractor,
    Forager,
    PatternAnalyzer,
    Perturbator,
    TeacherModel,
)
from src.coreason_synthesis.models import (
    Document,
    ProvenanceType,
    SeedCase,
    SynthesisTemplate,
    SyntheticTestCase,
)


def test_cannot_instantiate_interfaces() -> None:
    """Ensure that interfaces cannot be instantiated directly."""
    with pytest.raises(TypeError):
        TeacherModel()  # type: ignore
    with pytest.raises(TypeError):
        PatternAnalyzer()  # type: ignore
    with pytest.raises(TypeError):
        Forager()  # type: ignore
    with pytest.raises(TypeError):
        Extractor()  # type: ignore
    with pytest.raises(TypeError):
        Compositor()  # type: ignore
    with pytest.raises(TypeError):
        Perturbator()  # type: ignore
    with pytest.raises(TypeError):
        Appraiser()  # type: ignore


def test_concrete_implementations() -> None:
    """Ensure that concrete implementations work if they implement all abstract methods."""

    class ConcreteTeacher(TeacherModel):
        def generate(self, prompt: str, context: Optional[str] = None) -> str:
            return "generated"

    class ConcreteAnalyzer(PatternAnalyzer):
        def analyze(self, seeds: List[SeedCase]) -> SynthesisTemplate:
            return SynthesisTemplate(structure="s", complexity_description="c", domain="d", embedding_centroid=[0.1])

    class ConcreteForager(Forager):
        def forage(self, template: SynthesisTemplate, limit: int = 10) -> List[Document]:
            return [Document(content="c", source_urn="u")]

    class ConcreteExtractor(Extractor):
        def extract(self, documents: List[Document], template: SynthesisTemplate) -> List[str]:
            return ["slice"]

    class ConcreteCompositor(Compositor):
        def composite(self, context_slice: str, template: SynthesisTemplate) -> SyntheticTestCase:
            return SyntheticTestCase(
                verbatim_context="v",
                synthetic_question="q",
                golden_chain_of_thought="g",
                expected_json={},
                provenance=ProvenanceType.VERBATIM_SOURCE,
                source_urn="u",
                complexity=1.0,
                diversity=1.0,
                validity_confidence=1.0,
            )

    class ConcretePerturbator(Perturbator):
        def perturb(self, case: SyntheticTestCase) -> List[SyntheticTestCase]:
            return [case]

    class ConcreteAppraiser(Appraiser):
        def appraise(self, cases: List[SyntheticTestCase]) -> List[SyntheticTestCase]:
            return cases

    # Instantiate to verify no TypeError
    ConcreteTeacher()
    ConcreteAnalyzer()
    ConcreteForager()
    ConcreteExtractor()
    ConcreteCompositor()
    ConcretePerturbator()
    ConcreteAppraiser()
