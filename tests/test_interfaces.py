# tests/test_interfaces.py
from typing import List, Optional
from uuid import uuid4

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


def test_partial_implementation_fails() -> None:
    """Ensure that concrete classes failing to implement all abstract methods cannot be instantiated."""

    class PartialTeacher(TeacherModel):
        pass

    with pytest.raises(TypeError):
        PartialTeacher()  # type: ignore

    class PartialAnalyzer(PatternAnalyzer):
        pass

    with pytest.raises(TypeError):
        PartialAnalyzer()  # type: ignore

    class PartialForager(Forager):
        pass

    with pytest.raises(TypeError):
        PartialForager()  # type: ignore


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


def test_workflow_simulation() -> None:
    """
    Simulates a full workflow by chaining concrete implementations of the interfaces.
    This verifies that the output type of one component matches the input type of the next.
    """

    # 1. Define Concrete Implementations acting as Mocks
    class MockAnalyzer(PatternAnalyzer):
        def analyze(self, seeds: List[SeedCase]) -> SynthesisTemplate:
            return SynthesisTemplate(
                structure="QA_Format", complexity_description="High", domain="Finance", embedding_centroid=[0.5, 0.5]
            )

    class MockForager(Forager):
        def forage(self, template: SynthesisTemplate, limit: int = 10) -> List[Document]:
            assert template.domain == "Finance"
            return [Document(content="Financial Report 2024...", source_urn="http://example.com/report")]

    class MockExtractor(Extractor):
        def extract(self, documents: List[Document], template: SynthesisTemplate) -> List[str]:
            assert len(documents) > 0
            return [documents[0].content]

    class MockCompositor(Compositor):
        def composite(self, context_slice: str, template: SynthesisTemplate) -> SyntheticTestCase:
            return SyntheticTestCase(
                verbatim_context=context_slice,
                synthetic_question="What is the revenue?",
                golden_chain_of_thought="Revenue is listed as...",
                expected_json={"revenue": 100},
                provenance=ProvenanceType.VERBATIM_SOURCE,
                source_urn="http://example.com/report",
                complexity=5.0,
                diversity=0.8,
                validity_confidence=0.95,
            )

    class MockAppraiser(Appraiser):
        def appraise(self, cases: List[SyntheticTestCase]) -> List[SyntheticTestCase]:
            return sorted(cases, key=lambda c: c.complexity, reverse=True)

    # 2. Instantiate Components
    analyzer = MockAnalyzer()
    forager = MockForager()
    extractor = MockExtractor()
    compositor = MockCompositor()
    appraiser = MockAppraiser()

    # 3. Execute Workflow
    # Step A: Analyze Seeds
    seed = SeedCase(id=uuid4(), context="ctx", question="q", expected_output="a")
    template = analyzer.analyze([seed])
    assert isinstance(template, SynthesisTemplate)

    # Step B: Forage
    documents = forager.forage(template)
    assert isinstance(documents[0], Document)

    # Step C: Extract
    slices = extractor.extract(documents, template)
    assert len(slices) == 1
    assert slices[0] == "Financial Report 2024..."

    # Step D: Composite
    draft_case = compositor.composite(slices[0], template)
    assert isinstance(draft_case, SyntheticTestCase)
    assert draft_case.verbatim_context == "Financial Report 2024..."

    # Step E: Appraise
    final_cases = appraiser.appraise([draft_case])
    assert len(final_cases) == 1
    assert final_cases[0].validity_confidence == 0.95
