# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_synthesis

from typing import Any, List, Optional, Type, TypeVar
from uuid import uuid4

import pytest
from pydantic import BaseModel

from coreason_synthesis.interfaces import (
    Appraiser,
    Compositor,
    Extractor,
    Forager,
    PatternAnalyzer,
    Perturbator,
    TeacherModel,
)
from coreason_synthesis.models import (
    Document,
    ExtractedSlice,
    ProvenanceType,
    SeedCase,
    SynthesisTemplate,
    SyntheticTestCase,
)

T = TypeVar("T", bound=BaseModel)


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
        async def generate(self, prompt: str, context: Optional[str] = None) -> str:
            return "generated"

        async def generate_structured(self, prompt: str, response_model: Type[T], context: Optional[str] = None) -> T:
            # Simple dummy implementation for test
            try:
                return response_model()
            except Exception as e:
                # If model requires fields, we might fail here, but this test just checks instantiation of the class
                raise NotImplementedError from e

    class ConcreteAnalyzer(PatternAnalyzer):
        async def analyze(self, seeds: List[SeedCase]) -> SynthesisTemplate:
            return SynthesisTemplate(structure="s", complexity_description="c", domain="d", embedding_centroid=[0.1])

    class ConcreteForager(Forager):
        async def forage(
            self, template: SynthesisTemplate, user_context: dict[str, Any], limit: int = 10
        ) -> List[Document]:
            return [Document(content="c", source_urn="u")]

    class ConcreteExtractor(Extractor):
        async def extract(self, documents: List[Document], template: SynthesisTemplate) -> List[ExtractedSlice]:
            return [ExtractedSlice(content="slice", source_urn="u", page_number=1, pii_redacted=False, metadata={})]

    class ConcreteCompositor(Compositor):
        async def composite(self, context_slice: ExtractedSlice, template: SynthesisTemplate) -> SyntheticTestCase:
            return SyntheticTestCase(
                verbatim_context=context_slice.content,
                synthetic_question="q",
                golden_chain_of_thought="g",
                expected_json={},
                provenance=ProvenanceType.VERBATIM_SOURCE,
                source_urn=context_slice.source_urn,
                complexity=1.0,
                diversity=1.0,
                validity_confidence=1.0,
            )

    class ConcretePerturbator(Perturbator):
        async def perturb(self, case: SyntheticTestCase) -> List[SyntheticTestCase]:
            return [case]

    class ConcreteAppraiser(Appraiser):
        async def appraise(
            self,
            cases: List[SyntheticTestCase],
            template: SynthesisTemplate,
            sort_by: str = "complexity_desc",
            min_validity_score: float = 0.8,
        ) -> List[SyntheticTestCase]:
            return cases

    # Instantiate to verify no TypeError
    ConcreteTeacher()
    ConcreteAnalyzer()
    ConcreteForager()
    ConcreteExtractor()
    ConcreteCompositor()
    ConcretePerturbator()
    ConcreteAppraiser()


@pytest.mark.asyncio
async def test_workflow_simulation() -> None:
    """
    Simulates a full workflow by chaining concrete implementations of the interfaces.
    This verifies that the output type of one component matches the input type of the next.
    """

    # 1. Define Concrete Implementations acting as Mocks
    class MockAnalyzer(PatternAnalyzer):
        async def analyze(self, seeds: List[SeedCase]) -> SynthesisTemplate:
            return SynthesisTemplate(
                structure="QA_Format", complexity_description="High", domain="Finance", embedding_centroid=[0.5, 0.5]
            )

    class MockForager(Forager):
        async def forage(
            self, template: SynthesisTemplate, user_context: dict[str, Any], limit: int = 10
        ) -> List[Document]:
            assert template.domain == "Finance"
            return [Document(content="Financial Report 2024...", source_urn="http://example.com/report")]

    class MockExtractor(Extractor):
        async def extract(self, documents: List[Document], template: SynthesisTemplate) -> List[ExtractedSlice]:
            assert len(documents) > 0
            return [
                ExtractedSlice(
                    content=documents[0].content,
                    source_urn=documents[0].source_urn,
                    page_number=1,
                    pii_redacted=False,
                    metadata={},
                )
            ]

    class MockCompositor(Compositor):
        async def composite(self, context_slice: ExtractedSlice, template: SynthesisTemplate) -> SyntheticTestCase:
            return SyntheticTestCase(
                verbatim_context=context_slice.content,
                synthetic_question="What is the revenue?",
                golden_chain_of_thought="Revenue is listed as...",
                expected_json={"revenue": 100},
                provenance=ProvenanceType.VERBATIM_SOURCE,
                source_urn=context_slice.source_urn,
                complexity=5.0,
                diversity=0.8,
                validity_confidence=0.95,
            )

    class MockAppraiser(Appraiser):
        async def appraise(
            self,
            cases: List[SyntheticTestCase],
            template: SynthesisTemplate,
            sort_by: str = "complexity_desc",
            min_validity_score: float = 0.8,
        ) -> List[SyntheticTestCase]:
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
    template = await analyzer.analyze([seed])
    assert isinstance(template, SynthesisTemplate)

    # Step B: Forage
    documents = await forager.forage(template, user_context={})
    assert isinstance(documents[0], Document)

    # Step C: Extract
    slices = await extractor.extract(documents, template)
    assert len(slices) == 1
    assert slices[0].content == "Financial Report 2024..."

    # Step D: Composite
    draft_case = await compositor.composite(slices[0], template)
    assert isinstance(draft_case, SyntheticTestCase)
    assert draft_case.verbatim_context == "Financial Report 2024..."

    # Step E: Appraise
    final_cases = await appraiser.appraise([draft_case], template)
    assert len(final_cases) == 1
    assert final_cases[0].validity_confidence == 0.95
