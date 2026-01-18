# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_synthesis

from typing import Any, Dict

from pydantic import BaseModel, Field

from .interfaces import Compositor, TeacherModel
from .models import (
    ExtractedSlice,
    ProvenanceType,
    SynthesisTemplate,
    SyntheticTestCase,
)


class GenerationOutput(BaseModel):
    """Internal model for the Teacher's structured output."""

    synthetic_question: str = Field(..., description="The generated question based on the context")
    golden_chain_of_thought: str = Field(..., description="Step-by-step reasoning to reach the answer")
    expected_json: Dict[str, Any] = Field(..., description="The structured answer matching the template")


class CompositorImpl(Compositor):
    """
    Concrete implementation of the Compositor.
    Wraps real data in synthetic interactions using a Teacher Model.
    """

    def __init__(self, teacher: TeacherModel):
        self.teacher = teacher

    def composite(self, context_slice: ExtractedSlice, template: SynthesisTemplate) -> SyntheticTestCase:
        """
        Generates a single synthetic test case from a context slice.
        """
        # Construct the prompt
        prompt = self._construct_prompt(context_slice.content, template)

        # Generate structured output from Teacher
        # We pass context_slice.content as 'context' to the teacher as well.
        output = self.teacher.generate_structured(
            prompt=prompt, response_model=GenerationOutput, context=context_slice.content
        )

        # Construct the SyntheticTestCase
        return SyntheticTestCase(
            verbatim_context=context_slice.content,
            synthetic_question=output.synthetic_question,
            golden_chain_of_thought=output.golden_chain_of_thought,
            expected_json=output.expected_json,
            provenance=ProvenanceType.VERBATIM_SOURCE,
            source_urn=context_slice.source_urn,
            modifications=[],
            # Metrics initialized to 0.0, to be calculated by Appraiser
            complexity=0.0,
            diversity=0.0,
            validity_confidence=0.0,
        )

    def _construct_prompt(self, context: str, template: SynthesisTemplate) -> str:
        """
        Constructs the prompt for the Teacher Model.
        """
        return (
            f"You are an expert Data Synthesizer for the domain: {template.domain}.\n"
            f"Your task is to generate a test case based on the provided context.\n\n"
            f"Target Structure: {template.structure}\n"
            f"Target Complexity: {template.complexity_description}\n\n"
            f"Context:\n{context}\n\n"
            "Please generate:\n"
            "1. A synthetic question that tests the user's ability to reason over the context.\n"
            "2. A golden chain-of-thought explaining the correct logic.\n"
            "3. The expected output in JSON format."
        )
