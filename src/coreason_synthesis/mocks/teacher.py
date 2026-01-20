# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_synthesis

"""
Mock Teacher Model for testing.
"""

from typing import Optional, Type

from coreason_synthesis.interfaces import T, TeacherModel


class MockTeacher(TeacherModel):
    """Deterministic mock teacher model for testing."""

    def generate(self, prompt: str, context: Optional[str] = None) -> str:
        """Returns a mock response based on the prompt content.

        Args:
            prompt: The input prompt.
            context: Optional context.

        Returns:
            A string response.
        """
        if "structure" in prompt.lower():
            return (
                "Structure: Question + JSON Output\n"
                "Complexity: Requires multi-hop reasoning\n"
                "Domain: Oncology / Inclusion Criteria"
            )
        return "Mock generated response"

    def generate_structured(self, prompt: str, response_model: Type[T], context: Optional[str] = None) -> T:
        """Returns a mock structured response based on the prompt content and response model.

        Args:
            prompt: The input prompt.
            response_model: The Pydantic model to populate.
            context: Optional context.

        Returns:
            An instance of response_model populated with mock data.

        Raises:
            NotImplementedError: If the response_model is not supported by the mock.
        """
        # We need to construct a dummy instance of T.
        # This is tricky without knowing the exact structure of T, but for our tests we know what T will be.
        # However, to be generic in the mock, we can try to instantiate it with default values or known test values.

        # Check if the response_model is one we expect
        if "SynthesisTemplate" in response_model.__name__ or "TemplateAnalysis" in response_model.__name__:
            # We can return a dict compatible with the expected fields, validated by the model
            # Note: SynthesisTemplate requires embedding_centroid, but TemplateAnalysis (local model) might not.
            # We'll assume the caller uses a model compatible with these fields.
            try:
                # Attempt to instantiate with test data
                return response_model(
                    structure="Question + JSON Output",
                    complexity_description="Requires multi-hop reasoning",
                    domain="Oncology / Inclusion Criteria",
                    embedding_centroid=[0.1, 0.2, 0.3],  # Dummy centroid if needed
                )
            except Exception:  # pragma: no cover
                # If T doesn't match above, try to construct with defaults if possible, or raise
                pass
        elif "GenerationOutput" in response_model.__name__:
            try:
                return response_model(
                    synthetic_question="Synthetic question?",
                    golden_chain_of_thought="Step 1. Step 2.",
                    expected_json={"result": "value"},
                )
            except Exception:  # pragma: no cover
                pass
        elif "AppraisalAnalysis" in response_model.__name__:
            try:
                return response_model(
                    complexity_score=5.0,
                    ambiguity_score=2.0,
                    validity_confidence=0.9,
                )
            except Exception:  # pragma: no cover
                pass

        # If we can't determine what to return, we might need a more sophisticated mock or hardcode for specific tests.
        # For now, let's try to construct it with dummy data if it's a simple model, or raise NotImplementedError
        # allowing tests to patch it if needed.
        raise NotImplementedError(
            f"MockTeacher.generate_structured does not know how to mock {response_model.__name__}"
        )
