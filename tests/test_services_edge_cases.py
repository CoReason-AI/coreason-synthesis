import pytest
from pydantic import BaseModel

from coreason_synthesis.services import MockTeacher


class RandomModel(BaseModel):
    field: str


def test_mock_teacher_unknown_model_error() -> None:
    """Test that MockTeacher raises NotImplementedError for unknown models."""
    teacher = MockTeacher()
    with pytest.raises(
        NotImplementedError, match="MockTeacher.generate_structured does not know how to mock RandomModel"
    ):
        teacher.generate_structured("prompt", RandomModel)


def test_mock_teacher_synthesis_template_partial() -> None:
    """Test the exception block in MockTeacher by passing a model that matches name but not fields."""

    # Define a model with matching name but strict required fields that differ from default mock
    class SynthesisTemplate(BaseModel):
        required_field_not_in_mock: str

    teacher = MockTeacher()
    # This should hit the except Exception block and then fall through to NotImplementedError
    # or handle it gracefully if we change the implementation.
    # Current implementation: pass in except, then raise NotImplementedError at end.

    with pytest.raises(NotImplementedError):
        teacher.generate_structured("prompt", SynthesisTemplate)
