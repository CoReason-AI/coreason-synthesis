from coreason_synthesis.services import MockTeacher


def test_mock_teacher_generate_default() -> None:
    """Test that MockTeacher.generate returns default mock response."""
    teacher = MockTeacher()
    assert teacher.generate("some prompt") == "Mock generated response"


def test_mock_teacher_generate_structure() -> None:
    """Test that MockTeacher.generate returns structure-specific response."""
    teacher = MockTeacher()
    response = teacher.generate("Describe structure")
    assert "Structure: Question + JSON Output" in response
    assert "Complexity: Requires multi-hop reasoning" in response
