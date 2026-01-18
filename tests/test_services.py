# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the License).
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_synthesis

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
