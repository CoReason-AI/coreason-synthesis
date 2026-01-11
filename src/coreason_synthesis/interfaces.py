# src/coreason_synthesis/interfaces.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .models import (
    Document,
    SeedCase,
    SynthesisTemplate,
    SyntheticTestCase,
)


class TeacherModel(ABC):
    """Abstract interface for the Teacher Model (LLM)."""

    @abstractmethod
    def generate(self, prompt: str, context: Optional[str] = None) -> str:
        """
        Generates text based on a prompt and optional context.

        Args:
            prompt: The main prompt for the LLM.
            context: Optional background context (e.g., retrieval data).

        Returns:
            The generated text string.
        """
        pass


class PatternAnalyzer(ABC):
    """The Brain: Deconstructs User's Seeds."""

    @abstractmethod
    def analyze(self, seeds: List[SeedCase]) -> SynthesisTemplate:
        """
        Analyzes seed cases to extract a synthesis template and vector centroid.

        Args:
            seeds: List of user-provided seed cases.

        Returns:
            A SynthesisTemplate containing the extracted pattern and centroid.
        """
        pass


class Forager(ABC):
    """The Crawler: Retrieval engine."""

    @abstractmethod
    def forage(self, template: SynthesisTemplate, user_context: Dict[str, Any], limit: int = 10) -> List[Document]:
        """
        Retrieves documents based on the synthesis template's centroid.

        Args:
            template: The synthesis template containing the vector centroid.
            user_context: Context for RBAC (e.g., auth token, user ID).
            limit: Maximum number of documents to retrieve.

        Returns:
            List of retrieved Documents.
        """
        pass


class Extractor(ABC):
    """The Miner: Targeted mining of text slices."""

    @abstractmethod
    def extract(self, documents: List[Document], template: SynthesisTemplate) -> List[str]:
        """
        Extracts relevant text slices from documents matching the template structure.

        Args:
            documents: List of retrieved documents.
            template: The synthesis template describing the target structure.

        Returns:
            List of extracted text slices (verbatim).
        """
        pass


class Compositor(ABC):
    """The Generator: Wraps real data in synthetic interactions."""

    @abstractmethod
    def composite(self, context_slice: str, template: SynthesisTemplate) -> SyntheticTestCase:
        """
        Generates a single synthetic test case from a context slice.

        Args:
            context_slice: The verbatim text slice.
            template: The synthesis template to guide generation.

        Returns:
            A draft SyntheticTestCase (usually VERBATIM_SOURCE).
        """
        pass


class Perturbator(ABC):
    """The Red Team: Creates 'Hard Negatives' and 'Edge Cases'."""

    @abstractmethod
    def perturb(self, case: SyntheticTestCase) -> List[SyntheticTestCase]:
        """
        Applies perturbations to a test case to create variants.

        Args:
            case: The original synthetic test case.

        Returns:
            A list containing the original and/or perturbed cases.
        """
        pass


class Appraiser(ABC):
    """The Judge: Scoring engine that ranks quality."""

    @abstractmethod
    def appraise(self, cases: List[SyntheticTestCase]) -> List[SyntheticTestCase]:
        """
        Scores and filters test cases.

        Args:
            cases: List of generated test cases.

        Returns:
            List of appraised and ranked test cases.
        """
        pass
