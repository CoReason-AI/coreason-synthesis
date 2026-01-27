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
Interfaces for the core components of the synthesis pipeline.

This module defines the abstract base classes that enforce the
Pattern-Forage-Fabricate-Rank architecture described in the PRD.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Type, TypeVar

from coreason_identity.models import UserContext
from pydantic import BaseModel

from .models import (
    Document,
    ExtractedSlice,
    SeedCase,
    SynthesisTemplate,
    SyntheticTestCase,
)

T = TypeVar("T", bound=BaseModel)


class TeacherModel(ABC):
    """Abstract interface for the Teacher Model (LLM).

    The Teacher Model is responsible for high-reasoning tasks such as
    pattern extraction, content generation, and quality appraisal.
    It typically utilizes models like Claude 3.5 Sonnet/Opus or GPT-4o.
    """

    @abstractmethod
    async def generate(self, prompt: str, context: Optional[str] = None) -> str:
        """Generates text based on a prompt and optional context.

        Args:
            prompt: The main prompt for the LLM.
            context: Optional background context (e.g., retrieval data) to be
                included in the prompt construction.

        Returns:
            The generated text string.
        """
        pass

    @abstractmethod
    async def generate_structured(self, prompt: str, response_model: Type[T], context: Optional[str] = None) -> T:
        """Generates a structured object based on a prompt and optional context.

        Args:
            prompt: The main prompt for the LLM.
            response_model: The Pydantic model class to enforce the output structure.
            context: Optional background context (e.g., retrieval data).

        Returns:
            An instance of the response_model containing the structured output.
        """
        pass


class EmbeddingService(ABC):
    """Abstract interface for embedding generation.

    This service converts text into vector representations to support
    semantic search and diversity calculations.
    """

    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """Generates a vector embedding for the given text.

        Args:
            text: The input text string to be embedded.

        Returns:
            A list of floats representing the embedding vector.
        """
        pass


class MCPClient(ABC):
    """Abstract interface for the Model Context Protocol (MCP) client.

    The MCP Client handles communication with the external knowledge base,
    facilitating the retrieval of real-world documents.
    """

    @abstractmethod
    async def search(self, query_vector: List[float], user_context: UserContext, limit: int) -> List[Document]:
        """Searches the MCP for relevant documents using a vector query.

        Args:
            query_vector: The embedding vector to search with.
            user_context: Context for RBAC (e.g., auth token, user identity) to
                ensure the user is allowed to access the retrieved documents.
            limit: Maximum number of documents to retrieve.

        Returns:
            A list of retrieved Document objects.
        """
        pass


class PatternAnalyzer(ABC):
    """The Brain: Deconstructs User's Seeds.

    Responsible for analyzing the user's few-shot examples to infer the
    testing intent, structural pattern, and domain context.
    """

    @abstractmethod
    async def analyze(self, seeds: List[SeedCase]) -> SynthesisTemplate:
        """Analyzes seed cases to extract a synthesis template and vector centroid.

        Args:
            seeds: List of user-provided seed cases.

        Returns:
            A SynthesisTemplate containing the extracted pattern, domain,
            complexity description, and the vector centroid of the seeds.
        """
        pass


class Forager(ABC):
    """The Crawler: Retrieval engine.

    Responsible for finding raw material (documents) that matches the
    semantic neighborhood of the user's seeds while enforcing diversity.
    """

    @abstractmethod
    async def forage(self, template: SynthesisTemplate, user_context: UserContext, limit: int = 10) -> List[Document]:
        """Retrieves documents based on the synthesis template's centroid.

        Args:
            template: The synthesis template containing the vector centroid.
            user_context: Context for RBAC (e.g., auth token, user ID).
            limit: Maximum number of documents to retrieve. Defaults to 10.

        Returns:
            A list of retrieved Document objects, ideally diverse in content.
        """
        pass


class Extractor(ABC):
    """The Miner: Targeted mining of text slices.

    Responsible for identifying and copying text chunks (e.g., paragraphs,
    tables) from retrieved documents and sanitizing PII.
    """

    @abstractmethod
    async def extract(self, documents: List[Document], template: SynthesisTemplate) -> List[ExtractedSlice]:
        """Extracts relevant text slices from documents matching the template structure.

        Args:
            documents: List of retrieved documents.
            template: The synthesis template describing the target structure.

        Returns:
            A list of ExtractedSlice objects containing the verbatim text
            and lineage information.
        """
        pass


class Compositor(ABC):
    """The Generator: Wraps real data in synthetic interactions.

    Responsible for using a Teacher Model to read the verbatim text and
    generate a user prompt and a 'Golden Chain-of-Thought'.
    """

    @abstractmethod
    async def composite(self, context_slice: ExtractedSlice, template: SynthesisTemplate) -> SyntheticTestCase:
        """Generates a single synthetic test case from a context slice.

        Args:
            context_slice: The verbatim text slice (wrapped in ExtractedSlice for lineage).
            template: The synthesis template to guide generation.

        Returns:
            A draft SyntheticTestCase (usually with provenance VERBATIM_SOURCE).
        """
        pass


class Perturbator(ABC):
    """The Red Team: Creates 'Hard Negatives' and 'Edge Cases'.

    Responsible for applying mutations (e.g., value swaps, negations) to
    generated test cases to test robustness and failure modes.
    """

    @abstractmethod
    async def perturb(self, case: SyntheticTestCase) -> List[SyntheticTestCase]:
        """Applies perturbations to a test case to create variants.

        Args:
            case: The original synthetic test case.

        Returns:
            A list containing the perturbed variants (and optionally the original
            if the implementation decides so, though typically just variants).
        """
        pass


class Appraiser(ABC):
    """The Judge: Scoring engine that ranks quality.

    Responsible for scoring generated cases based on complexity, ambiguity,
    diversity, and validity, and filtering out low-quality ones.
    """

    @abstractmethod
    async def appraise(
        self,
        cases: List[SyntheticTestCase],
        template: SynthesisTemplate,
        sort_by: str = "complexity_desc",
        min_validity_score: float = 0.8,
    ) -> List[SyntheticTestCase]:
        """Scores and filters test cases.

        Args:
            cases: List of generated test cases to appraise.
            template: The synthesis template (needed for diversity calculation).
            sort_by: The metric to sort by (e.g., 'complexity_desc', 'diversity_desc').
                Defaults to 'complexity_desc'.
            min_validity_score: The minimum validity confidence required to keep a case.
                Defaults to 0.8.

        Returns:
            A sorted list of appraised SyntheticTestCase objects that met the
            validity threshold.
        """
        pass
