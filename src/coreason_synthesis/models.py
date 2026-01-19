# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_synthesis

from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ProvenanceType(str, Enum):
    """Enumeration for the provenance of a test case."""

    VERBATIM_SOURCE = "VERBATIM_SOURCE"
    SYNTHETIC_PERTURBED = "SYNTHETIC_PERTURBED"


class Diff(BaseModel):
    """Represents a modification made to the original text."""

    description: str = Field(..., description="Description of the change")
    original: Optional[str] = Field(None, description="The original text segment")
    new: Optional[str] = Field(None, description="The new text segment")


class SeedCase(BaseModel):
    """Represents a user-provided seed example for synthesis."""

    id: UUID = Field(..., description="Unique identifier for the seed case")
    context: str = Field(..., description="The context or background text for the seed case")
    question: str = Field(..., description="The question or prompt")
    expected_output: Union[str, Dict[str, Any]] = Field(..., description="The expected answer or output")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata for the seed")


class SyntheticJob(BaseModel):
    """Represents a job to generate synthetic test cases."""

    id: UUID = Field(..., description="Unique identifier for the job")
    seed_ids: List[UUID] = Field(..., description="List of seed case IDs used for this job")
    config: Dict[str, Any] = Field(..., description="Configuration for the job (e.g., perturbation_rate, target_count)")


class Document(BaseModel):
    """Represents a retrieved document from the Forager."""

    content: str = Field(..., description="The full text content of the document")
    source_urn: str = Field(..., description="Unique Resource Name or Source URL")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata (e.g., title, author)")


class ExtractedSlice(BaseModel):
    """Represents a mined text slice from a document, with PII handling and traceability."""

    content: str = Field(..., description="The sanitized text content")
    source_urn: str = Field(..., description="URN of the source document")
    page_number: Optional[int] = Field(None, description="Page number where the slice was found")
    pii_redacted: bool = Field(False, description="Flag indicating if PII was redacted")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class SynthesisTemplate(BaseModel):
    """Represents the extracted pattern from SeedCases."""

    structure: str = Field(..., description="Description of the question/output structure")
    complexity_description: str = Field(..., description="Description of the complexity")
    domain: str = Field(..., description="The identified domain of the seeds")
    embedding_centroid: Optional[List[float]] = Field(None, description="The vector centroid of the seeds")


class SyntheticTestCase(BaseModel):
    """Represents a generated synthetic test case."""

    # Content
    verbatim_context: str = Field(..., description="The real data context (verbatim)")
    synthetic_question: str = Field(..., description="The generated question")
    golden_chain_of_thought: str = Field(..., description="The teacher's reasoning logic")
    expected_json: Dict[str, Any] = Field(..., description="The expected JSON output")

    # Provenance
    provenance: ProvenanceType = Field(..., description="The origin type of the test case")
    source_urn: str = Field(..., description="URN of the source document")
    modifications: List[Union[Diff, str]] = Field(
        default_factory=list, description="List of modifications applied (if any)"
    )

    # Metrics
    complexity: float = Field(..., ge=0, le=10, description="Estimated logical steps required (0-10)")
    ambiguity: float = Field(..., ge=0, le=10, description="How implicit is the answer? (0-10)")
    diversity: float = Field(..., ge=0, le=1, description="Distance from the seed's centroid (0-1)")
    validity_confidence: float = Field(..., ge=0, le=1, description="Self-consistency score (0-1)")

    model_config = ConfigDict(use_enum_values=True)
