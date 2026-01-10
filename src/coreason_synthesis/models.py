# src/coreason_synthesis/models.py
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
    diversity: float = Field(..., ge=0, le=1, description="Distance from the seed's centroid (0-1)")
    validity_confidence: float = Field(..., ge=0, le=1, description="Self-consistency score (0-1)")

    model_config = ConfigDict(use_enum_values=True)
