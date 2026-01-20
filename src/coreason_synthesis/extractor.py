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
Extraction and sanitization module.

This module is responsible for mining usable text chunks from retrieved documents
and ensuring all Personally Identifiable Information (PII) is redacted before use.
"""

import re
from typing import List

from .interfaces import Extractor
from .models import Document, ExtractedSlice, SynthesisTemplate


class ExtractorImpl(Extractor):
    """Concrete implementation of the Extractor.

    Mines text slices using heuristic chunking and sanitizes PII using regex patterns.
    """

    def extract(self, documents: List[Document], template: SynthesisTemplate) -> List[ExtractedSlice]:
        """Extracts text slices from documents.

        Applies PII sanitization and maps back to source.

        Args:
            documents: List of retrieved documents.
            template: The synthesis template (used for potential structure matching).

        Returns:
            List of extracted text slices (verbatim).
        """
        extracted_slices: List[ExtractedSlice] = []

        for doc in documents:
            # 1. Heuristic Chunking (Paragraphs)
            chunks = self._chunk_content(doc.content)

            for i, chunk in enumerate(chunks):
                if not self._is_valid_chunk(chunk):
                    continue

                # 2. PII Sanitization
                sanitized_content, redacted = self._sanitize(chunk)

                # 3. Create ExtractedSlice
                extracted_slices.append(
                    ExtractedSlice(
                        content=sanitized_content,
                        source_urn=doc.source_urn,
                        # Fallback page logic or from metadata if available.
                        # Assuming metadata might contain page info, else None
                        page_number=doc.metadata.get("page_number"),
                        pii_redacted=redacted,
                        metadata={
                            "chunk_index": i,
                            "original_length": len(chunk),
                            "sanitized_length": len(sanitized_content),
                        },
                    )
                )

        return extracted_slices

    def _chunk_content(self, content: str) -> List[str]:
        """Splits content into paragraphs based on double newlines.

        Handles mixed line endings by normalizing to \\n.

        Args:
            content: The raw document content.

        Returns:
            List of text chunks (paragraphs).
        """
        if not content:
            return []
        # Normalize line endings
        normalized = content.replace("\r\n", "\n").replace("\r", "\n")
        # Split by double newline to identify paragraphs
        # Filter out empty strings after strip
        return [c.strip() for c in normalized.split("\n\n") if c.strip()]

    def _is_valid_chunk(self, chunk: str) -> bool:
        """Filters out chunks that are too short or irrelevant.

        Args:
            chunk: The text chunk to evaluate.

        Returns:
            True if the chunk is valid for synthesis, False otherwise.
        """
        # Minimum character count to be considered a useful context
        if len(chunk) < 50:
            return False
        return True

    def _sanitize(self, text: str) -> tuple[str, bool]:
        """Sanitizes PII from the text using Regex.

        Args:
            text: The text to sanitize.

        Returns:
            A tuple containing (sanitized_text, was_redacted).
        """
        sanitized_text = text
        redacted = False

        # Regex Patterns
        # Note: Order matters for overlapping patterns, though these are mostly distinct.
        patterns = {
            "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            # Simple US SSN: 000-00-0000
            "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
            # Phone: (123) 456-7890, 123-456-7890, 123.456.7890. Captures simple variants.
            # Uses \(?\b to handle optional parenthesis before the boundary check for the number.
            "PHONE": r"(?:\+?1[-. ]?)?\(?\b\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b",
            # MRN: Generic alphanumeric pattern as per specification (e.g., AB123456)
            # Matches 2-3 uppercase letters followed by 6-9 digits
            "MRN": r"\b[A-Z]{2,3}\d{6,9}\b",
        }

        for label, pattern in patterns.items():
            if re.search(pattern, sanitized_text):
                # Replacement uses [LABEL] instead of [LABEL_REDACTED]
                sanitized_text = re.sub(pattern, f"[{label}]", sanitized_text)
                redacted = True

        return sanitized_text, redacted
