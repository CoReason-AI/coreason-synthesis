import re
from typing import List

from .interfaces import Extractor
from .models import Document, ExtractedSlice, SynthesisTemplate


class ExtractorImpl(Extractor):
    """
    Concrete implementation of the Extractor.
    Mines text slices using heuristic chunking and sanitizes PII.
    """

    def extract(self, documents: List[Document], template: SynthesisTemplate) -> List[ExtractedSlice]:
        """
        Extracts text slices from documents.
        Applies PII sanitization and maps back to source.
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
        """
        Splits content into paragraphs based on double newlines.
        Handles mixed line endings by normalizing to \n.
        """
        if not content:
            return []
        # Normalize line endings
        normalized = content.replace("\r\n", "\n").replace("\r", "\n")
        # Split by double newline to identify paragraphs
        # Filter out empty strings after strip
        return [c.strip() for c in normalized.split("\n\n") if c.strip()]

    def _is_valid_chunk(self, chunk: str) -> bool:
        """
        Filters out chunks that are too short or irrelevant.
        """
        # Minimum character count to be considered a useful context
        if len(chunk) < 50:
            return False
        return True

    def _sanitize(self, text: str) -> tuple[str, bool]:
        """
        Sanitizes PII from the text using Regex.
        Returns (sanitized_text, was_redacted).
        """
        sanitized_text = text
        redacted = False

        # Regex Patterns
        patterns = {
            "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            # Simple US SSN: 000-00-0000
            "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
            # Phone: (123) 456-7890 or 123-456-7890. Captures simple variants.
            "PHONE": r"\b(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b",
            # Generic ID pattern (e.g., MRN: 123456) - avoiding false positives is hard without context
            # We will target explicit labels if possible, or just sequences of digits if they look like IDs
            # For this iteration, let's target labeled IDs often seen in medical docs: "MRN: 12345"
            "MRN": r"\b(MRN|ID)[:#]?\s*\d+\b",
        }

        for label, pattern in patterns.items():
            if re.search(pattern, sanitized_text):
                sanitized_text = re.sub(pattern, f"[{label}_REDACTED]", sanitized_text)
                redacted = True

        return sanitized_text, redacted
