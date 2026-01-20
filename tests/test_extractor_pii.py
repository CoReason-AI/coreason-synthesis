# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_synthesis

import pytest

from coreason_synthesis.extractor import ExtractorImpl
from coreason_synthesis.models import Document, SynthesisTemplate


class TestExtractorPII:
    @pytest.fixture
    def extractor(self) -> ExtractorImpl:
        return ExtractorImpl()

    @pytest.fixture
    def template(self) -> SynthesisTemplate:
        return SynthesisTemplate(
            structure="Q+A",
            complexity_description="Medium",
            domain="Test",
            embedding_centroid=[0.1],
        )

    @pytest.mark.asyncio
    async def test_extract_integration(self, extractor: ExtractorImpl, template: SynthesisTemplate) -> None:
        """Integration test for extraction flow."""
        raw_text = "Patient John Doe (MRN: AB123456) was admitted.\n\nContact: john.doe@example.com or 555-123-4567.\n"
        # Pad to ensure validity > 50 chars
        raw_text = raw_text.ljust(100, " ")

        doc = Document(content=raw_text, source_urn="urn:test:doc1")
        slices = await extractor.extract([doc], template)

        assert len(slices) >= 1
        content = slices[0].content

        # Check PII redaction
        assert "AB123456" not in content
        assert "[MRN]" in content
        assert "john.doe@example.com" not in content
        assert "[EMAIL]" in content
        assert "555-123-4567" not in content
        assert "[PHONE]" in content

    @pytest.mark.asyncio
    async def test_valid_chunk_filtering(self, extractor: ExtractorImpl, template: SynthesisTemplate) -> None:
        """Test filtering of short chunks."""
        short_text = "Short."
        long_text = "This is a long enough chunk to be preserved." * 5

        doc = Document(content=f"{short_text}\n\n{long_text}", source_urn="urn:test:doc2")
        slices = await extractor.extract([doc], template)

        assert len(slices) == 1
        assert slices[0].content == long_text
