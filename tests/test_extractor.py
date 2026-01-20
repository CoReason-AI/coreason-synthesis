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


@pytest.fixture
def extractor() -> ExtractorImpl:
    return ExtractorImpl()


@pytest.fixture
def sample_template() -> SynthesisTemplate:
    return SynthesisTemplate(
        structure="Q",
        complexity_description="L",
        domain="D",
        embedding_centroid=[],
    )


@pytest.mark.asyncio
async def test_chunking_logic(extractor: ExtractorImpl, sample_template: SynthesisTemplate) -> None:
    # We need chunks > 50 chars
    long_text = "A" * 60 + "\n\n" + "B" * 60
    doc = Document(content=long_text, source_urn="u1")

    slices = await extractor.extract([doc], sample_template)

    assert len(slices) == 2
    assert slices[0].content == "A" * 60
    assert slices[1].content == "B" * 60


@pytest.mark.asyncio
async def test_pii_sanitization(extractor: ExtractorImpl, sample_template: SynthesisTemplate) -> None:
    pii_text = "My email is test@example.com and ssn 123-45-6789 here."
    # Pad to > 50 chars
    content = pii_text.ljust(60, ".")
    doc = Document(content=content, source_urn="u1")

    slices = await extractor.extract([doc], sample_template)

    assert len(slices) == 1
    assert "[EMAIL]" in slices[0].content
    assert "[SSN]" in slices[0].content
    assert slices[0].pii_redacted is True


@pytest.mark.asyncio
async def test_no_pii_no_redaction(extractor: ExtractorImpl, sample_template: SynthesisTemplate) -> None:
    clean_text = "Just some safe text " * 5  # > 50 chars
    # Ensure no trailing space that might get stripped
    clean_text = clean_text.strip()
    doc = Document(content=clean_text, source_urn="u1")

    slices = await extractor.extract([doc], sample_template)

    assert len(slices) == 1
    assert slices[0].content == clean_text
    assert slices[0].pii_redacted is False


@pytest.mark.asyncio
async def test_empty_document(extractor: ExtractorImpl, sample_template: SynthesisTemplate) -> None:
    doc = Document(content="", source_urn="u1")
    slices = await extractor.extract([doc], sample_template)
    assert slices == []


@pytest.mark.asyncio
async def test_short_chunks_ignored(extractor: ExtractorImpl, sample_template: SynthesisTemplate) -> None:
    short_text = "Too short"
    doc = Document(content=short_text, source_urn="u1")
    slices = await extractor.extract([doc], sample_template)
    assert slices == []


@pytest.mark.asyncio
async def test_metadata_passthrough(extractor: ExtractorImpl, sample_template: SynthesisTemplate) -> None:
    content = "A" * 60
    doc = Document(content=content, source_urn="u1", metadata={"page_number": 5})

    slices = await extractor.extract([doc], sample_template)

    assert len(slices) == 1
    assert slices[0].page_number == 5
    assert slices[0].metadata["original_length"] == 60
