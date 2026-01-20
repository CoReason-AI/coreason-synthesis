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
def tmpl() -> SynthesisTemplate:
    return SynthesisTemplate(structure="", complexity_description="", domain="", embedding_centroid=[])


@pytest.mark.asyncio
async def test_mixed_line_endings(extractor: ExtractorImpl, tmpl: SynthesisTemplate) -> None:
    # \r\n vs \n
    text = ("A" * 60) + "\r\n\r\n" + ("B" * 60)
    doc = Document(content=text, source_urn="u")

    slices = await extractor.extract([doc], tmpl)
    assert len(slices) == 2


@pytest.mark.asyncio
async def test_giant_text_block(extractor: ExtractorImpl, tmpl: SynthesisTemplate) -> None:
    # No newlines, huge block
    text = "A" * 10000
    doc = Document(content=text, source_urn="u")

    slices = await extractor.extract([doc], tmpl)
    assert len(slices) == 1
    assert len(slices[0].content) == 10000


@pytest.mark.asyncio
async def test_pii_edge_cases_false_positives(extractor: ExtractorImpl, tmpl: SynthesisTemplate) -> None:
    # Email-like but not email? "user@domain" is valid usually.
    # Phone-like: "version 1.2.3.4"
    text = "Version 1.2.3.4 released today."
    padded = text.ljust(60, ".")
    doc = Document(content=padded, source_urn="u")

    slices = await extractor.extract([doc], tmpl)
    assert len(slices) == 1
    # Should NOT be redacted (regex for phone usually requires dashes/parens or specific length)
    assert "[PHONE]" not in slices[0].content


@pytest.mark.asyncio
async def test_pii_case_insensitivity(extractor: ExtractorImpl, tmpl: SynthesisTemplate) -> None:
    # Email is case insensitive
    text = "MAIL@EXAMPLE.COM"
    padded = text.ljust(60, ".")
    doc = Document(content=padded, source_urn="u")

    slices = await extractor.extract([doc], tmpl)
    assert "[EMAIL]" in slices[0].content


@pytest.mark.asyncio
async def test_messy_ocr_scenario(extractor: ExtractorImpl, tmpl: SynthesisTemplate) -> None:
    # Garbage chars
    text = " Some text " * 10
    doc = Document(content=text, source_urn="u")
    slices = await extractor.extract([doc], tmpl)
    assert len(slices) == 1
