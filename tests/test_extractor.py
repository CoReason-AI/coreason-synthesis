import pytest

from coreason_synthesis.extractor import ExtractorImpl
from coreason_synthesis.models import Document, SynthesisTemplate


@pytest.fixture
def extractor() -> ExtractorImpl:
    return ExtractorImpl()


@pytest.fixture
def dummy_template() -> SynthesisTemplate:
    return SynthesisTemplate(
        structure="Q+A", complexity_description="Medium", domain="Test", embedding_centroid=[0.1, 0.2]
    )


def test_chunking_logic(extractor: ExtractorImpl, dummy_template: SynthesisTemplate) -> None:
    """Test that content is correctly split into paragraphs."""
    content = "Paragraph 1.\n\nParagraph 2 is longer.\n\nParagraph 3."
    doc = Document(content=content, source_urn="urn:test", metadata={})

    slices = extractor.extract([doc], dummy_template)

    # Paragraph 1 is too short (<50 chars) -> "Paragraph 1." (12 chars)
    # Paragraph 2 is "Paragraph 2 is longer." (22 chars) -> Still too short?
    # Let's check the implementation limit: 50.
    # We need longer paragraphs for test.

    content_long = (
        "This is paragraph 1 which is definitely long enough to be considered a valid chunk "
        "for our extraction logic.\n\n"
        "This is paragraph 2 also long enough. It contains some useful info for the test case generation.\n\n"
        "Short."
    )
    doc_long = Document(content=content_long, source_urn="urn:test", metadata={})

    slices = extractor.extract([doc_long], dummy_template)

    assert len(slices) == 2
    assert slices[0].content.startswith("This is paragraph 1")
    assert slices[1].content.startswith("This is paragraph 2")
    assert slices[0].source_urn == "urn:test"
    assert not slices[0].pii_redacted


def test_pii_sanitization(extractor: ExtractorImpl, dummy_template: SynthesisTemplate) -> None:
    """Test PII redaction for Email, SSN, Phone, MRN."""
    pii_text = (
        "Patient email is test@example.com and phone is 555-123-4567. "
        "Their SSN is 123-45-6789. Also MRN: AB987654321 found. "
        "This text must be long enough to pass the filter limit of fifty characters to be extracted."
    )
    doc = Document(content=pii_text, source_urn="urn:pii", metadata={})

    slices = extractor.extract([doc], dummy_template)

    assert len(slices) == 1
    sanitized = slices[0].content

    assert "[EMAIL]" in sanitized
    assert "test@example.com" not in sanitized

    assert "[PHONE]" in sanitized
    assert "555-123-4567" not in sanitized

    assert "[SSN]" in sanitized
    assert "123-45-6789" not in sanitized

    assert "[MRN]" in sanitized
    assert "AB987654321" not in sanitized

    assert slices[0].pii_redacted is True


def test_no_pii_no_redaction(extractor: ExtractorImpl, dummy_template: SynthesisTemplate) -> None:
    """Test that clean text is not flagged as redacted."""
    clean_text = (
        "This is a clean paragraph with no personal information. "
        "It should be extracted exactly as is without any modification flags. "
        "Adding length to ensure it passes the length filter."
    )
    doc = Document(content=clean_text, source_urn="urn:clean", metadata={})

    slices = extractor.extract([doc], dummy_template)

    assert len(slices) == 1
    assert slices[0].content == clean_text
    assert slices[0].pii_redacted is False


def test_empty_document(extractor: ExtractorImpl, dummy_template: SynthesisTemplate) -> None:
    """Test handling of empty documents."""
    doc = Document(content="", source_urn="urn:empty", metadata={})
    slices = extractor.extract([doc], dummy_template)
    assert len(slices) == 0


def test_short_chunks_ignored(extractor: ExtractorImpl, dummy_template: SynthesisTemplate) -> None:
    """Test that short chunks are filtered out."""
    content = "Short.\n\nAlso short.\n\nTiny."
    doc = Document(content=content, source_urn="urn:short", metadata={})
    slices = extractor.extract([doc], dummy_template)
    assert len(slices) == 0


def test_metadata_passthrough(extractor: ExtractorImpl, dummy_template: SynthesisTemplate) -> None:
    """Test that page numbers and metadata are preserved."""
    content = (
        "This is a valid paragraph that should be extracted and have metadata attached. "
        "It is definitely long enough to pass the filter."
    )
    doc = Document(content=content, source_urn="urn:meta", metadata={"page_number": 42, "author": "Jules"})

    slices = extractor.extract([doc], dummy_template)

    assert len(slices) == 1
    assert slices[0].page_number == 42
    # Metadata includes chunk index and lengths
    assert slices[0].metadata["chunk_index"] == 0
    assert slices[0].metadata["original_length"] == len(content)
