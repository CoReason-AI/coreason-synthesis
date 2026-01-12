import pytest
from coreason_synthesis.extractor import ExtractorImpl
from coreason_synthesis.models import Document, SynthesisTemplate

@pytest.fixture
def extractor():
    return ExtractorImpl()

@pytest.fixture
def empty_template():
    return SynthesisTemplate(
        structure="test",
        complexity_description="test",
        domain="test",
        embedding_centroid=[0.1]
    )

class TestExtractorPII:
    """Test suite for PII sanitization in ExtractorImpl."""

    def test_sanitize_email(self, extractor):
        text = "Contact us at support@coreason.ai for help."
        sanitized, redacted = extractor._sanitize(text)
        assert redacted is True
        assert sanitized == "Contact us at [EMAIL] for help."

    def test_sanitize_ssn(self, extractor):
        text = "Patient SSN is 123-45-6789 in the file."
        sanitized, redacted = extractor._sanitize(text)
        assert redacted is True
        assert sanitized == "Patient SSN is [SSN] in the file."

    def test_sanitize_phone(self, extractor):
        variants = [
            ("Call (555) 123-4567 now.", "Call [PHONE] now."),
            ("Call 555-123-4567 now.", "Call [PHONE] now."),
            ("Call 555.123.4567 now.", "Call [PHONE] now."),
        ]
        for original, expected in variants:
            sanitized, redacted = extractor._sanitize(original)
            assert redacted is True
            assert sanitized == expected

    def test_sanitize_mrn(self, extractor):
        # Matches pattern \b[A-Z]{2,3}\d{6,9}\b
        variants = [
            ("Patient ID: AB123456 verified.", "Patient ID: [MRN] verified."),
            ("Ref XYZ123456789.", "Ref [MRN]."),
        ]
        for original, expected in variants:
            sanitized, redacted = extractor._sanitize(original)
            assert redacted is True
            assert sanitized == expected

    def test_sanitize_multiple_pii(self, extractor):
        text = "User john.doe@example.com (SSN: 987-65-4320) called 555-555-0199."
        sanitized, redacted = extractor._sanitize(text)
        assert redacted is True
        assert "[EMAIL]" in sanitized
        assert "[SSN]" in sanitized
        assert "[PHONE]" in sanitized
        # Order of replacement might vary but all should be present
        expected = "User [EMAIL] (SSN: [SSN]) called [PHONE]."
        assert sanitized == expected

    def test_no_pii(self, extractor):
        text = "This is a safe string with no PII 12345."
        sanitized, redacted = extractor._sanitize(text)
        assert redacted is False
        assert sanitized == text

    def test_mrn_edge_cases(self, extractor):
        # Should NOT match:
        negatives = [
            "A123456",       # Only 1 letter
            "ABCD123456",    # 4 letters
            "AB12345",       # 5 digits
            "AB1234567890",  # 10 digits
        ]
        for text in negatives:
            sanitized, redacted = extractor._sanitize(text)
            assert redacted is False, f"Matched incorrect MRN: {text}"
            assert sanitized == text

    def test_extract_integration(self, extractor, empty_template):
        """Verify _sanitize is called during extract workflow."""
        # Content must be > 50 chars to pass _is_valid_chunk
        doc = Document(
            content="Sensitive info: AB123456 needs redaction. This is a long enough paragraph to pass the chunk filter.\n\nParagraph 2 is also long enough to pass the filter if we add more text to it effectively.",
            source_urn="test://doc1",
            metadata={}
        )

        slices = extractor.extract([doc], template=empty_template)

        assert len(slices) == 2
        assert slices[0].pii_redacted is True
        assert "Sensitive info: [MRN] needs redaction." in slices[0].content
        assert slices[1].pii_redacted is False

    def test_chunking_edge_cases(self, extractor):
        """Test _chunk_content with empty input."""
        assert extractor._chunk_content("") == []
        assert extractor._chunk_content(None) == []

    def test_valid_chunk_filtering(self, extractor, empty_template):
        """Test that short chunks are filtered out."""
        # "Short" is < 50 chars. "Long" is >= 50 chars.
        short = "Too short."
        long_chunk = "This is a sufficiently long chunk that should be kept by the extractor logic."

        doc = Document(
            content=f"{short}\n\n{long_chunk}",
            source_urn="test",
            metadata={}
        )

        slices = extractor.extract([doc], template=empty_template)

        assert len(slices) == 1
        assert slices[0].content == long_chunk
        assert slices[0].metadata["chunk_index"] == 1  # Verify it was the second chunk

    def test_overlapping_pii(self, extractor):
        """Test strict priority when PII patterns overlap."""
        # Email "phone@example.com" contains "phone" but not a phone number pattern.
        # But consider an email that looks like a phone number? unlikely.
        # What about an email that contains digits matching SSN?
        # e.g. "user.123-45-6789@example.com"
        # This matches Email regex. It also matches SSN regex inside.
        # Since we use re.sub sequentially (Email first in dictionary iteration usually, but let's see),
        # If Email matches first, it consumes the text.
        # If SSN matches first, it breaks the email.
        # Python 3.7+ dicts preserve insertion order.
        # In ExtractorImpl, "EMAIL" is first. So it should win.
        text = "Contact user.123-45-6789@example.com for details."
        sanitized, redacted = extractor._sanitize(text)
        assert sanitized == "Contact [EMAIL] for details."
        assert "[SSN]" not in sanitized

    def test_pii_punctuation_adjacency(self, extractor):
        """Test PII adjacent to punctuation."""
        variants = [
            ("Call me at 555-123-4567.", "Call me at [PHONE]."),
            ("My SSN (123-45-6789) is secret.", "My SSN ([SSN]) is secret."),
            ("Email: test@example.com, or call.", "Email: [EMAIL], or call."),
            ("Check MRN: AB123456!", "Check MRN: [MRN]!"),
        ]
        for original, expected in variants:
            sanitized, redacted = extractor._sanitize(original)
            assert sanitized == expected, f"Failed for '{original}'"

    def test_complex_medical_narrative(self, extractor):
        """
        Simulates a complex clinical note with mixed PII, dates, and numbers.
        """
        note = (
            "Patient Name: John Doe (DOB: 01/01/1980)\n"
            "MRN: XY999999 seen in clinic on 12/12/2023.\n"
            "Reported valid contact: 555-010-9999 and backup 555.010.9999.\n"
            "Labs sent to lab-results@hospital.org.\n"
            "SSN 000-12-3456 verified on intake.\n"
            "Reference ID 12345 (not PII) and Room 404."
        )

        sanitized, redacted = extractor._sanitize(note)

        assert redacted is True

        # Check PII is gone
        assert "[MRN]" in sanitized
        assert "XY999999" not in sanitized

        assert "[PHONE]" in sanitized
        assert "555-010-9999" not in sanitized

        assert "[EMAIL]" in sanitized
        assert "lab-results@hospital.org" not in sanitized

        assert "[SSN]" in sanitized
        assert "000-12-3456" not in sanitized

        # Check Context is preserved
        assert "Patient Name: John Doe (DOB: 01/01/1980)" in sanitized # Names/Dates not redacted yet per spec
        assert "Reference ID 12345 (not PII) and Room 404" in sanitized

        # Verify multiple replacements
        assert sanitized.count("[PHONE]") == 2

    def test_large_document_performance(self, extractor):
        """Basic check for processing larger text blocks with PII."""
        # Create a text with 100 lines, each having an email
        base_line = "User user{}@example.com logged in at 12:00.\n"
        content = "".join([base_line.format(i) for i in range(100)])

        sanitized, redacted = extractor._sanitize(content)

        assert redacted is True
        assert sanitized.count("[EMAIL]") == 100
        assert "user0@example.com" not in sanitized
        assert "user99@example.com" not in sanitized
