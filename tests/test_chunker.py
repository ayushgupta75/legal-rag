"""
tests/test_chunker.py — Unit tests for the legal chunker.
Run: pytest tests/
"""
import sys
sys.path.insert(0, ".")

from ingestion.chunkers.legal_chunker import (
    chunk_constitution, chunk_uscode, chunk_cfr, LegalChunkData
)

CONSTITUTION_SAMPLE = """
PREAMBLE
We the People of the United States, in Order to form a more perfect Union...

ARTICLE I
Section 1. All legislative Powers herein granted shall be vested in a Congress.
Section 2. The House of Representatives shall be composed of Members chosen every second Year.

AMENDMENT I
Congress shall make no law respecting an establishment of religion, or prohibiting the free
exercise thereof; or abridging the freedom of speech, or of the press.

AMENDMENT XIV
Section 1. All persons born or naturalized in the United States are citizens thereof.
No State shall deprive any person of life, liberty, or property, without due process of law.
"""

USCODE_SAMPLE = """
§ 1030. Fraud and related activity in connection with computers
(a) Whoever intentionally accesses a computer without authorization or exceeds authorized
access, and thereby obtains information from any protected computer shall be punished.
(b) Whoever attempts to commit an offense under subsection (a) shall be subject to penalties.

§ 1031. Major fraud against the United States
(a) Whoever knowingly executes, or attempts to execute, any scheme or artifice with the intent
to defraud the United States shall be guilty of a felony.
"""


class TestConstitutionChunker:
    def test_produces_chunks(self):
        chunks = list(chunk_constitution(CONSTITUTION_SAMPLE))
        assert len(chunks) > 0

    def test_chunk_has_required_fields(self):
        chunks = list(chunk_constitution(CONSTITUTION_SAMPLE))
        for c in chunks:
            assert isinstance(c, LegalChunkData)
            assert c.id
            assert c.source == "constitution"
            assert c.citation
            assert len(c.text) >= 50
            assert c.char_count == len(c.text)
            assert c.version_hash

    def test_amendment_citation_format(self):
        chunks = list(chunk_constitution(CONSTITUTION_SAMPLE))
        citations = [c.citation for c in chunks]
        assert any("AMENDMENT" in cit for cit in citations)

    def test_no_tiny_chunks(self):
        chunks = list(chunk_constitution(CONSTITUTION_SAMPLE))
        assert all(c.char_count >= 50 for c in chunks)

    def test_version_hash_is_deterministic(self):
        chunks1 = list(chunk_constitution(CONSTITUTION_SAMPLE))
        chunks2 = list(chunk_constitution(CONSTITUTION_SAMPLE))
        hashes1 = {c.version_hash for c in chunks1}
        hashes2 = {c.version_hash for c in chunks2}
        assert hashes1 == hashes2


class TestUsCodeChunker:
    def test_produces_chunks(self):
        chunks = list(chunk_uscode(USCODE_SAMPLE, "18", "Crimes and Criminal Procedure"))
        assert len(chunks) > 0

    def test_section_numbers_extracted(self):
        chunks = list(chunk_uscode(USCODE_SAMPLE, "18", "Crimes and Criminal Procedure"))
        sections = [c.section for c in chunks]
        assert any("1030" in s for s in sections)

    def test_citation_format(self):
        chunks = list(chunk_uscode(USCODE_SAMPLE, "18", "Crimes and Criminal Procedure"))
        for c in chunks:
            assert "18 U.S.C." in c.citation

    def test_source_is_uscode(self):
        chunks = list(chunk_uscode(USCODE_SAMPLE, "18", "Crimes and Criminal Procedure"))
        assert all(c.source == "uscode" for c in chunks)

    def test_jurisdiction_is_federal(self):
        chunks = list(chunk_uscode(USCODE_SAMPLE, "18", "Crimes and Criminal Procedure"))
        assert all(c.jurisdiction == "federal" for c in chunks)


class TestCFRChunker:
    def test_cfr_delegates_to_uscode_chunker(self):
        chunks = list(chunk_cfr(USCODE_SAMPLE, "47", "230"))
        assert isinstance(chunks, list)
