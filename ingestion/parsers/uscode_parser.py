"""
parsers/uscode_parser.py

Parses US Code from:
  - PDF files downloaded from uscode.house.gov  (primary)
  - XML files in USLM format                    (fallback)
  - Plain text                                  (fallback)

Usage:
    from ingestion.parsers.uscode_parser import parse_uscode_pdf
    sections = parse_uscode_pdf("data/uscode/usc18.pdf", "18", "Crimes and Criminal Procedure")
"""
import re
import logging
from pathlib import Path
from xml.etree import ElementTree as ET
from dataclasses import dataclass
import fitz  # pymupdf

logger = logging.getLogger(__name__)


@dataclass
class RawSection:
    title_num: str
    title_name: str
    section_num: str
    section_name: str
    text: str


def _clean(text: str) -> str:
    """Strip excessive whitespace and XML artifacts."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_uscode_xml(xml_path: str | Path) -> list[RawSection]:
    """
    Parse a single US Code XML file (USLM format from uscode.house.gov).
    Returns a flat list of RawSection, one per statutory section.
    """
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    # USLM namespace
    ns = {"uslm": "http://xml.house.gov/schemas/uslm/1.0"}

    # Pull title metadata
    title_elem = root.find(".//uslm:title", ns) or root
    title_num = title_elem.get("identifier", "").lstrip("/").split("/")[0] or "?"
    title_name_elem = root.find(".//uslm:heading", ns)
    title_name = _clean(title_name_elem.text or "") if title_name_elem is not None else ""

    sections = []
    for sec in root.iter("{http://xml.house.gov/schemas/uslm/1.0}section"):
        sec_id = sec.get("identifier", "")
        sec_num = sec_id.rsplit("/s", 1)[-1] if "/s" in sec_id else "?"

        heading = sec.find("{http://xml.house.gov/schemas/uslm/1.0}heading")
        sec_name = _clean(heading.text or "") if heading is not None else ""

        # Collect all text recursively
        raw_text = " ".join(t for t in sec.itertext() if t.strip())
        cleaned = _clean(raw_text)

        if len(cleaned) < 80:
            continue

        sections.append(RawSection(
            title_num=title_num,
            title_name=title_name,
            section_num=sec_num,
            section_name=sec_name,
            text=cleaned,
        ))

    return sections


def parse_uscode_pdf(pdf_path: str | Path, title_num: str, title_name: str) -> list[RawSection]:
    """
    Extract text from a US Code PDF (from uscode.house.gov) and split into sections.
    Uses PyMuPDF for accurate text extraction, then splits on § markers.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        return []

    logger.info(f"Extracting text from {pdf_path.name} ({pdf_path.stat().st_size // 1024} KB)…")
    doc = fitz.open(str(pdf_path))

    pages_text = []
    for page in doc:
        text = page.get_text("text")
        # Strip page headers/footers: lines that are just a number or short boilerplate
        lines = text.splitlines()
        cleaned = []
        for line in lines:
            stripped = line.strip()
            # Skip bare page numbers and common header/footer patterns
            if re.fullmatch(r"\d{1,4}", stripped):
                continue
            if re.match(r"^(TITLE\s+\d+—|Sec\.\s+\d+\.|Page \d+)", stripped):
                continue
            cleaned.append(line)
        pages_text.append("\n".join(cleaned))

    doc.close()
    full_text = "\n".join(pages_text)
    logger.info(f"  Extracted {len(full_text):,} characters from {len(pages_text)} pages.")
    return parse_uscode_plaintext(full_text, title_num, title_name)


def parse_uscode_plaintext(text: str, title_num: str, title_name: str) -> list[RawSection]:
    """
    Fallback: parse plain-text US Code (e.g. copied from Cornell LII).
    Splits on § markers.
    """
    pattern = re.compile(r"§\s*([\d\w.–-]+)\s*[.–-]?\s*(.*?)(?=§|\Z)", re.S)
    sections = []
    for m in pattern.finditer(text):
        sec_num = m.group(1).strip()
        body = m.group(0).strip()
        name_match = re.match(r"§\s*[\d\w.–-]+\s*[.–-]?\s*(.+?)[\n\r]", body)
        sec_name = name_match.group(1).strip() if name_match else ""
        if len(body) < 80:
            continue
        sections.append(RawSection(
            title_num=title_num,
            title_name=title_name,
            section_num=sec_num,
            section_name=sec_name,
            text=body,
        ))
    return sections
