"""
parsers/cfr_parser.py

Fetches and parses CFR titles from the eCFR XML API (no API key needed).
API docs: https://www.ecfr.gov/developers/documentation/api/v1

Usage:
    from ingestion.parsers.cfr_parser import fetch_cfr_title
    sections = fetch_cfr_title(title=47, date="2024-01-01")
"""
import httpx
import logging
from xml.etree import ElementTree as ET
from dataclasses import dataclass
from datetime import date

logger = logging.getLogger(__name__)

ECFR_XML_BASE = "https://www.ecfr.gov/api/versioner/v1/full"


@dataclass
class CFRSection:
    title_num: str
    part: str
    section_num: str
    section_name: str
    text: str


def fetch_cfr_title(
    title: int,
    fetch_date: str | None = None,
    part: str | None = None,
) -> list[CFRSection]:
    """
    Fetch a full CFR title (or a single part) as XML and parse into sections.

    Args:
        title:      CFR title number (1–50)
        fetch_date: ISO date string e.g. "2024-01-01". Defaults to today.
        part:       Optional part number to narrow download (e.g. "230")
    """
    if fetch_date is None:
        fetch_date = date.today().isoformat()

    url = f"{ECFR_XML_BASE}/{fetch_date}/title-{title}.xml"
    params = {}
    if part:
        params["part"] = part

    logger.info(f"Fetching CFR Title {title} from eCFR API ({fetch_date})…")
    try:
        r = httpx.get(url, params=params, timeout=60, follow_redirects=True)
        r.raise_for_status()
    except Exception as e:
        logger.error(f"Failed to fetch CFR title {title}: {e}")
        return []

    return _parse_cfr_xml(r.text, str(title))


def _parse_cfr_xml(xml_text: str, title_num: str) -> list[CFRSection]:
    """Parse eCFR XML into a flat list of CFRSection objects."""
    try:
        root = ET.fromstring(xml_text.encode())
    except ET.ParseError as e:
        logger.error(f"XML parse error: {e}")
        return []

    sections = []
    # eCFR XML uses <PART>, <SECTION>, <SUBJECT>, <P> tags
    for part_elem in root.iter("PART"):
        part_num_elem = part_elem.find("EAR")
        part_num = part_num_elem.text.strip() if part_num_elem is not None else "?"

        for sec_elem in part_elem.iter("SECTION"):
            sec_num_elem = sec_elem.find("SECTNO")
            sec_name_elem = sec_elem.find("SUBJECT")

            sec_num = sec_num_elem.text.strip() if sec_num_elem is not None else "?"
            sec_name = sec_name_elem.text.strip() if sec_name_elem is not None else ""

            paragraphs = []
            for p in sec_elem.iter("P"):
                text = "".join(p.itertext()).strip()
                if text:
                    paragraphs.append(text)

            full_text = f"{sec_num} {sec_name}\n\n" + "\n\n".join(paragraphs)
            if len(full_text.strip()) < 80:
                continue

            sections.append(CFRSection(
                title_num=title_num,
                part=part_num,
                section_num=sec_num,
                section_name=sec_name,
                text=full_text.strip(),
            ))

    logger.info(f"Parsed {len(sections)} CFR sections from title {title_num}.")
    return sections
