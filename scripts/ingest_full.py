"""
scripts/ingest_full.py

Full ingestion pipeline:
  1. US Constitution (plain text via govinfo.gov)
  2. US Code titles (XML from uscode.house.gov)
  3. CFR titles (XML from eCFR API)

Run with:
    python scripts/ingest_full.py --source all
    python scripts/ingest_full.py --source constitution
    python scripts/ingest_full.py --source uscode --titles 18 42
    python scripts/ingest_full.py --source cfr --titles 47
"""
import sys
import argparse
import logging
from pathlib import Path
sys.path.insert(0, ".")

from sqlalchemy import text
from ingestion.db import init_db, engine
from ingestion.chunkers.legal_chunker import (
    chunk_constitution, chunk_uscode, chunk_cfr
)
from ingestion.embedders.embedder import upsert_chunks
from ingestion.parsers.cfr_parser import fetch_cfr_title
from ingestion.parsers.uscode_parser import parse_uscode_pdf

USCODE_PDF_DIR = Path("data/pdf_uscAll@119-73")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# All active US Code titles (Title 34 repealed/transferred, Title 53 reserved)
USCODE_TITLES = {
    "1":  "General Provisions",
    "2":  "The Congress",
    "3":  "The President",
    "4":  "Flag and Seal, Seat of Government, and the States",
    "5":  "Government Organization and Employees",
    "6":  "Domestic Security",
    "7":  "Agriculture",
    "8":  "Aliens and Nationality",
    "9":  "Arbitration",
    "10": "Armed Forces",
    "11": "Bankruptcy",
    "12": "Banks and Banking",
    "13": "Census",
    "14": "Coast Guard",
    "15": "Commerce and Trade",
    "16": "Conservation",
    "17": "Copyrights",
    "18": "Crimes and Criminal Procedure",
    "19": "Customs Duties",
    "20": "Education",
    "21": "Food and Drugs",
    "22": "Foreign Relations and Intercourse",
    "23": "Highways",
    "24": "Hospitals and Asylums",
    "25": "Indians",
    "26": "Internal Revenue Code",
    "27": "Intoxicating Liquors",
    "28": "Judiciary and Judicial Procedure",
    "29": "Labor",
    "30": "Mineral Lands and Mining",
    "31": "Money and Finance",
    "32": "National Guard",
    "33": "Navigation and Navigable Waters",
    "34": "Crime Control and Law Enforcement",
    "35": "Patents",
    "36": "Patriotic and National Observances, Ceremonies, and Organizations",
    "37": "Pay and Allowances of the Uniformed Services",
    "38": "Veterans Benefits",
    "39": "Postal Service",
    "40": "Public Buildings, Property, and Works",
    "41": "Public Contracts",
    "42": "The Public Health and Welfare",
    "43": "Public Lands",
    "44": "Public Printing and Documents",
    "45": "Railroads",
    "46": "Shipping",
    "47": "Telecommunications",
    "48": "Territories and Insular Possessions",
    "49": "Transportation",
    "50": "War and National Defense",
    "51": "National and Commercial Space Programs",
    "52": "Voting and Elections",
    "54": "National Park Service and Related Programs",
}

# All 50 CFR titles
CFR_TITLES = {
    "1":  "General Provisions",
    "2":  "Grants and Agreements",
    "3":  "The President",
    "4":  "Accounts",
    "5":  "Administrative Personnel",
    "6":  "Domestic Security",
    "7":  "Agriculture",
    "8":  "Aliens and Nationality",
    "9":  "Animals and Animal Products",
    "10": "Energy",
    "11": "Federal Elections",
    "12": "Banks and Banking",
    "13": "Business Credit and Assistance",
    "14": "Aeronautics and Space",
    "15": "Commerce and Foreign Trade",
    "16": "Commercial Practices",
    "17": "Commodity and Securities Exchanges",
    "18": "Conservation of Power and Water Resources",
    "19": "Customs Duties",
    "20": "Employees Benefits",
    "21": "Food and Drugs",
    "22": "Foreign Relations",
    "23": "Highways",
    "24": "Housing and Urban Development",
    "25": "Indians",
    "26": "Internal Revenue",
    "27": "Alcohol, Tobacco Products and Firearms",
    "28": "Judicial Administration",
    "29": "Labor",
    "30": "Mineral Resources",
    "31": "Money and Finance: Treasury",
    "32": "National Defense",
    "33": "Navigation and Navigable Waters",
    "34": "Education",
    "35": "Panama Canal",
    "36": "Parks, Forests, and Public Property",
    "37": "Patents, Trademarks, and Copyrights",
    "38": "Pensions, Bonuses, and Veterans Relief",
    "39": "Postal Service",
    "40": "Protection of Environment",
    "41": "Public Contracts and Property Management",
    "42": "Public Health",
    "43": "Public Lands: Interior",
    "44": "Emergency Management and Assistance",
    "45": "Public Welfare",
    "46": "Shipping",
    "47": "Telecommunication",
    "48": "Federal Acquisition Regulations System",
    "49": "Transportation",
    "50": "Wildlife and Fisheries",
}

CONSTITUTION_FILE = "data/constitution.txt"


def drop_hnsw_index():
    with engine.connect() as conn:
        conn.execute(text("DROP INDEX IF EXISTS legal_chunks_embedding_idx"))
        conn.commit()
    logger.info("HNSW index dropped — inserts will be fast.")


def rebuild_hnsw_index():
    logger.info("Rebuilding HNSW index over all data (this takes a minute)…")
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS legal_chunks_embedding_idx
            ON legal_chunks
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        """))
        conn.commit()
    logger.info("HNSW index ready.")


def ingest_constitution():
    logger.info("── Ingesting US Constitution ──")
    with open(CONSTITUTION_FILE, "r", encoding="utf-8") as f:
        constitution_text = f.read()
    chunks = list(chunk_constitution(constitution_text))
    logger.info(f"  {len(chunks)} chunks")
    n = upsert_chunks(chunks)
    logger.info(f"  {n} inserted")


def ingest_uscode(title_nums: list[str]):
    for num in title_nums:
        name = USCODE_TITLES.get(num, "")
        logger.info(f"── Ingesting US Code Title {num} — {name} ──")

        # Find all PDFs for this title (handles appendix files like 05A, and
        # split titles like 42 which has 6 separate part files)
        pdf_files = sorted(USCODE_PDF_DIR.glob(f"usc{num.zfill(2)}*.pdf"))
        if not pdf_files:
            logger.warning(f"  No PDF found for Title {num} in {USCODE_PDF_DIR}, skipping.")
            continue

        all_chunks = []
        for pdf_path in pdf_files:
            logger.info(f"  Reading {pdf_path.name} …")
            raw_sections = parse_uscode_pdf(pdf_path, num, name)
            for sec in raw_sections:
                all_chunks.extend(chunk_uscode(sec.text, sec.title_num, sec.title_name))

        logger.info(f"  {len(all_chunks)} chunks")
        n = upsert_chunks(all_chunks)
        logger.info(f"  {n} inserted")


def ingest_cfr(title_nums: list[str]):
    for num in title_nums:
        name = CFR_TITLES.get(num, "")
        logger.info(f"── Ingesting CFR Title {num} — {name} ──")
        sections = fetch_cfr_title(int(num))
        all_chunks = []
        for sec in sections:
            all_chunks.extend(chunk_cfr(sec.text, sec.title_num, sec.part))
        logger.info(f"  {len(all_chunks)} chunks")
        n = upsert_chunks(all_chunks)
        logger.info(f"  {n} inserted")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=["all", "constitution", "uscode", "cfr"],
        default="constitution",
    )
    parser.add_argument(
        "--titles",
        nargs="*",
        help="Title numbers to ingest (for uscode or cfr)",
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Skip rebuilding the HNSW index at the end (use when more titles still to come)",
    )
    args = parser.parse_args()

    logger.info("=== Initialising database ===")
    init_db()
    drop_hnsw_index()

    if args.source in ("all", "constitution"):
        ingest_constitution()

    if args.source in ("all", "uscode"):
        titles = args.titles or list(USCODE_TITLES.keys())
        ingest_uscode(titles)

    if args.source in ("all", "cfr"):
        titles = args.titles or list(CFR_TITLES.keys())
        ingest_cfr(titles)

    if args.skip_index:
        logger.info("Skipping HNSW index rebuild (--skip-index). Run without --skip-index on the last title.")
    else:
        rebuild_hnsw_index()
    logger.info("=== Ingestion complete ===")


if __name__ == "__main__":
    main()
