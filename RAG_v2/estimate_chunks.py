"""
RAG_v2/estimate_chunks.py

Calculates the total number of chunks that LlamaIndex SentenceSplitter
would produce across the full US legal corpus — without embedding or uploading.

Run:
    python RAG_v2/estimate_chunks.py
    python RAG_v2/estimate_chunks.py --titles 18 26 42
"""
import sys
import time
import argparse
import logging
sys.path.insert(0, ".")

from pathlib import Path
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document
from ingestion.parsers.uscode_parser import parse_uscode_pdf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

CONSTITUTION_FILE = "data/constitution.txt"
USCODE_PDF_DIR    = Path("data/pdf_uscAll@119-73")
CHUNK_SIZE        = 512
CHUNK_OVERLAP     = 40

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
    "36": "Patriotic and National Observances",
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

SEPARATOR = "─" * 65


def count_nodes(text: str, splitter: SentenceSplitter) -> tuple[int, float, int, int]:
    doc   = Document(text=text)
    nodes = splitter.get_nodes_from_documents([doc])
    sizes = [len(n.text) for n in nodes]
    avg   = sum(sizes) // len(sizes) if sizes else 0
    return len(nodes), avg, min(sizes) if sizes else 0, max(sizes) if sizes else 0


def human_size(n_chunks: int) -> str:
    total_gb = (n_chunks * 1536 * 4 + n_chunks * 2048) / 1e9
    return f"{total_gb:.1f} GB"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--titles", nargs="*", help="Specific title numbers (default: all)")
    args = parser.parse_args()

    titles_to_process = args.titles or list(USCODE_TITLES.keys())
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    print(f"\n{SEPARATOR}")
    print("  CHUNK ESTIMATOR — RAG v2 (LlamaIndex SentenceSplitter)")
    print(f"  chunk_size={CHUNK_SIZE} tokens  chunk_overlap={CHUNK_OVERLAP} tokens")
    print(SEPARATOR)

    grand_total   = 0
    title_results = []
    t_start       = time.time()

    # Constitution
    with open(CONSTITUTION_FILE, "r", encoding="utf-8") as f:
        raw = f.read()
    count, avg, mn, mx = count_nodes(raw, splitter)
    grand_total += count
    print(f"  {'Constitution':<45} {count:>8} chunks  avg={avg:>5} chars")

    print(f"\n  {'Title':<45} {'Chunks':>8}   {'Avg':>6}   {'Est. Size':>10}")
    print(f"  {'─'*45} {'─'*8}   {'─'*6}   {'─'*10}")

    for num in titles_to_process:
        name      = USCODE_TITLES.get(num, "Unknown")
        pdf_files = sorted(USCODE_PDF_DIR.glob(f"usc{num.zfill(2)}*.pdf"))

        if not pdf_files:
            print(f"  Title {num:<2} {name:<38} {'NO PDF':>8}")
            continue

        title_chunks = 0
        for pdf in pdf_files:
            logging.info(f"  Parsing {pdf.name} …")
            sections = parse_uscode_pdf(pdf, num, name)
            for s in sections:
                c, _, _, _ = count_nodes(s.text, splitter)
                title_chunks += c
            logging.info(f"  Done    {pdf.name} — {title_chunks:,} chunks so far")

        est_mb = title_chunks * (1536 * 4 + 2048) / 1e6
        grand_total += title_chunks
        title_results.append((num, name, title_chunks))
        print(f"  Title {num:<2} {name:<38} {title_chunks:>8,}   {'—':>5}   {est_mb:>8.1f} MB")

    elapsed = time.time() - t_start

    print(f"\n{SEPARATOR}")
    print(f"  {'TOTAL CHUNKS':<45} {grand_total:>8,}")
    print(f"  {'Est. Qdrant storage':<45} {human_size(grand_total):>8}")
    print(f"  {'Time to estimate':<45} {elapsed:>7.1f}s")
    print(SEPARATOR)

    if len(title_results) > 1:
        top5 = sorted(title_results, key=lambda x: x[2], reverse=True)[:5]
        print("\n  Top 5 largest titles:")
        for num, name, count in top5:
            pct = count * 100 // grand_total
            print(f"    Title {num:<2} {name:<35} {count:>8,}  ({pct}%)")

    print()


if __name__ == "__main__":
    main()
