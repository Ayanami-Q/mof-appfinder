import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

DEFAULT_RESULTS_DIR = os.getenv("KB_RESULTS_DIR", "")
DEFAULT_DEDUP_META  = os.getenv("KB_DEDUP_META", "")
DEFAULT_OUTPUT_DIR  = os.getenv("KB_OUTPUT_DIR", "")

# ---------------------------------------------------------------------------
# Field definitions for the extracted content
# ---------------------------------------------------------------------------
EXTRACTION_FIELDS = [
    "synthesis_summary",
    "characterization_summary",
    "applications_summary",
    "metal_valence",
    "structural_description",
    "structure_activity_relationship",
    "justification",
]

# File extensions to strip from the end of custom_id DOI slugs
KNOWN_EXT_SUFFIXES = ["_html", "_htm", "_xml", "_pdf", "_md"]


# ---------------------------------------------------------------------------
# DOI / custom_id normalization for matching
# ---------------------------------------------------------------------------

def doi_to_slug(doi: str) -> str:
    """Normalize a canonical DOI into a matchable slug."""
    return re.sub(r'[^\w]', '_', doi).strip('_').lower()


def extract_doi_slug_from_custom_id(custom_id: str) -> str:
    """Pull a DOI-like slug from a batch custom_id of the form '00000_10_1016_...'.

    Steps:
    1. Strip the leading 5-digit index + underscore.
    2. Strip a known file-extension suffix.
    """
    # Drop the 6-char prefix "XXXXX_"
    if len(custom_id) > 6 and custom_id[5] == '_' and custom_id[:5].isdigit():
        slug = custom_id[6:]
    else:
        slug = custom_id

    for ext in KNOWN_EXT_SUFFIXES:
        if slug.endswith(ext):
            slug = slug[:-len(ext)]
            break

    return slug.lower()


# ---------------------------------------------------------------------------
# Parsing the LLM response body into structured fields
# ---------------------------------------------------------------------------

def parse_llm_content(raw_text: str, custom_id: str = "") -> Dict[str, str]:
    """Parse the JSON buried inside the LLM's markdown/json code fence."""
    cleaned = raw_text.strip()

    # Strip ```json ... ``` fences
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
        return {
            "synthesis_summary":             data.get("synthesis_summary", "Not provided"),
            "characterization_summary":      data.get("characterization_summary", "Not provided"),
            "applications_summary":          data.get("applications_summary", "Not provided"),
            "metal_valence":                 data.get("metal_valence", "Not provided"),
            "structural_description":        data.get("structural_description", "Not provided"),
            "structure_activity_relationship": data.get("structure_activity_relationship", "Not provided"),
            "justification":                 data.get("justification", "Not found in text"),
        }
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        return {
            "synthesis_summary":             f"[Parse error] {raw_text[:200]}",
            "characterization_summary":      "Not provided",
            "applications_summary":          "Not provided",
            "metal_valence":                 "Not provided",
            "structural_description":        "Not provided",
            "structure_activity_relationship": "Not provided",
            "justification":                 f"JSON parse failed: {e}",
        }


def extract_batch_entry(entry: dict) -> Optional[dict]:
    """Extract structured content from a single batch JSONL line.

    Returns a dict with extraction fields + custom_id and parse status, or None.
    """
    custom_id = entry.get("custom_id", "unknown")
    resp = entry.get("response", {})

    if resp.get("status_code") != 200:
        return None

    choices = resp.get("body", {}).get("choices", [])
    if not choices:
        return None

    raw_text = choices[0].get("message", {}).get("content", "")
    parsed = parse_llm_content(raw_text, custom_id)
    parsed["custom_id"] = custom_id
    parsed["batch_id"] = entry.get("id", "")

    # Flag whether the parse succeeded
    j = parsed.get("justification", "")
    parsed["_parse_ok"] = not (isinstance(j, str) and j.startswith("JSON parse failed"))
    return parsed


# ---------------------------------------------------------------------------
# Building the DOI → metadata lookup from dedup_metadata.json
# ---------------------------------------------------------------------------

def load_dedup_metadata(meta_path: str) -> Dict[str, dict]:
    """Load dedup_metadata.json, return {normalized_doi_slug: metadata}."""
    with open(meta_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    lookup = {}
    for filename, meta in raw.items():
        doi = meta.get("doi", "")
        if not doi:
            continue
        slug = doi_to_slug(doi)
        lookup[slug] = {
            "doi": doi,
            "refcodes": meta.get("refcodes", []),
            "mof_names": meta.get("mof_names", []),
            "filename": filename,
            "original_files": meta.get("original_files", []),
        }
    return lookup


# ---------------------------------------------------------------------------
# Core matching logic
# ---------------------------------------------------------------------------

def match_custom_id_to_doi(
    custom_id_slug: str, dedup_lookup: Dict[str, dict]
) -> Optional[dict]:
    """Try to match a custom_id's DOI slug to an entry in dedup_metadata.

    Strategy (in order):
    1. Exact match on the full slug.
    2. Check if any known DOI slug is a substring of the custom_id slug.
    3. Check if the custom_id slug is a substring of any known DOI slug.
    """
    # 1. Exact match
    if custom_id_slug in dedup_lookup:
        return dedup_lookup[custom_id_slug]

    # 2. Known DOI slug is a substring of custom_id slug
    for known_slug, meta in dedup_lookup.items():
        if len(known_slug) >= 10 and known_slug in custom_id_slug:
            return meta

    # 3. Custom_id slug is a substring of a known DOI slug
    if len(custom_id_slug) >= 10:
        for known_slug, meta in dedup_lookup.items():
            if custom_id_slug in known_slug:
                return meta

    return None


# ---------------------------------------------------------------------------
# Main merge procedure
# ---------------------------------------------------------------------------

def merge_results(
    results_dir: str,
    dedup_meta_path: str,
) -> Tuple[Dict[str, dict], dict]:
    """Read all JSONL files, match to CSD refcodes, return indexed knowledge base.

    Returns:
        kb:       {refcode: {extraction_data, doi, mof_names, ...}}
        report:   summary statistics dict
    """
    dedup_lookup = load_dedup_metadata(dedup_meta_path)
    print(f"Loaded {len(dedup_lookup)} DOI entries from dedup_metadata.json")

    # ---- Collect all results ----
    all_entries: List[dict] = []
    jsonl_files = sorted(
        f for f in os.listdir(results_dir) if f.endswith(".jsonl")
    )

    print(f"Found {len(jsonl_files)} JSONL files")
    for fname in jsonl_files:
        fpath = os.path.join(results_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    parsed = extract_batch_entry(entry)
                    if parsed:
                        all_entries.append(parsed)
                except json.JSONDecodeError:
                    continue

    print(f"Parsed {len(all_entries)} valid entries from JSONL files")

    # ---- Match to CSD refcodes ----
    kb: Dict[str, dict] = {}  # refcode → info
    refcode_seen: Dict[str, list] = defaultdict(list)  # track duplicates
    unmatched: List[dict] = []
    multi_refcode_count = 0

    for entry in all_entries:
        slug = extract_doi_slug_from_custom_id(entry["custom_id"])
        meta = match_custom_id_to_doi(slug, dedup_lookup)

        if meta is None:
            unmatched.append({
                "custom_id": entry["custom_id"],
                "slug": slug,
                "parse_ok": entry["_parse_ok"],
            })
            continue

        refcodes = meta["refcodes"]
        if not refcodes:
            unmatched.append({
                "custom_id": entry["custom_id"],
                "slug": slug,
                "doi": meta["doi"],
                "reason": "no_refcodes",
            })
            continue

        if len(refcodes) > 1:
            multi_refcode_count += 1

        # Build the record
        record = {
            "doi": meta["doi"],
            "refcodes": refcodes,
            "mof_names": meta["mof_names"],
            "filename": meta["filename"],
            "original_files": meta["original_files"],
            "custom_id": entry["custom_id"],
            "batch_id": entry["batch_id"],
        }
        for field in EXTRACTION_FIELDS:
            record[field] = entry.get(field, "Not provided")
        record["_parse_ok"] = entry["_parse_ok"]

        # Index by every refcode (a paper with N refcodes is reachable by any)
        for rc in refcodes:
            rc_upper = rc.upper()
            if rc_upper in refcode_seen:
                # This refcode appears in multiple papers; keep the one with parse_ok=True
                existing = kb.get(rc_upper)
                if existing and not existing.get("_parse_ok") and record["_parse_ok"]:
                    kb[rc_upper] = record
            else:
                kb[rc_upper] = record
            refcode_seen[rc_upper].append(record["doi"])

    # ---- Build report ----
    duplicate_refcodes = {
        rc: dois for rc, dois in refcode_seen.items() if len(dois) > 1
    }

    report = {
        "generated_at": datetime.now().isoformat(),
        "total_jsonl_entries": len(all_entries),
        "total_matched": len(all_entries) - len(unmatched),
        "total_unmatched": len(unmatched),
        "unique_refcodes_in_kb": len(kb),
        "papers_in_kb": len(set(tuple(r.get("refcodes", [])) for r in kb.values())),
        "multi_refcode_papers": multi_refcode_count,
        "duplicate_refcodes": {
            rc: dois
            for rc, dois in duplicate_refcodes.items()
        },
        "unmatched_entries": unmatched,
        "parse_success_rate": (
            sum(1 for e in all_entries if e["_parse_ok"]) / max(len(all_entries), 1)
        ),
    }

    return kb, report


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def save_knowledge_base(kb: Dict[str, dict], report: dict, output_dir: str):
    """Write the knowledge base to JSON, CSV, and a merge report."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---- JSON ----
    json_path = os.path.join(output_dir, "knowledge_base.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(kb, f, ensure_ascii=False, indent=2)
    print(f"Knowledge base JSON: {json_path}  ({len(kb)} refcodes)")

    # ---- CSV (flattened) ----
    csv_rows = []
    for refcode, record in sorted(kb.items()):
        row = {
            "refcode": refcode,
            "doi": record.get("doi", ""),
            "mof_names": " | ".join(record.get("mof_names", [])),
            "all_refcodes": " | ".join(record.get("refcodes", [])),
            "filename": record.get("filename", ""),
        }
        for field in EXTRACTION_FIELDS:
            val = record.get(field, "")
            # Truncate long text for CSV readability
            if isinstance(val, str) and len(val) > 500:
                val = val[:500] + "..."
            row[field] = val
        csv_rows.append(row)

    csv_path = os.path.join(output_dir, "knowledge_base.csv")
    df = pd.DataFrame(csv_rows)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Knowledge base CSV:  {csv_path}  ({len(df)} rows)")

    # ---- Merge Report ----
    report_path = os.path.join(output_dir, "merge_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Merge report:        {report_path}")


def print_report(report: dict):
    """Print a human-readable summary."""
    print("\n" + "=" * 60)
    print("KNOWLEDGE BASE MERGE REPORT")
    print("=" * 60)
    print(f"  JSONL entries parsed:       {report['total_jsonl_entries']}")
    print(f"  Matched to CSD refcodes:    {report['total_matched']}")
    print(f"  Unmatched:                  {report['total_unmatched']}")
    print(f"  Unique refcodes in KB:      {report['unique_refcodes_in_kb']}")
    print(f"  Papers in KB:               {report['papers_in_kb']}")
    print(f"  Multi-refcode papers:       {report['multi_refcode_papers']}")
    print(f"  Parse success rate:         {report['parse_success_rate']:.1%}")
    dup_count = len(report["duplicate_refcodes"])
    if dup_count:
        print(f"  Duplicate refcodes:         {dup_count} (see merge_report.json for details)")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Quick-lookup utility
# ---------------------------------------------------------------------------

def lookup_refcode(kb: Dict[str, dict], refcode: str) -> Optional[dict]:
    """Query the knowledge base by CSD refcode (case-insensitive)."""
    return kb.get(refcode.upper().strip())


def search_by_mof_name(kb: Dict[str, dict], keyword: str) -> List[Tuple[str, dict]]:
    """Fuzzy search by MOF name substring."""
    results = []
    kw = keyword.lower()
    for rc, record in kb.items():
        names = " ".join(record.get("mof_names", [])).lower()
        if kw in names:
            results.append((rc, record))
    return results


def search_by_doi(kb: Dict[str, dict], doi: str) -> List[Tuple[str, dict]]:
    """Search by DOI substring."""
    results = []
    kw = doi.lower().strip()
    for rc, record in kb.items():
        if kw in record.get("doi", "").lower():
            results.append((rc, record))
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Merge GLM batch results into a CSD-refcode-indexed knowledge base"
    )
    parser.add_argument(
        "--results-dir", default=DEFAULT_RESULTS_DIR,
        help="Directory containing JSONL result files",
    )
    parser.add_argument(
        "--dedup-meta", default=DEFAULT_DEDUP_META,
        help="Path to dedup_metadata.json",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help="Output directory for knowledge_base.json/.csv",
    )
    parser.add_argument(
        "--query", default=None,
        help="Quick query: refcode, 'doi:...', or 'name:...'",
    )
    args = parser.parse_args()

    kb_path = os.path.join(args.output_dir, "knowledge_base.json")

    # ---- Quick query mode (load cached KB when possible) ----
    if args.query:
        if os.path.isfile(kb_path):
            print(f"Loading cached knowledge base: {kb_path}")
            with open(kb_path, "r", encoding="utf-8") as f:
                kb = json.load(f)
        else:
            if not os.path.isdir(args.results_dir):
                print(f"ERROR: results-dir not found: {args.results_dir}")
                sys.exit(1)
            if not os.path.isfile(args.dedup_meta):
                print(f"ERROR: dedup_metadata.json not found: {args.dedup_meta}")
                sys.exit(1)
            kb, report = merge_results(args.results_dir, args.dedup_meta)
            save_knowledge_base(kb, report, args.output_dir)
            print_report(report)
        _do_query(kb, args.query)
        return

    # ---- Build mode ----
    if not os.path.isdir(args.results_dir):
        print(f"ERROR: results-dir not found: {args.results_dir}")
        sys.exit(1)
    if not os.path.isfile(args.dedup_meta):
        print(f"ERROR: dedup_metadata.json not found: {args.dedup_meta}")
        sys.exit(1)

    kb, report = merge_results(args.results_dir, args.dedup_meta)
    save_knowledge_base(kb, report, args.output_dir)
    print_report(report)

    if args.query:
        _do_query(kb, args.query)


def _do_query(kb: dict, query: str):
    """Run a query against the knowledge base and print results."""
    q = query.strip()
    print("\n" + "-" * 60)
    if q.startswith("doi:"):
        results = search_by_doi(kb, q[4:])
        print(f"Search by DOI '{q[4:]}': {len(results)} result(s)")
        for rc, rec in results[:5]:
            print(f"  {rc}: {rec.get('doi')}  |  {rec.get('mof_names', [])}")
    elif q.startswith("name:"):
        results = search_by_mof_name(kb, q[5:])
        print(f"Search by MOF name '{q[5:]}': {len(results)} result(s)")
        for rc, rec in results[:5]:
            print(f"  {rc}: {rec.get('mof_names', [])}")
    else:
        rec = lookup_refcode(kb, q)
        if rec:
            print(f"Refcode: {q.upper()}")
            print(f"  DOI:      {rec.get('doi')}")
            print(f"  MOF:      {rec.get('mof_names')}")
            print(f"  Refcodes: {rec.get('refcodes')}")
            for field in EXTRACTION_FIELDS:
                val = rec.get(field, "")
                if val and val != "Not provided":
                    print(f"  {field}: {str(val)[:200]}...")
        else:
            print(f"Refcode '{q}' not found in knowledge base.")


if __name__ == "__main__":
    main()
