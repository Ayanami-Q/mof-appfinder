import os
import re
import csv
from typing import Dict, Optional


class CSDLookup:
    """Extract CSD refcode from CoRE-MOF ID and look up corresponding DOI and MOF name."""

    def __init__(self, csv_path: str):
        self._refcode_to_doi: Dict[str, str] = {}
        self._refcode_to_name: Dict[str, str] = {}

        if not os.path.exists(csv_path):
            print(f"[CSDLookup] Warning: CSV file not found: {csv_path}, DOI lookup will degrade")
            return

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                refcode = (row.get("Ref Code") or "").strip().upper()
                doi = (row.get("Reference") or "").strip()
                name = (row.get("MOF Name") or "").strip()
                if refcode and doi:
                    if refcode not in self._refcode_to_doi:
                        self._refcode_to_doi[refcode] = doi
                        self._refcode_to_name[refcode] = name

        print(f"[CSDLookup] Loaded {len(self._refcode_to_doi)} CSD->DOI mappings")

    def extract_refcode(self, cif_id: str) -> Optional[str]:
        base = cif_id.replace("_clean", "").replace("_charged", "").replace("_auto", "")
        match = re.match(r"^([A-Z]{6})", base.upper())
        if match:
            return match.group(1)
        return None

    def get_doi(self, cif_id: str) -> Optional[str]:
        refcode = self.extract_refcode(cif_id)
        if refcode and refcode in self._refcode_to_doi:
            return self._refcode_to_doi[refcode]
        return None

    def get_name(self, cif_id: str) -> Optional[str]:
        refcode = self.extract_refcode(cif_id)
        if refcode and refcode in self._refcode_to_name:
            return self._refcode_to_name[refcode]
        return None
