import json
from typing import Optional, Dict, List, Tuple

from mof_agent.database.whitebox_db import WhiteboxDBManager
from mof_agent.lookup.csd_lookup import CSDLookup
from mof_agent.tools.retrieve_tool import WeightedParallelRetrieveTool

_whitebox_db: Optional[WhiteboxDBManager] = None
_csd_lookup: Optional[CSDLookup] = None
_retrieve_tool: Optional[WeightedParallelRetrieveTool] = None
_knowledge_base: Optional["KnowledgeBase"] = None


class KnowledgeBase:
    """CSD refcode-indexed knowledge base of batch-extracted structured literature data."""

    def __init__(self, kb_path: str):
        with open(kb_path, "r", encoding="utf-8") as f:
            self._data: Dict[str, dict] = json.load(f)
        print(f"[KnowledgeBase] Loaded {len(self._data)} records: {kb_path}")

    def lookup(self, refcode: str) -> Optional[dict]:
        if not refcode:
            return None
        return self._data.get(refcode.upper().strip())

    def search_by_name(self, keyword: str) -> List[Tuple[str, dict]]:
        results = []
        kw = keyword.lower()
        for rc, record in self._data.items():
            names = " ".join(record.get("mof_names", [])).lower()
            if kw in names:
                results.append((rc, record))
        return results

    def __len__(self) -> int:
        return len(self._data)


def init_whitebox_db(db: WhiteboxDBManager) -> None:
    global _whitebox_db
    _whitebox_db = db


def get_whitebox_db() -> Optional[WhiteboxDBManager]:
    return _whitebox_db


def init_csd_lookup(lookup: CSDLookup) -> None:
    global _csd_lookup
    _csd_lookup = lookup


def get_csd_lookup() -> Optional[CSDLookup]:
    return _csd_lookup


def init_retrieve_tool(tool: WeightedParallelRetrieveTool) -> None:
    global _retrieve_tool
    _retrieve_tool = tool


def get_retrieve_tool() -> Optional[WeightedParallelRetrieveTool]:
    return _retrieve_tool


def init_knowledge_base(kb: "KnowledgeBase") -> None:
    global _knowledge_base
    _knowledge_base = kb


def get_knowledge_base() -> Optional["KnowledgeBase"]:
    return _knowledge_base
