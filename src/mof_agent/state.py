from typing import TypedDict, List, Dict, Any


class AgentState(TypedDict, total=False):
    cif_path: str
    target_features: Dict[str, Any]
    sbu_vec: Any
    ligand_vec: Any
    retrieval_strategy: Dict[str, Any]
    top15_candidates: List[Dict[str, Any]]
    top5_selected: List[Dict[str, Any]]
    literature_context: str
    final_analysis: str
    extraction_errors: List[str]
