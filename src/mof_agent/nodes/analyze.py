from typing import Dict, Any
import numpy as np

from mof_agent.state import AgentState
from mof_agent.tools.analyze_tool import AnalyzeMOFTool


def node_analyze(state: AgentState) -> Dict[str, Any]:
    cif_path = state["cif_path"]
    result = AnalyzeMOFTool.run(cif_path)

    sbu_vec = result.get("sbu_vec")
    ligand_vec = result.get("ligand_vec")

    def _to_native(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _to_native(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_native(v) for v in obj]
        return obj

    return {
        "target_features": _to_native(result["target_features"]),
        "sbu_vec": sbu_vec.tolist() if sbu_vec is not None else None,
        "ligand_vec": ligand_vec.tolist() if ligand_vec is not None else None,
        "extraction_errors": result.get("extraction_errors", []),
    }
