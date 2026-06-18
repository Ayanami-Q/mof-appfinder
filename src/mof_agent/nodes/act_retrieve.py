from typing import Dict, Any

import numpy as np

from mof_agent.config import DEFAULT_WEIGHTS
from mof_agent.state import AgentState
from mof_agent.context import get_retrieve_tool


def node_act_retrieve(state: AgentState) -> Dict[str, Any]:
    print(f"\n{'='*60}")
    print(f"[Node 3] Weighted Parallel Retrieval (Act)")
    print(f"{'='*60}")

    sbu_raw = state.get("sbu_vec")
    ligand_raw = state.get("ligand_vec")
    sbu_vec = np.array(sbu_raw, dtype=np.float32) if sbu_raw is not None else None
    ligand_vec = np.array(ligand_raw, dtype=np.float32) if ligand_raw is not None else None

    strategy = state.get("retrieval_strategy", DEFAULT_WEIGHTS)
    phys = state.get("target_features", {}).get("physical_properties", {})

    tool = get_retrieve_tool()
    if tool is None:
        raise RuntimeError("WeightedParallelRetrieveTool not initialized")

    top15 = tool.run(
        sbu_vec=sbu_vec,
        ligand_vec=ligand_vec,
        phys_dict=phys,
        weights=strategy,
    )

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

    top15 = [_to_native(c) for c in top15]

    return {"top15_candidates": top15}
