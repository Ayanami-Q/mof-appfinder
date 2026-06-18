from typing import Dict, List, Any, Optional

import numpy as np

from mof_agent.database.whitebox_db import WhiteboxDBManager


class WeightedParallelRetrieveTool:
    """Dynamic weighted parallel retrieval tool."""

    def __init__(self, db: WhiteboxDBManager):
        self.db = db

    def run(
        self,
        sbu_vec: Optional[np.ndarray],
        ligand_vec: Optional[np.ndarray],
        phys_dict: Dict[str, Any],
        weights: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        return self.db.weighted_parallel_search(
            sbu_vec=sbu_vec,
            ligand_vec=ligand_vec,
            phys_dict=phys_dict,
            weights=weights,
            top_k=15,
        )
