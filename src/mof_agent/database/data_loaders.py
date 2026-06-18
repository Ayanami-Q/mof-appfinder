import os
import json
from typing import Dict, List, Optional

from mof_agent.config import SBU_FORMULAS_JSON, LIGAND_SMILES_JSON

_sbu_formulas_cache: Optional[Dict[str, List[str]]] = None
_ligand_smiles_cache: Optional[Dict[str, List[str]]] = None


def load_sbu_formulas() -> Dict[str, List[str]]:
    global _sbu_formulas_cache
    if _sbu_formulas_cache is not None:
        return _sbu_formulas_cache
    if os.path.exists(SBU_FORMULAS_JSON):
        with open(SBU_FORMULAS_JSON, "r") as f:
            _sbu_formulas_cache = json.load(f)
        print(f"[Data] Loaded {len(_sbu_formulas_cache)} SBU metal formulas")
    else:
        print(f"[Data] Warning: SBU formulas JSON not found: {SBU_FORMULAS_JSON}")
        _sbu_formulas_cache = {}
    return _sbu_formulas_cache


def load_ligand_smiles() -> Dict[str, List[str]]:
    global _ligand_smiles_cache
    if _ligand_smiles_cache is not None:
        return _ligand_smiles_cache
    if os.path.exists(LIGAND_SMILES_JSON):
        with open(LIGAND_SMILES_JSON, "r") as f:
            _ligand_smiles_cache = json.load(f)
        print(f"[Data] Loaded {len(_ligand_smiles_cache)} ligand SMILES")
    else:
        print(f"[Data] Warning: Ligand SMILES JSON not found: {LIGAND_SMILES_JSON}")
        _ligand_smiles_cache = {}
    return _ligand_smiles_cache
