"""Global configuration. All secrets and paths read from environment variables."""

import os
import random
import numpy as np

LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

GLM_API_KEY = LLM_API_KEY
GLM_BASE_URL = LLM_BASE_URL
GLM_MODEL = LLM_MODEL

_use_real_llm = os.getenv("USE_REAL_LLM", "false").lower() in ("1", "true", "yes")


def set_use_real_llm(v: bool) -> None:
    global _use_real_llm
    _use_real_llm = v


def get_use_real_llm() -> bool:
    return _use_real_llm


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

DEFAULT_PHYSICAL_DIR = os.getenv("PHYSICAL_DB_DIR", "")
DEFAULT_LIGAND_DB = os.getenv("LIGAND_DB_PATH", "")
DEFAULT_LIGAND_INDEX = os.getenv("LIGAND_INDEX_PATH", "")
DEFAULT_SBU_DB = os.getenv("SBU_DB_PATH", "")
DEFAULT_SBU_INDEX = os.getenv("SBU_INDEX_PATH", "")

SBU_FORMULAS_JSON = os.getenv("SBU_FORMULAS_JSON", "")
LIGAND_SMILES_JSON = os.getenv("LIGAND_SMILES_JSON", "")

_script_dir = os.path.dirname(os.path.abspath(__file__))
CSD_CSV_PATH = os.path.join(_script_dir, "data", "MOF_names_and_CSD_codes.csv")

KNOWLEDGE_BASE_PATH = os.getenv("KNOWLEDGE_BASE_PATH", "")

PUBLISHER_API_KEYS = {
    "elsevier": os.getenv("ELSEVIER_API_KEY", ""),
    "springer": os.getenv("SPRINGER_API_KEY", ""),
    "wiley": os.getenv("WILEY_TDM_TOKEN", ""),
}
PUBLISHER_API_ENABLED = os.getenv("PUBLISHER_API_ENABLED", "false").lower() in ("1", "true", "yes")

DEFAULT_WEIGHTS = {"sbu_weight": 0.3, "ligand_weight": 0.3, "physical_weight": 0.4}
DEFAULT_WEIGHT_REASONING = "LLM output parsing failed, falling back to default equal-weight strategy."
