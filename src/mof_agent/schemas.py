from typing import List
from pydantic import BaseModel, Field


class RetrievalStrategy(BaseModel):
    """LLM-produced three-channel retrieval weights. Must sum to 1.0."""
    sbu_weight: float = Field(..., ge=0.0, le=1.0,
                              description="SBU/metal-cluster SOAP vector similarity weight")
    ligand_weight: float = Field(..., ge=0.0, le=1.0,
                                 description="Ligand Morgan fingerprint similarity weight")
    physical_weight: float = Field(..., ge=0.0, le=1.0,
                                   description="Physical pore feature (LCD/PLD/VF) weight")
    reasoning: str = Field(..., description="Scientific reasoning for the weight assignment")


class Top5Selection(BaseModel):
    """LLM-selected top-5 candidates from the top-15 pool."""
    selected_indices: List[int] = Field(..., description="Selected candidate indices (0-based, max 5)")
    selection_reasoning: str = Field(..., description="Chemical rationale for selecting these 5")


def validate_weights(strategy: RetrievalStrategy) -> bool:
    total = strategy.sbu_weight + strategy.ligand_weight + strategy.physical_weight
    return abs(total - 1.0) < 0.02
