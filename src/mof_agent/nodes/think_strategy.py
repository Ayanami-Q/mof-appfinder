import json
import re
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from mof_agent.config import (
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL,
    get_use_real_llm, DEFAULT_WEIGHTS, DEFAULT_WEIGHT_REASONING,
)
from mof_agent.schemas import RetrievalStrategy, validate_weights
from mof_agent.state import AgentState

STRATEGY_SYSTEM_PROMPT = """You are a senior computational chemist. Your task is to analyze the target MOF's features, determine its most likely application scenarios, and output a strictly valid JSON retrieval weight scheme. Output ONLY JSON, no other text."""


def _build_strategy_prompt(features: Dict[str, Any]) -> str:
    phys = features.get("physical_properties", {})
    cif = features.get("cif_metadata", {})
    pore = features.get("pore_topology", {})
    surface = features.get("surface_environment", {})
    sbu_f = features.get("sbu_formulas", [])
    lig_s = features.get("ligand_smiles", [])

    return f"""Analyze the following target MOF features and output a JSON object containing sbu_weight, ligand_weight, physical_weight (three floats, sum=1.0) and reasoning (string).

## Target MOF Features

### Basic Info
- Formula: {cif.get('reduced_formula', 'N/A')}
- Main metal: {cif.get('main_metal', 'N/A')}
- Metal types: {cif.get('metal_elements', [])}
- All elements: {cif.get('elements', [])}

### Metal Cluster (SBU)
- Formulas: {sbu_f if sbu_f else 'No metal clusters extracted'}

### Ligand
- SMILES: {lig_s[:5] if lig_s else 'No ligand SMILES extracted'}

### Physical Pore Features
- LCD (largest cavity diameter): {phys.get('lcd', 'N/A')} A
- PLD (pore limiting diameter): {phys.get('pld', 'N/A')} A
- VF (void fraction): {phys.get('vf', 'N/A')}
- Cell volume: {phys.get('volume', 'N/A')} A3

### Porosity Source and Uncertainty
- Data source: {phys.get('porosity_source', 'N/A')}
- Confidence: {phys.get('porosity_confidence', 'N/A')} (high/medium/low)
- 3D percolation: {phys.get('percolates', 'N/A')}
- Accessible VF (N2 probe): {phys.get('vf_accessible_by_probe', {}).get('1.82', 'N/A')}
- Warnings: {phys.get('porosity_warnings', [])}

### Pore Topology (pyzeo)
- Di: {pore.get('di_A', 'N/A')} A
- Df: {pore.get('df_A', 'N/A')} A
- PLD: {pore.get('pld_A', 'N/A')} A
- Accessible volume fraction: {pore.get('accessible_volume_fraction', 'N/A')}

### Surface Environment
- Surface composition: {surface.get('surface_composition', 'N/A')}
- Polarity assessment: {surface.get('polarity_assessment', 'N/A')}

## Weight Guidelines
1. **sbu_weight**: catalytic metals (Cu, Fe, Co, Ni, Pd, Pt, Cr, etc.) -> 0.35-0.5; simple metal nodes (Zn, Zr) -> 0.2-0.3
2. **ligand_weight**: special functional groups (-NH2, -OH, -COOH, porphyrins, etc.) -> 0.3-0.4; simple aromatic carboxylates -> 0.2-0.3
3. **physical_weight**:
   - Do NOT treat low-confidence porosity data as precise
   - porosity_confidence == "low" -> physical_weight must not exceed 0.25
   - pld < 2.4 A or vf < 0.10 or percolates is False -> not suitable for gas storage, physical_weight <= 0.30
   - If catalytic metals present with low/narrow pores, increase sbu_weight and surface/ligand weights
   - Only when confidence is medium/high AND percolates=True: LCD < 5A (gas sieving) -> 0.45-0.6; VF > 0.7 (gas storage) -> 0.4-0.55

Output format (strict JSON):
{{"sbu_weight": 0.XX, "ligand_weight": 0.XX, "physical_weight": 0.XX, "reasoning": "Brief scientific justification for weight assignment"}}"""


def _parse_strategy_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    match = re.search(r'```(?:json)?\s*({.*?})\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    match = re.search(r'\{[^{}]*"sbu_weight"[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Cannot parse JSON from LLM output: {text[:200]}")


def _clamp_weights(weights: Dict[str, float], features: Dict[str, Any]) -> Dict[str, float]:
    phys = features.get("physical_properties", {})
    pconf = phys.get("porosity_confidence", "low")
    percolates = phys.get("percolates", False)
    pld = phys.get("pld")
    vf = phys.get("vf")
    metals = features.get("cif_metadata", {}).get("metal_elements", [])
    catalytic_metals = {"Cu", "Fe", "Co", "Ni", "Pd", "Pt", "Cr", "Mn", "Mo", "Ru", "Rh", "Ir"}
    has_catalytic = bool(set(metals) & catalytic_metals)

    clamped = dict(weights)

    if pconf == "low":
        clamped["physical_weight"] = min(clamped["physical_weight"], 0.25)

    is_narrow = (pld is not None and pld < 2.4) or (vf is not None and vf < 0.10) or (percolates is False)
    if is_narrow:
        clamped["physical_weight"] = min(clamped["physical_weight"], 0.30)

    if has_catalytic and is_narrow:
        clamped["sbu_weight"] = max(clamped["sbu_weight"], 0.35)

    # Re-normalize to sum=1.0
    total = sum(clamped[k] for k in ("sbu_weight", "ligand_weight", "physical_weight"))
    if total > 1e-12:
        for k in ("sbu_weight", "ligand_weight", "physical_weight"):
            clamped[k] /= total

    return clamped


def node_think_strategy(state: AgentState) -> Dict[str, Any]:
    print(f"\n{'='*60}")
    print(f"[Node 2] LLM Strategy Thinking (JSON Output)")
    print(f"{'='*60}")

    features = state.get("target_features", {})
    user_prompt = _build_strategy_prompt(features)

    if get_use_real_llm() and LLM_API_KEY not in ("", "sk-placeholder-key"):
        try:
            llm = ChatOpenAI(
                model=LLM_MODEL,
                api_key=LLM_API_KEY,
                base_url=LLM_BASE_URL,
                temperature=0.3,
                max_tokens=1024,
                timeout=60,
            )
            messages = [
                SystemMessage(content=STRATEGY_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
            response = llm.invoke(messages)
            raw_text = response.content.strip()

            parsed = _parse_strategy_json(raw_text)
            strategy = RetrievalStrategy(**parsed)

            # Apply porosity-aware clamping
            weights_dict = _clamp_weights(strategy.model_dump(), features)

            if validate_weights(strategy):
                print(f"  [OK] LLM weights: SBU={weights_dict['sbu_weight']:.2f}, "
                      f"Ligand={weights_dict['ligand_weight']:.2f}, "
                      f"Physical={weights_dict['physical_weight']:.2f}")
                print(f"  Reasoning: {strategy.reasoning[:120]}...")
                return {
                    "retrieval_strategy": weights_dict,
                }
            else:
                total = strategy.sbu_weight + strategy.ligand_weight + strategy.physical_weight
                print(f"  [!] Weight sum={total:.2f} != 1.0, falling back to defaults")
        except Exception as e:
            print(f"  [!] LLM parsing failed: {e}")
    else:
        print(f"  [LLM] Mock mode, using heuristic weight assignment")

    # Fallback: heuristic weight assignment with porosity-aware clamping
    phys = features.get("physical_properties", {})
    cif = features.get("cif_metadata", {})
    lcd = phys.get("lcd") or 999
    vf = phys.get("vf") or 0.5
    pld = phys.get("pld")
    pconf = phys.get("porosity_confidence", "low")
    percolates = phys.get("percolates", False)
    metals = cif.get("metal_elements", [])

    catalytic_metals = {"Cu", "Fe", "Co", "Ni", "Pd", "Pt", "Cr", "Mn", "Mo", "Ru", "Rh", "Ir"}
    has_catalytic = bool(set(metals) & catalytic_metals)

    is_narrow = (pld is not None and pld < 2.4) or (vf is not None and vf < 0.10) or (percolates is False)

    if pconf == "low":
        weights = {"sbu_weight": 0.40, "ligand_weight": 0.35, "physical_weight": 0.25}
        reasoning = "Porosity confidence is low; physical_weight capped at 0.25."
    elif is_narrow:
        weights = {"sbu_weight": 0.45 if has_catalytic else 0.35,
                   "ligand_weight": 0.30,
                   "physical_weight": 0.25 if has_catalytic else 0.30}
        reasoning = "PLD/VF very low or structure non-percolating; physical_weight capped <= 0.30; gas storage not recommended."
    elif lcd < 5 and percolates:
        weights = {"sbu_weight": 0.2, "ligand_weight": 0.25, "physical_weight": 0.55}
        reasoning = "LCD < 5A and percolating -> gas sieving scenario; physical pore weight dominates."
    elif has_catalytic and vf > 0.5 and percolates:
        weights = {"sbu_weight": 0.45, "ligand_weight": 0.25, "physical_weight": 0.3}
        reasoning = "Catalytic active metal + moderate+ porosity + percolation -> SBU weight dominates."
    elif vf > 0.7 and percolates:
        weights = {"sbu_weight": 0.25, "ligand_weight": 0.25, "physical_weight": 0.5}
        reasoning = "High porosity + percolation -> gas storage scenario; physical pore weight dominates."
    else:
        weights = DEFAULT_WEIGHTS.copy()
        reasoning = DEFAULT_WEIGHT_REASONING

    # Apply final safety clamp
    weights = _clamp_weights(weights, features)

    print(f"  [Fallback] Weights: SBU={weights['sbu_weight']:.2f}, "
          f"Ligand={weights['ligand_weight']:.2f}, "
          f"Physical={weights['physical_weight']:.2f}")
    return {
        "retrieval_strategy": {
            **weights,
            "reasoning": reasoning,
        }
    }
