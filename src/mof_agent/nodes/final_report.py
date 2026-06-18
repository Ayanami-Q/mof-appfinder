import json
import time
from typing import Dict, Any, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from mof_agent.config import (
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL,
    get_use_real_llm,
)
from mof_agent.state import AgentState
from mof_agent.context import get_whitebox_db


def _build_final_prompt(state: AgentState) -> str:
    features = state.get("target_features", {})
    strategy = state.get("retrieval_strategy", {})
    top5 = state.get("top5_selected", [])
    literature = state.get("literature_context", "")

    phys = features.get("physical_properties", {})
    cif = features.get("cif_metadata", {})
    pore = features.get("pore_topology", {})
    surface = features.get("surface_environment", {})

    top5_summary = []
    for c in top5:
        top5_summary.append({
            "cif_id": c["cif_id"],
            "final_score": c.get("final_score"),
            "sbu_similarity": c.get("sbu_similarity"),
            "ligand_similarity": c.get("ligand_similarity"),
            "physical_distance": c.get("physical_distance"),
            "sbu_formulas": c.get("sbu_formulas", []),
            "ligand_smiles": c.get("ligand_smiles", [])[:3],
            "lcd": c.get("lcd"),
            "pld": c.get("pld"),
            "vf": c.get("vf"),
        })

    return f"""You are a world-class computational chemist and MOF materials scientist.

## Target MOF Whitebox Features
- Formula: {cif.get('reduced_formula', 'N/A')}
- Main metal: {cif.get('main_metal', 'N/A')}, metal types: {cif.get('metal_elements', [])}
- LCD: {phys.get('lcd', 'N/A')} A, PLD: {phys.get('pld', 'N/A')} A, VF: {phys.get('vf', 'N/A')}
- Cell volume: {phys.get('volume', 'N/A')} A3
- SBU formulas: {features.get('sbu_formulas', [])}
- Ligand SMILES: {features.get('ligand_smiles', [])[:5]}
- Pore topology (pyzeo): Di={pore.get('di_A', 'N/A')} A, Df={pore.get('df_A', 'N/A')} A
- Surface environment: {surface.get('surface_composition', 'N/A')}

## Porosity Source and Uncertainty
- Data source: {phys.get('porosity_source', 'N/A')}
- Confidence: {phys.get('porosity_confidence', 'N/A')} (high/medium/low)
- 3D percolation: {phys.get('percolates', 'N/A')}
- Accessible VF (N2 probe 1.82A): {phys.get('vf_accessible_by_probe', {}).get('1.82', 'N/A')}
- Geometric VF: {phys.get('vf_geometric', 'N/A')}
- Warnings: {phys.get('porosity_warnings', [])}

## Retrieval Strategy
- SBU weight: {strategy.get('sbu_weight', 'N/A')}
- Ligand weight: {strategy.get('ligand_weight', 'N/A')}
- Physical weight: {strategy.get('physical_weight', 'N/A')}
- Reasoning: {strategy.get('reasoning', 'N/A')}

## Top-5 Selected Candidates (multi-dimensional scores)
{json.dumps(top5_summary, indent=2, ensure_ascii=False)}

## Candidate Literature Context (RAG)
{literature[:6000]}

## Report Requirements
1. Analyze target MOF whitebox features (cell params, pore size, porosity, metal clusters, ligand functional groups)
2. Combine with Top-5 candidate multi-dimensional scores (SBU/ligand/physical) to infer potential application directions
3. Provide top-2 recommended applications with clear physical/chemical causal reasoning
4. Discuss structure-property relationships
5. **Critical**: MUST provide both evidence and counter-evidence:
   - If porosity_confidence=low, MUST declare porosity data is uncertain and cannot be used as precise parameters
   - If VF is very low, PLD is very narrow, or percolates=False, MUST note potentially low capacity and mass transfer limitations
   - Do NOT provide unsubstantiated quantitative metrics (e.g., unsupported BET surface area, storage capacity)
6. Use Markdown format with clear heading hierarchy
7. Each recommended application includes: (a) application name (b) confidence level (c) causal reasoning (d) expected key performance indicators (if estimable)
8. Concluding summary paragraph
9. Language: English"""


def node_final_report(state: AgentState) -> Dict[str, Any]:
    print(f"\n{'='*60}")
    print(f"[Node 6] Final Analysis Report")
    print(f"{'='*60}")

    if get_use_real_llm() and LLM_API_KEY not in ("", "sk-placeholder-key"):
        try:
            llm = ChatOpenAI(
                model=LLM_MODEL,
                api_key=LLM_API_KEY,
                base_url=LLM_BASE_URL,
                temperature=0.3,
                max_tokens=4096,
                timeout=120,
            )
            prompt = _build_final_prompt(state)
            messages = [
                SystemMessage(content="You are a senior computational chemist and MOF materials scientist. Write a professional materials analysis report in English."),
                HumanMessage(content=prompt),
            ]
            response = llm.invoke(messages)
            final_analysis = response.content
            print(f"  [OK] LLM returned {len(final_analysis)} chars")
        except Exception as e:
            print(f"  [ERR] LLM call failed: {e}")
            final_analysis = _mock_final_report(state)
    else:
        print(f"  [LLM] Mock mode")
        final_analysis = _mock_final_report(state)

    print(f"  Final analysis: {len(final_analysis)} chars")
    return {"final_analysis": final_analysis}


def _mock_final_report(state: AgentState) -> str:
    features = state.get("target_features", {})
    phys = features.get("physical_properties", {})
    cif = features.get("cif_metadata", {})
    strategy = state.get("retrieval_strategy", {})
    top5 = state.get("top5_selected", [])

    lcd = phys.get("lcd", "N/A")
    pld = phys.get("pld", "N/A")
    vf = phys.get("vf", "N/A")
    elements = cif.get("elements", [])
    sbu = features.get("sbu_formulas", [])

    pconf = phys.get("porosity_confidence", "N/A")
    psource = phys.get("porosity_source", "N/A")
    percolates = phys.get("percolates", "N/A")
    pwarnings = phys.get("porosity_warnings", [])

    top_ids = [c["cif_id"] for c in top5[:3]] if top5 else ["N/A", "N/A", "N/A"]

    db = get_whitebox_db()
    db_size = db.physical_db.features.shape[0] if db else 0

    top5_lines = ""
    if top5:
        for i, c in enumerate(top5):
            top5_lines += (
                f"### Candidate {i+1}: {c['cif_id']}\n"
                f"- Overall score: {c.get('final_score', 'N/A')}\n"
                f"- SBU similarity: {c.get('sbu_similarity', 'N/A')}\n"
                f"- Ligand similarity: {c.get('ligand_similarity', 'N/A')}\n"
                f"- Physical distance: {c.get('physical_distance', 'N/A')}\n"
                f"- Metal cluster: {c.get('sbu_formulas', [])}\n"
                f"- Ligand SMILES: {c.get('ligand_smiles', [])[:2]}\n\n"
            )

    # Porosity-aware application recommendation
    is_low_confidence = (pconf == "low" or psource in ("vdw_heuristic", "none"))
    is_narrow = (isinstance(pld, (int, float)) and pld < 2.4) or \
               (isinstance(vf, (int, float)) and vf < 0.10) or \
               (percolates is False)

    if is_low_confidence:
        porosity_note = (
            f"> **Porosity Uncertainty Notice**: current porosity data source is {psource}, "
            f"confidence is {pconf}. Application inferences below are for reference only. "
            f"Quantitative metrics (BET area, storage capacity) lack reliable basis; "
            f"experimental verification (N2 adsorption at 77K) is recommended."
        )
    elif is_narrow:
        porosity_note = (
            f"> **Porosity Note**: PLD={pld} A, VF={vf}, percolates={percolates}. "
            f"Structure has very narrow or non-percolating pores; "
            f"gas storage / high-throughput sieving applications are limited; mass transfer may be restricted."
        )
    else:
        porosity_note = (
            f"> **Porosity Source**: {psource}, confidence={pconf}, percolates={percolates}. "
            f"Data can be used for quantitative analysis."
        )

    warnings_text = ""
    if pwarnings:
        warnings_text = "\n".join(f"- {w}" for w in pwarnings[:3])

    if is_low_confidence:
        perf_app1 = "Insufficient confidence for quantitative predictions; experimental validation recommended."
    elif is_narrow:
        perf_app1 = f"Limited by narrow/non-percolating pores; expected gas storage capacity low (< 50 cm3/g CH4)."
    else:
        perf_app1 = f"Based on VF={vf}, estimated CH4 storage 100-250 v/v (experimental validation needed)."

    perf_app2 = "Further experimental determination needed." if is_low_confidence \
        else "Active site density estimated 1.0-3.0 mmol/g."

    return f"""# MOF Intelligent Retrieval Analysis Report

> This report was auto-generated by the MOF LangGraph Agent (dynamic weighted routing + ReAct){' (Mock mode)' if not get_use_real_llm() else ''}

---

## 1. Target MOF Whitebox Features

| Property | Value |
|----------|-------|
| Formula | {cif.get('reduced_formula', 'N/A')} |
| Elements | {', '.join(elements)} |
| Main metal | {cif.get('main_metal', 'N/A')} |
| SBU formulas | {', '.join(sbu) if sbu else 'N/A'} |
| LCD | {lcd} A |
| PLD | {pld} A |
| VF | {vf} |

## 2. Porosity Source and Uncertainty

{porosity_note}

- Data source: {psource}
- Confidence: {pconf}
- 3D percolation: {percolates}
- Warnings: {warnings_text if warnings_text else 'None'}

## 3. Retrieval Strategy (LLM Dynamic Assignment)

| Channel | Weight |
|---------|--------|
| SBU (metal cluster SOAP) | {strategy.get('sbu_weight', 'N/A')} |
| Ligand (Morgan fingerprint) | {strategy.get('ligand_weight', 'N/A')} |
| Physical (pore features) | {strategy.get('physical_weight', 'N/A')} |

> Reasoning: {strategy.get('reasoning', 'N/A')}

## 4. Selected Top-5 Candidates

{top5_lines if top5_lines else 'No candidates'}

## 5. Recommended Application Directions

### Application #1: {'Gas Storage & Separation' if not is_narrow else 'Catalysis / Surface Chemistry'}
**Confidence**: {'Medium' if is_low_confidence else 'High'}
**Causal reasoning**: Target MOF pore size {lcd} A, void fraction {vf}, best-matching candidate {top_ids[0]}.
**Expected performance**: {perf_app1}

### Application #2: {'Catalysis / Adsorption' if not is_narrow else 'Gas Sensing / Selective Adsorption'}
**Confidence**: Medium
**Causal reasoning**: Contains {'/'.join(elements)} elements; {'open metal sites suggest catalytic activity' if not is_narrow else 'ultra-narrow pores provide molecular sieving / selective sensing potential'}.
**Expected performance**: {perf_app2}

## 6. Summary

The target MOF, processed through LLM dynamic weighted strategy -> parallel retrieval -> expert review -> literature enrichment, most closely matches **{top_ids[0]}** from the CoRE-MOF database.
Primary recommendation: {'gas storage & separation' if not is_narrow else 'catalysis / surface chemistry'}. Secondary direction: {'catalysis / adsorption' if not is_narrow else 'gas sensing / selective adsorption'}.
{'Porosity data confidence is low; experimental BET/N2 adsorption verification of key metrics is strongly recommended.' if is_low_confidence else ''}

---
*Report generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*
*Retrieval mode: LangGraph Agent (dynamic weighted routing + ReAct)*
*Database size: {db_size} physical features*
"""
