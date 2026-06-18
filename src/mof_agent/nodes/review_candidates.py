import json
import re
from typing import Dict, Any, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from mof_agent.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, get_use_real_llm
from mof_agent.state import AgentState

REVIEW_SYSTEM_PROMPT = """You are a senior MOF materials scientist. Review the candidate list and select the 5 materials with the strongest chemical causal relationships. Output ONLY a JSON object, no other text."""


def _build_review_prompt(top15: List[Dict[str, Any]], features: Dict[str, Any]) -> str:
    summary = []
    for i, c in enumerate(top15):
        summary.append({
            "index": i,
            "cif_id": c["cif_id"],
            "final_score": c["final_score"],
            "sbu_sim": c["sbu_similarity"],
            "lig_sim": c["ligand_similarity"],
            "phys_dist": c["physical_distance"],
            "sbu_formulas": c.get("sbu_formulas", []),
            "ligand_smiles": c.get("ligand_smiles", [])[:3],
            "lcd": c.get("lcd"),
            "pld": c.get("pld"),
            "vf": c.get("vf"),
            "elements": c.get("elements", []),
            "porosity_confidence_candidate": c.get("porosity_confidence", "N/A"),
            "porosity_source_candidate": c.get("porosity_source", "N/A"),
            "percolates_candidate": c.get("percolates", "N/A"),
            "vf_accessible_candidate": c.get("vf_accessible_by_probe", {}),
        })

    phys = features.get("physical_properties", {})
    target_summary = {
        "lcd": phys.get("lcd"),
        "pld": phys.get("pld"),
        "vf": phys.get("vf"),
        "porosity_confidence": phys.get("porosity_confidence", "N/A"),
        "porosity_source": phys.get("porosity_source", "N/A"),
        "percolates": phys.get("percolates", "N/A"),
        "vf_accessible_by_probe": phys.get("vf_accessible_by_probe", {}),
        "sbu_formulas": features.get("sbu_formulas", []),
        "ligand_smiles": features.get("ligand_smiles", [])[:3],
        "elements": features.get("cif_metadata", {}).get("elements", []),
    }

    return f"""Select the 5 MOF candidates with the strongest chemical causal relationships from the following 15.

## Target MOF Features
{json.dumps(target_summary, indent=2, ensure_ascii=False)}

## Top-15 Candidate Summaries
{json.dumps(summary, indent=2, ensure_ascii=False)}

## Selection Criteria
1. SBU chemical dimension: metal cluster formula or metal element similarity
2. Ligand chemical dimension: ligand SMILES / functional group similarity
3. Physical dimension: LCD/PLD/VF close to target
4. Overall: candidates with balanced and high scores across all three channels preferred

## Risk Flagging Rules
- If candidate physical pore data is missing (porosity_source=N/A or physical_distance=null) or confidence=low, you MUST flag the risk in reasoning; do not claim physical similarity based solely on final_score
- If target or candidate percolates=False or vf < 0.10, reduce the weight of physical similarity

Output strictly valid JSON:
{{"selected_indices": [0, 3, 7, 11, 14], "selection_reasoning": "Brief chemical justification for each selected candidate"}}"""


def _parse_review_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    match = re.search(r'```(?:json)?\s*({.*?})\s*```', text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    match = re.search(r'\{[^{}]*"selected_indices"[^{}]*\}', text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"Cannot parse JSON from LLM output: {text[:200]}")


def node_review_candidates(state: AgentState) -> Dict[str, Any]:
    print(f"\n{'='*60}")
    print(f"[Node 4] LLM Candidate Review")
    print(f"{'='*60}")

    top15 = state.get("top15_candidates", [])
    features = state.get("target_features", {})

    if not top15:
        print(f"  [!] Top-15 is empty, skipping review")
        return {"top5_selected": []}

    if get_use_real_llm() and LLM_API_KEY not in ("", "sk-placeholder-key"):
        try:
            llm = ChatOpenAI(
                model=LLM_MODEL,
                api_key=LLM_API_KEY,
                base_url=LLM_BASE_URL,
                temperature=0.3,
                max_tokens=2048,
                timeout=90,
            )
            user_prompt = _build_review_prompt(top15, features)
            messages = [
                SystemMessage(content=REVIEW_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
            response = llm.invoke(messages)
            raw_text = response.content.strip()

            parsed = _parse_review_json(raw_text)
            valid_indices = [int(i) for i in parsed.get("selected_indices", [])
                            if 0 <= int(i) < len(top15)]
            top5 = [top15[i] for i in valid_indices[:5]]
            reasoning = parsed.get("selection_reasoning", "")

            print(f"  [OK] LLM selected {len(top5)} candidates")
            for i, c in enumerate(top5, 1):
                print(f"      #{i}: {c['cif_id']} (index={valid_indices[i-1]})")
            print(f"  Reasoning: {reasoning[:120]}...")
            return {"top5_selected": top5}
        except Exception as e:
            print(f"  [!] LLM review failed: {e}, falling back to Top-5 by final_score")

    # Fallback
    top5 = sorted(top15, key=lambda x: x["final_score"], reverse=True)[:5]
    print(f"  [Fallback] Top-5 by final_score")
    for i, c in enumerate(top5, 1):
        print(f"      #{i}: {c['cif_id']} final={c['final_score']:.4f}")
    return {"top5_selected": top5}
