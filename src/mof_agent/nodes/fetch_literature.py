from typing import Dict, Any

from mof_agent.state import AgentState
from mof_agent.tools.lit_fetch_tool import LitFetchTool


def node_fetch_literature(state: AgentState) -> Dict[str, Any]:
    print(f"\n{'='*60}")
    print(f"[Node 5] Literature Retrieval (Top-5)")
    print(f"{'='*60}")

    top5 = state.get("top5_selected", [])
    if not top5:
        print(f"  [!] Top-5 is empty, skipping literature retrieval")
        return {"literature_context": "No candidates available for literature retrieval."}

    sections = []
    for i, cand in enumerate(top5, 1):
        cif_id = cand["cif_id"]
        physical_record = {
            "elements": cand.get("elements", []),
            "lcd": cand.get("lcd"),
            "pld": cand.get("pld"),
            "vf": cand.get("vf"),
            "volume": cand.get("volume"),
        }
        match_score = {
            "final_score": cand.get("final_score"),
            "sbu_similarity": cand.get("sbu_similarity"),
            "ligand_similarity": cand.get("ligand_similarity"),
            "physical_distance": cand.get("physical_distance"),
        }

        print(f"  [{i}/{len(top5)}] Retrieving: {cif_id}")
        lit = LitFetchTool.fetch(cif_id, physical_record, match_score)
        source_labels = {
            "knowledge_base": "[KB]",
            "publisher": "[PUB]",
            "pdf": "[PDF]",
            "crossref": "[API]",
            "heuristic": "[!]",
        }
        label = source_labels.get(lit["source"], "[?]")
        print(f"    {label} source={lit['source']} ({len(lit['content'])} chars)")

        score_lines = [
            f"\n**Multi-dimensional match scores**:",
            f"- Overall score: {match_score['final_score']}",
            f"- SBU similarity: {match_score['sbu_similarity']}",
            f"- Ligand similarity: {match_score['ligand_similarity']}",
            f"- Physical distance: {match_score['physical_distance']}",
        ]
        sections.append(
            f"## Candidate {i}: {cif_id} | source={lit['source']}\n"
            f"{lit['content']}\n" + "\n".join(score_lines) + "\n"
        )

    literature_context = "\n---\n".join(sections)
    print(f"\n  Literature context total length: {len(literature_context)} chars")
    return {"literature_context": literature_context}
