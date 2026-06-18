import os
import sys

import mof_agent.config as config
from mof_agent.config import (
    DEFAULT_PHYSICAL_DIR, DEFAULT_LIGAND_DB, DEFAULT_LIGAND_INDEX,
    DEFAULT_SBU_DB, DEFAULT_SBU_INDEX, CSD_CSV_PATH, KNOWLEDGE_BASE_PATH,
)
from mof_agent.state import AgentState
from mof_agent.graph import build_graph
from mof_agent.context import (
    init_whitebox_db, init_csd_lookup, init_retrieve_tool, init_knowledge_base,
    get_whitebox_db, get_csd_lookup, KnowledgeBase,
)
from mof_agent.database import WhiteboxDBManager
from mof_agent.lookup import CSDLookup
from mof_agent.tools import WeightedParallelRetrieveTool
from mof_agent.dummy_cif import generate_dummy_cif


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="MOF Intelligent Retrieval Agent (LangGraph dynamic weighted routing + ReAct)")
    parser.add_argument("--cif_path", type=str, default=None,
                        help="Input CIF file path (default: auto-generate dummy MOF-5)")
    parser.add_argument("--use-real-llm", action="store_true",
                        help="Enable real LLM API calls")
    parser.add_argument("--physical_dir", type=str, default=DEFAULT_PHYSICAL_DIR or None,
                        help="Physical feature database directory")
    parser.add_argument("--ligand_db", type=str, default=DEFAULT_LIGAND_DB or None,
                        help="Ligand FAISS SQLite path")
    parser.add_argument("--ligand_index", type=str, default=DEFAULT_LIGAND_INDEX or None,
                        help="Ligand FAISS index path")
    parser.add_argument("--sbu_db", type=str, default=DEFAULT_SBU_DB or None,
                        help="SBU FAISS SQLite path")
    parser.add_argument("--sbu_index", type=str, default=DEFAULT_SBU_INDEX or None,
                        help="SBU FAISS index path")
    parser.add_argument("--thread-id", type=str, default="mof_agent_session_1",
                        help="MemorySaver session ID (for cross-turn persistence)")
    args = parser.parse_args()

    if args.use_real_llm:
        config.set_use_real_llm(True)

    missing = []
    for name, val in [("--physical_dir", args.physical_dir),
                      ("--ligand_db", args.ligand_db),
                      ("--ligand_index", args.ligand_index),
                      ("--sbu_db", args.sbu_db),
                      ("--sbu_index", args.sbu_index)]:
        if not val:
            missing.append(name)
    if missing:
        print(f"Error: missing required database arguments: {', '.join(missing)}")
        print("Set them via CLI or environment variables (PHYSICAL_DB_DIR, LIGAND_DB_PATH, etc.)")
        sys.exit(1)

    cif_path = args.cif_path or generate_dummy_cif("./dummy_mof5.cif")

    if os.path.isdir(cif_path):
        cif_files = sorted([
            os.path.join(cif_path, f) for f in os.listdir(cif_path)
            if f.endswith(".cif")
        ])
        if not cif_files:
            print(f"Error: no .cif files found in directory {cif_path}")
            sys.exit(1)
        cif_path = cif_files[0]
        if len(cif_files) > 1:
            print(f"[Info] {len(cif_files)} CIF files in directory, auto-selecting first: "
                  f"{os.path.basename(cif_path)}")

    if not os.path.isfile(cif_path):
        print(f"Error: CIF file not found: {cif_path}")
        sys.exit(1)

    use_real = config.get_use_real_llm()
    print(f"\n{'#'*60}")
    print(f"# MOF Intelligent Retrieval Agent — Dynamic Weighted Routing + ReAct")
    print(f"# CIF:          {cif_path}")
    print(f"# Physical Dir: {args.physical_dir}")
    print(f"# Ligand DB:    {args.ligand_db}")
    print(f"# SBU DB:       {args.sbu_db}")
    print(f"# LLM Mode:     {'REAL' if use_real else 'MOCK/Heuristic'}")
    print(f"# Session ID:   {args.thread_id}")
    print(f"{'#'*60}")

    print(f"\n[Init] Loading multi-channel databases...")

    wb_db = WhiteboxDBManager(
        physical_dir=args.physical_dir,
        ligand_db=args.ligand_db,
        ligand_index=args.ligand_index,
        sbu_db=args.sbu_db,
        sbu_index=args.sbu_index,
    )
    init_whitebox_db(wb_db)

    retrieve_tool = WeightedParallelRetrieveTool(wb_db)
    init_retrieve_tool(retrieve_tool)

    csd = CSDLookup(CSD_CSV_PATH)
    init_csd_lookup(csd)

    kb = None
    if os.path.isfile(KNOWLEDGE_BASE_PATH):
        kb = KnowledgeBase(KNOWLEDGE_BASE_PATH)
        init_knowledge_base(kb)
    else:
        print(f"[Init] Warning: knowledge base file not found: {KNOWLEDGE_BASE_PATH}, "
              f"will use literature fetch only")

    print(f"[Init] Databases ready: "
          f"{wb_db.physical_db.features.shape[0]} physical features, "
          f"{'SBU OK' if wb_db._sbu_loaded else 'SBU MISSING'}, "
          f"{'Ligand OK' if wb_db._ligand_loaded else 'Ligand MISSING'}, "
          f"{len(csd._refcode_to_doi)} CSD->DOI mappings, "
          f"KB={len(kb) if kb else 0} records")

    graph = build_graph()
    initial_state: AgentState = {"cif_path": cif_path}
    run_config = {"configurable": {"thread_id": args.thread_id}}

    print(f"\n{'~'*60}")
    print(f" Starting LangGraph Agent pipeline...")
    print(f" Node order: analyze -> think_strategy -> act_retrieve -> review -> fetch_lit -> final_report")
    print(f"{'~'*60}")

    result = graph.invoke(initial_state, run_config)

    print(f"\n{'='*60}")
    print(f" Agent State Summary")
    print(f"{'='*60}")

    tf = result.get("target_features", {})
    phys = tf.get("physical_properties", {})
    cif_meta = tf.get("cif_metadata", {})
    print(f" [node_analyze]        Target features extracted")
    print(f"   - Formula: {cif_meta.get('reduced_formula', 'N/A')}")
    print(f"   - LCD={phys.get('lcd', 'N/A')} A, PLD={phys.get('pld', 'N/A')} A, VF={phys.get('vf', 'N/A')}")
    print(f"   - SBU formulas: {tf.get('sbu_formulas', [])}")
    print(f"   - Extraction errors: {result.get('extraction_errors', [])}")

    strategy = result.get("retrieval_strategy", {})
    print(f" [node_think_strategy] Retrieval strategy")
    print(f"   - SBU={strategy.get('sbu_weight', 'N/A'):.3f}  "
          f"Ligand={strategy.get('ligand_weight', 'N/A'):.3f}  "
          f"Physical={strategy.get('physical_weight', 'N/A'):.3f}")
    print(f"   - Reasoning: {str(strategy.get('reasoning', 'N/A'))[:100]}")

    top15 = result.get("top15_candidates", [])
    print(f" [node_act_retrieve]   Weighted parallel retrieval -> Top-{len(top15)}")
    for i, c in enumerate(top15[:5], 1):
        print(f"   #{i}: {c['cif_id']} final={c['final_score']:.4f}")

    top5 = result.get("top5_selected", [])
    print(f" [node_review]         LLM review -> Top-{len(top5)} selected")
    for i, c in enumerate(top5, 1):
        pd = c.get('physical_distance')
        pd_str = f"{pd:.3f}" if pd is not None else "MISSING"
        print(f"   #{i}: {c['cif_id']} "
              f"sbu={c.get('sbu_similarity', 'N/A'):.3f} "
              f"lig={c.get('ligand_similarity', 'N/A'):.3f} "
              f"phys={pd_str} "
              f"sbu_f={c.get('sbu_formulas', [])} "
              f"lig_s={c.get('ligand_smiles', [])[:2]}")

    lit = result.get("literature_context", "")
    print(f" [node_fetch_lit]      RAG literature context: {len(lit)} chars")
    print(f" [node_final_report]   Final report: {len(result.get('final_analysis', ''))} chars")

    print(f"\n{'='*60}")
    print(f" Final Analysis Report")
    print(f"{'='*60}")
    print(result.get("final_analysis", "(no analysis result)"))
    print(f"\n{'#'*60}")
    print(f"# Pipeline complete | Session: {args.thread_id}")
    print(f"{'#'*60}")

    return result


if __name__ == "__main__":
    main()
