# Architecture

## Pipeline overview

1. **Feature Extraction** — Parse the target MOF CIF, run MOFCleaner, MOFCIFAnalyzer (pymatgen + pyzeo), and whitebox extraction to produce SBU SOAP vectors, ligand Morgan fingerprints, and physical pore features (LCD, PLD, VF).

2. **Multi-Channel Retrieval** — Query three parallel indices:
   - SBU channel: FAISS index of metal cluster SOAP descriptors
   - Ligand channel: FAISS index of Morgan fingerprints
   - Physical channel: KNN over cell parameters and pore features
   Results are combined with LLM-determined dynamic weights.

3. **Candidate Review** — The LLM reviews condensed Top-15 summaries (with porosity confidence/risk annotations) and selects the 5 with the strongest chemical causal relationships.

4. **Literature Enrichment** — For each Top-5 candidate, fetch literature context from the knowledge base, publisher APIs (Elsevier/Springer/Wiley), CrossRef metadata, and PDF sources.

5. **Final Report** — An expert-level English report combining all data, with explicit evidence and counter-evidence, including porosity uncertainty disclosure and application recommendations.

## Key design decisions

- **Offline-first**: Default mock/heuristic mode works without any LLM API keys.
- **Uncertainty-aware**: Low-confidence porosity data is explicitly downgraded in weight assignment and flagged in the final report.
- **Package isolation**: All source code lives under `src/mof_agent/` with package-relative imports.
- **External databases**: Large FAISS indexes, SQLite databases, and knowledge base files are external artifacts — paths are configured via environment variables or CLI.
