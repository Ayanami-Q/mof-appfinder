# MOF Agent

Intelligent MOF (Metal-Organic Framework) retrieval agent that uses LangGraph with dynamic weighted routing and ReAct to analyze a target MOF CIF file, search across multi-channel vector databases (SBU SOAP, ligand Morgan fingerprints, physical pore features), and produce an expert-level analysis report with literature enrichment.

## Installation

```bash
pip install -e .
```

## Usage

### Offline / mock mode (no API keys required)

```bash
# Using the console script
mof-agent --physical_dir /path/to/physical_db \
          --ligand_db /path/to/ligand.db --ligand_index /path/to/ligand.index \
          --sbu_db /path/to/sbu.db --sbu_index /path/to/sbu.index

# Using python -m
python -m mof_agent.cli --physical_dir /path/to/physical_db \
          --ligand_db /path/to/ligand.db --ligand_index /path/to/ligand.index \
          --sbu_db /path/to/sbu.db --sbu_index /path/to/sbu.index
```

Without `--cif_path`, a dummy MOF-5 CIF file is auto-generated for testing.

### Real LLM mode

Set environment variables (or a `.env` file) and add `--use-real-llm`:

```bash
export LLM_API_KEY=your-api-key
export LLM_BASE_URL=https://api.deepseek.com/v1
export LLM_MODEL=deepseek-chat
export USE_REAL_LLM=true

mof-agent --use-real-llm --cif_path /path/to/mof.cif \
          --physical_dir /path/to/physical_db \
          --ligand_db /path/to/ligand.db --ligand_index /path/to/ligand.index \
          --sbu_db /path/to/sbu.db --sbu_index /path/to/sbu.index
```

## External artifacts

Large MOF databases (FAISS indexes, SQLite databases, physical feature `.npy`/`.pkl` files) and knowledge base JSON files are **external artifacts** and are not committed to this repository. You must build or obtain them separately. Set their paths via environment variables or CLI arguments.

The only committed data file is `src/mof_agent/data/MOF_names_and_CSD_codes.csv` (~1 MB), which maps CoRE-MOF IDs to CSD refcodes and DOIs.

## Architecture

```
CIF input -> [Node 1: Feature Analysis] -> [Node 2: Strategy Thinking]
         -> [Node 3: Weighted Parallel Retrieval] -> [Node 4: Candidate Review]
         -> [Node 5: Literature Retrieval] -> [Node 6: Final Report]
```

The agent uses a LangGraph StateGraph DAG with MemorySaver checkpointing. Three parallel retrieval channels (SBU FAISS, Ligand FAISS, Physical KNN) are combined with LLM-determined dynamic weights. Top-15 candidates are reviewed by the LLM to select a Top-5, and literature context is fetched via CrossRef, Unpaywall, and publisher APIs before producing the final report.

## Security

- **Never commit API keys.** Use environment variables or a `.env` file (excluded via `.gitignore`).
- All API keys and database paths default to empty values in the source code.
- See `.env.example` for required configuration.
