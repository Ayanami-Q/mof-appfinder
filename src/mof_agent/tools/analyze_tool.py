import os
import tempfile
from typing import Dict, Any, List

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter

from mof_agent.tools.mof_cleaner import MOFCleaner
from mof_agent.tools.mof_analyzer_pro import MOFCIFAnalyzer, METAL_ELEMENTS
from mof_agent.tools.whitebox_extractor import (
    extract_all_features,
    extract_metal_formulas_from_structure,
)


class AnalyzeMOFTool:
    """Target MOF feature analysis: cleaning, deep analysis, and whitebox extraction."""

    @staticmethod
    def run(cif_path: str) -> Dict[str, Any]:
        print(f"\n{'='*60}")
        print(f"[AnalyzeMOFTool] Starting analysis: {cif_path}")
        print(f"{'='*60}")

        errors: List[str] = []

        structure = Structure.from_file(cif_path)

        try:
            cleaned_structure, clean_log = MOFCleaner.clean(structure)
            print(f"  [Clean] Atoms: {clean_log['initial_atoms']} -> {clean_log['final_atoms']}")
            if clean_log.get("removed_solvents"):
                print(f"    Removed solvents: {clean_log['removed_solvents']}")
        except Exception as e:
            print(f"  [Clean] Failed: {e}, using original structure")
            cleaned_structure = structure
            clean_log = {"error": str(e)}

        with tempfile.NamedTemporaryFile(suffix=".cif", delete=False, prefix="cleaned_") as f:
            CifWriter(cleaned_structure).write_file(f.name)
            cleaned_cif_path = f.name

        try:
            try:
                analyzer = MOFCIFAnalyzer(cleaned_cif_path)
                pore_info = analyzer.get_pore_and_window_topology()
                surface_info = analyzer.assess_surface_polarity()

                elements = [str(site.specie) for site in cleaned_structure]
                metals = [e for e in elements if e in METAL_ELEMENTS]
                main_metal = max(set(metals), key=metals.count) if metals else None

                cif_metadata = {
                    "reduced_formula": str(cleaned_structure.composition.reduced_formula),
                    "num_atoms": len(cleaned_structure),
                    "elements": sorted(set(elements)),
                    "metal_elements": sorted(set(metals)),
                    "main_metal": main_metal,
                    "space_group": cleaned_structure.get_space_group_info()[0],
                    "volume_A3": round(float(cleaned_structure.lattice.volume), 3),
                }
                print(f"  [Analyze] Formula: {cif_metadata['reduced_formula']}, "
                      f"main metal: {main_metal}, atom count: {cif_metadata['num_atoms']}")
            except Exception as e:
                print(f"  [Analyze] MOFCIFAnalyzer failed: {e}")
                errors.append(f"MOFCIFAnalyzer: {e}")
                pore_info = {}
                surface_info = {}
                cif_metadata = {
                    "reduced_formula": str(cleaned_structure.composition.reduced_formula),
                    "num_atoms": len(cleaned_structure),
                    "elements": sorted(set(str(site.specie) for site in cleaned_structure)),
                    "metal_elements": [],
                }

            try:
                wb = extract_all_features(cleaned_cif_path)
                errors.extend(wb.get("errors", []))
            except Exception as e:
                print(f"  [Extract] whitebox_extractor failed: {e}")
                errors.append(f"whitebox_extractor: {e}")
                wb = {"sbu_vec": None, "ligand_vec": None, "physical_properties": {},
                      "sbu_chemical_formulas": [], "ligand_smiles_strings": [], "errors": [str(e)]}

            sbu_formulas = wb.get("sbu_chemical_formulas", [])
            if not sbu_formulas:
                sbu_formulas = extract_metal_formulas_from_structure(cleaned_structure)

            target_features = {
                "cif_metadata": cif_metadata,
                "pore_topology": pore_info,
                "surface_environment": surface_info,
                "physical_properties": wb.get("physical_properties", {}),
                "sbu_formulas": sbu_formulas,
                "ligand_smiles": wb.get("ligand_smiles_strings", []),
                "clean_log": clean_log,
            }

            return {
                "target_features": target_features,
                "sbu_vec": wb.get("sbu_vec"),
                "ligand_vec": wb.get("ligand_vec"),
                "extraction_errors": errors,
            }

        finally:
            if os.path.exists(cleaned_cif_path):
                os.unlink(cleaned_cif_path)
