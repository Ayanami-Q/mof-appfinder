import warnings
import networkx as nx
from typing import Tuple, Dict, Any, List

from pymatgen.core import Structure, Composition
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import JmolNN


class MOFCleaner:
    COMMON_SOLVENTS = {
        "H2O", "C3H7NO", "C5H11NO", "CH4O", "C2H6O", "C3H8O",
        "C4H8O", "C3H6O", "C2H3N", "CHCl3", "CH2Cl2", "C4H10O",
        "C6H6", "C7H8", "C2H6OS", "O"
    }

    COMMON_ORG_IONS = {
        "H4N", "C2H8N", "C4H12N", "C8H20N", "C16H36N"
    }

    IONIC_ELEMENTS = {
        "F", "Cl", "Br", "I", "B", "P", "S",
        "Li", "Na", "K", "Rb", "Cs", "Mg", "Ca", "Sr", "Ba", "Ag"
    }

    @classmethod
    def _match_formula(cls, comp: Composition, formula_set: set) -> bool:
        if not comp.almost_valid:
            return False
        red_form = comp.reduced_formula
        for f in formula_set:
            if Composition(f).reduced_formula == red_form:
                return True
        return False

    @classmethod
    def clean(cls, structure: Structure, min_framework_size: int = 50) -> Tuple[Structure, Dict[str, Any]]:
        cleaned_sites = []
        for site in structure:
            if not site.is_ordered:
                dominant_species = max(site.species.items(), key=lambda x: x[1])[0]
                new_site = site.copy()
                new_site.species = {dominant_species: 1.0}
                cleaned_sites.append(new_site)
            else:
                cleaned_sites.append(site)

        working_struct = Structure.from_sites(cleaned_sites)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sg = StructureGraph.with_local_env_strategy(working_struct, JmolNN())

        components = list(nx.connected_components(sg.graph.to_undirected()))

        keep_indices = set()
        log_removed = []
        log_kept_ions = []

        for comp in components:
            if len(comp) >= min_framework_size:
                keep_indices.update(comp)
                continue

            sub_struct = Structure.from_sites([working_struct[i] for i in comp])
            comp_obj = sub_struct.composition

            is_solvent = cls._match_formula(comp_obj, cls.COMMON_SOLVENTS)
            is_ion = cls._match_formula(comp_obj, cls.COMMON_ORG_IONS)

            if not is_solvent and not is_ion:
                elements_in_comp = {str(el) for el in comp_obj.elements}
                if elements_in_comp.intersection(cls.IONIC_ELEMENTS):
                    is_ion = True

            if is_ion:
                keep_indices.update(comp)
                log_kept_ions.append(comp_obj.hill_formula)
            else:
                log_removed.append(comp_obj.hill_formula)

        final_struct = Structure.from_sites([working_struct[i] for i in keep_indices])

        log = {
            "initial_atoms": len(structure),
            "final_atoms": len(final_struct),
            "removed_solvents": log_removed,
            "kept_counter_ions": log_kept_ions
        }

        return final_struct, log
