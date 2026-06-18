import os
import json
import sys
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import networkx as nx

warnings.filterwarnings("ignore")

from pymatgen.core import Structure, Composition
from pymatgen.analysis.local_env import CrystalNN

os.environ.setdefault("RDKIT_LOG_LEVEL", "ERROR")
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

from dscribe.descriptors import SOAP

try:
    from mof_agent.tools.mof_sbu_analyzer import SBUExtractor, compute_soap_vector
    _SBU_AVAILABLE = True
except ImportError:
    _SBU_AVAILABLE = False

from mof_agent.tools.mof_analyzer_pro import VDW_RADII, DEFAULT_VDW_RADIUS


FINGERPRINT_RADIUS = 2
FINGERPRINT_BITS = 2048
SOAP_R_CUT = 6.0
SOAP_N_MAX = 4
SOAP_L_MAX = 4
SOAP_SIGMA = 0.5

SUPPORTED_ELEMENTS = [
    "H", "C", "N", "O", "F", "Cl", "S", "P",
    "Zn", "Cu", "Co", "Ni", "Fe", "Mn", "Zr", "Ti", "V", "Cr", "Cd", "Ag",
]

METAL_ELEMENTS = {
    "Zn", "Cu", "Co", "Ni", "Fe", "Mn", "Zr", "Ti", "V", "Cr",
    "Cd", "Ag", "Au", "Pt", "Pd", "Ru", "Rh", "Ir", "Os", "Mo", "W",
    "Al", "Ga", "In", "Mg", "Ca", "Sr", "Ba", "Li", "Na", "K", "Rb", "Cs",
    "Sc", "Y", "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Nb", "Ta", "Re", "Sb", "Bi",
    "Sn", "Pb", "Th", "U",
}

_soap_generator: Optional[SOAP] = None


def _get_soap_generator() -> SOAP:
    global _soap_generator
    if _soap_generator is None:
        _soap_generator = SOAP(
            species=SUPPORTED_ELEMENTS, periodic=False, r_cut=SOAP_R_CUT,
            n_max=SOAP_N_MAX, l_max=SOAP_L_MAX, sigma=SOAP_SIGMA,
            average="off", sparse=False,
        )
    return _soap_generator


def _build_global_graph(structure: Structure) -> nx.Graph:
    cnn = CrystalNN(distance_cutoffs=None, x_diff_weight=0.0)
    g = nx.Graph()
    for i, site in enumerate(structure):
        g.add_node(i, element=str(site.specie), coords=site.coords)
    for i in range(len(structure)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nn_info = cnn.get_nn_info(structure, i)
        for nn in nn_info:
            j = nn["site_index"]
            if i != j:
                g.add_edge(i, j)
    return g


def extract_sbu_features_enhanced(cif_path: str) -> Tuple[Optional[np.ndarray], List[str]]:
    formulas = []
    if not _SBU_AVAILABLE:
        return None, formulas

    try:
        structure = Structure.from_file(cif_path)
        extractor = SBUExtractor(cif_path)
        clusters = extractor.extract_clusters()

        if not clusters:
            metal_species = sorted(set(
                str(site.specie) for site in structure
                if str(site.specie) in METAL_ELEMENTS
            ))
            return None, metal_species

        for cluster in clusters:
            elements = cluster["elements"]
            comp = Composition("".join(elements))
            formulas.append(comp.reduced_formula)

        formulas = sorted(list(set(formulas)))

        soap = _get_soap_generator()
        cluster_vectors = []
        for cluster in clusters:
            vec = compute_soap_vector(cluster, soap)
            if vec is not None:
                cluster_vectors.append(vec)

        if not cluster_vectors:
            return None, formulas

        sbu_vec = np.mean(cluster_vectors, axis=0).astype(np.float32)
        norm = np.linalg.norm(sbu_vec)
        if norm > 1e-12:
            sbu_vec /= norm
        return sbu_vec, formulas

    except Exception as e:
        print(f"  [SBU Extract] Failed: {e}", file=sys.stderr)
        return None, formulas


def _fragment_to_morgan_and_smiles(
    atom_elements: List[str], atom_coords: np.ndarray, bonds: List[Tuple[int, int]]
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    try:
        mol = Chem.RWMol()
        idx_map = {}
        for i, elem in enumerate(atom_elements):
            atom = Chem.Atom(elem)
            idx = mol.AddAtom(atom)
            idx_map[i] = idx

        for i, j in bonds:
            if i in idx_map and j in idx_map:
                try:
                    mol.AddBond(idx_map[i], idx_map[j], Chem.BondType.SINGLE)
                except Exception:
                    pass

        mol = mol.GetMol()
        Chem.SanitizeMol(mol)
        smiles = Chem.MolToSmiles(mol)
        if not smiles:
            return None, None

        fp = AllChem.GetMorganFingerprintAsBitVect(
            Chem.MolFromSmiles(smiles), FINGERPRINT_RADIUS, nBits=FINGERPRINT_BITS
        )
        arr = np.zeros((1,), dtype=np.float32)
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
        return arr, smiles
    except Exception:
        return None, None


def _fragment_to_graph_fingerprint(atom_elements: List[str]) -> np.ndarray:
    from collections import Counter
    counts = Counter(atom_elements)
    total = len(atom_elements) or 1
    base_elements = ["H", "C", "N", "O", "F", "Cl", "S", "P", "Br", "I"]
    vec = np.zeros(128, dtype=np.float32)
    for i, el in enumerate(base_elements):
        vec[i] = counts.get(el, 0) / total
    c, n, o, h = counts.get("C", 0), counts.get("N", 0), counts.get("O", 0), counts.get("H", 0)
    vec[64] = (n + o + counts.get("S", 0) + counts.get("F", 0) + counts.get("Cl", 0) + counts.get("Br", 0)) / total
    vec[65] = c / max(h, 1)
    vec[66] = c / max(n, 1)
    rng = np.random.RandomState(
        sum(ord(c) * (i + 1) for i, c in enumerate(sorted(counts.keys())[:5])) % (2**31))
    vec[67:128] = rng.randn(61).astype(np.float32) * 0.01
    norm = np.linalg.norm(vec)
    if norm > 1e-12:
        vec /= norm
    return vec


def extract_ligand_features_enhanced(cif_path: str) -> Tuple[Optional[np.ndarray], List[str]]:
    smiles_list = []
    try:
        structure = Structure.from_file(cif_path)
        g = _build_global_graph(structure)

        metal_indices = {n for n, attr in g.nodes(data=True) if attr["element"] in METAL_ELEMENTS}
        organic_indices = set(g.nodes()) - metal_indices

        if not organic_indices:
            return None, smiles_list

        organic_subgraph = g.subgraph(organic_indices)
        ligand_vecs = []

        for component in nx.connected_components(organic_subgraph):
            if len(component) < 3:
                continue

            comp_list = list(component)
            elements = [g.nodes[idx]["element"] for idx in comp_list]
            coords = np.array([g.nodes[idx]["coords"] for idx in comp_list])

            idx_to_pos = {idx: pos for pos, idx in enumerate(comp_list)}
            bonds = []
            for idx in comp_list:
                for neighbor in g.neighbors(idx):
                    if neighbor in idx_to_pos and idx < neighbor:
                        bonds.append((idx_to_pos[idx], idx_to_pos[neighbor]))

            if not bonds and len(elements) > 10:
                from scipy.spatial import cKDTree
                tree = cKDTree(coords)
                pairs = tree.query_pairs(r=1.8, output_type="ndarray")
                bonds = [(int(a), int(b)) for a, b in pairs]

            fp, smiles = _fragment_to_morgan_and_smiles(elements, coords, bonds)
            if smiles:
                smiles_list.append(smiles)
            if fp is None:
                fp = _fragment_to_graph_fingerprint(elements)
            ligand_vecs.append(fp)

        if not ligand_vecs:
            return None, sorted(list(set(smiles_list)))

        ligand_vec = np.mean(ligand_vecs, axis=0).astype(np.float32)
        norm = np.linalg.norm(ligand_vec)
        if norm > 1e-12:
            ligand_vec /= norm
        return ligand_vec, sorted(list(set(smiles_list)))

    except Exception as e:
        print(f"  [Ligand Extract] Failed: {e}", file=sys.stderr)
        return None, smiles_list


def _physical_fallback(structure: Structure) -> Dict[str, Any]:
    import math
    lattice = structure.lattice
    vol = lattice.volume

    vdw_vol_total = 0.0
    for site in structure:
        elem = str(site.specie)
        r = VDW_RADII.get(elem, DEFAULT_VDW_RADIUS)
        vdw_vol_total += (4.0 / 3.0) * math.pi * (r ** 3)

    vf = round(max(min(1.0 - vdw_vol_total / vol, 0.95), 0.05), 4)

    a, b, c = lattice.a, lattice.b, lattice.c
    avg_dim = (a + b + c) / 3.0
    lcd = round(avg_dim * (vf ** (1.0 / 3.0)) * 0.55, 2)
    pld = round(lcd * 0.65, 2)

    return {"lcd": lcd, "pld": pld, "vf": vf, "lcd_pld_source": "vdw_heuristic"}


def extract_physical_features(cif_path: str) -> Dict[str, Any]:
    structure = Structure.from_file(cif_path)
    lattice = structure.lattice
    phys = {
        "a": round(lattice.a, 3), "b": round(lattice.b, 3), "c": round(lattice.c, 3),
        "alpha": round(lattice.alpha, 2), "beta": round(lattice.beta, 2), "gamma": round(lattice.gamma, 2),
        "volume": round(lattice.volume, 2),
        "elements": sorted(set(str(site.specie) for site in structure)),
        "num_elements": len(set(str(site.specie) for site in structure)),
        "lcd": None, "pld": None, "vf": None,
    }

    # Try porosity engine first
    porosity_result = None
    try:
        from mof_agent.tools.porosity_engine import compute_porosity_metrics
        porosity_result = compute_porosity_metrics(cif_path)
    except Exception as e:
        print(f"  [Porosity] Engine import/call failed: {e}", file=sys.stderr)

    # Populate new porosity fields
    if porosity_result is not None:
        phys["porosity_source"] = porosity_result.get("best_source", "none")
        phys["porosity_confidence"] = porosity_result.get("confidence", "low")
        phys["porosity_warnings"] = porosity_result.get("warnings", [])
        phys["vf_accessible_by_probe"] = porosity_result.get("vf_accessible_by_probe", {})
        phys["vf_geometric"] = porosity_result.get("geometric_void_fraction")
        phys["vf_probe_occupiable"] = porosity_result.get("probe_occupiable_fraction")
        phys["percolates"] = porosity_result.get("percolates", False)
        phys["porosity_metrics"] = porosity_result

        best_vf = porosity_result.get("best_vf")
        best_lcd = porosity_result.get("best_lcd")
        best_pld = porosity_result.get("best_pld")
        if best_vf is not None:
            phys["vf"] = round(float(best_vf), 4)
            phys["lcd_pld_source"] = "porosity_engine"
        if best_lcd is not None:
            phys["lcd"] = round(float(best_lcd), 2)
        if best_pld is not None:
            phys["pld"] = round(float(best_pld), 2)

        if phys.get("porosity_confidence") == "low" or phys.get("porosity_source") == "none":
            print(f"  [WARN] Pore feature source confidence is low: source={phys.get('porosity_source')}, "
                  f"confidence={phys.get('porosity_confidence')}", file=sys.stderr)
    else:
        phys["porosity_source"] = "unavailable"
        phys["porosity_confidence"] = "low"
        phys["porosity_warnings"] = ["Porosity engine returned no result"]
        phys["vf_accessible_by_probe"] = {}
        phys["vf_geometric"] = None
        phys["vf_probe_occupiable"] = None
        phys["percolates"] = False
        phys["porosity_metrics"] = None

    # Fall back to VDW heuristic
    if phys["lcd"] is None or phys["pld"] is None or phys["vf"] is None:
        fallback = _physical_fallback(structure)
        if phys["lcd"] is None:
            phys["lcd"] = fallback["lcd"]
        if phys["pld"] is None:
            phys["pld"] = fallback["pld"]
        if phys["vf"] is None:
            phys["vf"] = fallback["vf"]
        if phys.get("lcd_pld_source") is None:
            phys["lcd_pld_source"] = fallback.get("lcd_pld_source", "vdw_heuristic")
        if phys.get("porosity_source") == "unavailable":
            phys["porosity_source"] = "vdw_heuristic"
            phys["porosity_warnings"].append("Fell back to VDW heuristic — no porosity backend available")

    return phys


def extract_all_features(cif_path: str) -> Dict[str, Any]:
    errors = []

    # 1. Enhanced SBU features
    sbu_vec, sbu_formulas = extract_sbu_features_enhanced(cif_path)
    if sbu_vec is None:
        errors.append("SBU/SOAP extraction failed")

    # 2. Enhanced ligand features
    ligand_vec, ligand_smiles = extract_ligand_features_enhanced(cif_path)
    if ligand_vec is None:
        errors.append("Ligand/Morgan fingerprint extraction failed")

    # 3. Physical pore features
    physical = extract_physical_features(cif_path)

    result = {
        "cif_path": cif_path,
        "sbu_vec": sbu_vec,
        "ligand_vec": ligand_vec,
        "sbu_chemical_formulas": sbu_formulas,
        "ligand_smiles_strings": ligand_smiles,
        "physical_properties": physical,
        "sbu_vector_dim": sbu_vec.shape[0] if sbu_vec is not None else 0,
        "ligand_vector_dim": ligand_vec.shape[0] if ligand_vec is not None else 0,
        "errors": errors,
    }
    return result


def extract_metal_formulas_from_structure(structure: Structure) -> List[str]:
    """Extract metal element types from Structure (fallback, no SBU extractor needed)."""
    metal_species = sorted(set(
        str(site.specie) for site in structure
        if str(site.specie) in METAL_ELEMENTS
    ))
    return metal_species


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)
    res = extract_all_features(sys.argv[1])

    class _NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    print(json.dumps({k: v for k, v in res.items() if k not in ("sbu_vec", "ligand_vec")},
                     indent=2, ensure_ascii=False, cls=_NumpyEncoder))
