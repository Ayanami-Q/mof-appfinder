from __future__ import annotations

import json
import math
import os
import tempfile
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN

try:
    import pyzeo
    from pyzeo.extension import AtomNetwork as PyzeoAtomNetwork
    PYSEO_AVAILABLE = True
except ImportError:
    pyzeo = None  # type: ignore[assignment]
    PyzeoAtomNetwork = None  # type: ignore[assignment]
    PYSEO_AVAILABLE = False

VDW_RADII: Dict[str, float] = {
    "H": 1.20, "He": 1.40,
    "B": 1.92, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47,
    "Si": 2.10, "P": 1.80, "S": 1.80, "Cl": 1.75, "Br": 1.85, "I": 1.98,
    "Cu": 1.40, "Zn": 1.39, "Co": 1.63, "Ni": 1.63, "Fe": 1.56,
    "Mn": 1.61, "Cd": 1.58, "Zr": 1.86, "Hf": 1.82,
    "Cr": 1.69, "V": 1.73, "Ti": 1.80, "Mg": 1.73, "Ca": 2.31,
    "Mo": 1.90, "W": 1.90, "Pd": 1.63, "Pt": 1.75, "Ag": 1.72,
    "Au": 1.66, "Al": 1.84, "Ga": 1.87, "In": 1.93, "Sn": 2.17,
    "Pb": 2.02, "Li": 1.82, "Na": 2.27, "K": 2.75, "Rb": 3.03,
    "Cs": 3.43, "Sr": 2.49, "Ba": 2.68, "Sc": 2.11, "Y": 2.19,
    "La": 2.43, "Ru": 1.84, "Rh": 1.82, "Os": 1.88, "Ir": 1.86,
}
DEFAULT_VDW_RADIUS: float = 1.70


@contextmanager
def _suppress_stdout():
    """Temporarily suppress stdout (including C/C++ extension printf output)."""
    old_fd = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(old_fd, 1)
        os.close(old_fd)


METAL_ELEMENTS: set[str] = {
    "Cu", "Zn", "Co", "Ni", "Fe", "Mn", "Cd", "Zr", "Hf",
    "Cr", "V", "Ti", "Mg", "Ca", "Mo", "W", "Pd", "Pt", "Ag",
    "Au", "Al", "Ga", "In", "Sb", "Bi", "Pb", "Sn", "Ru", "Os",
    "Rh", "Ir", "Sc", "Y", "La", "Ce", "Pr", "Nd", "Sm", "Eu",
    "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Li", "Na",
    "K", "Rb", "Cs", "Sr", "Ba", "Th", "U",
}

EQEQ_PARAMS: Dict[str, Tuple[float, float]] = {
    "H":  (13.60, 0.75), "He": (24.59, 0.00), "Li": (5.39, 0.62),
    "Be": (9.32, 0.00),  "B":  (8.30, 0.28),  "C":  (11.26, 1.26),
    "N":  (14.53, 0.07), "O":  (13.62, 1.46), "F":  (17.42, 3.40),
    "Ne": (21.56, 0.00), "Na": (5.14, 0.55),  "Mg": (7.65, 0.00),
    "Al": (5.99, 0.43),  "Si": (8.15, 1.39),  "P":  (10.49, 0.75),
    "S":  (10.36, 2.08), "Cl": (12.97, 3.61), "K":  (4.34, 0.50),
    "Ca": (6.11, 0.02),  "Sc": (6.56, 0.19),  "Ti": (6.83, 0.08),
    "V":  (6.75, 0.53),  "Cr": (6.77, 0.67),  "Mn": (7.43, 0.00),
    "Fe": (7.90, 0.15),  "Co": (7.88, 0.66),  "Ni": (7.64, 1.16),
    "Cu": (7.73, 1.24),  "Zn": (9.39, 0.00),  "Ga": (6.00, 0.30),
    "Ge": (7.90, 1.23),  "As": (9.82, 0.81),  "Se": (9.75, 2.02),
    "Br": (11.81, 3.36), "Kr": (14.00, 0.00), "Rb": (4.18, 0.49),
    "Sr": (5.69, 0.05),  "Y":  (6.22, 0.31),  "Zr": (6.63, 0.43),
    "Nb": (6.76, 0.89),  "Mo": (7.09, 0.75),  "Ru": (7.36, 1.05),
    "Rh": (7.46, 1.14),  "Pd": (8.34, 0.56),  "Ag": (7.58, 1.30),
    "Cd": (8.99, 0.00),  "In": (5.79, 0.30),  "Sn": (7.34, 1.11),
    "Sb": (8.64, 1.05),  "Te": (9.01, 1.97),  "I":  (10.45, 3.06),
    "Xe": (12.13, 0.00), "Cs": (3.89, 0.47),  "Ba": (5.21, 0.14),
    "La": (5.58, 0.50),  "Hf": (6.83, 0.00),  "Ta": (7.89, 0.32),
    "W":  (7.98, 0.82),  "Re": (7.88, 0.15),  "Os": (8.70, 1.10),
    "Ir": (9.10, 1.56),  "Pt": (9.00, 2.13),  "Au": (9.22, 2.31),
    "Hg": (10.44, 0.00), "Pb": (7.42, 0.36),  "Bi": (7.29, 0.95),
}
_DEFAULT_IP = 8.0   # eV
_DEFAULT_EA = 1.0   # eV
EQEQ_WIDTH = 0.5  # sigma = 1 / (width * sqrt(2))


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating):
            return None if np.isnan(obj) or np.isinf(obj) else float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return super().default(obj)


def _safe_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_json(item) for item in obj]
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) or np.isinf(obj) else float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, np.bool_): return bool(obj)
    if isinstance(obj, (complex, np.complexfloating)): return [float(obj.real), float(obj.imag)]
    if isinstance(obj, bytes): return obj.decode("utf-8", errors="replace")
    return obj


def _get_vdw_radius(element: str) -> float:
    return VDW_RADII.get(element, DEFAULT_VDW_RADIUS)


def _compute_eqeq_charges(structure: Structure) -> np.ndarray:
    n_atoms = len(structure)
    chi = np.zeros(n_atoms, dtype=np.float64)
    j_self = np.zeros(n_atoms, dtype=np.float64)

    for i, site in enumerate(structure):
        elem = str(site.specie)
        ip, ea = EQEQ_PARAMS.get(elem, (_DEFAULT_IP, _DEFAULT_EA))
        chi[i] = 0.5 * (ip + ea)
        j_self[i] = ip - ea

    sigma = EQEQ_WIDTH
    cutoff = 15.0

    J = lil_matrix((n_atoms, n_atoms), dtype=np.float64)
    for i in range(n_atoms):
        J[i, i] = j_self[i] + (2.0 * sigma / np.sqrt(np.pi))

    c_idx, n_idx, _, dists = structure.get_neighbor_list(cutoff)

    for c, n, d in zip(c_idx, n_idx, dists):
        if d < 1e-8:
            continue
        J[c, n] += math.erf(sigma * d) / d

    J = J.tocsr()

    A = lil_matrix((n_atoms + 1, n_atoms + 1), dtype=np.float64)
    A[:n_atoms, :n_atoms] = J
    A[:n_atoms, n_atoms] = 1.0
    A[n_atoms, :n_atoms] = 1.0
    A[n_atoms, n_atoms] = 0.0
    A = A.tocsr()

    b = np.zeros(n_atoms + 1, dtype=np.float64)
    b[:n_atoms] = -chi

    try:
        sol = spsolve(A, b)
        charges = sol[:n_atoms]
    except Exception:
        from scipy.sparse.linalg import lsqr
        sol, _exit_code = lsqr(A, b)[:2]
        charges = sol[:n_atoms]

    return charges


class MOFCIFAnalyzer:
    def __init__(self, cif_path: str) -> None:
        _path = Path(cif_path)
        if not _path.is_file():
            raise FileNotFoundError(f"CIF file not found: {cif_path}")
        self.cif_path: str = str(_path.absolute())
        self.cif_name: str = _path.stem

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.structure = Structure.from_file(self.cif_path, merge_tol=0.05)
            except Exception:
                self.structure = Structure.from_file(self.cif_path)

        self._cnn = CrystalNN(distance_cutoffs=None, x_diff_weight=0.0)
        self._nn_cache: Dict[int, List[Dict[str, Any]]] = {}

        self._elements: List[str] = [str(site.specie) for site in self.structure]
        self._unique_elements: List[str] = sorted(set(self._elements))
        self._num_atoms: int = len(self.structure)
        self._coords: np.ndarray = np.array(
            [site.coords for site in self.structure], dtype=np.float64)

        self._pyzeo_net: Any = None
        self._pyzeo_radius_file: Optional[str] = None
        self._init_pyzeo_network()

    def _init_pyzeo_network(self) -> None:
        if not PYSEO_AVAILABLE:
            warnings.warn("pyzeo not installed. Pore topology and surface analysis will fall back.")
            return

        unique_elem = set(self._elements)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".rad", delete=False, prefix="pyzeo_rad_") as fh:
            for elem in sorted(unique_elem):
                r = VDW_RADII.get(elem, DEFAULT_VDW_RADIUS)
                fh.write(f"{elem}  {r}\n")
            self._pyzeo_radius_file = fh.name

        try:
            with _suppress_stdout():
                self._pyzeo_net = PyzeoAtomNetwork.read_from_CIF(
                    self.cif_path, rad_flag=True, rad_file=self._pyzeo_radius_file
                )
        except Exception as e:
            warnings.warn(f"pyzeo CIF read failed: {e}. Falling back to heuristic evaluation.")
            self._pyzeo_net = None

    def __del__(self) -> None:
        if self._pyzeo_radius_file and os.path.exists(self._pyzeo_radius_file):
            try:
                os.unlink(self._pyzeo_radius_file)
            except OSError:
                pass

    def _get_nn_info(self, site_index: int) -> List[Dict[str, Any]]:
        if site_index not in self._nn_cache:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._nn_cache[site_index] = self._cnn.get_nn_info(self.structure, site_index)
        return self._nn_cache[site_index]

    @staticmethod
    def _parse_pyzeo_kv(raw_text: str) -> Dict[str, str]:
        import re
        pattern = r'([A-Za-z_/^0-9]+):\s*([^\s\n]+)'
        matches = re.findall(pattern, raw_text)
        return {key: val for key, val in matches}

    @staticmethod
    def _compute_angle(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        cos = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b) + 1e-12)
        return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

    def _find_sites_by_element(self, element: str) -> List[Tuple[int, np.ndarray]]:
        return [(i, site.coords.copy()) for i, site in enumerate(self.structure)
                if str(site.specie) == element]

    def _get_crystal_system(self) -> str:
        sg_mapping = {
            range(1, 3):    "Triclinic",
            range(3, 16):   "Monoclinic",
            range(16, 75):  "Orthorhombic",
            range(75, 143): "Tetragonal",
            range(143, 168): "Trigonal",
            range(168, 195): "Hexagonal",
            range(195, 231): "Cubic",
        }
        try:
            _, sg_number = self.structure.get_space_group_info()
            for num_range, name in sg_mapping.items():
                if sg_number in num_range: return name
            return f"Unknown (SG #{sg_number})"
        except Exception:
            return "Unknown"

    def get_coordination_geometry(
        self, central_element: str, ligand_element: str, verbose: bool = False
    ) -> Dict[str, Any]:
        central_sites = self._find_sites_by_element(central_element)
        if not central_sites:
            return {"error": f"Element '{central_element}' not found in structure"}

        all_bond_lengths, cn_per_center, all_bond_angles = [], [], []

        for site_idx, center_coords in central_sites:
            nn_list = self._get_nn_info(site_idx)
            ligand_neighbors = [nn["site"].coords for nn in nn_list
                               if str(nn["site"].specie) == ligand_element]

            cn = len(ligand_neighbors)
            cn_per_center.append(cn)

            for neighbor_coords in ligand_neighbors:
                all_bond_lengths.append(float(np.linalg.norm(center_coords - neighbor_coords)))

            if cn >= 2:
                for i in range(cn):
                    for j in range(i + 1, cn):
                        vec_i = ligand_neighbors[i] - center_coords
                        vec_j = ligand_neighbors[j] - center_coords
                        all_bond_angles.append(self._compute_angle(vec_i, vec_j))

        if not all_bond_lengths:
            return {"error": f"No {central_element}–{ligand_element} bonds found."}

        lengths_arr = np.array(all_bond_lengths)
        result: Dict[str, Any] = {
            "central_element": central_element,
            "ligand_element": ligand_element,
            "num_centers": len(central_sites),
            "mean_bond_length_A": round(float(np.mean(lengths_arr)), 4),
            "bond_length_range_A": [round(float(np.min(lengths_arr)), 4),
                                     round(float(np.max(lengths_arr)), 4)],
            "mean_cn": round(float(np.mean(cn_per_center)), 2) if cn_per_center else 0,
        }

        if all_bond_angles:
            angles_arr = np.array(all_bond_angles)
            result["mean_bond_angle_deg"] = round(float(np.mean(angles_arr)), 2)
            result["bond_angle_range_deg"] = [round(float(np.min(angles_arr)), 2),
                                              round(float(np.max(angles_arr)), 2)]

        if verbose:
            result["all_bond_lengths_A"] = [round(v, 4) for v in all_bond_lengths]
            result["cn_per_center"] = cn_per_center
            if all_bond_angles:
                result["all_bond_angles_deg"] = [round(v, 2) for v in all_bond_angles]

        return result

    def get_pore_and_window_topology(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "di_A": None, "df_A": None, "pld_A": None,
            "accessible_volume_A3": None, "accessible_volume_fraction": None,
            "asa_A2": None, "asa_m2_cm3": None,
            "num_channels": None, "num_pockets": None,
            "window_analysis": [], "pyzeo_available": PYSEO_AVAILABLE and self._pyzeo_net is not None,
        }
        if self._pyzeo_net is None: return result

        try:
            with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, prefix="fsp_") as fh:
                fsp_file = fh.name
            with _suppress_stdout():
                self._pyzeo_net.calculate_free_sphere_parameters(fsp_file)
            with open(fsp_file, "r") as fh:
                fsp_content = fh.read().strip()
            os.unlink(fsp_file)

            parts = fsp_content.split()
            if len(parts) >= 3:
                result["di_A"], result["df_A"], result["pld_A"] = map(
                    lambda x: round(float(x), 4), parts[-3:])

            self._rebuild_pyzeo_net()
            with _suppress_stdout():
                vol_bytes = pyzeo.volume(self._pyzeo_net, 0.0, 1.86, 10000)
            vol_kv = self._parse_pyzeo_kv(
                vol_bytes.decode() if isinstance(vol_bytes, bytes) else str(vol_bytes))
            result["accessible_volume_fraction"] = round(float(vol_kv.get("AV_Volume_fraction", 0)), 4)
            result["num_channels"] = int(vol_kv.get("Number_of_channels", 0))

            self._rebuild_pyzeo_net()
            with _suppress_stdout():
                sa_bytes = pyzeo.surface_area(self._pyzeo_net, 0.0, 1.86, 10000)
            sa_kv = self._parse_pyzeo_kv(
                sa_bytes.decode() if isinstance(sa_bytes, bytes) else str(sa_bytes))
            result["asa_m2_cm3"] = round(float(sa_kv.get("ASA_m^2/cm^3", 0)), 4)

            if result["df_A"] is not None and result["df_A"] > 1.0:
                df = result["df_A"]
                window_centers = self._find_window_centers(df)
                search_radius = max(df / 2.0 + 1.52, 2.0)

                for center in window_centers:
                    neighbors = self.structure.get_sites_in_sphere(
                        center, search_radius, include_index=True)

                    metal_atoms, ligand_atoms = [], []
                    for site, *_ in neighbors:
                        elem = str(site.specie)
                        if elem in METAL_ELEMENTS:
                            metal_atoms.append(elem)
                        else:
                            ligand_atoms.append(elem)

                    if metal_atoms or ligand_atoms:
                        result["window_analysis"].append({
                            "window_center_cartesian": [round(c, 3) for c in center],
                            "search_radius_A": round(search_radius, 2),
                            "metal_atoms": sorted(set(metal_atoms)),
                            "ligand_atoms": sorted(set(ligand_atoms)),
                        })

        except Exception as e:
            result["pyzeo_error"] = str(e)
        return result

    def _find_window_centers(self, df_A: float, grid_res: float = 0.5,
                             tolerance: float = 0.3) -> List[np.ndarray]:
        """Grid-scan based search for real window (bottleneck) geometric centers."""
        from scipy.spatial import cKDTree

        supercell = self.structure * (3, 3, 3)
        coords = supercell.cart_coords
        elements = [str(site.specie) for site in supercell]
        vdw_radii = np.array([_get_vdw_radius(el) for el in elements])
        kdtree = cKDTree(coords)

        lattice = self.structure.lattice
        na = min(max(int(lattice.a / grid_res), 5), 40)
        nb = min(max(int(lattice.b / grid_res), 5), 40)
        nc = min(max(int(lattice.c / grid_res), 5), 40)

        frac_points = np.vstack(np.meshgrid(
            np.linspace(0, 1, na, endpoint=False),
            np.linspace(0, 1, nb, endpoint=False),
            np.linspace(0, 1, nc, endpoint=False),
            indexing='ij'
        )).reshape(3, -1).T

        center_frac_points = frac_points + np.array([1.0, 1.0, 1.0])
        test_points = supercell.lattice.get_cartesian_coords(center_frac_points)

        dists, indices = kdtree.query(test_points, k=1)
        void_radii = dists - vdw_radii[indices]

        target_r = df_A / 2.0
        mask = np.abs(void_radii - target_r) < tolerance
        window_points_cart = test_points[mask]

        if len(window_points_cart) == 0:
            mask = np.abs(void_radii - target_r) < (tolerance * 2.5)
            window_points_cart = test_points[mask]

        if len(window_points_cart) == 0:
            return []

        window_frac = supercell.lattice.get_fractional_coords(window_points_cart)
        window_frac = window_frac % 1.0
        final_window_cart = self.structure.lattice.get_cartesian_coords(window_frac)

        clusters = []
        cluster_radius = max(target_r, 2.5)
        for pt in final_window_cart:
            added = False
            for cluster in clusters:
                if np.linalg.norm(pt - cluster["center"]) < cluster_radius:
                    cluster["points"].append(pt)
                    cluster["center"] = np.mean(cluster["points"], axis=0)
                    added = True
                    break
            if not added:
                clusters.append({"center": pt, "points": [pt]})

        return [c["center"] for c in clusters]

    def _rebuild_pyzeo_net(self) -> None:
        if self._pyzeo_radius_file and os.path.exists(self._pyzeo_radius_file):
            with _suppress_stdout():
                self._pyzeo_net = PyzeoAtomNetwork.read_from_CIF(
                    self.cif_path, rad_flag=True, rad_file=self._pyzeo_radius_file
                )

    def assess_surface_polarity(self, probe_radius: float = 1.3) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "probe_radius_A": probe_radius, "total_asa_A2": None,
            "asa_element_breakdown": {}, "surface_composition": "",
            "polarity_assessment": "",
        }

        try:
            charges = _compute_eqeq_charges(self.structure)
        except Exception as e:
            result["charge_error"] = str(e)
            charges = np.zeros(self._num_atoms, dtype=np.float64)

        surface_indices = self._identify_surface_atoms(probe_radius)

        if surface_indices:
            surface_elements = [self._elements[i] for i in surface_indices]
            surface_charges = charges[surface_indices]

            from collections import Counter
            elem_counts = Counter(surface_elements)
            total_surface = len(surface_indices)
            breakdown = {el: {"count": c, "fraction": round(c / total_surface, 4)}
                        for el, c in sorted(elem_counts.items(), key=lambda x: -x[1])}
            result["asa_element_breakdown"] = breakdown

            hydrophobic_frac = sum(
                breakdown.get(el, {}).get("fraction", 0) for el in ("C", "H"))
            polar_frac = sum(
                breakdown.get(el, {}).get("fraction", 0) for el in ("O", "N", "F", "Cl", "S", "P"))
            metal_frac = sum(
                breakdown.get(el, {}).get("fraction", 0) for el in METAL_ELEMENTS)

            result["surface_composition"] = (
                f"C+H: {hydrophobic_frac:.1%}, O+N+X: {polar_frac:.1%}, Metals: {metal_frac:.1%}")

            area_pol = ("Hydrophobic" if hydrophobic_frac > 0.80 else
                       "Hydrophilic" if (polar_frac + metal_frac > 0.40) else "Mixed")
            charge_var = float(np.var(np.abs(surface_charges)))
            charge_pol = ("Hydrophilic" if charge_var > 0.05 else
                         "Mixed" if charge_var > 0.01 else "Hydrophobic")

            result["polarity_assessment"] = (
                f"By Area: {area_pol} | By Charge Variance ({charge_var:.4f}): {charge_pol}")

        return result

    def _identify_surface_atoms(self, probe_radius: float) -> List[int]:
        n_atoms = self._num_atoms
        surface: List[int] = []

        max_vdw = max(_get_vdw_radius(elem) for elem in set(self._elements))
        search_radius = max_vdw * 2.0 + probe_radius + 2.0

        c_idx, n_idx, _, dists = self.structure.get_neighbor_list(search_radius)
        neighbors_dict: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(n_atoms)}

        for c, n, d in zip(c_idx, n_idx, dists):
            if c != n or d > 1e-8:
                neighbors_dict[c].append((n, d))

        for i in range(n_atoms):
            elem_i = self._elements[i]
            vdw_i = _get_vdw_radius(elem_i)
            probe_center_dist = vdw_i + probe_radius

            accessible = True
            for j, dist in neighbors_dict[i]:
                elem_j = self._elements[j]
                vdw_j = _get_vdw_radius(elem_j)
                if dist < probe_center_dist + vdw_j:
                    accessible = False
                    break

            if accessible:
                surface.append(i)

        if len(surface) < n_atoms * 0.1:
            surface = list(range(n_atoms))

        return surface

    def export_llm_report(
        self, central_element: str = "Zn", ligand_element: str = "O", verbose: bool = False
    ) -> Dict[str, Any]:
        lattice = self.structure.lattice
        cif_metadata = {
            "cif_name": self.cif_name,
            "reduced_formula": str(self.structure.composition.reduced_formula),
            "num_atoms": self._num_atoms,
            "space_group": self.structure.get_space_group_info()[0],
            "crystal_system": self._get_crystal_system(),
            "volume_A3": round(float(lattice.volume), 3),
        }
        return _safe_json({
            "cif_metadata": cif_metadata,
            "coordination_info": self.get_coordination_geometry(
                central_element, ligand_element, verbose=verbose),
            "pore_topology": self.get_pore_and_window_topology(),
            "surface_environment": self.assess_surface_polarity(),
        })

    def export_llm_report_json(
        self, central_element: str = "Zn", ligand_element: str = "O", verbose: bool = False
    ) -> str:
        report = self.export_llm_report(central_element, ligand_element, verbose=verbose)
        return json.dumps(report, ensure_ascii=False, indent=2, cls=_NumpyEncoder)


if __name__ == "__main__":
    import sys
    test_cif = sys.argv[1] if len(sys.argv) > 1 else "dummy.cif"
    try:
        analyzer = MOFCIFAnalyzer(test_cif)
        print(analyzer.export_llm_report_json(verbose=False))
    except Exception as e:
        print(f"Execution failed: {e}")
