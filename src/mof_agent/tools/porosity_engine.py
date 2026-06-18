import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from pymatgen.core import Structure

VDW_RADII = {
    "H": 1.20, "He": 1.40, "Li": 1.82, "Be": 1.53, "B": 1.92,
    "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47, "Ne": 1.54,
    "Na": 2.27, "Mg": 1.73, "Al": 1.84, "Si": 2.10, "P": 1.80,
    "S": 1.80, "Cl": 1.75, "Ar": 1.88, "K": 2.75, "Ca": 2.31,
    "Sc": 2.15, "Ti": 2.11, "V": 2.07, "Cr": 2.06, "Mn": 2.05,
    "Fe": 2.04, "Co": 2.00, "Ni": 1.97, "Cu": 1.96, "Zn": 2.01,
    "Ga": 1.87, "Ge": 2.11, "As": 1.85, "Se": 1.90, "Br": 1.85,
    "Kr": 2.02, "Rb": 3.03, "Sr": 2.49, "Y": 2.40, "Zr": 2.30,
    "Nb": 2.15, "Mo": 2.10, "Tc": 2.05, "Ru": 2.05, "Rh": 2.00,
    "Pd": 2.05, "Ag": 2.11, "Cd": 2.18, "In": 1.93, "Sn": 2.17,
    "Sb": 2.06, "Te": 2.06, "I": 1.98, "Xe": 2.16, "Cs": 3.43,
    "Ba": 2.68, "La": 2.43, "Ce": 2.42, "Pr": 2.40, "Nd": 2.39,
    "Sm": 2.38, "Eu": 2.31, "Gd": 2.33, "Tb": 2.25, "Dy": 2.28,
    "Ho": 2.26, "Er": 2.26, "Tm": 2.22, "Yb": 2.20, "Lu": 2.17,
    "Hf": 2.25, "Ta": 2.20, "W": 2.10, "Re": 2.05, "Os": 2.00,
    "Ir": 2.00, "Pt": 2.05, "Au": 2.10, "Hg": 2.05, "Tl": 2.10,
    "Pb": 2.02, "Bi": 2.07, "Po": 2.00, "At": 2.00, "Rn": 2.20,
    "Fr": 3.48, "Ra": 2.83, "Ac": 2.60, "Th": 2.37, "Pa": 2.43,
    "U": 2.40, "Np": 2.40, "Pu": 2.40,
}

PROBE_RADII = [0.0, 1.2, 1.5, 1.65, 1.82, 1.9]
DEFAULT_GRID_SPACING = 0.35


@dataclass
class PorosityMetric:
    name: str
    value: float
    source: str
    probe_radius_A: Optional[float] = None


@dataclass
class PorosityResult:
    source: str = "heuristic"
    method: str = "none"
    probe_radius_A: Optional[float] = None
    geometric_void_fraction: Optional[float] = None
    accessible_void_fraction: Optional[float] = None
    probe_occupiable_fraction: Optional[float] = None
    helium_void_fraction: Optional[float] = None
    pore_volume_cm3_g: Optional[float] = None
    lcd_A: Optional[float] = None
    pld_A: Optional[float] = None
    channel_dimensionality: int = 0
    percolates: bool = False
    confidence: str = "low"
    warnings: List[str] = field(default_factory=list)
    raw: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class ZeoPlusPlusBackend:
    """Zeo++ backend using the 'network' binary."""

    @staticmethod
    def _find_binary() -> Optional[str]:
        for env_var in ("ZEO_NETWORK_BIN", "ZEO_PP_BIN"):
            path = os.getenv(env_var, "")
            if path and os.path.isfile(path):
                return path
        return shutil.which("network")

    @classmethod
    def is_available(cls) -> bool:
        return cls._find_binary() is not None

    @classmethod
    def compute(cls, cif_path: str) -> Optional[PorosityResult]:
        binary = cls._find_binary()
        if binary is None:
            return None

        warnings_list: List[str] = []
        tmpdir = tempfile.mkdtemp(prefix="zeopp_")
        try:
            probe = 1.82
            cmd = [
                binary, "-ha", "-res", str(probe),
                "-str", "1.0", cif_path,
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120,
                cwd=tmpdir,
            )
            if result.returncode != 0:
                warnings_list.append(f"Zeo++ exit code {result.returncode}: {result.stderr[:200]}")
                return None

            vf_geom = None
            vf_acc = None
            lcd = None
            pld = None
            for line in result.stdout.splitlines():
                line_lower = line.lower()
                if "accessible_volume_fraction" in line_lower:
                    parts = line.split()
                    for p in parts:
                        try:
                            vf_acc = float(p)
                            break
                        except ValueError:
                            pass
                if "unit_cell_volume" in line_lower:
                    parts = line.split()
                    for p in parts:
                        try:
                            lcd = float(p)
                            break
                        except ValueError:
                            pass
                if "pld" in line_lower or "pore_limiting_diameter" in line_lower:
                    parts = line.split()
                    for p in parts:
                        try:
                            pld = float(p)
                            break
                        except ValueError:
                            pass

            percolates = vf_acc is not None and vf_acc > 0.001

            return PorosityResult(
                source="zeo++",
                method="geometric_probe",
                probe_radius_A=probe,
                geometric_void_fraction=vf_geom,
                accessible_void_fraction=vf_acc,
                probe_occupiable_fraction=vf_acc,
                lcd_A=lcd,
                pld_A=pld,
                percolates=percolates,
                confidence="high" if vf_acc is not None else "medium",
                warnings=warnings_list,
                raw={"stdout": result.stdout[:2000], "stderr": result.stderr[:1000]},
            )
        except subprocess.TimeoutExpired:
            warnings_list.append("Zeo++ timed out (>120s)")
            return None
        except Exception as e:
            warnings_list.append(f"Zeo++ error: {e}")
            return None
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class RASBABackend:
    """RASPA backend for Helium void fraction — stub implementation."""

    @staticmethod
    def _find_binary() -> Optional[str]:
        for env_var in ("RASPA_BIN",):
            path = os.getenv(env_var, "")
            if path and os.path.isfile(path):
                return path
        return shutil.which("raspa_sim") or shutil.which("raspa")

    @classmethod
    def is_available(cls) -> bool:
        return cls._find_binary() is not None

    @classmethod
    def compute(cls, cif_path: str) -> Optional[PorosityResult]:
        if not cls.is_available():
            return None
        return PorosityResult(
            source="raspa",
            method="helium_insertion",
            warnings=["RASPA backend not yet implemented — returning stub"],
            confidence="low",
        )


class Olex2Backend:
    """Olex2/PLATON crystallographic void cross-check — stub implementation."""

    @staticmethod
    def _find_binary() -> Optional[str]:
        for candidate in ("olex2", "platon", "shelx"):
            found = shutil.which(candidate)
            if found:
                return found
        return None

    @classmethod
    def is_available(cls) -> bool:
        return cls._find_binary() is not None

    @classmethod
    def compute(cls, cif_path: str) -> Optional[PorosityResult]:
        if not cls.is_available():
            return None
        return PorosityResult(
            source="olex2_platon",
            method="crystallographic_void",
            warnings=["Olex2/PLATON cross-check not yet implemented — returning stub"],
            confidence="low",
        )


class VoxelBackend:

    @classmethod
    def compute(cls, cif_path: str) -> PorosityResult:
        spacing = float(os.getenv("POROSITY_GRID_SPACING", str(DEFAULT_GRID_SPACING)))
        warnings_list: List[str] = []

        try:
            structure = Structure.from_file(cif_path)
        except Exception as e:
            return PorosityResult(
                source="voxel",
                method="grid_based",
                confidence="low",
                warnings=[f"Failed to parse CIF: {e}"],
            )

        lattice = structure.lattice
        a_vec, b_vec, c_vec = lattice.matrix
        a_len = float(np.linalg.norm(a_vec))
        b_len = float(np.linalg.norm(b_vec))
        c_len = float(np.linalg.norm(c_vec))

        na = max(4, int(a_len / spacing))
        nb = max(4, int(b_len / spacing))
        nc = max(4, int(c_len / spacing))
        total_voxels = na * nb * nc

        cart_coords = structure.cart_coords
        elements = [str(el) for el in structure.species]
        radii = np.array([VDW_RADII.get(el, 1.70) for el in elements])

        results_by_probe: Dict[float, Dict] = {}
        frac_coords_all = np.array([site.frac_coords for site in structure.sites])

        for probe in PROBE_RADII:
            occ_thresholds = radii + probe
            occ_thresholds_cart = occ_thresholds

            i_vals = np.arange(na)
            j_vals = np.arange(nb)
            k_vals = np.arange(nc)

            occupied = np.zeros((na, nb, nc), dtype=np.bool_)

            for atom_idx in range(len(cart_coords)):
                atom_cart = cart_coords[atom_idx]
                radius = occ_thresholds_cart[atom_idx]
                radius_sq = radius * radius

                atom_frac = frac_coords_all[atom_idx]

                di = max(1, int(radius / spacing) + 2)
                dj = max(1, int(radius / spacing) + 2)
                dk = max(1, int(radius / spacing) + 2)

                ci = int(round(atom_frac[0] * na))
                cj = int(round(atom_frac[1] * nb))
                ck = int(round(atom_frac[2] * nc))

                for ii_rel in range(-di, di + 1):
                    for jj_rel in range(-dj, dj + 1):
                        for kk_rel in range(-dk, dk + 1):
                            gi = (ci + ii_rel) % na
                            gj = (cj + jj_rel) % nb
                            gk = (ck + kk_rel) % nc

                            frac_i = gi / na
                            frac_j = gj / nb
                            frac_k = gk / nc

                            cart_i = (frac_i * a_vec[0] + frac_j * b_vec[0] + frac_k * c_vec[0])
                            cart_j = (frac_i * a_vec[1] + frac_j * b_vec[1] + frac_k * c_vec[1])
                            cart_k = (frac_i * a_vec[2] + frac_j * b_vec[2] + frac_k * c_vec[2])

                            dx = cart_i - atom_cart[0]
                            dy = cart_j - atom_cart[1]
                            dz = cart_k - atom_cart[2]

                            dx -= a_len * round(dx / a_len) if abs(dx) > a_len / 2 else 0
                            dy -= b_len * round(dy / b_len) if abs(dy) > b_len / 2 else 0
                            dz -= c_len * round(dz / c_len) if abs(dz) > c_len / 2 else 0

                            dist_sq = dx * dx + dy * dy + dz * dz
                            if dist_sq <= radius_sq:
                                occupied[gi, gj, gk] = True

            void = ~occupied

            accessible = np.zeros((na, nb, nc), dtype=np.bool_)
            seeds = [
                (0, 0, 0), (na - 1, 0, 0), (0, nb - 1, 0), (0, 0, nc - 1),
                (na - 1, nb - 1, 0), (na - 1, 0, nc - 1), (0, nb - 1, nc - 1),
                (na - 1, nb - 1, nc - 1),
            ]

            neighbors_26 = [
                (di, dj, dk) for di in (-1, 0, 1) for dj in (-1, 0, 1) for dk in (-1, 0, 1)
                if not (di == 0 and dj == 0 and dk == 0)
            ]

            for si, sj, sk in seeds:
                if not void[si, sj, sk] or accessible[si, sj, sk]:
                    continue
                stack = [(si, sj, sk)]
                accessible[si, sj, sk] = True
                while stack:
                    vi, vj, vk = stack.pop()
                    for di, dj, dk in neighbors_26:
                        ni = (vi + di) % na
                        nj = (vj + dj) % nb
                        nk = (vk + dk) % nc
                        if void[ni, nj, nk] and not accessible[ni, nj, nk]:
                            accessible[ni, nj, nk] = True
                            stack.append((ni, nj, nk))

            n_void = int(void.sum())
            n_accessible = int(accessible.sum())
            results_by_probe[probe] = {
                "geometric_void_fraction": n_void / total_voxels,
                "accessible_void_fraction": n_accessible / total_voxels,
                "n_void": n_void,
                "n_accessible": n_accessible,
                "total_voxels": total_voxels,
                "grid_spacing": spacing,
            }

        probe_n2 = 1.82
        closest = min(results_by_probe.keys(), key=lambda x: abs(x - probe_n2))
        if probe_n2 in results_by_probe:
            percolates = results_by_probe[probe_n2]["accessible_void_fraction"] > 0.001
        else:
            percolates = results_by_probe[closest]["accessible_void_fraction"] > 0.001

        vf_geom = results_by_probe[0.0]["geometric_void_fraction"]
        cell_dims = np.array([a_len, b_len, c_len])
        avg_cell_dim = float(np.mean(cell_dims))

        lcd_est = avg_cell_dim * (vf_geom ** (1.0 / 3.0)) * 0.8

        pld_est = None
        for probe in sorted(PROBE_RADII):
            if results_by_probe[probe]["accessible_void_fraction"] > 0.001:
                pld_est = probe * 2.0
        if pld_est is None:
            pld_est = 0.0

        acc_vals = [results_by_probe[p]["accessible_void_fraction"] for p in PROBE_RADII]
        acc_variance = float(np.var(acc_vals))
        if acc_variance < 0.02:
            confidence = "medium"
        else:
            confidence = "low"

        if vf_geom < 0.01:
            warnings_list.append("Geometric VF < 1%: structure is effectively dense (non-porous)")
        if not percolates:
            warnings_list.append(
                "Structure does not percolate at N2 probe radius (1.82A) — "
                "no accessible porosity for N2-sized guests")
        if na * nb * nc > 500000:
            warnings_list.append(
                f"Large grid ({na}x{nb}x{nc}={total_voxels} voxels) may be computationally expensive")

        n2_result = results_by_probe.get(probe_n2, results_by_probe[closest])

        return PorosityResult(
            source="voxel",
            method=f"grid_{spacing:.2f}A",
            probe_radius_A=probe_n2,
            geometric_void_fraction=round(vf_geom, 4),
            accessible_void_fraction=round(n2_result["accessible_void_fraction"], 4),
            probe_occupiable_fraction=round(n2_result["accessible_void_fraction"], 4),
            helium_void_fraction=None,
            pore_volume_cm3_g=round(
                vf_geom * float(structure.volume) / float(structure.composition.weight), 4),
            lcd_A=round(lcd_est, 3),
            pld_A=round(pld_est, 3),
            channel_dimensionality=1 if percolates else 0,
            percolates=percolates,
            confidence=confidence,
            warnings=warnings_list,
            raw={
                "probe_results": {
                    str(p): {k: v for k, v in d.items() if k != "n_void" and k != "n_accessible"}
                    for p, d in results_by_probe.items()
                },
                "grid_shape": [na, nb, nc],
                "grid_spacing": spacing,
            },
        )


def compute_porosity_metrics(cif_path: str) -> dict:
    backends = [
        ("zeo++", ZeoPlusPlusBackend),
        ("raspa", RASBABackend),
        ("olex2", Olex2Backend),
        ("voxel", VoxelBackend),
    ]

    results: Dict[str, PorosityResult] = {}
    all_warnings: List[str] = []

    for name, backend_cls in backends:
        try:
            result = backend_cls.compute(cif_path)
            if result is not None:
                results[name] = result
                all_warnings.extend(result.warnings)
        except Exception as e:
            all_warnings.append(f"{name} backend raised exception: {e}")

    if not results:
        return {
            "best_vf": None, "best_lcd": None, "best_pld": None,
            "best_source": "none", "confidence": "low",
            "warnings": all_warnings + ["No porosity backend produced a result"],
            "geometric_void_fraction": None, "accessible_void_fraction": None,
            "probe_occupiable_fraction": None, "helium_void_fraction": None,
            "pore_volume_cm3_g": None, "lcd_A": None, "pld_A": None,
            "channel_dimensionality": 0, "percolates": False,
            "source": "none", "method": "none", "probe_radius_A": None,
            "raw": {}, "vf_accessible_by_probe": {},
        }

    priority_order = ["zeo++", "raspa", "olex2", "voxel"]
    best_name = None
    for name in priority_order:
        if name in results:
            best_name = name
            break

    best = results[best_name]
    overrides = {}
    if best_name != "zeo++" and "zeo++" in results:
        overrides["_zeopp_failed"] = True
        overrides["_zeopp_warnings"] = results["zeo++"].warnings

    vf_by_probe: Dict[str, float] = {}
    if "voxel" in results and results["voxel"].raw:
        probe_results = results["voxel"].raw.get("probe_results", {})
        for probe_str, pdict in probe_results.items():
            vf_by_probe[probe_str] = pdict.get("accessible_void_fraction", 0.0)

    output = {
        "best_vf": best.accessible_void_fraction,
        "best_lcd": best.lcd_A,
        "best_pld": best.pld_A,
        "best_source": best.source,
        "confidence": best.confidence,
        "warnings": all_warnings,
        "geometric_void_fraction": best.geometric_void_fraction,
        "accessible_void_fraction": best.accessible_void_fraction,
        "probe_occupiable_fraction": best.probe_occupiable_fraction,
        "helium_void_fraction": best.helium_void_fraction,
        "pore_volume_cm3_g": best.pore_volume_cm3_g,
        "lcd_A": best.lcd_A,
        "pld_A": best.pld_A,
        "channel_dimensionality": best.channel_dimensionality,
        "percolates": best.percolates,
        "source": best.source,
        "method": best.method,
        "probe_radius_A": best.probe_radius_A,
        "raw": best.raw,
        "vf_accessible_by_probe": vf_by_probe,
    }
    output.update(overrides)
    return output
