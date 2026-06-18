import os
from pymatgen.core import Structure, Lattice
from pymatgen.io.cif import CifWriter


def generate_dummy_cif(output_path: str = "./dummy_mof5.cif") -> str:
    """Generate a MOF-5-like CIF using pymatgen for testing."""
    if os.path.exists(output_path):
        print(f"[Setup] Dummy CIF already exists: {output_path}")
        return output_path

    lattice = Lattice.cubic(25.8849)
    species = ["Zn", "Zn", "O", "O", "O", "C", "C", "C", "C", "C", "C", "H", "H"]
    coords = [
        [0.2934, 0.2066, 0.5000], [0.2066, 0.2934, 0.5000],
        [0.2500, 0.2500, 0.5000], [0.2158, 0.2842, 0.4461],
        [0.1531, 0.1531, 0.5000], [0.1710, 0.1710, 0.5621],
        [0.1116, 0.1116, 0.5000], [0.0752, 0.0752, 0.5526],
        [0.0536, 0.1116, 0.5000], [0.2500, 0.2500, 0.6359],
        [0.3109, 0.2324, 0.6359], [0.0553, 0.0553, 0.5875],
        [0.0292, 0.1233, 0.5875],
    ]
    structure = Structure(lattice, species, coords, coords_are_cartesian=False)
    CifWriter(structure).write_file(output_path)
    print(f"[Setup] Generated dummy CIF: {output_path} (MOF-5-like)")
    return output_path
