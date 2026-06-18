import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple, Any

import numpy as np
import networkx as nx
from ase import Atoms
from dscribe.descriptors import SOAP
from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN
import faiss

import warnings
warnings.filterwarnings("ignore")

SUPPORTED_ELEMENTS = [
    "H", "C", "N", "O", "F", "Cl", "S", "P",
    "Zn", "Cu", "Co", "Ni", "Fe", "Mn", "Zr", "Ti", "V", "Cr", "Cd", "Ag"
]

METAL_ELEMENTS = {
    "Zn", "Cu", "Co", "Ni", "Fe", "Mn", "Zr", "Ti", "V", "Cr",
    "Cd", "Ag", "Au", "Pt", "Pd", "Ru", "Rh", "Ir", "Os", "Mo", "W"
}

SOAP_RCUT = 6.0
SOAP_NMAX = 8
SOAP_LMAX = 6
SOAP_SIGMA = 0.5


class SBUExtractor:
    def __init__(self, cif_path: str):
        self.cif_path = cif_path
        self.cif_name = Path(cif_path).stem
        self.structure = Structure.from_file(cif_path)
        self.cnn = CrystalNN(distance_cutoffs=None, x_diff_weight=0.0)
        self.global_graph = nx.Graph()
        self._build_global_graph()

    def _build_global_graph(self):
        for i, site in enumerate(self.structure):
            self.global_graph.add_node(i, element=str(site.specie), coords=site.coords)

        for i in range(len(self.structure)):
            nn_info = self.cnn.get_nn_info(self.structure, i)
            for nn in nn_info:
                j = nn['site_index']
                if i != j:
                    self.global_graph.add_edge(i, j)

    def extract_clusters(self) -> List[Dict[str, Any]]:
        metal_indices = [
            n for n, attr in self.global_graph.nodes(data=True)
            if attr['element'] in METAL_ELEMENTS
        ]

        metal_graph = nx.Graph()
        metal_graph.add_nodes_from(metal_indices)

        for i in range(len(metal_indices)):
            for j in range(i + 1, len(metal_indices)):
                m1, m2 = metal_indices[i], metal_indices[j]
                neighbors1 = set(self.global_graph.neighbors(m1))
                neighbors2 = set(self.global_graph.neighbors(m2))
                common_neighbors = neighbors1.intersection(neighbors2)

                if common_neighbors or self.global_graph.has_edge(m1, m2):
                    metal_graph.add_edge(m1, m2)

        clusters = []
        for component in nx.connected_components(metal_graph):
            core_metals = list(component)
            cluster_type = "Single_Node" if len(core_metals) == 1 else "Cluster"

            cluster_indices = set(core_metals)
            for m in core_metals:
                cluster_indices.update(self.global_graph.neighbors(m))

            cluster_indices = list(cluster_indices)
            elements = [self.global_graph.nodes[idx]['element'] for idx in cluster_indices]
            coords = np.array([self.global_graph.nodes[idx]['coords'] for idx in cluster_indices])

            centroid = np.mean(coords, axis=0)
            coords -= centroid

            clusters.append({
                "type": cluster_type,
                "num_metals": len(core_metals),
                "total_atoms": len(cluster_indices),
                "elements": elements,
                "coords": coords.tolist(),
            })

        return clusters


def compute_soap_vector(cluster_data: Dict[str, Any], soap_generator: SOAP) -> np.ndarray:
    elements = cluster_data['elements']
    coords = np.array(cluster_data['coords'])

    valid_indices = [i for i, el in enumerate(elements) if el in SUPPORTED_ELEMENTS]
    if not valid_indices:
        return None

    valid_elements = [elements[i] for i in valid_indices]
    valid_coords = coords[valid_indices]

    ase_atoms = Atoms(symbols=valid_elements, positions=valid_coords)

    try:
        soap_features = soap_generator.create(ase_atoms)
        cluster_vector = np.mean(soap_features, axis=0).astype(np.float32)
        return cluster_vector
    except Exception as e:
        print(f"  SOAP calculation failed: {e}")
        return None


def build_database(input_dir: str, db_path: str, index_path: str):
    print(f"Initializing SOAP generator...")
    soap = SOAP(
        species=SUPPORTED_ELEMENTS,
        periodic=False,
        rcut=SOAP_RCUT,
        nmax=SOAP_NMAX,
        lmax=SOAP_LMAX,
        sigma=SOAP_SIGMA,
        average="off",
        sparse=False
    )

    core_dir = Path(input_dir)
    files = sorted(list(core_dir.glob("*.cif")))
    print(f"Found {len(files)} CIF files.")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS sbus")
    cur.execute("""
        CREATE TABLE sbus (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cif_name TEXT NOT NULL,
            cluster_type TEXT NOT NULL,
            num_metals INTEGER,
            total_atoms INTEGER,
            structure_json TEXT NOT NULL
        )
    """)

    vectors = []
    records = []

    t0 = time.time()
    for i, filepath in enumerate(files):
        print(f"Processing {i+1}/{len(files)}: {filepath.name}")
        try:
            extractor = SBUExtractor(str(filepath))
            clusters = extractor.extract_clusters()

            for cluster in clusters:
                vec = compute_soap_vector(cluster, soap)
                if vec is not None:
                    vectors.append(vec)
                    records.append((
                        filepath.stem,
                        cluster["type"],
                        cluster["num_metals"],
                        cluster["total_atoms"],
                        json.dumps({"elements": cluster["elements"], "coords": cluster["coords"]})
                    ))
        except Exception as e:
            print(f"  Failed parsing {filepath.name}: {e}")

    print(f"Extraction done. {len(vectors)} clusters found in {time.time() - t0:.1f}s.")

    if not vectors:
        print("No valid clusters found. Exiting.")
        sys.exit(1)

    cur.executemany(
        "INSERT INTO sbus (cif_name, cluster_type, num_metals, total_atoms, structure_json) "
        "VALUES (?, ?, ?, ?, ?)",
        records
    )
    conn.commit()
    conn.close()

    vec_matrix = np.array(vectors, dtype=np.float32)
    faiss.normalize_L2(vec_matrix)
    dimension = vec_matrix.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(vec_matrix)
    faiss.write_index(index, index_path)

    print(f"Database saved. SQLite: {db_path}, FAISS: {index_path} (dim={dimension})")


def main():
    parser = argparse.ArgumentParser(description="MOF SBU Extractor & Vector Database Builder")
    subparsers = parser.add_subparsers(dest="command")

    build_parser = subparsers.add_parser("build", help="Build SBU database from CIF files")
    build_parser.add_argument("--input_dir", type=str, required=True,
                              help="Directory containing CIF files")
    build_parser.add_argument("--db_path", type=str, default="mof_sbu.db",
                              help="Output SQLite database path")
    build_parser.add_argument("--index_path", type=str, default="mof_sbu.index",
                              help="Output FAISS index path")

    args = parser.parse_args()

    if args.command == "build":
        build_database(args.input_dir, args.db_path, args.index_path)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
