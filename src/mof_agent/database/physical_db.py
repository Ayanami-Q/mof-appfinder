import os
import json
import pickle
from typing import List, Dict, Any, Optional

import numpy as np

FEATURE_NAMES = ["a", "b", "c", "alpha", "beta", "gamma",
                 "volume", "num_elements", "lcd", "pld"]


def build_feature_vector(physical: Dict[str, Any]) -> np.ndarray:
    return np.array([physical[name] for name in FEATURE_NAMES], dtype=np.float64)


class PhysicalDB:
    """Physical feature KNN database."""

    def __init__(self, db_dir: str):
        features_path = os.path.join(db_dir, "physical_features.npy")
        scaler_path = os.path.join(db_dir, "physical_scaler.pkl")
        knn_path = os.path.join(db_dir, "physical_knn.pkl")
        meta_path = os.path.join(db_dir, "physical_metadata.json")

        for p in [features_path, scaler_path, knn_path, meta_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"Physical database incomplete, missing: {p}\n"
                    f"Run: python pmtransformer_model/build_physical_db.py first"
                )

        self.features = np.load(features_path)
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        with open(knn_path, "rb") as f:
            self.knn = pickle.load(f)
        with open(meta_path) as f:
            self.metadata = json.load(f)

        self.cif_ids: List[str] = self.metadata["cif_ids"]
        self.records: List[Dict] = self.metadata["records"]
        self._record_map: Dict[str, Dict] = {r["cif_id"]: r for r in self.records}

        print(f"[PhysicalDB] Loaded {len(self.cif_ids)} MOF physical features "
              f"(dim={self.features.shape[1]})")

    def query(self, physical_features: Dict[str, Any], k: int = 100) -> List[Dict[str, Any]]:
        vec = build_feature_vector(physical_features).reshape(1, -1)
        vec_scaled = self.scaler.transform(vec)
        distances, indices = self.knn.kneighbors(vec_scaled, n_neighbors=min(k, len(self.cif_ids)))

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            record = self.records[idx]
            results.append({
                "cif_id": self.cif_ids[idx],
                "distance": round(float(dist), 4),
                "elements": record.get("elements", []),
                "lcd": record.get("lcd", 0),
                "pld": record.get("pld", 0),
                "vf": record.get("vf", 0),
                "volume": record.get("volume", 0),
            })
        return results

    def get_record(self, cif_id: str) -> Optional[Dict]:
        return self._record_map.get(cif_id)

    def get_all_cif_ids(self) -> set:
        return set(self.cif_ids)
