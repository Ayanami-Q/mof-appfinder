import os
import sqlite3
from typing import Dict, List, Any, Optional

import numpy as np
import faiss

from mof_agent.config import DEFAULT_WEIGHTS
from mof_agent.database.physical_db import PhysicalDB
from mof_agent.database.data_loaders import load_sbu_formulas, load_ligand_smiles


class WhiteboxDBManager:
    """Whitebox multi-channel vector database manager."""

    def __init__(
        self,
        physical_dir: str,
        ligand_db: str,
        ligand_index: str,
        sbu_db: str,
        sbu_index: str,
    ):
        self.physical_dir = physical_dir
        self.ligand_db_path = ligand_db
        self.ligand_index_path = ligand_index
        self.sbu_db_path = sbu_db
        self.sbu_index_path = sbu_index

        self.physical_db = PhysicalDB(physical_dir)

        self._ligand_conn: Optional[sqlite3.Connection] = None
        self._ligand_index: Optional[faiss.Index] = None
        self._ligand_loaded = self._load_ligand_db()

        self._sbu_conn: Optional[sqlite3.Connection] = None
        self._sbu_index: Optional[faiss.Index] = None
        self._sbu_loaded = self._load_sbu_db()

        self._build_id_mappings()

        self._sbu_formulas = load_sbu_formulas()
        self._ligand_smiles = load_ligand_smiles()

        print(f"[WhiteboxDB] Initialization complete: "
              f"physical={self.physical_db.features.shape[0]} entries, "
              f"ligand={'OK' if self._ligand_loaded else 'MISSING'}, "
              f"SBU={'OK' if self._sbu_loaded else 'MISSING'}")

    def _load_ligand_db(self) -> bool:
        if not os.path.exists(self.ligand_db_path):
            print(f"[WhiteboxDB] Warning: ligand database not found: {self.ligand_db_path}")
            return False
        if not os.path.exists(self.ligand_index_path):
            print(f"[WhiteboxDB] Warning: ligand FAISS index not found: {self.ligand_index_path}")
            return False
        try:
            self._ligand_conn = sqlite3.connect(self.ligand_db_path)
            self._ligand_index = faiss.read_index(self.ligand_index_path)
            print(f"[WhiteboxDB] Ligand DB: {self._ligand_index.ntotal} vectors, "
                  f"dim={self._ligand_index.d}")
            return True
        except Exception as e:
            print(f"[WhiteboxDB] Failed to load ligand DB: {e}")
            return False

    def _load_sbu_db(self) -> bool:
        if not os.path.exists(self.sbu_db_path):
            print(f"[WhiteboxDB] Warning: SBU database not found: {self.sbu_db_path}")
            return False
        if not os.path.exists(self.sbu_index_path):
            print(f"[WhiteboxDB] Warning: SBU FAISS index not found: {self.sbu_index_path}")
            return False
        try:
            self._sbu_conn = sqlite3.connect(self.sbu_db_path)
            self._sbu_index = faiss.read_index(self.sbu_index_path)
            print(f"[WhiteboxDB] SBU DB: {self._sbu_index.ntotal} vectors, "
                  f"dim={self._sbu_index.d}")
            return True
        except Exception as e:
            print(f"[WhiteboxDB] Failed to load SBU DB: {e}")
            return False

    def _build_id_mappings(self) -> None:
        self._phys_id_set = set(self.physical_db.cif_ids)

        self._ligand_id_to_cif: Dict[int, str] = {}
        if self._ligand_conn:
            cur = self._ligand_conn.cursor()
            cur.execute("SELECT id, file_name FROM mofs")
            for row_id, file_name in cur.fetchall():
                cif_id = file_name.replace(".txt", "")
                self._ligand_id_to_cif[row_id] = cif_id

        self._sbu_id_to_cif: Dict[int, str] = {}
        if self._sbu_conn:
            cur = self._sbu_conn.cursor()
            cur.execute("SELECT id, cif_name FROM sbus")
            for row_id, cif_name in cur.fetchall():
                self._sbu_id_to_cif[row_id] = cif_name

    def _search_sbu_broad(self, sbu_vec: np.ndarray, k: int = 500) -> Dict[str, float]:
        if not self._sbu_loaded or self._sbu_index is None:
            return {}
        vec = sbu_vec.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(vec)
        scores, ids = self._sbu_index.search(vec, min(k, self._sbu_index.ntotal))
        scores, ids = scores[0], ids[0]
        cif_best: Dict[str, float] = {}
        for score, faiss_id in zip(scores, ids):
            faiss_id_int = int(faiss_id) + 1
            cif_name = self._sbu_id_to_cif.get(faiss_id_int)
            if cif_name:
                sim = float(score)
                if cif_name not in cif_best or sim > cif_best[cif_name]:
                    cif_best[cif_name] = sim
        return cif_best

    def _search_ligand_broad(self, ligand_vec: np.ndarray, k: int = 500) -> Dict[str, float]:
        if not self._ligand_loaded or self._ligand_index is None:
            return {}
        vec = ligand_vec.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(vec)
        scores, ids = self._ligand_index.search(vec, min(k, self._ligand_index.ntotal))
        scores, ids = scores[0], ids[0]
        result: Dict[str, float] = {}
        for score, faiss_id in zip(scores, ids):
            faiss_id_int = int(faiss_id) + 1
            cif_id = self._ligand_id_to_cif.get(faiss_id_int)
            if cif_id and cif_id not in result:
                result[cif_id] = float(score)
            if len(result) >= k:
                break
        return result

    def weighted_parallel_search(
        self,
        sbu_vec: Optional[np.ndarray],
        ligand_vec: Optional[np.ndarray],
        phys_dict: Dict[str, Any],
        weights: Dict[str, float],
        top_k: int = 15,
    ) -> List[Dict[str, Any]]:
        w_sbu = weights.get("sbu_weight", DEFAULT_WEIGHTS["sbu_weight"])
        w_ligand = weights.get("ligand_weight", DEFAULT_WEIGHTS["ligand_weight"])
        w_phys = weights.get("physical_weight", DEFAULT_WEIGHTS["physical_weight"])

        print(f"\n{'='*60}")
        print(f"[Weighted Parallel Search] Dynamic weighted retrieval")
        print(f"  Weights: SBU={w_sbu:.2f} | Ligand={w_ligand:.2f} | Physical={w_phys:.2f}")
        print(f"{'='*60}")

        sbu_scores: Dict[str, float] = {}
        if sbu_vec is not None and w_sbu > 0:
            sbu_scores = self._search_sbu_broad(sbu_vec, k=500)
            print(f"  SBU FAISS:    recalled {len(sbu_scores)} candidates")
        else:
            print(f"  SBU FAISS:    SKIP (vec=None or weight=0)")

        ligand_scores: Dict[str, float] = {}
        if ligand_vec is not None and w_ligand > 0:
            ligand_scores = self._search_ligand_broad(ligand_vec, k=500)
            print(f"  Ligand FAISS:  recalled {len(ligand_scores)} candidates")
        else:
            print(f"  Ligand FAISS: SKIP (vec=None or weight=0)")

        phys_candidates = self.physical_db.query(phys_dict, k=200)
        phys_scores: Dict[str, float] = {}
        phys_records: Dict[str, Dict] = {}
        for c in phys_candidates:
            phys_scores[c["cif_id"]] = c["distance"]
            phys_records[c["cif_id"]] = c
        print(f"  Physical KNN:  recalled {len(phys_scores)} candidates")

        all_ids: set = set()
        all_ids.update(sbu_scores.keys())
        all_ids.update(ligand_scores.keys())
        all_ids.update(phys_scores.keys())

        if not all_ids:
            print(f"  [!] All three channels returned empty; cannot retrieve")
            return []

        phys_distances = np.array([phys_scores.get(cid, 1.0) for cid in all_ids])
        phys_max = float(phys_distances.max()) if len(phys_distances) > 0 else 1.0
        if phys_max < 1e-12:
            phys_max = 1.0

        scored = []
        for cif_id in all_ids:
            s_sbu = float(sbu_scores.get(cif_id, 0.0))
            s_lig = float(ligand_scores.get(cif_id, 0.0))
            sbu_missing = cif_id not in sbu_scores
            lig_missing = cif_id not in ligand_scores

            phys_missing = cif_id not in phys_scores
            if phys_missing:
                d_phys = None
                d_norm = None
                phys_score = w_phys * 0.5
            else:
                d_phys = float(phys_scores[cif_id])
                d_norm = d_phys / phys_max
                phys_score = w_phys * (1.0 - d_norm)

            final_score = w_sbu * s_sbu + w_ligand * s_lig + phys_score
            sbu_score = w_sbu * s_sbu
            lig_score = w_ligand * s_lig

            missing_channels = []
            if sbu_missing:
                missing_channels.append("SBU")
            if lig_missing:
                missing_channels.append("ligand")
            if phys_missing:
                missing_channels.append("physical")

            phys_rec = phys_records.get(cif_id, {})
            sbu_formulas = self._sbu_formulas.get(cif_id, [])
            lig_smiles = self._ligand_smiles.get(cif_id, [])

            scored.append({
                "cif_id": cif_id,
                "final_score": round(final_score, 6),
                "sbu_similarity": round(s_sbu, 4),
                "ligand_similarity": round(s_lig, 4),
                "physical_distance": round(d_phys, 4) if d_phys is not None else None,
                "physical_distance_norm": round(d_norm, 4) if d_norm is not None else None,
                "physical_missing": phys_missing,
                "score_breakdown": {
                    "sbu_score": round(sbu_score, 4),
                    "ligand_score": round(lig_score, 4),
                    "physical_score": round(phys_score, 4),
                    "used_weights": {"sbu": w_sbu, "ligand": w_ligand, "physical": w_phys},
                    "missing_channels": missing_channels,
                },
                "sbu_formulas": sbu_formulas,
                "ligand_smiles": lig_smiles,
                "lcd": phys_rec.get("lcd"),
                "pld": phys_rec.get("pld"),
                "vf": phys_rec.get("vf"),
                "elements": phys_rec.get("elements", []),
                "volume": phys_rec.get("volume"),
            })

        scored.sort(key=lambda x: x["final_score"], reverse=True)
        top_k_results = scored[:top_k]

        print(f"\n  >>> Weighted scoring Top-{len(top_k_results)} candidates:")
        for i, c in enumerate(top_k_results, 1):
            pd = c.get("physical_distance")
            pd_str = f"{pd:.4f}" if pd is not None else "MISSING"
            missing_flag = " [NO_PHYS]" if c.get("physical_missing") else ""
            print(f"      #{i}: {c['cif_id']} "
                  f"final={c['final_score']:.4f} "
                  f"sbu={c['sbu_similarity']:.4f} "
                  f"lig={c['ligand_similarity']:.4f} "
                  f"phys_dist={pd_str}{missing_flag}")

        return top_k_results
