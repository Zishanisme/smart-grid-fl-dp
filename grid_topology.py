"""
grid_topology.py
================
Builds realistic distribution network graph topology.

Replaces the previous "connect feeders in same utility" hack.

Implements:
  1. Radial feeder topology (how real distribution networks are structured)
     - Substations at root nodes
     - Feeders branch radially from substations
     - Lateral branches off main feeders

  2. Electrical distance proxy
     - Impedance-weighted path length between feeders
     - Feeders electrically closer → stronger spatial correlation

  3. Shared substation edges
     - Feeders sharing a substation are directly connected
     - Substation health propagates to all child feeders

  4. Correlated outage edges (data-driven)
     - Two feeders with high outage co-occurrence get an edge
     - Captures common-mode failures (shared weather exposure, shared crew)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch


# ---------------------------------------------------------------------------
# Radial feeder topology builder
# ---------------------------------------------------------------------------

class RadialGridTopology:
    """
    

    Here we construct a synthetic but structurally realistic radial network.
    """

    def __init__(
        self,
        assets:           pd.DataFrame,
        n_substations:    int   = 5,
        seed:             int   = 42,
    ):
        self.assets        = assets.reset_index(drop=True)
        self.n_feeders     = len(assets)
        self.n_substations = min(n_substations, self.n_feeders // 3)
        self.rng           = np.random.default_rng(seed)

        self.feeder_ids    = assets["feeder_id"].tolist()
        self.fid_to_idx    = {f: i for i, f in enumerate(self.feeder_ids)}

        self.substation_map: Dict[int, int] = {}   # feeder_idx → substation_idx
        self.edge_types:     Dict[Tuple[int,int], str] = {}
        self.impedance:      Dict[Tuple[int,int], float] = {}

        self._assign_substations()
        self._build_radial_tree()

    def _assign_substations(self):
        """
        Assign feeders to substations based on:
          1. Utility_id grouping (same utility → same substations)
          2. Geographic clustering proxy (loading profile similarity)
        """
        utility_groups = {}
        for i, row in self.assets.iterrows():
            uid = row["utility_id"]
            utility_groups.setdefault(uid, []).append(i)

        sub_idx = 0
        self.substation_to_feeders: Dict[int, List[int]] = {}

        for uid, feeder_indices in utility_groups.items():
            # Each utility gets 1-2 substations depending on size
            n_subs = max(1, len(feeder_indices) // 8)
            chunks = np.array_split(feeder_indices, n_subs)

            for chunk in chunks:
                sub = sub_idx
                self.substation_to_feeders[sub] = list(chunk)
                for fi in chunk:
                    self.substation_map[fi] = sub
                sub_idx += 1

        self.n_substations = sub_idx

    def _build_radial_tree(self):
        """
        Within each substation zone, build a radial tree:
          - Sort feeders by asset_age (older = closer to substation = trunk)
          - Connect in a tree structure with impedance weights
        """
        self.edges_src: List[int] = []
        self.edges_dst: List[int] = []

        for sub, feeders in self.substation_to_feeders.items():
            if len(feeders) < 2:
                continue

            # Sort by health_index ascending (healthier = closer to substation trunk)
            sorted_feeders = sorted(
                feeders,
                key=lambda i: self.assets.iloc[i]["health_index"]
            )

            # Build radial tree: each feeder connects to 1-2 upstream feeders
            for depth, fi in enumerate(sorted_feeders[1:], start=1):
                # Primary connection: to parent in tree
                parent = sorted_feeders[max(0, depth - 1)]
                self._add_edge(fi, parent, etype="radial_parent",
                               impedance=self._impedance(fi, parent))

                # Secondary connection: lateral tap to sibling (if branch exists)
                if depth > 1 and self.rng.random() < 0.4:
                    sibling = sorted_feeders[depth - 2]
                    self._add_edge(fi, sibling, etype="lateral_tap",
                                   impedance=self._impedance(fi, sibling) * 1.5)

            # Shared substation edges: all feeders in zone share substation node
            # → model as clique over top-3 trunk feeders (electrically proximate)
            trunk = sorted_feeders[:min(3, len(sorted_feeders))]
            for i, fi in enumerate(trunk):
                for fj in trunk[i+1:]:
                    self._add_edge(fi, fj, etype="shared_substation",
                                   impedance=self._impedance(fi, fj) * 0.5)

    def _add_edge(self, i: int, j: int, etype: str, impedance: float):
        # Undirected: add both directions
        self.edges_src.extend([i, j])
        self.edges_dst.extend([j, i])
        self.edge_types[(i, j)] = etype
        self.edge_types[(j, i)] = etype
        self.impedance[(i, j)]  = impedance
        self.impedance[(j, i)]  = impedance

    def _impedance(self, i: int, j: int) -> float:
        """
        Electrical distance proxy:
          Z ∝ cable_length × (1 / underground_ratio + 0.5) × loading_factor
        Overhead lines have higher impedance than underground cables.
        """
        a = self.assets.iloc[i]
        b = self.assets.iloc[j]

        def z(row):
            length  = row.get("cable_length_km", 5.0)
            ug      = max(0.1, row.get("underground_ratio", 0.5))
            loading = row.get("loading_pct", 60.0) / 100.0
            return length * (1.0 / ug) * (0.8 + 0.4 * loading)

        return float((z(a) + z(b)) / 2)

    def add_correlated_outage_edges(
        self,
        timeseries: pd.DataFrame,
        min_correlation: float = 0.3,
        top_k_per_feeder: int  = 2,
    ):
        """
        Data-driven edges: connect feeder pairs with correlated outage histories.
        Captures common-mode failures (shared weather, shared crew zones).
        """
        outage_pivot = (
            timeseries[["date", "feeder_id", "outage_flag"]]
            .pivot(index="date", columns="feeder_id", values="outage_flag")
            .fillna(0)
        )

        # Only use feeders present in both assets and timeseries
        common_fids = [f for f in self.feeder_ids if f in outage_pivot.columns]
        sub_pivot   = outage_pivot[common_fids]
        corr_matrix = sub_pivot.corr()

        n_added = 0
        for fi_name in common_fids:
            if fi_name not in self.fid_to_idx:
                continue
            fi = self.fid_to_idx[fi_name]
            row = corr_matrix[fi_name].drop(fi_name)
            top = row[row >= min_correlation].nlargest(top_k_per_feeder)
            for fj_name, corr_val in top.items():
                if fj_name not in self.fid_to_idx:
                    continue
                fj = self.fid_to_idx[fj_name]
                if (fi, fj) not in self.edge_types:
                    self._add_edge(fi, fj, etype="correlated_outage",
                                   impedance=1.0 - float(corr_val))
                    n_added += 1

        print(f"  [Topology] Added {n_added} correlated-outage edges "
              f"(min_correlation={min_correlation})")

    def to_edge_index(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (edge_index [2,E], edge_attr [E,2]) tensors."""
        edge_index = torch.tensor(
            [self.edges_src, self.edges_dst], dtype=torch.long
        )
        # Edge features: [normalised_impedance, is_shared_substation]
        impedances = [
            self.impedance.get((s, d), 1.0)
            for s, d in zip(self.edges_src, self.edges_dst)
        ]
        max_z = max(impedances) if impedances else 1.0
        is_sub = [
            1.0 if self.edge_types.get((s, d)) == "shared_substation" else 0.0
            for s, d in zip(self.edges_src, self.edges_dst)
        ]
        edge_attr = torch.tensor(
            [[z / max_z, s] for z, s in zip(impedances, is_sub)],
            dtype=torch.float32
        )
        return edge_index, edge_attr

    def topology_summary(self) -> Dict:
        n_edges = len(self.edges_src) // 2  # undirected
        type_counts: Dict[str, int] = {}
        for etype in self.edge_types.values():
            type_counts[etype] = type_counts.get(etype, 0) + 1
        type_counts = {k: v // 2 for k, v in type_counts.items()}

        print(f"\n  [Grid topology] Feeders={self.n_feeders} | "
              f"Substations={self.n_substations} | Edges={n_edges}")
        for etype, count in type_counts.items():
            print(f"    {etype:<25}: {count}")
        avg_degree = (len(self.edges_src) / self.n_feeders)
        print(f"    Avg degree: {avg_degree:.2f}")

        return {
            "n_feeders":     self.n_feeders,
            "n_substations": self.n_substations,
            "n_edges":       n_edges,
            "edge_types":    type_counts,
            "avg_degree":    round(avg_degree, 2),
        }
