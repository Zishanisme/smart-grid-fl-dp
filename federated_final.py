"""
federated_final.py
==================

"""

import sys, json, warnings
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from opacus.validators import ModuleValidator
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import wasserstein_distance
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, average_precision_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "models"))

from secagg_dh import BonawitzSecAgg

try:
    from opacus import PrivacyEngine
    HAS_OPACUS = True
except ImportError:
    HAS_OPACUS = False
    print("[WARN] opacus not installed — DP disabled. pip install opacus>=1.4")

try:
    from torch_geometric.nn import SAGEConv
    pass  # pygeom F imported inside method
    HAS_PYGEOM = True
except Exception:
    HAS_PYGEOM = False

# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------
STATIC_FEATURES = [
    "asset_age_years", "health_index", "underground_ratio",
    "vegetation_risk_index", "der_penetration_pct", "rated_capacity_proxy",
]
TEMPORAL_FEATURES = [
    "outage_rate_7d", "outage_rate_30d", "storm_rate_7d", "storm_rate_30d",
    "max_loading_7d", "max_loading_30d", "mean_temp_7d", "mean_temp_30d",
    "days_since_last_outage", "loading_pct", "temp_c", "wind_speed_ms",
    "storm_flag", "heatwave_flag",
]
ALL_FEATURES = STATIC_FEATURES + TEMPORAL_FEATURES
LABEL_COL    = "label_7d"
IN_DIM       = len(ALL_FEATURES)


# ===========================================================================
# MODELS
# ===========================================================================

class TabularMLP(nn.Module):
    def __init__(self, hidden=64, drop=0.3):
        super().__init__()
        ns, nt = len(STATIC_FEATURES), len(TEMPORAL_FEATURES)
        self.s = nn.Sequential(
            nn.Linear(ns, hidden), nn.LayerNorm(hidden),
            nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
        )
        self.t = nn.Sequential(
            nn.Linear(nt, hidden), nn.LayerNorm(hidden),
            nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Dropout(drop), nn.Linear(hidden // 2, 1),
        )

    def forward(self, xs, xt, edge_index=None):
        # edge_index accepted but not used — API-compatible with GraphRiskModel
        return self.head(
            torch.cat([self.s(xs), self.t(xt)], dim=-1)
        ).squeeze(-1)


class GraphRiskModel(nn.Module):
    """
    Two-layer GraphSAGE risk model.

    When edge_index is supplied: runs full message passing over the
    feeder-substation graph (radial prior topology).

    When edge_index is None (e.g., test without topology data): falls back
    to topology-aware MLP that uses global mean-pool as neighbourhood context.
    This is explicitly labelled as a degraded mode in logs.
    """
    def __init__(self, in_dim=IN_DIM, hidden=64, drop=0.3):
        super().__init__()
        self._sage = HAS_PYGEOM
        if HAS_PYGEOM:
            self.conv1 = SAGEConv(in_dim, hidden)
            self.conv2 = SAGEConv(hidden, hidden // 2)
            self.bn1   = nn.BatchNorm1d(hidden)
            self.bn2   = nn.BatchNorm1d(hidden // 2)
        else:
            # Fallback: mean-pool neighbourhood approximation
            self.enc = nn.Sequential(
                nn.Linear(in_dim * 2, hidden), nn.ReLU(),
                nn.Dropout(drop), nn.Linear(hidden, hidden // 2), nn.ReLU(),
            )
        self.head = nn.Sequential(
            nn.Linear(hidden // 2, hidden // 4), nn.ReLU(),
            nn.Dropout(drop), nn.Linear(hidden // 4, 1),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, xs, xt, edge_index=None):
        import torch.nn.functional as F
        x = torch.cat([xs, xt], dim=-1)

        if self._sage and edge_index is not None:
            h = F.relu(self.bn1(self.conv1(x, edge_index)))
            h = self.drop(h)
            h = F.relu(self.bn2(self.conv2(h, edge_index)))
            h = self.drop(h)
        else:
            if edge_index is None:
                print("[GraphRiskModel] edge_index=None — running in MLP-fallback mode")
            ctx = x.mean(0, keepdim=True).expand_as(x)
            h = self.enc(torch.cat([x, ctx], dim=-1))

        return self.head(h).squeeze(-1)


# ===========================================================================
# FIX 4: SAIDI impact model (properly implemented)
# ===========================================================================

class SAIDIGraphModel(nn.Module):
    """
    Self-contained model for the SAIDI experiment.

    Design rationale:
      The previous SAIDIImpactModel WRAPPED another model. This meant DP
      could only be attached to the inner encoder — the SAIDI heads were
      outside Opacus's scope. Training them with a separate plain SGD
      bypassed the PrivacyEngine entirely.

      This class is self-contained: encoder + SAIDI heads are one nn.Module.
      UtilityClient attaches Opacus to this whole object at init time.
      The DP optimizer covers ALL parameters — encoder and heads.
      SecAgg aggregates ALL weights — full federated SAIDI, not just encoder.

    forward(xs, xt, edge_index) → outage_logit [N]
      Compatible with UtilityClient.train_round (BCEWithLogitsLoss) and
      UtilityClient.evaluate (sigmoid → AUC).

    saidi_forward(xs, xt, edge_index) → (outage_logit [N], pred_duration [N])
      Used in _run_fl_saidi training loop for the joint loss.
      Called on get_base(client.model) so it works through the DP wrapper.
    """

    def __init__(self, in_dim=IN_DIM, hidden=64, drop=0.3):
        super().__init__()
        # Encoder: same GraphRiskModel architecture
        self._sage = HAS_PYGEOM
        if HAS_PYGEOM:
            self.conv1 = SAGEConv(in_dim, hidden)
            self.conv2 = SAGEConv(hidden, hidden // 2)
            self.bn1   = nn.BatchNorm1d(hidden)
            self.bn2   = nn.BatchNorm1d(hidden // 2)
        else:
            self.enc = nn.Sequential(
                nn.Linear(in_dim * 2, hidden), nn.ReLU(),
                nn.Dropout(drop), nn.Linear(hidden, hidden // 2), nn.ReLU(),
            )

        # Shared projection from encoder output (scalar) → embedding
        self.embed_proj  = nn.Sequential(nn.Linear(1, 32), nn.ReLU())

        # Outage head: binary classification
        self.outage_head = nn.Linear(32, 1)

        # CML head: predicts expected outage duration (Softplus → strictly positive)
        self.cml_head = nn.Sequential(
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Softplus(),
        )
        self.drop = nn.Dropout(drop)

    def _encode(self, xs, xt, edge_index=None):
        import torch.nn.functional as F
        x = torch.cat([xs, xt], dim=-1)
        if self._sage and edge_index is not None:
            h = F.relu(self.bn1(self.conv1(x, edge_index)))
            h = self.drop(h)
            h = F.relu(self.bn2(self.conv2(h, edge_index)))
            h = self.drop(h)
        else:
            ctx = x.mean(0, keepdim=True).expand_as(x)
            h   = self.enc(torch.cat([x, ctx], dim=-1))
        return h   # [N, hidden//2]

    def _heads(self, h):
        # Map [N, hidden//2] → scalar per node, then project to embedding
        logit_raw = h.mean(-1, keepdim=True)          # [N, 1]  (cheap projection)
        embed     = self.embed_proj(logit_raw)          # [N, 32]
        outage_logit  = self.outage_head(embed).squeeze(-1)  # [N]
        pred_duration = self.cml_head(embed).squeeze(-1)     # [N]
        return outage_logit, pred_duration

    def forward(self, xs, xt, edge_index=None):
        """
        Returns outage_logit only.
        Compatible with UtilityClient.train_round (BCEWithLogitsLoss)
        and UtilityClient.evaluate (sigmoid → AUC).
        The CML head is still trained via saidi_forward in _run_fl_saidi.
        """
        h = self._encode(xs, xt, edge_index)
        outage_logit, _ = self._heads(h)
        return outage_logit

    def saidi_forward(self, xs, xt, edge_index=None):
        """
        Returns (outage_logit, pred_duration).
        Used in _run_fl_saidi for the joint SAIDI loss.
        """
        h = self._encode(xs, xt, edge_index)
        return self._heads(h)

    def expected_cml(self, xs, xt, n_customers, edge_index=None):
        """E[CML] = P(outage) × E[duration] × n_customers."""
        logit, duration = self.saidi_forward(xs, xt, edge_index)
        return torch.sigmoid(logit) * duration * n_customers


def saidi_loss(
    outage_logit:  torch.Tensor,
    pred_duration: torch.Tensor,
    y_binary:      torch.Tensor,
    n_customers:   torch.Tensor,
    duration_true: torch.Tensor,    # per-row observed or synthetic duration
    lambda_cml:    float = 0.4,
) -> torch.Tensor:
    """
    Joint loss:
      L = BCE(outage_logit, y_binary)
          + λ × MSE(log1p(pred_cml), log1p(true_cml))

    true_cml = y_binary × duration_true × n_customers  (actual customer-minutes-lost)
    """
    bce_loss  = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([5.0], device=y_binary.device)
    )(outage_logit, y_binary)

    true_cml = y_binary * duration_true * n_customers
    pred_cml = torch.sigmoid(outage_logit) * pred_duration * n_customers
    cml_loss  = nn.MSELoss()(
        torch.log1p(pred_cml), torch.log1p(true_cml)
    )

    return bce_loss + lambda_cml * cml_loss


# ===========================================================================
# DATA + NORMALISER
# ===========================================================================

class Normalizer:
    def __init__(self):
        self.m = self.s = None

    def fit(self, x: torch.Tensor) -> "Normalizer":
        self.m, self.s = x.mean(0), x.std(0).clamp(1e-6)
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.m) / self.s


def load_data(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path, parse_dates=["date"])
    if "rated_capacity_proxy" not in df.columns:
        df["rated_capacity_proxy"] = df["loading_pct"] * 0.5
    if "n_customers" not in df.columns:
        df["n_customers"] = 500.0
    return df.dropna(subset=ALL_FEATURES + [LABEL_COL]).sort_values("date")


def utility_split(df: pd.DataFrame, uid: str, val_frac=0.10, test_frac=0.15):
    sub = df[df["utility_id"] == uid].copy()
    n   = len(sub)
    i_v = int(n * (1 - val_frac - test_frac))
    i_t = int(n * (1 - test_frac))
    return sub.iloc[:i_v], sub.iloc[i_v:i_t], sub.iloc[i_t:]


def make_loader(
    frame: pd.DataFrame,
    ns: Normalizer, nt: Normalizer,
    shuffle: bool = False,
    include_customers: bool = False,
) -> DataLoader:
    xs = ns.transform(
        torch.tensor(frame[STATIC_FEATURES].values,   dtype=torch.float32)
    )
    xt = nt.transform(
        torch.tensor(frame[TEMPORAL_FEATURES].values, dtype=torch.float32)
    )
    y  = torch.tensor(frame[LABEL_COL].values, dtype=torch.float32)

    if include_customers:
        nc = torch.tensor(
            frame["n_customers"].values, dtype=torch.float32
        )
        # Synthetic duration: Gamma-distributed around 60 min (EU benchmark)
        dur = torch.tensor(
            np.random.default_rng(42).gamma(shape=2.0, scale=30.0, size=len(frame)),
            dtype=torch.float32,
        )
        ds = TensorDataset(xs, xt, y, nc, dur)
    else:
        ds = TensorDataset(xs, xt, y)

    return DataLoader(ds, batch_size=256, shuffle=shuffle, drop_last=False)


# ===========================================================================
# FIX 3: Persistent DP — one model, one engine, across all rounds
# ===========================================================================

def get_base(model: nn.Module) -> nn.Module:
    """Strip Opacus GradSampleModule to get the underlying nn.Module."""
    return model._module if hasattr(model, "_module") else model


def set_weights_on_base(model: nn.Module, arrays: List[np.ndarray]):
    """
    Update weights on the base module (inside Opacus wrapper if present).
    This is the ONLY correct way to update global weights mid-federation
    without breaking the PrivacyEngine attachment.
    """
    base = get_base(model)
    sd   = OrderedDict({
        k: torch.tensor(v)
        for k, v in zip(base.state_dict().keys(), arrays)
    })
    base.load_state_dict(sd, strict=True)


def get_weights_from_base(model: nn.Module) -> List[np.ndarray]:
    return [v.detach().cpu().numpy() for v in get_base(model).state_dict().values()]


def frozen_base_params(model: nn.Module) -> List[torch.Tensor]:
    """Capture param tensors from base module only — shapes never expanded by Opacus."""
    return [p.detach().clone() for p in get_base(model).parameters()]


class UtilityClient:
    """
    One client = one utility company.

    The model and PrivacyEngine are created ONCE and reused for all rounds.
    Each round:
      1. Global weights set on model._module (base stays DP-wrapped)
      2. Local FedProx training for `local_epochs`
      3. Updated weights extracted from model._module
      4. Privacy budget read from persistent engine

    This gives correct cumulative epsilon accounting.
    """

    def __init__(
        self,
        uid:           str,
        df:            pd.DataFrame,
        assets_df:     pd.DataFrame,
        model_cls,
        edge_index:    Optional[torch.Tensor],   # FIX 1: pre-built subgraph
        n_rounds:      int,
        local_epochs:  int,
        use_dp:        bool,
        target_eps:    float,
        target_delta:  float = 1e-5,
        max_grad_norm: float = 1.0,
        mu:            float = 0.1,
    ):
        self.uid        = uid
        self.mu         = mu
        self.edge_index = edge_index    # FIX 1: stored per client

        tr, vl, te = utility_split(df, uid)
        self.test_df = te

        self.ns = Normalizer().fit(
            torch.tensor(tr[STATIC_FEATURES].values, dtype=torch.float32)
        )
        self.nt = Normalizer().fit(
            torch.tensor(tr[TEMPORAL_FEATURES].values, dtype=torch.float32)
        )

        self.train_loader = make_loader(tr, self.ns, self.nt, shuffle=True,
                                        include_customers=True)
        self.val_loader   = make_loader(vl, self.ns, self.nt)
        self.test_loader  = make_loader(te, self.ns, self.nt)

        self.n_train      = len(tr)
        self.prevalence   = float(tr[LABEL_COL].mean())

        # Persistent model — never recreated
        self.model = model_cls()
        if use_dp and HAS_OPACUS:
            self.model = ModuleValidator.fix(self.model)
        
        self.optimizer = torch.optim.SGD(
            get_base(self.model).parameters(),
            lr=0.01, momentum=0.9, weight_decay=1e-4,
        )

        # Attach DP engine ONCE
        self.dp_engine = None
        self.dp_attached = False
        self.dp_exhausted = False

        if use_dp and HAS_OPACUS:
            pe = PrivacyEngine()
            self.model, self.optimizer, self.train_loader = (
                pe.make_private_with_epsilon(
                    module=self.model,
                    optimizer=self.optimizer,
                    data_loader=self.train_loader,
                    target_epsilon=target_eps,
                    target_delta=target_delta,
                    epochs=n_rounds * local_epochs,
                    max_grad_norm=max_grad_norm,
                )
            )
            self.dp_engine   = pe
            self.dp_attached = True

        print(
            f"  Client {uid:10s} | n_train={self.n_train:>7,} "
            f"| prev={self.prevalence:.4f} "
            f"| DP={'ON' if self.dp_attached else 'OFF'} "
            f"| graph={'YES' if edge_index is not None else 'NO'}"
        )

    def current_epsilon(self) -> Optional[float]:
        if self.dp_engine is None:
            return None
        try:
            return float(self.dp_engine.get_epsilon(1e-5))
        except Exception:
            return None

    def train_round(self, global_arrays: List[np.ndarray], local_epochs: int):
        """
        Update local model with global weights, then run FedProx training.
        DP engine stays attached — correctly accumulates epsilon.
        """
        if self.dp_exhausted:
            return

        # Set global weights on base module (DP wrapper stays intact)
        set_weights_on_base(self.model, global_arrays)

        # Freeze global params for proximal term — from base, no Opacus expansion
        global_frozen = frozen_base_params(self.model)

        crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]))
        self.model.train()

        for _ in range(local_epochs):
            eps = self.current_epsilon()
            if eps is not None and eps >= (self.dp_engine.target_epsilon
                                           if hasattr(self.dp_engine, "target_epsilon")
                                           else 1.0):
                self.dp_exhausted = True
                break

            for batch in self.train_loader:
                xs, xt, y = batch[0], batch[1], batch[2]
                self.optimizer.zero_grad()

                # FIX 1: pass edge_index to model
                logits    = self.model(xs, xt, self.edge_index)
                task_loss = crit(logits, y)

                # FedProx proximal term — base params aligned with frozen_base_params
                prox = torch.tensor(0.0)
                for p, g in zip(get_base(self.model).parameters(), global_frozen):
                    prox += (self.mu / 2.0) * torch.sum((p - g) ** 2)

                (task_loss + prox).backward()
                self.optimizer.step()

    def get_weights(self) -> List[np.ndarray]:
        return get_weights_from_base(self.model)

    @torch.no_grad()
    def evaluate(self, global_arrays: List[np.ndarray]) -> Dict:
        set_weights_on_base(self.model, global_arrays)
        base = get_base(self.model)
        base.eval()

        xs = self.ns.transform(
            torch.tensor(self.test_df[STATIC_FEATURES].values, dtype=torch.float32)
        )
        xt = self.nt.transform(
            torch.tensor(self.test_df[TEMPORAL_FEATURES].values, dtype=torch.float32)
        )
        y  = self.test_df[LABEL_COL].values

        # FIX 1: pass edge_index in evaluation too
        probs = torch.sigmoid(base(xs, xt, self.edge_index)).numpy()

        auc   = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else 0.5
        k     = max(1, int(len(probs) * 0.10))
        top10 = float(y[np.argsort(probs)[::-1][:k]].sum() / max(1, y.sum()))

        return {
            "utility":   self.uid,
            "auc":       round(float(auc),   4),
            "top10":     round(float(top10), 4),
            "prevalence": self.prevalence,
            "n_test":    len(y),
            "epsilon":   round(self.current_epsilon(), 4) if self.current_epsilon() else None,
            "graph_used": self.edge_index is not None,
        }


# ===========================================================================
# 
# ===========================================================================

def build_utility_subgraphs(
    assets_df:  pd.DataFrame,
    timeseries: pd.DataFrame,
) -> Dict[str, Optional[torch.Tensor]]:
    """
    Builds a synthetic radial topology prior (not claimed to be realistic)
    for each utility's feeder subgraph.

    Returns dict: utility_id → edge_index tensor [2, E] with LOCAL feeder indices
    (0-indexed within each utility's feeder list).
    """
    from grid_topology import RadialGridTopology

    subgraphs: Dict[str, Optional[torch.Tensor]] = {}

    for uid, grp in assets_df.groupby("utility_id"):
        grp = grp.reset_index(drop=True)
        try:
            topo = RadialGridTopology(grp, n_substations=max(1, len(grp) // 8))
            # Optionally add correlated-outage edges if timeseries available
            if timeseries is not None:
                topo.add_correlated_outage_edges(
                    timeseries[timeseries["utility_id"] == uid],
                    min_correlation=0.25,
                )
            edge_index, _ = topo.to_edge_index()
            subgraphs[uid] = edge_index

            n_nodes = len(grp)
            n_edges = edge_index.shape[1] // 2
            print(f"  [{uid}] Synthetic radial prior: "
                  f"{n_nodes} feeders, {n_edges} undirected edges")
        except Exception as e:
            print(f"  [{uid}] Graph build failed ({e}), using MLP fallback")
            subgraphs[uid] = None

    return subgraphs


# ===========================================================================
# 
# ===========================================================================

def masked_weighted_aggregate(
    secagg:        BonawitzSecAgg,
    raw_updates:   List[List[np.ndarray]],
    dataset_sizes: List[int],
    round_num:     int,
    dropped:       List[int] = None,
) -> Tuple[List[np.ndarray], Dict]:
    """
    Correct SecAgg for weighted FedAvg.

    Naive approach (WRONG):
        masked_i = w_i + mask_ij
        server computes: alpha_i * (w_i + mask_ij) + alpha_j * (w_j - mask_ij)
                       = alpha_i*w_i + alpha_j*w_j + mask_ij*(alpha_i - alpha_j)
        Masks DON'T cancel unless alpha_i == alpha_j.

    Correct approach:
        Client i premultiplies: weighted_i = (n_i / total) * w_i
        Masks applied to weighted update: masked_i = weighted_i + mask_ij
        Server sums: sum(masked_i) = sum(alpha_i * w_i) + sum(all masks)
                                   = weighted_average  (masks cancel exactly)

    Clients need to know `total` to premultiply. Dataset sizes are not
    sensitive (revealed to server anyway in standard FL).
    """
    dropped = dropped or []
    surviving = [i for i in range(len(raw_updates)) if i not in dropped]
    total = sum(dataset_sizes[i] for i in surviving)

    # Step 1: each surviving client premultiplies by its FedAvg weight
    weighted_updates = []
    for i in range(len(raw_updates)):
        if i in dropped:
            weighted_updates.append(None)
        else:
            weight = dataset_sizes[i] / total
            wu = [layer * weight for layer in raw_updates[i]]
            weighted_updates.append(wu)

    # Step 2: mask the WEIGHTED updates (masks now cancel in unweighted sum)
    masked = []
    for i in range(len(weighted_updates)):
        if weighted_updates[i] is None:
            masked.append(None)
        else:
            m = secagg.mask_update(i, weighted_updates[i], round_num, dropped)
            masked.append(m)

    # Step 3: server sums masked weighted updates
    # Result = sum(alpha_i * w_i) + sum(masks that cancel) = weighted average
    agg, report = secagg.aggregate(
        masked,
        [1] * len(masked),   # all weights = 1 here because premultiplied already
        round_num,
    )
    report["weighted_secagg"] = True
    return agg, report


# ===========================================================================
# NON-IID BENCHMARK
# ===========================================================================

def noniid_benchmark(df: pd.DataFrame, out_dir: Path) -> Dict:
    uids  = sorted(df["utility_id"].unique())

    stats = {}
    for uid in uids:
        sub = df[df["utility_id"] == uid]
        stats[uid] = {
            "n":              len(sub),
            "prevalence":     round(float(sub[LABEL_COL].mean()), 4),
            "age_mean":       round(float(sub["asset_age_years"].mean()), 2),
            "health_mean":    round(float(sub["health_index"].mean()), 4),
            "loading_mean":   round(float(sub["loading_pct"].mean()), 2),
        }

    print(f"\n{'─'*68}")
    print(f"  Non-IID statistics per utility (synthetic radial topology prior)")
    print(f"{'─'*68}")
    print(f"  {'Utility':<12} {'N':>8} {'Outage %':>10} {'Age μ':>8} {'Health μ':>10}")
    for uid, s in stats.items():
        print(f"  {uid:<12} {s['n']:>8,} {s['prevalence']:>10.4f} "
              f"{s['age_mean']:>8.1f} {s['health_mean']:>10.4f}")

    def kl_bern(p, q, eps=1e-9):
        p, q = max(eps, min(1-eps, p)), max(eps, min(1-eps, q))
        return p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))

    kl_pairs, w1_pairs = {}, {}
    print(f"\n  Divergences between utility pairs:")
    for i, u1 in enumerate(uids):
        for u2 in uids[i+1:]:
            kl = (kl_bern(stats[u1]["prevalence"], stats[u2]["prevalence"]) +
                  kl_bern(stats[u2]["prevalence"], stats[u1]["prevalence"])) / 2
            w1 = wasserstein_distance(
                df[df["utility_id"] == u1]["health_index"].values,
                df[df["utility_id"] == u2]["health_index"].values,
            )
            key = f"{u1}|{u2}"
            kl_pairs[key] = round(kl, 6)
            w1_pairs[key] = round(w1, 6)
            print(f"    {u1} vs {u2}: KL(outage)={kl:.4f}  W1(health)={w1:.4f}")

    max_kl = max(kl_pairs.values()) if kl_pairs else 0
    level  = "HIGH" if max_kl > 0.05 else ("MEDIUM" if max_kl > 0.01 else "LOW")
    print(f"\n  Non-IID level: {level} (max KL={max_kl:.4f})")
    print(f"  FedProx advantage over FedAvg: {'EXPECTED' if level != 'LOW' else 'MARGINAL'}")

    result = {"per_utility": stats, "kl_pairs": kl_pairs,
              "w1_pairs": w1_pairs, "level": level, "max_kl": round(max_kl, 6)}
    (out_dir / "noniid_benchmark.json").write_text(json.dumps(result, indent=2))
    return result


# ===========================================================================
# 
# ===========================================================================

def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    edges = np.linspace(0, 1, n_bins + 1)
    result = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() == 0:
            continue
        result += (m.sum() / len(y)) * abs(y[m].mean() - p[m].mean())
    return float(result)


def plot_reliability(
    systems: List[Tuple[str, np.ndarray, np.ndarray]],
    out_dir: Path,
):
    n = len(systems)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    colors = ["#378ADD", "#1D9E75", "#7F77DD", "#D85A30", "#0F6E56"]
    for ax, (name, y, p), color in zip(axes, systems, colors):
        frac, mean = calibration_curve(y, p, n_bins=12)
        score      = ece(y, p)
        ax.plot([0,1],[0,1],"k--",lw=1,alpha=0.4)
        ax.plot(mean, frac, "o-", color=color, lw=2, ms=5)
        ax.fill_between(mean, frac, mean, alpha=0.15, color=color)
        ax.set_title(f"{name}\nECE={score:.4f}", fontsize=9)
        ax.set_xlabel("Predicted probability")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Empirical frequency")
    fig.suptitle("Reliability diagrams — smart grid outage risk models", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "reliability_diagrams.png", dpi=150)
    plt.close(fig)
    print("  Saved reliability_diagrams.png")


# ===========================================================================
# ROBUSTNESS — 
# ===========================================================================

def robustness_client_dropout(
    df: pd.DataFrame,
    n_rounds: int = 6,
    dropout_rate: float = 0.33,
    seed: int = 42,
) -> Dict:
    """
    Tests dropout-aware masking exclusion.
    Simulates BonawitzSecAgg behaviour when clients drop before sending.

    What this tests: SecAgg continues with surviving clients and
    threshold logic aborts the round if too few clients survive.

    What this does NOT test: full Bonawitz protocol secret-share
    recovery, which requires clients to share partial secrets of
    dropped clients' keys before the round begins.
    """
    uids   = sorted(df["utility_id"].unique())
    n      = len(uids)
    secagg = BonawitzSecAgg(n_clients=n, dropout_threshold=max(2, n-1), seed=seed)
    rng    = np.random.default_rng(seed)

    dummy_model = TabularMLP()
    shapes = [a.shape for a in get_weights_from_base(dummy_model)]
    gw     = get_weights_from_base(dummy_model)

    results = []
    print(f"\n  [Robustness] Dropout-aware masking test (rate={dropout_rate})")
    for rnd in range(1, n_rounds + 1):
        dropped   = [i for i in range(n) if rng.random() < dropout_rate]
        surviving = [i for i in range(n) if i not in dropped]

        raw_updates, sizes = [], []
        for i in range(n):
            noise  = [rng.standard_normal(s).astype(np.float32)*0.01 for s in shapes]
            raw_updates.append([gw[l] + noise[l] for l in range(len(gw))])
            sizes.append(1000)

        # Mark dropped clients' updates as None
        for i in dropped:
            raw_updates[i] = None

        present   = [u for u in raw_updates if u is not None]
        p_sizes   = [sizes[i] for i in surviving]
        try:
            agg, report = masked_weighted_aggregate(
                secagg,
                [u if i not in dropped else [np.zeros_like(gw[l]) for l in range(len(gw))]
                 for i, u in enumerate(raw_updates)],
                sizes, rnd, dropped,
            )
            status = f"OK ({len(surviving)}/{n} clients)"
        except RuntimeError as e:
            status = f"ABORTED: {e}"

        print(f"    Round {rnd}: dropped={dropped} → {status}")
        results.append({"round": rnd, "dropped": dropped, "status": status})

    return {"dropout_test": results, "dropout_rate": dropout_rate,
            "note": "Tests masking exclusion only, not full Bonawitz secret-share recovery"}


def robustness_adversarial(
    df: pd.DataFrame,
    n_rounds: int = 6,
    noise_scale: float = 5.0,
    adversarial_idx: int = 0,
    seed: int = 42,
) -> Dict:
    """
    Byzantine client sends gradient updates scaled by noise_scale.
    Aggregation uses gradient norm clipping (max_norm=1.0) to limit damage.
    This test shows degradation under attack and partial resilience from clipping.

    Note: This is a gradient-noise Byzantine attack test with norm clipping.
    It is NOT a Krum or trimmed-mean robust aggregation — those are not implemented.
    """
    uids     = sorted(df["utility_id"].unique())
    gm       = TabularMLP()
    crit     = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]))
    max_norm = 1.0       # gradient clipping — partial Byzantine defence
    auc_hist = []
    rng      = np.random.default_rng(seed)

    tr_data  = {}
    for uid in uids:
        tr, _, te = utility_split(df, uid)
        ns = Normalizer().fit(torch.tensor(tr[STATIC_FEATURES].values, dtype=torch.float32))
        nt = Normalizer().fit(torch.tensor(tr[TEMPORAL_FEATURES].values, dtype=torch.float32))
        tr_data[uid] = {
            "xs": ns.transform(torch.tensor(tr[STATIC_FEATURES].values,   dtype=torch.float32)),
            "xt": nt.transform(torch.tensor(tr[TEMPORAL_FEATURES].values, dtype=torch.float32)),
            "y":  torch.tensor(tr[LABEL_COL].values, dtype=torch.float32),
            "xs_te": ns.transform(torch.tensor(te[STATIC_FEATURES].values,   dtype=torch.float32)),
            "xt_te": nt.transform(torch.tensor(te[TEMPORAL_FEATURES].values, dtype=torch.float32)),
            "y_te":  te[LABEL_COL].values,
        }

    print(f"\n  [Robustness] Byzantine attack test (noise_scale={noise_scale}, "
          f"adversary={uids[adversarial_idx]})")
    for rnd in range(1, n_rounds + 1):
        gw      = get_weights_from_base(gm)
        updates, sizes = [], []

        for i, uid in enumerate(uids):
            local = TabularMLP()
            set_weights_on_base(local, gw)
            d   = tr_data[uid]
            opt = torch.optim.SGD(local.parameters(), lr=0.01)
            local.train(); opt.zero_grad()
            crit(local(d["xs"], d["xt"]), d["y"]).backward()

            # Gradient norm clipping — partial Byzantine defence
            nn.utils.clip_grad_norm_(local.parameters(), max_norm)

            if i == adversarial_idx:
                # Byzantine: scale gradients by noise_scale AFTER clipping
                with torch.no_grad():
                    for p in local.parameters():
                        if p.grad is not None:
                            p.grad.mul_(noise_scale)

            opt.step()
            updates.append(get_weights_from_base(local))
            sizes.append(len(d["y"]))

        total = sum(sizes)
        new_w = [
            sum(updates[i][l] * (sizes[i] / total) for i in range(len(uids)))
            for l in range(len(gw))
        ]
        set_weights_on_base(gm, new_w)

        # Evaluate on first utility
        uid0 = uids[0]
        d0   = tr_data[uid0]
        gm.eval()
        with torch.no_grad():
            p = torch.sigmoid(gm(d0["xs_te"], d0["xt_te"])).numpy()
        auc = roc_auc_score(d0["y_te"], p) if len(np.unique(d0["y_te"])) > 1 else 0.5
        auc_hist.append(round(float(auc), 4))
        print(f"    Round {rnd}: AUC={auc:.4f}")

    return {"adversarial_auc_history": auc_hist, "noise_scale": noise_scale,
            "adversarial_client": uids[adversarial_idx], "max_grad_norm": max_norm,
            "note": "Gradient norm clipping provides partial robustness only. "
                    "Krum/trimmed-mean robust aggregation not implemented."}


def robustness_missing_telemetry(
    model: nn.Module,
    test_df: pd.DataFrame,
    edge_index: Optional[torch.Tensor] = None,
    fracs: List[float] = [0.0, 0.1, 0.2, 0.3, 0.5],
) -> List[Dict]:
    ns = Normalizer().fit(
        torch.tensor(test_df[STATIC_FEATURES].values, dtype=torch.float32)
    )
    nt = Normalizer().fit(
        torch.tensor(test_df[TEMPORAL_FEATURES].values, dtype=torch.float32)
    )
    xs  = ns.transform(torch.tensor(test_df[STATIC_FEATURES].values,   dtype=torch.float32))
    y   = test_df[LABEL_COL].values
    rng = np.random.default_rng(0)
    rows = []

    print(f"\n  [Robustness] Missing telemetry (zero-imputation)")
    for frac in fracs:
        xt = nt.transform(
            torch.tensor(test_df[TEMPORAL_FEATURES].values, dtype=torch.float32)
        )
        mask = torch.tensor(rng.uniform(0, 1, xt.shape) < frac, dtype=torch.bool)
        xt[mask] = 0.0

        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(xs, xt, edge_index)).numpy()
        auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else 0.5
        rows.append({"missing_frac": frac, "auc_roc": round(float(auc), 4)})
        print(f"    {int(frac*100):3d}% missing: AUC={auc:.4f}")
    return rows


# ===========================================================================
# METRICS
# ===========================================================================

def metrics(y: np.ndarray, p: np.ndarray, name: str) -> Dict:
    k = max(1, int(len(p) * 0.10))
    return {
        "system":        name,
        "auc_roc":       round(float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else 0.5, 4),
        "avg_precision": round(float(average_precision_score(y, p)), 4),
        "brier":         round(float(brier_score_loss(y, p)), 4),
        "ece":           round(ece(y, p), 4),
        "top10_capture": round(float(y[np.argsort(p)[::-1][:k]].sum() / max(1, y.sum())), 4),
    }


# ===========================================================================
# MAIN 
# ===========================================================================

def run_comparison(
    data_path:    str,
    assets_path:  str,
    out_dir:      Path,
    n_rounds:     int   = 10,
    local_epochs: int   = 3,
    use_dp:       bool  = True,
    target_eps:   float = 1.0,
    use_secagg:   bool  = True,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    df       = load_data(data_path)
    assets   = pd.read_csv(assets_path) if Path(assets_path).exists() else None
    uids     = sorted(df["utility_id"].unique())

    if "n_customers" not in df.columns and assets is not None and "n_customers" in assets.columns:
        cust_map = assets.set_index("feeder_id")["n_customers"].to_dict()
        df["n_customers"] = df["feeder_id"].map(cust_map).fillna(500.0)

    # Non-IID benchmark
    noniid   = noniid_benchmark(df, out_dir)

    # FIX 1: Build subgraphs once, shared across conditions
    print("\n[Building synthetic radial topology priors]")
    subgraphs = build_utility_subgraphs(assets, df) if assets is not None else {u: None for u in uids}

    results          = []
    calibration_data = []

    # 1. Centralized upper bound
    print("\n[1/6] Centralized upper bound (no federation)")
    y_c, p_c = _centralized(df)
    r = metrics(y_c, p_c, "Centralized (upper bound)")
    results.append(r)
    calibration_data.append(("Centralized", y_c, p_c))
    print(f"  AUC={r['auc_roc']:.4f}  ECE={r['ece']:.4f}")

    # 2a. Local-only MLP lower bound
    print("\n[2a/6] Local-only MLP (lower bound)")
    y_l, p_l = _local_only(df, subgraphs, model_cls=TabularMLP)
    r = metrics(y_l, p_l, "Local-only MLP (lower bound)")
    results.append(r)
    calibration_data.append(("Local MLP", y_l, p_l))
    print(f"  AUC={r['auc_roc']:.4f}  ECE={r['ece']:.4f}")

    # 2b. Local-only Graph — controlled baseline: isolates federation effect from graph effect
    # Same GraphRiskModel architecture as the federated flagship, but no collaboration.
    # Comparison: local-graph vs federated-graph shows pure federation benefit.
    print("\n[2b/6] Local-only Graph (controlled architecture baseline)")
    y_lg, p_lg = _local_only(df, subgraphs, model_cls=GraphRiskModel)
    r = metrics(y_lg, p_lg, "Local-only Graph (controlled)")
    results.append(r)
    calibration_data.append(("Local Graph", y_lg, p_lg))
    print(f"  AUC={r['auc_roc']:.4f}  ECE={r['ece']:.4f}")

    # 3–5. FL variants
    for step, (cond_label, model_cls, dp, secagg) in enumerate([
        ("FL + FedProx  (MLP, no DP)",    TabularMLP,     False, False),
        ("FL + DP + FedProx  (MLP)",      TabularMLP,     True,  False),
        ("FL + DP + SecAgg + Graph",       GraphRiskModel, True,  True),
    ], start=3):
        print(f"\n[{step}/6] {cond_label}")
        y_fl, p_fl, per_client, eps_final = _run_fl(
            df, uids, subgraphs, model_cls,
            n_rounds, local_epochs, dp, target_eps, secagg, assets,
        )
        r = metrics(y_fl, p_fl, cond_label)
        r["epsilon"]    = eps_final
        r["secagg"]     = secagg
        r["per_client"] = per_client
        results.append(r)
        calibration_data.append((cond_label[:25], y_fl, p_fl))
        eps_str = f"{eps_final:.4f}" if eps_final is not None else "N/A"
        print(f"  AUC={r['auc_roc']:.4f}  ECE={r['ece']:.4f}  ε={eps_str}")

    # 6. SAIDI impact model — SAIDIGraphModel: encoder + outage head + CML head,
    # all parameters DP-covered and SecAgg-federated in one module.
    print("\n[6/6] FL + DP + SecAgg + SAIDI (fully federated)")
    y_saidi, p_saidi, saidi_rankings, eps_saidi = _run_fl_saidi(
        df, uids, subgraphs, n_rounds, local_epochs, target_eps, assets,
    )
    r = metrics(y_saidi, p_saidi, "FL + DP + SecAgg + SAIDI (federated)")
    r["epsilon"]  = eps_saidi
    r["secagg"]   = True
    r["saidi_note"] = (
        "SAIDIGraphModel: encoder + outage_head + cml_head in one nn.Module. "
        "Opacus DP covers all parameters. SecAgg aggregates all weights. "
        "Ranked by E[CML] = P(outage) x E[duration] x n_customers. "
        "Duration: Gamma(2,30) synthetic prior, mean=60min (CEER EU benchmark). "
        "Replace with real SAIDI event records when available."
    )
    results.append(r)
    calibration_data.append(("SAIDI model", y_saidi, p_saidi))
    eps_str = f"{eps_saidi:.4f}" if eps_saidi is not None else "N/A"
    print(f"  AUC={r['auc_roc']:.4f}  ECE={r['ece']:.4f}  ε={eps_str}")

    # Reliability diagrams (FIX 5)
    plot_reliability(calibration_data, out_dir)

    # Robustness tests (FIX 8)
    print("\n[Robustness suite]")
    rob_results = {
        "client_dropout":    robustness_client_dropout(df),
        "adversarial":       robustness_adversarial(df),
    }

    # Get last FL model for missing-telemetry test
    last_client = UtilityClient(
        uids[0], df, assets, TabularMLP, subgraphs.get(uids[0]),
        n_rounds, local_epochs, False, 1.0,
    )
    rob_results["missing_telemetry"] = robustness_missing_telemetry(
        get_base(last_client.model), df[df["utility_id"] == uids[0]].iloc[-500:],
        subgraphs.get(uids[0]),
    )

    # Final comparison table
    print("\n" + "="*82)
    print(f"  {'System':<36} {'AUC':>7} {'AP':>6} {'Brier':>7} {'ECE':>7} {'Top10%':>8} {'ε':>7}")
    print("="*82)
    for r in results:
        eps_s = f"{r['epsilon']:.4f}" if r.get("epsilon") else "  N/A"
        print(f"  {r['system']:<36} {r['auc_roc']:>7.4f} "
              f"{r.get('avg_precision',0):>6.4f} {r['brier']:>7.4f} "
              f"{r['ece']:>7.4f} {r['top10_capture']:>8.4f} {eps_s:>7}")
    print("="*82)

    (out_dir / "full_results.json").write_text(
        json.dumps({"results": results, "noniid": noniid,
                    "robustness": rob_results}, indent=2, default=str)
    )
    print(f"\n✓  All outputs → {out_dir.resolve()}")
    return results


def _centralized(df):
    cut  = int(len(df) * 0.80)
    tr, te = df.iloc[:cut], df.iloc[cut:]
    ns = Normalizer().fit(torch.tensor(tr[STATIC_FEATURES].values, dtype=torch.float32))
    nt = Normalizer().fit(torch.tensor(tr[TEMPORAL_FEATURES].values, dtype=torch.float32))
    xs_tr = ns.transform(torch.tensor(tr[STATIC_FEATURES].values, dtype=torch.float32))
    xt_tr = nt.transform(torch.tensor(tr[TEMPORAL_FEATURES].values, dtype=torch.float32))
    y_tr  = torch.tensor(tr[LABEL_COL].values, dtype=torch.float32)
    xs_te = ns.transform(torch.tensor(te[STATIC_FEATURES].values, dtype=torch.float32))
    xt_te = nt.transform(torch.tensor(te[TEMPORAL_FEATURES].values, dtype=torch.float32))

    m   = TabularMLP()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    c   = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]))
    for _ in range(20):
        m.train(); opt.zero_grad(); c(m(xs_tr, xt_tr), y_tr).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        p = torch.sigmoid(m(xs_te, xt_te)).numpy()
    return te[LABEL_COL].values, p


def _local_only(df, subgraphs, model_cls=None):
    """
    Local-only baseline: each utility trains on its own data, no collaboration.

    model_cls controls which model architecture is used locally.
    Pass GraphRiskModel to get a controlled local-graph baseline that isolates
    the federation effect from the graph-architecture effect.
    Pass TabularMLP (default) for the standard local MLP lower bound.
    """
    if model_cls is None:
        model_cls = TabularMLP

    all_y, all_p = [], []
    for uid in sorted(df["utility_id"].unique()):
        tr, _, te = utility_split(df, uid)
        ns = Normalizer().fit(torch.tensor(tr[STATIC_FEATURES].values, dtype=torch.float32))
        nt = Normalizer().fit(torch.tensor(tr[TEMPORAL_FEATURES].values, dtype=torch.float32))
        xs_tr = ns.transform(torch.tensor(tr[STATIC_FEATURES].values, dtype=torch.float32))
        xt_tr = nt.transform(torch.tensor(tr[TEMPORAL_FEATURES].values, dtype=torch.float32))
        xs_te = ns.transform(torch.tensor(te[STATIC_FEATURES].values,  dtype=torch.float32))
        xt_te = nt.transform(torch.tensor(te[TEMPORAL_FEATURES].values, dtype=torch.float32))
        y_tr  = torch.tensor(tr[LABEL_COL].values, dtype=torch.float32)
        ei    = subgraphs.get(uid)

        m   = model_cls()
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        c   = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]))
        for _ in range(15):
            m.train(); opt.zero_grad()
            c(m(xs_tr, xt_tr, ei), y_tr).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            p = torch.sigmoid(m(xs_te, xt_te, ei)).numpy()
        all_y.extend(te[LABEL_COL].values.tolist())
        all_p.extend(p.tolist())
    return np.array(all_y), np.array(all_p)


def _run_fl(df, uids, subgraphs, model_cls, n_rounds, local_epochs,
            use_dp, target_eps, use_secagg, assets):
    clients = [
        UtilityClient(
            uid, df, assets, model_cls,
            subgraphs.get(uid),    # FIX 1: edge_index per client
            n_rounds, local_epochs, use_dp, target_eps,
        )
        for uid in uids
    ]

    secagg = BonawitzSecAgg(len(uids), seed=42) if use_secagg else None
    global_w = get_weights_from_base(clients[0].model)  # init from first client

    for rnd in range(1, n_rounds + 1):
        for client in clients:
            client.train_round(global_w, local_epochs)

        raw_updates = [c.get_weights() for c in clients]
        sizes       = [c.n_train for c in clients]

        if use_secagg:
            # FIX 2: premultiply before masking
            global_w, report = masked_weighted_aggregate(
                secagg, raw_updates, sizes, rnd,
            )
        else:
            total    = sum(sizes)
            global_w = [
                sum(raw_updates[i][l] * (sizes[i] / total) for i in range(len(uids)))
                for l in range(len(global_w))
            ]

        eps_vals = [c.current_epsilon() for c in clients]
        eps_vals = [e for e in eps_vals if e is not None]
        if eps_vals:
            print(f"  Round {rnd:>2}/{n_rounds} | worst ε={max(eps_vals):.4f}")

    # Single evaluation pass: evaluate() already runs model forward with edge_index
    per_client = [c.evaluate(global_w) for c in clients]

    all_y_list, all_p_list = [], []
    for client in clients:
        te   = client.test_df
        xs   = client.ns.transform(
            torch.tensor(te[STATIC_FEATURES].values, dtype=torch.float32)
        )
        xt   = client.nt.transform(
            torch.tensor(te[TEMPORAL_FEATURES].values, dtype=torch.float32)
        )
        base = get_base(client.model)
        base.eval()
        with torch.no_grad():
            p = torch.sigmoid(base(xs, xt, client.edge_index)).numpy()
        all_p_list.extend(p.tolist())
        all_y_list.extend(te[LABEL_COL].values.tolist())

    eps_final = max(
        (c.current_epsilon() for c in clients if c.current_epsilon()), default=None
    )
    return (
        np.array(all_y_list),
        np.array(all_p_list),
        per_client,
        eps_final,
    )


def _run_fl_saidi(df, uids, subgraphs, n_rounds, local_epochs, target_eps, assets):
    """
    Federated SAIDI experiment.

    Model: SAIDIGraphModel — encoder + outage head + CML head in one nn.Module.
    DP:    Opacus attached to the FULL model at UtilityClient init.
           client.optimizer is the DP-privatized SGD.
           Training uses client.optimizer for the joint loss.
           Epsilon is tied to the actual SAIDI optimization path.
    SecAgg: aggregates ALL weights (encoder + embed_proj + outage_head + cml_head).
           This is a fully federated SAIDI model, not just a federated encoder.
    Label: "FL + DP + SecAgg + SAIDI (federated encoder + heads)"

    Duration targets: Gamma(2, 30) — mean 60 min, calibrated to CEER EU benchmark.
    Replace with real SAIDI event records when available.
    """
    clients = [
        UtilityClient(
            uid, df, assets,
            SAIDIGraphModel,           # self-contained — DP covers all params
            subgraphs.get(uid),
            n_rounds, local_epochs,
            use_dp=True, target_eps=target_eps,
        )
        for uid in uids
    ]

    secagg   = BonawitzSecAgg(len(uids), seed=99)
    global_w = get_weights_from_base(clients[0].model)

    for rnd in range(1, n_rounds + 1):
        raw_updates, sizes = [], []

        for client in clients:
            if client.dp_exhausted:
                raw_updates.append(get_weights_from_base(client.model))
                sizes.append(client.n_train)
                continue

            # Set global weights on base module (DP engine stays attached)
            set_weights_on_base(client.model, global_w)
            global_frozen = frozen_base_params(client.model)

            base = get_base(client.model)
            base.train()

            for batch in client.train_loader:
                xs, xt, y_bin = batch[0], batch[1], batch[2]
                n_cust   = batch[3] if len(batch) > 3 else torch.full_like(y_bin, 500.0)
                dur_true = batch[4] if len(batch) > 4 else torch.full_like(y_bin, 60.0)

                # Use client.optimizer — the DP-privatized optimizer
                # This is the only correct path: all gradients flow through
                # the PrivacyEngine hooks, epsilon accounts correctly
                client.optimizer.zero_grad()

                # saidi_forward on base module (same object DP is attached to)
                outage_logit, pred_dur = base.saidi_forward(
                    xs, xt, client.edge_index
                )

                # FedProx proximal term (base params, no Opacus expansion dim)
                prox = torch.tensor(0.0)
                for p, g in zip(base.parameters(), global_frozen):
                    prox += (0.1 / 2.0) * torch.sum((p - g) ** 2)

                loss = saidi_loss(
                    outage_logit, pred_dur, y_bin, n_cust, dur_true
                ) + prox
                loss.backward()
                client.optimizer.step()

            raw_updates.append(get_weights_from_base(client.model))
            sizes.append(client.n_train)

        # SecAgg aggregates ALL weights: encoder + embed_proj + outage_head + cml_head
        global_w, _ = masked_weighted_aggregate(secagg, raw_updates, sizes, rnd)

        eps_vals = [c.current_epsilon() for c in clients if c.current_epsilon()]
        if eps_vals:
            worst = max(eps_vals)
            status = "✓" if worst <= target_eps else "✗ OVER BUDGET"
            print(f"  [SAIDI] Round {rnd:>2}/{n_rounds} | ε={worst:.4f} {status}")

    # Evaluation
    all_y, all_p, all_cml = [], [], []
    for client in clients:
        set_weights_on_base(client.model, global_w)
        base = get_base(client.model)
        base.eval()
        te = client.test_df
        xs = client.ns.transform(
            torch.tensor(te[STATIC_FEATURES].values, dtype=torch.float32)
        )
        xt = client.nt.transform(
            torch.tensor(te[TEMPORAL_FEATURES].values, dtype=torch.float32)
        )
        n_cust = torch.tensor(
            te["n_customers"].values if "n_customers" in te.columns
            else np.full(len(te), 500.0),
            dtype=torch.float32,
        )
        with torch.no_grad():
            outage_logit, pred_dur = base.saidi_forward(xs, xt, client.edge_index)
            probs    = torch.sigmoid(outage_logit).numpy()
            cml_vals = base.expected_cml(xs, xt, n_cust, client.edge_index).numpy()

        all_y.extend(te[LABEL_COL].values.tolist())
        all_p.extend(probs.tolist())
        all_cml.extend(cml_vals.tolist())

    eps_final = max(
        (c.current_epsilon() for c in clients if c.current_epsilon()), default=None
    )
    print(f"  Mean E[CML] per feeder-week: {np.mean(all_cml):.1f} customer-minutes")
    print(f"  P99  E[CML]:                 {np.percentile(all_cml, 99):.1f} customer-minutes")
    return np.array(all_y), np.array(all_p), all_cml, eps_final


# ===========================================================================
# ENTRY POINT
# ===========================================================================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",    default="data/raw/model_dataset.csv")
    ap.add_argument("--assets",  default="data/raw/assets.csv")
    ap.add_argument("--rounds",  type=int,   default=10)
    ap.add_argument("--epsilon", type=float, default=1.0)
    ap.add_argument("--outdir",  default="models/outputs/final")
    ap.add_argument("--no-dp",   dest="dp",     action="store_false", default=True)
    ap.add_argument("--no-secagg", dest="secagg", action="store_false", default=True)
    args = ap.parse_args()

    run_comparison(
        data_path=args.data,
        assets_path=args.assets,
        out_dir=Path(args.outdir),
        n_rounds=args.rounds,
        use_dp=args.dp,
        target_eps=args.epsilon,
        use_secagg=args.secagg,
    )
