#!/usr/bin/env python3
"""
main2.py — Triple-Objective Multi-Objective Clustering (MOC)
           for Rolling Element Bearing Fault Diagnosis
           CWRU Bearing Dataset — ENTIRE DATASET

Three NSGA-II objectives
------------------------
  f1  Compactness   — Total Within-Cluster Variance (TWCV) computed in the
                      FULL 7-D feature space; cluster assignment uses only
                      the ACTIVE (masked) features. This keeps TWCV comparable
                      across different mask configurations.

  f2  Connectedness — Handl-Knowles k-NN connectivity penalty. KNN graph is
                      pre-computed on the full feature space (topology is fixed);
                      only cluster assignment is mask-dependent.

  f3  Simplicity    — L0 norm of the feature mask: count of features whose
                      mask weight exceeds 0.5. Minimising this objective forces
                      the Pareto front to reveal which 2-3 mechanical indicators
                      (RMS, Kurtosis, …) are sufficient to explain a given
                      fault partition, making clustering results auditable.

Decision-Making Advantage (three-objective extension)
------------------------------------------------------
The 3-D Pareto surface exposes TWO independent trade-off axes:

  Axis A  (f1 ↔ f2)  Coarse fault states (routine maintenance scheduling)
                      ↔ Fine-grained severity tiers (root-cause analysis)

  Axis B  (f3)        Interpretable 2-feature explanation
                      ↔ Comprehensive 7-feature clustering

An engineer navigates the surface based on operational context:
  • Condition-monitoring dashboard  → low K, ≤3 features (fast, explainable)
  • Root-cause failure report       → high K, all features (maximum discrimination)
  • Regulatory compliance audit     → moderate K, named features auditable by law

The KNEE POINT (minimum normalised distance to the utopia point) is the
recommended default; practitioners override by filtering the Pareto set on a
feature budget (e.g., keep only solutions where f3 ≤ 3).
"""

import os
import glob
import csv
import time
import warnings
from pathlib import Path

import numpy as np
import scipy.io
from scipy.spatial.distance import cdist
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, adjusted_rand_score,
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3-D projection

from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.core.mutation import Mutation
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.optimize import minimize
from pymoo.termination import get_termination

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Global constants
# ─────────────────────────────────────────────────────────────────────────────
WINDOW_SIZE      = 1024     # samples per segment
K_MIN            = 2        # min clusters
K_MAX            = 10       # max clusters
D                = 7        # feature dimensions
KNN_K            = 10       # neighbours for connectivity objective
POP_SIZE         = 150      # initial population (100–200 per spec)
N_GEN            = 500      # NSGA-II generations

# Chromosome layout (78 variables total):
#   x[0]                        → K gene (float, rounded to int ∈ [K_MIN, K_MAX])
#   x[1 : K_MAX*D + 1]          → centroid pool, K_MAX × D values (padded)
#   x[K_MAX*D + 1 : N_VARS]     → feature mask weights, D values ∈ [0, 1]
N_CENTROID_GENES = K_MAX * D           # 70
N_MASK_GENES     = D                   # 7
N_VARS           = 1 + N_CENTROID_GENES + N_MASK_GENES   # 78
MASK_START       = 1 + N_CENTROID_GENES                  # index 71

FEATURE_NAMES = [
    "RMS", "Kurtosis", "Skewness", "CrestFactor",
    "PeakToPeak", "StdDev", "SpectralCentroid",
]

# Full-dataset paths
DATASET_ROOT = "/home/cherishhh/Work/Acads/LBP/MOML_LBP/CWRU-dataset"
SAVE_DIR     = "/home/cherishhh/Work/Acads/LBP/MOML_LBP"

# Ground-truth label scheme (fault type, independent of severity / drive speed)
FAULT_TYPE_LABEL = {"Normal": 0, "B": 1, "IR": 2, "OR": 3}
FAULT_TYPE_NAMES = ["Normal", "Ball", "Inner Race", "Outer Race"]
GT_COLOURS       = ["#95a5a6", "#e74c3c", "#3498db", "#2ecc71"]


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA ENGINEERING & SIGNAL PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def _infer_metadata(mat_path: str):
    """
    Parse directory structure to extract fault type, severity, and fs.

    Directory naming conventions in CWRU dataset:
      <root>/Normal/                                    → Normal, no severity
      <root>/12k_Drive_End_Bearing_Fault_Data/B/007/   → Ball, 0.007", 12 kHz
      <root>/48k_Drive_End_Bearing_Fault_Data/IR/021/  → Inner Race, 0.021", 48 kHz
      <root>/12k_Fan_End_Bearing_Fault_Data/OR/007/@6/ → Outer Race (fan end), 12 kHz
    """
    rel   = os.path.relpath(mat_path, DATASET_ROOT)
    parts = set(Path(rel).parts)

    # Sampling frequency from directory name
    fs = 48_000.0 if any("48k" in p for p in parts) else 12_000.0

    # Fault type
    if "Normal" in parts:
        fault_type = "Normal"
    elif "B" in parts:
        fault_type = "B"
    elif "IR" in parts:
        fault_type = "IR"
    elif "OR" in parts:
        fault_type = "OR"
    else:
        fault_type = "Unknown"

    # Fault severity (sub-directory named '007', '014', '021', '028')
    severity = None
    for p in Path(rel).parts:
        if p in {"007", "014", "021", "028"}:
            severity = p
            break

    label = FAULT_TYPE_LABEL.get(fault_type, -1)
    return label, fault_type, severity, fs


def load_full_dataset(root_dir: str = DATASET_ROOT) -> list:
    """
    Recursively load every .mat file in the CWRU dataset.

    Strategy
    --------
    • Extract the Drive-End (DE_time) channel from every file.
    • Infer fault type, severity, and sampling frequency from the directory path.
    • Skip files without a recognisable DE_time key or unknown fault type.

    Returns
    -------
    list of (signal_1d, fault_label, fault_type_str, severity_str, fs, fpath)
    """
    mat_files = sorted(
        glob.glob(os.path.join(root_dir, "**", "*.mat"), recursive=True)
    )
    records = []
    skipped = 0

    for fpath in mat_files:
        label, fault_type, severity, fs = _infer_metadata(fpath)
        if label == -1:
            skipped += 1
            continue

        mat = scipy.io.loadmat(fpath, squeeze_me=True)
        # Locate the Drive-End time-series key
        de_key = next(
            (k for k in mat.keys() if "DE_time" in k and not k.startswith("_")),
            None,
        )
        if de_key is None:
            skipped += 1
            continue

        signal = np.asarray(mat[de_key], dtype=np.float64).ravel()
        if signal.size < WINDOW_SIZE:
            skipped += 1
            continue

        records.append((signal, label, fault_type, severity, fs, fpath))
        print(
            f"  {fault_type:8s} | sev={severity or 'none':4s} | "
            f"fs={int(fs/1000)}kHz | {len(signal):>8d} samps | "
            f"{os.path.basename(fpath)}"
        )

    print(f"\n  Loaded {len(records)} files  ({skipped} skipped)")
    return records


def segment_signals(records: list, window: int = WINDOW_SIZE):
    """
    Chop each signal into non-overlapping windows of `window` samples.

    Returns
    -------
    segments  : ndarray (N, window)  — raw vibration windows
    labels_gt : ndarray (N,)         — ground-truth fault type label
    fs_arr    : ndarray (N,)         — sampling frequency per segment
    """
    segs, labs, fss = [], [], []
    for signal, label, _ft, _sev, fs, _fp in records:
        n = len(signal) // window
        for i in range(n):
            segs.append(signal[i * window: (i + 1) * window])
            labs.append(label)
            fss.append(fs)
    return (
        np.array(segs, dtype=np.float64),
        np.array(labs,  dtype=np.int32),
        np.array(fss,   dtype=np.float64),
    )


def extract_features(segments: np.ndarray, fs_arr: np.ndarray) -> np.ndarray:
    """
    Compute 7 time/frequency-domain features per segment.

    Feature index map
    -----------------
    0  RMS               Energy content
    1  Kurtosis          Impulsiveness — spikes from fault impacts (excess)
    2  Skewness          Asymmetry — shifts as fault grows
    3  Crest Factor      Peak/RMS — early-stage shock indicator
    4  Peak-to-Peak      Total amplitude swing
    5  Std Deviation     Spread of vibration amplitude
    6  Spectral Centroid Frequency centre-of-mass (shifts near fault frequency)

    Each segment uses its own recorded fs for Spectral Centroid so that
    12 kHz and 48 kHz recordings are frequency-calibrated correctly before
    StandardScaler normalises the feature matrix.
    """
    N, W = segments.shape
    feats = np.zeros((N, 7), dtype=np.float64)

    for i, (seg, fs) in enumerate(zip(segments, fs_arr)):
        rms   = np.sqrt(np.mean(seg ** 2))
        kurt  = kurtosis(seg, fisher=True)
        skw   = skew(seg)
        peak  = np.max(np.abs(seg))
        crest = peak / (rms + 1e-12)
        p2p   = np.ptp(seg)
        std   = np.std(seg)

        freqs    = np.fft.rfftfreq(W, d=1.0 / fs)
        spectrum = np.abs(np.fft.rfft(seg))
        sc       = np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-12)

        feats[i] = [rms, kurt, skw, crest, p2p, std, sc]

    return feats


def preprocess(X: np.ndarray):
    """Zero-mean unit-variance standardisation. Returns (X_scaled, scaler)."""
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler


# ─────────────────────────────────────────────────────────────────────────────
# 2.  TRIPLE-OBJECTIVE MOC PROBLEM
# ─────────────────────────────────────────────────────────────────────────────

class MOCProblem3(Problem):
    """
    NSGA-II problem for variable-K clustering with feature selection.

    Chromosome layout (N_VARS = 78)
    ────────────────────────────────
    x[0]                : K  (float → int ∈ [K_MIN, K_MAX])
    x[1 : K*D+1]        : active cluster centres (K × D values)
    x[K*D+1 : K_MAX*D+1]: padding centroids (ignored)
    x[K_MAX*D+1 : 78]   : feature mask weights  (D floats ∈ [0, 1])
                          active when weight > 0.5 (L0 threshold)

    Objectives
    ──────────
    f1  TWCV          Full-space compactness (assign via masked, measure in 7D)
    f2  Connectivity  k-NN topology penalty  (KNN on full space)
    f3  Simplicity    count(mask_weight > 0.5)  — L0 feature count (min 2)
    """

    def __init__(self, X: np.ndarray, knn_k: int = KNN_K):
        self.X    = X
        self.N, _ = X.shape   # D is global constant

        # Pre-compute k-NN graph on the FULL feature space once.
        # Topology is fixed; only cluster assignment varies with the mask.
        print("  Pre-computing k-NN graph (full space) ...")
        nbrs = NearestNeighbors(
            n_neighbors=knn_k + 1, algorithm="ball_tree", n_jobs=-1
        )
        nbrs.fit(X)
        _, indices = nbrs.kneighbors(X)
        self.knn_indices = indices[:, 1:]                      # (N, knn_k)
        self.knn_weights = 1.0 / np.arange(1, knn_k + 1)     # [1, 1/2, …, 1/k]

        # Variable bounds
        xl = np.empty(N_VARS)
        xu = np.empty(N_VARS)
        xl[0], xu[0]              = K_MIN, K_MAX               # K gene
        xl[1:MASK_START]          = np.tile(X.min(axis=0), K_MAX)
        xu[1:MASK_START]          = np.tile(X.max(axis=0), K_MAX)
        xl[MASK_START:]           = 0.0                        # mask weights
        xu[MASK_START:]           = 1.0

        super().__init__(n_var=N_VARS, n_obj=3, n_ieq_constr=0, xl=xl, xu=xu)

    # ── Chromosome decoders ───────────────────────────────────────────────────

    def decode(self, x: np.ndarray):
        """
        Decode chromosome into (K, centres, mask_bool).

        mask_bool enforces a minimum of 2 active features:
        if fewer than 2 weights exceed 0.5, the top-2 by weight are activated.
        """
        K       = int(np.clip(round(x[0]), K_MIN, K_MAX))
        centres = x[1: K * D + 1].reshape(K, D)
        weights = x[MASK_START:]                   # shape (D,)

        mask_bool = weights > 0.5
        if mask_bool.sum() < 2:
            # Guarantee at least 2 features are active
            top2 = np.argsort(weights)[-2:]
            mask_bool = np.zeros(D, dtype=bool)
            mask_bool[top2] = True

        return K, centres, mask_bool

    def assign(self, X_sub: np.ndarray, centres_sub: np.ndarray) -> np.ndarray:
        """Assign each point to its nearest centre in the masked subspace."""
        return np.argmin(cdist(X_sub, centres_sub), axis=1)

    # ── Objective functions ───────────────────────────────────────────────────

    def _twcv_full(self, labels: np.ndarray, centres: np.ndarray) -> float:
        """
        Total Within-Cluster Variance in the FULL 7-D space.

        Using the full space keeps TWCV comparable across solutions with
        different active-feature sets, so f1 and f3 remain in genuine tension:
        fewer features → worse cluster assignment → higher full-space TWCV.
        """
        return float(np.sum((self.X - centres[labels]) ** 2))

    def _connectivity(self, labels: np.ndarray) -> float:
        """
        Handl-Knowles (2007) connectivity penalty.

        For point i and ranked k-NN neighbour j:
          Conn += (1/rank) × I(cluster(i) ≠ cluster(j))

        Penalises solutions that split topologically proximate points
        across different clusters, ensuring structural coherence.
        """
        neighbour_labels = labels[self.knn_indices]        # (N, knn_k)
        different        = (labels[:, None] != neighbour_labels)
        return float(np.sum(different * self.knn_weights))

    # ── pymoo evaluation hook ─────────────────────────────────────────────────

    def _evaluate(self, X_pop, out, *args, **kwargs):
        n = len(X_pop)
        F = np.empty((n, 3))

        for i, x in enumerate(X_pop):
            K, centres, mask_bool = self.decode(x)

            # Project data and centres into the active feature subspace
            X_sub = self.X[:, mask_bool]
            C_sub = centres[:, mask_bool]

            labels   = self.assign(X_sub, C_sub)
            F[i, 0]  = self._twcv_full(labels, centres)   # full-space TWCV
            F[i, 1]  = self._connectivity(labels)
            F[i, 2]  = float(mask_bool.sum())             # L0 feature count

        out["F"] = F


# ─────────────────────────────────────────────────────────────────────────────
# 3.  SMART INITIALIZATION  (1/3 split)
# ─────────────────────────────────────────────────────────────────────────────

class SmartSampling3(Sampling):
    """
    Population initialisation with three equal thirds:

    ┌────────────────────────────────────────────────────────────────────────┐
    │ 1/3  K-Means seeded   Centroids from K-Means; fast convergence        │
    │ 1/3  Random data pts  Actual dataset samples as centres; realism      │
    │ 1/3  Pure random      Uniform bounds sampling; genetic diversity      │
    └────────────────────────────────────────────────────────────────────────┘

    For all three strata, feature mask weights are initialised uniformly in
    [0, 1] to avoid biasing the search toward any particular feature subset.
    """

    def __init__(self, X: np.ndarray):
        super().__init__()
        self.X = X

    def _do(self, problem, n_samples, **kwargs):
        xl, xu = problem.xl, problem.xu
        pop    = np.random.uniform(xl, xu, size=(n_samples, N_VARS))

        n_km    = n_samples // 3
        n_rdata = n_samples // 3
        # Pure-random stratum fills the remainder (already set above)

        # ── Stratum 1: K-Means seeded ──────────────────────────────────────
        for i in range(n_km):
            K = np.random.randint(K_MIN, K_MAX + 1)
            pop[i, 0] = float(K)
            try:
                km = KMeans(n_clusters=K, n_init=3, max_iter=100,
                            random_state=None)
                km.fit(self.X)
                pop[i, 1: K * D + 1] = km.cluster_centers_.flatten()
            except Exception:
                pass  # fallback: random centroids already set

        # ── Stratum 2: Random data points as centres ───────────────────────
        for i in range(n_km, n_km + n_rdata):
            K = np.random.randint(K_MIN, K_MAX + 1)
            pop[i, 0] = float(K)
            chosen = self.X[np.random.choice(len(self.X), K, replace=False)]
            pop[i, 1: K * D + 1] = chosen.flatten()

        # Stratum 3: pure random — already filled by np.random.uniform above
        return pop


# ─────────────────────────────────────────────────────────────────────────────
# 4.  STRUCTURAL MUTATION  (centroids + feature mask)
# ─────────────────────────────────────────────────────────────────────────────

class StructuralMutation3(Mutation):
    """
    Three-phase mutation operator:

    Phase A — Polynomial Mutation (PM) on active centroid coordinates
        Probability 1/(K·D) per gene; eta=20 for moderate perturbations.

    Phase B — Structural Mutation (prob p_struct)
        ADD  a cluster (K < K_MAX): insert midpoint of two random centres
             plus Gaussian noise, growing fault granularity.
        DROP a cluster (K > K_MIN): remove the least-populated centre,
             simplifying the partition.

    Phase C — Feature-Mask Mutation (prob p_feat per feature)
        Randomly perturbs each mask weight with PM-style update.
        Additionally, with prob p_flip, hard-flips one feature's binary
        state (active ↔ inactive) to allow the GA to explore feature
        subsets more aggressively.
    """

    def __init__(
        self,
        X: np.ndarray,
        prob_struct: float = 0.10,
        prob_flip:   float = 0.15,
        eta:         float = 20.0,
    ):
        super().__init__()
        self.X           = X
        self.prob_struct = prob_struct
        self.prob_flip   = prob_flip
        self.eta         = eta

    def _do(self, problem, X_pop, **kwargs):
        xl, xu = problem.xl, problem.xu
        X_mut  = X_pop.copy()

        for idx in range(len(X_mut)):
            x = X_mut[idx]
            K = int(np.clip(round(x[0]), K_MIN, K_MAX))

            # ── Phase A: PM on active centroid coordinates ─────────────────
            pm_prob = 1.0 / max(K * D, 1)
            for j in range(1, K * D + 1):
                if np.random.random() >= pm_prob:
                    continue
                y        = x[j]
                lo, hi   = xl[j], xu[j]
                span     = hi - lo + 1e-12
                d1       = (y - lo)  / span
                d2       = (hi - y)  / span
                r        = np.random.random()
                mu       = 1.0 / (self.eta + 1.0)
                if r < 0.5:
                    v  = 2.0 * r + (1.0 - 2.0 * r) * (1.0 - d1) ** (self.eta + 1)
                    dq = v ** mu - 1.0
                else:
                    v  = 2.0 * (1.0 - r) + 2.0 * (r - 0.5) * (1.0 - d2) ** (self.eta + 1)
                    dq = 1.0 - v ** mu
                x[j] = float(np.clip(y + dq * span, lo, hi))

            # ── Phase B: Structural mutation (add / remove a cluster) ──────
            if np.random.random() < self.prob_struct:
                centres = x[1: K * D + 1].reshape(K, D)

                if np.random.random() < 0.5 and K < K_MAX:
                    # ADD cluster: midpoint of two random centres + noise
                    i1, i2 = np.random.choice(K, 2, replace=False)
                    new_c  = (centres[i1] + centres[i2]) * 0.5
                    noise  = np.random.normal(0, 0.05, D)
                    new_c  = np.clip(new_c + noise, xl[1: D + 1], xu[1: D + 1])
                    K_new  = K + 1
                    x[0]   = float(K_new)
                    x[K * D + 1: K_new * D + 1] = new_c

                elif K > K_MIN:
                    # DROP cluster: remove least-populated centre
                    labels = np.argmin(cdist(self.X, centres), axis=1)
                    counts = np.bincount(labels, minlength=K)
                    drop   = int(np.argmin(counts))
                    new_c  = np.delete(centres, drop, axis=0)   # (K-1, D)
                    K_new  = K - 1
                    x[0]   = float(K_new)
                    x[1: K_new * D + 1] = new_c.flatten()
                    # Randomise the now-unused centroid tail
                    tail = K_new * D + 1
                    x[tail: MASK_START] = np.random.uniform(
                        xl[tail: MASK_START], xu[tail: MASK_START]
                    )

            # ── Phase C: Feature mask mutation ────────────────────────────
            # Soft PM perturbation on each mask weight
            for j in range(MASK_START, N_VARS):
                if np.random.random() < (1.0 / D):
                    y      = x[j]
                    span   = 1.0   # mask bounds are [0, 1]
                    d1     = y
                    d2     = 1.0 - y
                    r      = np.random.random()
                    mu     = 1.0 / (self.eta + 1.0)
                    if r < 0.5:
                        v  = 2.0 * r + (1.0 - 2.0 * r) * (1.0 - d1) ** (self.eta + 1)
                        dq = v ** mu - 1.0
                    else:
                        v  = 2.0 * (1.0 - r) + 2.0 * (r - 0.5) * (1.0 - d2) ** (self.eta + 1)
                        dq = 1.0 - v ** mu
                    x[j] = float(np.clip(y + dq * span, 0.0, 1.0))

            # Hard flip: force one random feature across the 0.5 boundary
            if np.random.random() < self.prob_flip:
                j    = np.random.randint(MASK_START, N_VARS)
                # Flip to the other side of 0.5
                x[j] = 1.0 - x[j]

        return X_mut


# ─────────────────────────────────────────────────────────────────────────────
# 5.  PARETO ANALYSIS — KNEE POINT (3-D)
# ─────────────────────────────────────────────────────────────────────────────

def find_knee_point_3d(F: np.ndarray) -> int:
    """
    Locate the knee point on a 3-objective Pareto front.

    Method: minimum normalised Euclidean distance to the UTOPIA point.

    The utopia point is the vector of individual-objective minima —
    the (unachievable) ideal solution. The Pareto solution closest to it
    offers the best simultaneous trade-off across all three objectives.

    In the context of fault diagnosis:
      Utopia = (lowest TWCV, lowest connectivity penalty, fewest features)
    """
    eps    = 1e-12
    F_min  = F.min(axis=0)
    F_max  = F.max(axis=0)
    F_norm = (F - F_min) / (F_max - F_min + eps)
    # Utopia point in normalised space is (0, 0, 0)
    dist   = np.linalg.norm(F_norm, axis=1)
    return int(np.argmin(dist))


# ─────────────────────────────────────────────────────────────────────────────
# 6.  BASELINE: standard K-Means
# ─────────────────────────────────────────────────────────────────────────────

def kmeans_baseline(X: np.ndarray, labels_gt: np.ndarray, K: int = 4) -> dict:
    """Fit K-Means with K=n_fault_types and collect all evaluation metrics."""
    km      = KMeans(n_clusters=K, n_init=10, random_state=42)
    labels  = km.fit_predict(X)
    centres = km.cluster_centers_
    twcv    = float(sum(
        np.sum((X[labels == k] - centres[k]) ** 2)
        for k in range(K) if (labels == k).any()
    ))
    return {
        "labels": labels,
        "sil":    silhouette_score(X, labels),
        "db":     davies_bouldin_score(X, labels),
        "ari":    adjusted_rand_score(labels_gt, labels),
        "twcv":   twcv,
        "K":      K,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7.  FINAL POPULATION LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def log_final_population(
    res,
    problem: MOCProblem3,
    save_dir: str = SAVE_DIR,
) -> str:
    """
    Log the ENTIRE final population (all individuals, not just Pareto front)
    to a CSV file and print a summary to stdout.

    Columns
    -------
    index, rank, crowding_dist,
    K, n_active_features, active_feature_names,
    f1_TWCV, f2_Connectivity, f3_N_Features,
    mask_weights (one column per feature)
    """
    pop  = res.pop
    X_dv = pop.get("X")           # decision variables  (pop_size × N_VARS)
    F_dv = pop.get("F")           # objective values    (pop_size × 3)

    # pymoo stores rank and crowding distance in the population
    try:
        ranks  = pop.get("rank").astype(int)
    except Exception:
        ranks  = np.zeros(len(X_dv), dtype=int)
    try:
        cd     = pop.get("crowding")
    except Exception:
        cd     = np.full(len(X_dv), np.nan)

    out_path = os.path.join(save_dir, "final_population.csv")
    header   = (
        ["index", "rank", "crowding_dist",
         "K", "n_active_features", "active_feature_names",
         "f1_TWCV", "f2_Connectivity", "f3_N_Features"]
        + [f"mask_{fn}" for fn in FEATURE_NAMES]
    )

    print("\n" + "=" * 80)
    print("  FINAL POPULATION LOG")
    print("=" * 80)
    hdr = f"{'idx':>5} {'rank':>4} {'K':>3} {'n_feat':>6}  "
    hdr += f"{'active features':<40}  {'TWCV':>12}  {'Conn':>10}  {'Nfeat':>6}"
    print(hdr)
    print("-" * 90)

    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)

        for i, (x, f) in enumerate(zip(X_dv, F_dv)):
            _, _, mask_bool = problem.decode(x)
            n_active        = int(mask_bool.sum())
            active_names    = "|".join(
                fn for fn, m in zip(FEATURE_NAMES, mask_bool) if m
            )
            K_val           = int(np.clip(round(x[0]), K_MIN, K_MAX))
            mask_weights    = x[MASK_START:]

            row = (
                [i, ranks[i], f"{cd[i]:.4f}" if not np.isnan(cd[i]) else "inf",
                 K_val, n_active, active_names,
                 f"{f[0]:.4f}", f"{f[1]:.4f}", f"{f[2]:.0f}"]
                + [f"{w:.4f}" for w in mask_weights]
            )
            writer.writerow(row)

            # Print every individual to stdout
            print(
                f"{i:>5} {ranks[i]:>4} {K_val:>3} {n_active:>6}  "
                f"{active_names:<40}  {f[0]:>12.2f}  {f[1]:>10.2f}  {f[2]:>6.0f}"
            )

    print("=" * 80)
    print(f"[LOG] Final population ({len(X_dv)} individuals) saved → {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# 8.  VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_results_3d(
    F_pareto, knee_idx,
    X_scaled, labels_moc, labels_gt,
    metrics_moc, metrics_km,
    k_vals_pareto, mask_bool_knee,
    save_dir: str = SAVE_DIR,
) -> str:
    """
    2 × 3 diagnostic figure for the triple-objective MOC result.

    Layout
    ──────
    [0,0] 3-D Pareto front (TWCV, Connectivity, N_features), coloured by K
    [0,1] 2-D projection: TWCV vs Connectivity
    [0,2] 2-D projection: TWCV vs N_features  |  Connectivity vs N_features
    [1,0] PCA scatter: MOC knee clusters
    [1,1] PCA scatter: Ground Truth
    [1,2] Metrics table  +  Active-feature bar chart
    """
    fig = plt.figure(figsize=(22, 14))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.38)

    K_knee = metrics_moc["K"]
    n_feat_knee = int(mask_bool_knee.sum())

    # ── [0,0]  3-D Pareto front ──────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0], projection="3d")
    sc0 = ax0.scatter(
        F_pareto[:, 0], F_pareto[:, 1], F_pareto[:, 2],
        c=k_vals_pareto, cmap="viridis",
        s=50, alpha=0.80, depthshade=True,
    )
    ax0.scatter(
        F_pareto[knee_idx, 0], F_pareto[knee_idx, 1], F_pareto[knee_idx, 2],
        c="red", s=300, marker="*", zorder=10,
        label=f"Knee  K={K_knee}, feats={n_feat_knee}",
    )
    ax0.set_xlabel("TWCV (f1)", fontsize=8, labelpad=6)
    ax0.set_ylabel("Connectivity (f2)", fontsize=8, labelpad=6)
    ax0.set_zlabel("# Features (f3)", fontsize=8, labelpad=6)
    ax0.set_title(
        "3-D Pareto Front\n(coloured by K)",
        fontsize=9,
    )
    ax0.legend(fontsize=7, loc="upper right")
    fig.colorbar(sc0, ax=ax0, shrink=0.55, pad=0.12, label="K")

    # ── [0,1]  2-D: TWCV vs Connectivity ────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    sc1 = ax1.scatter(
        F_pareto[:, 0], F_pareto[:, 1],
        c=F_pareto[:, 2], cmap="plasma",
        s=60, alpha=0.80,
    )
    ax1.scatter(
        F_pareto[knee_idx, 0], F_pareto[knee_idx, 1],
        c="red", s=260, marker="*", zorder=6, label="Knee",
    )
    fig.colorbar(sc1, ax=ax1, label="# Active Features (f3)")
    ax1.set_xlabel("TWCV — Compactness (f1 ↓)")
    ax1.set_ylabel("Connectivity Penalty (f2 ↓)")
    ax1.set_title("Pareto Projection: TWCV vs Connectivity\nColoured by feature count")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.25)

    # Engineering decision zone annotations
    xlo, xhi = F_pareto[:, 0].min(), F_pareto[:, 0].max()
    ylo, yhi = F_pareto[:, 1].min(), F_pareto[:, 1].max()
    ax1.annotate(
        "◀ Coarse\nroutine maint.",
        xy=(xlo + (xhi - xlo) * 0.01, yhi * 0.88),
        fontsize=7.5, color="#1a5276",
        bbox=dict(boxstyle="round,pad=0.3", fc="#d6eaf8", alpha=0.85),
    )
    ax1.annotate(
        "Fine-grained ▶\nroot-cause",
        xy=(xhi * 0.72, ylo + (yhi - ylo) * 0.03),
        fontsize=7.5, color="#1e8449",
        bbox=dict(boxstyle="round,pad=0.3", fc="#d5f5e3", alpha=0.85),
    )

    # ── [0,2]  2-D: Connectivity vs N_features ───────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    sc2 = ax2.scatter(
        F_pareto[:, 2], F_pareto[:, 1],
        c=k_vals_pareto, cmap="viridis",
        s=60, alpha=0.80,
    )
    ax2.scatter(
        F_pareto[knee_idx, 2], F_pareto[knee_idx, 1],
        c="red", s=260, marker="*", zorder=6, label="Knee",
    )
    fig.colorbar(sc2, ax=ax2, label="K (# clusters)")
    ax2.set_xlabel("# Active Features (f3 ↓)  — Interpretability axis")
    ax2.set_ylabel("Connectivity Penalty (f2 ↓)")
    ax2.set_title(
        "Pareto Projection: Connectivity vs Simplicity\n"
        "← Fewer features (more interpretable)",
    )
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.25)
    # Annotate interpretability zones
    ax2.annotate(
        "← Interpretable\n(2–3 features)",
        xy=(2.2, F_pareto[:, 1].max() * 0.85),
        fontsize=7.5, color="#7d3c98",
        bbox=dict(boxstyle="round,pad=0.3", fc="#f9ebff", alpha=0.85),
    )

    # ── [1,0]  PCA: MOC knee clusters ────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    pca  = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_scaled)
    ev   = pca.explained_variance_ratio_
    cmap_moc = plt.cm.get_cmap("tab10", K_knee)

    for k in range(K_knee):
        mask_k = labels_moc == k
        ax3.scatter(
            X_2d[mask_k, 0], X_2d[mask_k, 1],
            color=cmap_moc(k), s=10, alpha=0.50, label=f"C{k}",
        )
    active_str = ", ".join(fn for fn, m in zip(FEATURE_NAMES, mask_bool_knee) if m)
    ax3.set_title(
        f"MOC Knee Solution  (K={K_knee})\n"
        f"Active: {active_str}",
        fontsize=8,
    )
    ax3.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)")
    ax3.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")
    ax3.legend(fontsize=6, markerscale=2, ncol=3)
    ax3.grid(True, alpha=0.2)

    # ── [1,1]  PCA: Ground Truth ─────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    for lbl, (col, name) in enumerate(zip(GT_COLOURS, FAULT_TYPE_NAMES)):
        m = labels_gt == lbl
        if m.any():
            ax4.scatter(
                X_2d[m, 0], X_2d[m, 1],
                color=col, s=10, alpha=0.50, label=name,
            )
    ax4.set_title("Ground Truth (Fault Type)", fontsize=9)
    ax4.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)")
    ax4.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")
    ax4.legend(fontsize=7, markerscale=2)
    ax4.grid(True, alpha=0.2)

    # ── [1,2]  Metrics table ──────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")

    n_gt_classes = len(np.unique(labels_gt))

    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    header = ["Metric", "MOC-3obj (Knee)", f"K-Means (K={n_gt_classes})"]
    rows = [
        ["Silhouette ↑",     fmt(metrics_moc["sil"]),  fmt(metrics_km["sil"])],
        ["Davies-Bouldin ↓", fmt(metrics_moc["db"]),   fmt(metrics_km["db"])],
        ["ARI ↑",            fmt(metrics_moc["ari"]),  fmt(metrics_km["ari"])],
        ["TWCV ↓",           fmt(metrics_moc["twcv"]), fmt(metrics_km["twcv"])],
        ["K (#clusters)",    str(metrics_moc["K"]),    str(n_gt_classes)],
        ["Active Features",  str(n_feat_knee),         "7 (all)"],
    ]

    tbl = ax5.table(
        cellText=rows, colLabels=header,
        loc="upper center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.1, 1.9)

    for col in range(3):
        tbl[(0, col)].set_facecolor("#2c3e50")
        tbl[(0, col)].set_text_props(color="white", fontweight="bold")

    higher_better = [True, False, True, False, None, None]
    for r, hb in enumerate(higher_better, start=1):
        if hb is None:
            continue
        try:
            vm = float(rows[r - 1][1])
            vk = float(rows[r - 1][2])
            good = 1 if (hb and vm >= vk) or (not hb and vm <= vk) else 2
            tbl[(r, good)].set_facecolor("#d5f5e3")
        except ValueError:
            pass

    ax5.set_title(
        "Evaluation Metrics — MOC-3obj vs K-Means",
        fontsize=9, pad=12,
    )

    # ── Overall title ─────────────────────────────────────────────────────────
    fig.suptitle(
        "Triple-Objective MOC — CWRU Full Dataset Bearing Fault Diagnosis\n"
        "NSGA-II  ·  Compactness (f1) + Connectivity (f2) + Feature Sparsity (f3)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    out_path = os.path.join(save_dir, "main2_moc_3obj.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[PLOT] Saved → {out_path}")
    return out_path


def plot_feature_importance(
    F_pareto: np.ndarray,
    X_pareto: np.ndarray,
    problem: MOCProblem3,
    knee_idx: int,
    save_dir: str = SAVE_DIR,
) -> str:
    """
    Feature usage analysis across the entire Pareto front.

    Panel A: bar chart of how often each feature appears on the Pareto front
    Panel B: heatmap of mask weights for every Pareto solution (sorted by f3)
    """
    n_sol       = len(X_pareto)
    usage_count = np.zeros(D, dtype=int)
    mask_matrix = np.zeros((n_sol, D))

    for i, x in enumerate(X_pareto):
        _, _, mask_bool = problem.decode(x)
        usage_count    += mask_bool.astype(int)
        mask_matrix[i]  = x[MASK_START:]

    # Sort solutions by f3 (feature count) then f1 (TWCV)
    sort_order  = np.lexsort((F_pareto[:, 0], F_pareto[:, 2]))
    mask_sorted = mask_matrix[sort_order]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: feature usage frequency
    ax = axes[0]
    colours = ["#e74c3c" if (x[MASK_START + d] > 0.5) else "#3498db"
               for d in range(D)
               for x in [X_pareto[knee_idx]]]
    colours = [
        "#e74c3c" if X_pareto[knee_idx][MASK_START + d] > 0.5 else "#3498db"
        for d in range(D)
    ]
    bars = ax.bar(FEATURE_NAMES, usage_count, color=colours, edgecolor="black", linewidth=0.6)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Frequency on Pareto Front")
    ax.set_title(
        "Feature Usage Frequency Across Pareto Front\n"
        "(red = active in knee solution)"
    )
    ax.set_ylim(0, n_sol * 1.1)
    for bar, cnt in zip(bars, usage_count):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + n_sol * 0.01,
                str(cnt), ha="center", va="bottom", fontsize=9)

    # Panel B: mask weight heatmap
    ax = axes[1]
    im = ax.imshow(
        mask_sorted.T, aspect="auto", cmap="RdYlGn",
        vmin=0, vmax=1, interpolation="nearest",
    )
    ax.set_yticks(range(D))
    ax.set_yticklabels(FEATURE_NAMES, fontsize=9)
    ax.set_xlabel("Pareto Solution (sorted by feature count, then TWCV)")
    ax.set_title(
        "Feature Mask Weight Heatmap\n"
        "(green = active, red = inactive; sorted by simplicity)"
    )
    ax.axhline(-0.5, color="black", linewidth=0.5)
    plt.colorbar(im, ax=ax, label="Mask Weight")

    plt.tight_layout()
    out_path = os.path.join(save_dir, "main2_feature_analysis.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved → {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# 9.  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t_total = time.time()

    print("=" * 72)
    print("  Triple-Objective MOC — CWRU Full Dataset Bearing Fault Diagnosis")
    print("  Objectives: Compactness (f1) | Connectivity (f2) | Simplicity (f3)")
    print("=" * 72)

    # ── Step 1: Load full dataset ─────────────────────────────────────────────
    print(f"\n[1/7] Loading full CWRU dataset from:\n      {DATASET_ROOT}")
    records          = load_full_dataset(DATASET_ROOT)
    segs, labels_gt, fs_arr = segment_signals(records)
    n_classes        = len(np.unique(labels_gt))
    print(f"\n      Segments : {segs.shape[0]:,}   Window : {WINDOW_SIZE} samples")
    print(f"      Classes  : {n_classes} — {FAULT_TYPE_NAMES}")
    print(f"      Distribution: { {k: int((labels_gt==k).sum()) for k in np.unique(labels_gt)} }")

    # ── Step 2: Feature extraction ────────────────────────────────────────────
    print(f"\n[2/7] Extracting {D} features per segment ...")
    t0    = time.time()
    feats = extract_features(segs, fs_arr)
    X, _  = preprocess(feats)
    print(f"      Feature matrix: {X.shape}  ({time.time()-t0:.1f}s)")

    # ── Step 3: Build problem & algorithm ─────────────────────────────────────
    print(f"\n[3/7] Building MOCProblem3  (N_VARS={N_VARS}, n_obj=3) ...")
    problem   = MOCProblem3(X, knn_k=KNN_K)
    sampling  = SmartSampling3(X)
    crossover = SBX(prob=0.9, eta=15, vtype=float)
    mutation  = StructuralMutation3(X, prob_struct=0.10, prob_flip=0.15, eta=20.0)

    algorithm = NSGA2(
        pop_size=POP_SIZE,
        sampling=sampling,
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=True,
    )
    termination = get_termination("n_gen", N_GEN)

    # ── Step 4: Optimise ──────────────────────────────────────────────────────
    print(f"\n[4/7] Running NSGA-II  (pop={POP_SIZE}, gen={N_GEN}) ...")
    print("      Progress printed every 100 generations ...\n")
    t1  = time.time()
    res = minimize(
        problem, algorithm, termination,
        seed=42, verbose=True, save_history=False,
    )
    elapsed = time.time() - t1
    print(f"\n      Optimisation complete in {elapsed/60:.1f} min")

    # ── Step 5: Log final population ─────────────────────────────────────────
    print("\n[5/7] Logging final population ...")
    log_final_population(res, problem, save_dir=SAVE_DIR)

    # ── Step 6: Pareto & knee analysis ───────────────────────────────────────
    print("\n[6/7] Analysing 3-objective Pareto front ...")
    F_pareto = res.F
    X_pareto = res.X

    k_vals = np.array([
        int(np.clip(round(x[0]), K_MIN, K_MAX)) for x in X_pareto
    ])
    print(f"      Pareto front size  : {len(F_pareto)}")
    print(f"      K values present   : {sorted(set(k_vals.tolist()))}")
    print(f"      f1 TWCV    range   : [{F_pareto[:,0].min():.2f}, {F_pareto[:,0].max():.2f}]")
    print(f"      f2 Conn    range   : [{F_pareto[:,1].min():.2f}, {F_pareto[:,1].max():.2f}]")
    print(f"      f3 N_feats range   : [{F_pareto[:,2].min():.0f}, {F_pareto[:,2].max():.0f}]")

    knee_idx              = find_knee_point_3d(F_pareto)
    K_knee, centres_knee, mask_bool_knee = problem.decode(X_pareto[knee_idx])
    X_sub_knee            = X[:, mask_bool_knee]
    C_sub_knee            = centres_knee[:, mask_bool_knee]
    labels_moc            = problem.assign(X_sub_knee, C_sub_knee)

    active_feat_names = [fn for fn, m in zip(FEATURE_NAMES, mask_bool_knee) if m]
    print(f"\n      Knee point → K={K_knee}  |  "
          f"TWCV={F_pareto[knee_idx,0]:.2f}  |  "
          f"Conn={F_pareto[knee_idx,1]:.2f}  |  "
          f"N_feat={int(mask_bool_knee.sum())}")
    print(f"      Active features: {active_feat_names}")

    # ── Step 7: Metrics, baseline & visualisation ─────────────────────────────
    print("\n[7/7] Evaluating and generating plots ...")
    metrics_moc = {
        "sil":  silhouette_score(X, labels_moc),
        "db":   davies_bouldin_score(X, labels_moc),
        "ari":  adjusted_rand_score(labels_gt, labels_moc),
        "twcv": float(F_pareto[knee_idx, 0]),
        "K":    K_knee,
    }
    metrics_km = kmeans_baseline(X, labels_gt, K=n_classes)

    # Console comparison table
    print("\n  ┌────────────────────┬──────────────────┬──────────────────────┐")
    print( "  │ Metric             │  MOC-3obj (Knee) │  K-Means Baseline    │")
    print( "  ├────────────────────┼──────────────────┼──────────────────────┤")
    for name, mk in [
        ("Silhouette  ↑",    "sil"),
        ("Davies-Bouldin ↓", "db"),
        ("ARI  ↑",           "ari"),
        ("TWCV  ↓",          "twcv"),
    ]:
        print(f"  │ {name:<18s} │ {metrics_moc[mk]:>16.4f} │ {metrics_km[mk]:>20.4f} │")
    print(f"  │ {'K (#clusters)':<18s} │ {metrics_moc['K']:>16} │ {metrics_km['K']:>20} │")
    print(f"  │ {'Active features':<18s} │ {len(active_feat_names):>16} │ {'7 (all)':>20} │")
    print("  └────────────────────┴──────────────────┴──────────────────────┘")

    plot_results_3d(
        F_pareto, knee_idx,
        X, labels_moc, labels_gt,
        metrics_moc, metrics_km,
        k_vals, mask_bool_knee,
        save_dir=SAVE_DIR,
    )

    plot_feature_importance(
        F_pareto, X_pareto, problem, knee_idx,
        save_dir=SAVE_DIR,
    )

    total_min = (time.time() - t_total) / 60
    print(f"\n[DONE]  Total runtime: {total_min:.1f} min")
    print(f"        Outputs saved to: {SAVE_DIR}")
    print(f"          • main2_moc_3obj.png")
    print(f"          • main2_feature_analysis.png")
    print(f"          • final_population.csv")

    return res, metrics_moc, metrics_km


if __name__ == "__main__":
    main()
