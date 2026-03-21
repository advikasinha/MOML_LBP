#!/usr/bin/env python3
"""
Multi-Objective Clustering (MOC) for Rolling Element Bearing Fault Diagnosis
CWRU Bearing Dataset — 12k Drive End, Ball Fault (B folder)

Pipeline stages
---------------
1. Data Engineering & Signal Processing
2. Multi-Objective Optimization via NSGA-II (pymoo)
3. Smart Initialization (1/3 split strategy)
4. Genetic Operators: SBX crossover + PM + Structural Mutation
5. Validation & Decision-Making Visualization

Decision-Making Advantage
-------------------------
The Pareto front exposes a trade-off spectrum an engineer can exploit:

  LEFT end  (low TWCV / low K)  → COARSE clusters
    • Broad health states (fault vs. no-fault)
    • Best for routine maintenance scheduling or automated alarming
    • High confidence, few false positives

  RIGHT end (low connectivity / high K) → FINE-GRAINED clusters
    • Distinguishes fault severity tiers (0.007" vs 0.014" vs 0.021" crack)
    • Best for root-cause failure analysis and remaining-useful-life models
    • Requires more samples per cluster to be statistically reliable

  KNEE POINT → Best general-purpose monitoring trade-off
    • Maximum curvature on normalized Pareto front
    • Chosen automatically; can be overridden by engineer based on context
"""

import os
import glob
import warnings
import numpy as np
import scipy.io
from scipy.spatial.distance import cdist
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, adjusted_rand_score
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
WINDOW_SIZE = 1024      # samples per segment
K_MIN       = 2         # minimum clusters
K_MAX       = 10        # maximum clusters
D           = 7         # feature dimensions
N_VARS      = 1 + K_MAX * D   # K gene + K_MAX centroid vectors (padded)
KNN_K       = 10        # neighbours for connectivity objective
POP_SIZE    = 150       # initial population size (100–200 per spec)
N_GEN       = 500       # number of generations
FS          = 12_000.0  # sampling frequency (Hz)

BASE_DIR = (
    "/opt/watchdog/users/cherish/MOML_LBP/"
    "CWRU-dataset/12k_Drive_End_Bearing_Fault_Data/B"
)
SAVE_DIR = "/opt/watchdog/users/cherish/MOML_LBP"

# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA ENGINEERING & SIGNAL PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def load_cwru_b(base_dir: str) -> list:
    """
    Load every .mat file under B/{007,014,021,028}/.

    Returns
    -------
    list of (signal_1d_array, int_label)
        Label encodes fault severity: 007→0, 014→1, 021→2, 028→3
    """
    fault_dirs = {"007": 0, "014": 1, "021": 2, "028": 3}
    records = []
    for folder_name, label in fault_dirs.items():
        folder = os.path.join(base_dir, folder_name)
        mat_files = sorted(glob.glob(os.path.join(folder, "*.mat")))
        if not mat_files:
            raise FileNotFoundError(f"No .mat files found in {folder}")
        for fpath in mat_files:
            mat = scipy.io.loadmat(fpath)
            # Locate the Drive-End time-series key (contains 'DE_time')
            de_key = next(
                k for k in mat.keys()
                if "DE_time" in k and not k.startswith("_")
            )
            signal = mat[de_key].flatten().astype(np.float64)
            records.append((signal, label))
            print(f"  Loaded {os.path.basename(fpath):20s} "
                  f"| label={label} | {len(signal):>7d} samples")
    return records


def segment_signals(records: list, window: int = WINDOW_SIZE):
    """
    Chop each signal into non-overlapping windows.

    Returns
    -------
    segments : ndarray (N, window)
    labels   : ndarray (N,)  — ground-truth fault severity
    """
    segs, labs = [], []
    for signal, label in records:
        n_windows = len(signal) // window
        for i in range(n_windows):
            segs.append(signal[i * window: (i + 1) * window])
            labs.append(label)
    return np.array(segs, dtype=np.float64), np.array(labs, dtype=int)


def extract_features(segments: np.ndarray, fs: float = FS) -> np.ndarray:
    """
    Compute 7 statistical / spectral features per segment.

    Feature index map
    -----------------
    0  RMS              Energy content of vibration
    1  Kurtosis         Impulsiveness — spikes from fault impacts
    2  Skewness         Asymmetry — shifts as fault grows
    3  Crest Factor     Peak/RMS — early shock indicator
    4  Peak-to-Peak     Total amplitude swing
    5  Std Deviation    Spread of vibration amplitude
    6  Spectral Centroid  Frequency center-of-mass (shifts near fault freq.)
    """
    N, W = segments.shape
    feats = np.zeros((N, 7), dtype=np.float64)
    freqs = np.fft.rfftfreq(W, d=1.0 / fs)  # frequency axis for SC

    for i, seg in enumerate(segments):
        rms   = np.sqrt(np.mean(seg ** 2))
        kurt  = kurtosis(seg, fisher=True)   # excess kurtosis
        skw   = skew(seg)
        peak  = np.max(np.abs(seg))
        crest = peak / (rms + 1e-12)
        p2p   = np.ptp(seg)
        std   = np.std(seg)

        spectrum = np.abs(np.fft.rfft(seg))
        sc = np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-12)

        feats[i] = [rms, kurt, skw, crest, p2p, std, sc]

    return feats


def preprocess(X: np.ndarray):
    """Zero-mean, unit-variance standardization. Returns (X_scaled, scaler)."""
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler


# ─────────────────────────────────────────────────────────────────────────────
# 2.  MULTI-OBJECTIVE CLUSTERING PROBLEM
# ─────────────────────────────────────────────────────────────────────────────

class MOCProblem(Problem):
    """
    NSGA-II problem for variable-K clustering.

    Chromosome layout (length = N_VARS = 1 + K_MAX * D)
    ────────────────────────────────────────────────────
    x[0]              : K  (float; rounded to int in [K_MIN, K_MAX])
    x[1 : K*D + 1]    : active cluster centres  (K × D values)
    x[K*D + 1 :]      : padding (ignored during evaluation)

    Objectives
    ──────────
    f1  TWCV         — Total Within-Cluster Variance  (compactness)
    f2  Connectivity — Handl-Knowles KNN penalty      (topology)
    """

    def __init__(self, X: np.ndarray, knn_k: int = KNN_K):
        self.X = X
        self.N, self.D = X.shape
        self.knn_k = knn_k

        # ── Pre-compute KNN graph once (topology is fixed) ──────────────────
        nbrs = NearestNeighbors(
            n_neighbors=knn_k + 1, algorithm="ball_tree", n_jobs=-1
        )
        nbrs.fit(X)
        _, indices = nbrs.kneighbors(X)
        self.knn_indices = indices[:, 1:]           # (N, knn_k) — skip self
        self.knn_weights = 1.0 / np.arange(1, knn_k + 1)  # [1, 1/2, …, 1/k]

        # ── Variable bounds ─────────────────────────────────────────────────
        xl = np.empty(N_VARS)
        xu = np.empty(N_VARS)
        xl[0], xu[0] = K_MIN, K_MAX                # K gene
        xl[1:] = np.tile(X.min(axis=0), K_MAX)     # centroid lower bounds
        xu[1:] = np.tile(X.max(axis=0), K_MAX)     # centroid upper bounds

        super().__init__(
            n_var=N_VARS, n_obj=2, n_ieq_constr=0, xl=xl, xu=xu
        )

    # ── Decoding helpers ─────────────────────────────────────────────────────

    def decode(self, x: np.ndarray):
        """Extract (K: int, centres: K×D array) from a chromosome."""
        K = int(np.clip(round(x[0]), K_MIN, K_MAX))
        centres = x[1: K * self.D + 1].reshape(K, self.D)
        return K, centres

    def assign(self, centres: np.ndarray) -> np.ndarray:
        """Hard-assign every point to its nearest centre."""
        return np.argmin(cdist(self.X, centres), axis=1)

    # ── Objective functions ──────────────────────────────────────────────────

    def _twcv(self, labels: np.ndarray, centres: np.ndarray) -> float:
        """
        Total Within-Cluster Variance — measures compactness.
        Minimising TWCV pushes points tightly around their centroid.
        """
        return float(np.sum((self.X - centres[labels]) ** 2))

    def _connectivity(self, labels: np.ndarray) -> float:
        """
        Connectivity penalty (Handl & Knowles, 2007).

        For each point i and its k ranked nearest neighbours:
          Conn += (1/rank) * I(cluster(i) ≠ cluster(neighbour))

        Penalises solutions that separate topologically close points,
        ensuring clusters reflect the true manifold of the vibration data.
        """
        # knn_indices : (N, k)  → gather neighbour labels
        neighbour_labels = labels[self.knn_indices]           # (N, k)
        different = (labels[:, None] != neighbour_labels)     # (N, k) bool
        return float(np.sum(different * self.knn_weights))    # broadcast 1/rank

    # ── pymoo evaluation entry point ─────────────────────────────────────────

    def _evaluate(self, X_pop, out, *args, **kwargs):
        f1_vals = np.empty(len(X_pop))
        f2_vals = np.empty(len(X_pop))

        for i, x in enumerate(X_pop):
            _, centres = self.decode(x)
            labels     = self.assign(centres)
            f1_vals[i] = self._twcv(labels, centres)
            f2_vals[i] = self._connectivity(labels)

        out["F"] = np.column_stack([f1_vals, f2_vals])


# ─────────────────────────────────────────────────────────────────────────────
# 3.  SMART INITIALIZATION  (1/3 split)
# ─────────────────────────────────────────────────────────────────────────────

class SmartSampling(Sampling):
    """
    Population initialisation with three equal thirds:

    ┌─────────────────────────────────────────────────────────────────┐
    │ 1/3  K-Means seeded   — fast convergence to good regions        │
    │ 1/3  Random data pts  — realistic feature-space coverage        │
    │ 1/3  Pure random      — genetic diversity / prevent stagnation  │
    └─────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, X: np.ndarray):
        super().__init__()
        self.X = X
        self.N, self.D = X.shape

    def _do(self, problem, n_samples, **kwargs):
        xl, xu = problem.xl, problem.xu
        pop = np.zeros((n_samples, N_VARS))

        n_km    = n_samples // 3
        n_rdata = n_samples // 3
        n_rand  = n_samples - n_km - n_rdata
        idx = 0

        # ── Slice 1: K-Means seeded ──────────────────────────────────────────
        for _ in range(n_km):
            K = np.random.randint(K_MIN, K_MAX + 1)
            x = np.random.uniform(xl, xu)
            x[0] = K
            try:
                km = KMeans(n_clusters=K, n_init=3, max_iter=100, random_state=None)
                km.fit(self.X)
                x[1: K * self.D + 1] = km.cluster_centers_.flatten()
            except Exception:
                pass   # fallback: keep random centres already set
            pop[idx] = x
            idx += 1

        # ── Slice 2: Random data points as centres ───────────────────────────
        for _ in range(n_rdata):
            K = np.random.randint(K_MIN, K_MAX + 1)
            x = np.random.uniform(xl, xu)
            x[0] = K
            chosen = self.X[np.random.choice(self.N, K, replace=False)]
            x[1: K * self.D + 1] = chosen.flatten()
            pop[idx] = x
            idx += 1

        # ── Slice 3: Pure random ─────────────────────────────────────────────
        for _ in range(n_rand):
            x = np.random.uniform(xl, xu)
            x[0] = float(np.random.randint(K_MIN, K_MAX + 1))
            pop[idx] = x
            idx += 1

        return pop


# ─────────────────────────────────────────────────────────────────────────────
# 4.  STRUCTURAL MUTATION
#     Polynomial mutation on centroids  +  add / remove a cluster
# ─────────────────────────────────────────────────────────────────────────────

class StructuralMutation(Mutation):
    """
    Two-phase mutation operator:

    Phase A — Polynomial Mutation (PM)
        Applied to each active centroid coordinate with prob 1/(K·D).
        Eta=20 keeps perturbations moderate.

    Phase B — Structural Mutation  (applied with prob p_struct)
        ADD  a cluster  (K < K_MAX): insert midpoint of two random centres
            plus Gaussian noise — explores finer fault granularity.
        DROP a cluster  (K > K_MIN): remove the least-populated centre
            (merge step) — simplifies the partition toward coarser states.

    Together these let the GA traverse the discrete K-space and discover
    the optimal number of fault severity tiers without manual tuning.
    """

    def __init__(self, X: np.ndarray, prob_struct: float = 0.10, eta: float = 20.0):
        super().__init__()
        self.X = X
        self.N, self.D = X.shape
        self.prob_struct = prob_struct
        self.eta = eta

    def _do(self, problem, X_pop, **kwargs):
        xl, xu = problem.xl, problem.xu
        X_mut  = X_pop.copy()

        for i in range(len(X_mut)):
            x = X_mut[i]
            K = int(np.clip(round(x[0]), K_MIN, K_MAX))

            # ── Phase A: Polynomial mutation on active centroid coords ────────
            pm_prob = 1.0 / max(K * self.D, 1)
            for j in range(1, K * self.D + 1):
                if np.random.random() >= pm_prob:
                    continue
                y    = x[j]
                lo, hi = xl[j], xu[j]
                span = hi - lo + 1e-12
                d1   = (y - lo)  / span
                d2   = (hi - y)  / span
                r    = np.random.random()
                mu   = 1.0 / (self.eta + 1.0)
                if r < 0.5:
                    v  = 2.0 * r + (1.0 - 2.0 * r) * (1.0 - d1) ** (self.eta + 1)
                    dq = v ** mu - 1.0
                else:
                    v  = 2.0 * (1.0 - r) + 2.0 * (r - 0.5) * (1.0 - d2) ** (self.eta + 1)
                    dq = 1.0 - v ** mu
                x[j] = float(np.clip(y + dq * span, lo, hi))

            # ── Phase B: Structural mutation ──────────────────────────────────
            if np.random.random() < self.prob_struct:
                centres = x[1: K * self.D + 1].reshape(K, self.D)

                if np.random.random() < 0.5 and K < K_MAX:
                    # ADD: midpoint of two random centres + noise
                    i1, i2 = np.random.choice(K, 2, replace=False)
                    new_c  = (centres[i1] + centres[i2]) * 0.5
                    noise  = np.random.normal(0, 0.05, self.D)
                    new_c  = np.clip(new_c + noise, xl[1: self.D + 1], xu[1: self.D + 1])
                    K_new  = K + 1
                    x[0]   = float(K_new)
                    x[K * self.D + 1: K_new * self.D + 1] = new_c

                elif K > K_MIN:
                    # DROP: remove centre with fewest assigned points
                    labels = np.argmin(cdist(self.X, centres), axis=1)
                    counts = np.bincount(labels, minlength=K)
                    drop   = int(np.argmin(counts))
                    new_c  = np.delete(centres, drop, axis=0)   # (K-1, D)
                    K_new  = K - 1
                    x[0]   = float(K_new)
                    x[1: K_new * self.D + 1] = new_c.flatten()
                    # Re-randomise the now-unused tail
                    tail_start = K_new * self.D + 1
                    x[tail_start:] = np.random.uniform(
                        xl[tail_start:], xu[tail_start:]
                    )

        return X_mut


# ─────────────────────────────────────────────────────────────────────────────
# 5.  PARETO / KNEE-POINT ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def find_knee_point(F: np.ndarray) -> int:
    """
    Locate the knee point on the Pareto front via maximum-curvature method.

    Steps
    -----
    1. Normalise both objectives to [0, 1].
    2. Connect the extreme Pareto points with a reference line.
    3. Return the index of the solution furthest from that line.

    This point offers the best compromise between cluster compactness
    (few, tight clusters) and topological fidelity (many, well-separated
    clusters) — the recommended default for general maintenance monitoring.
    """
    eps = 1e-12
    F_norm = (F - F.min(axis=0)) / ((F.max(axis=0) - F.min(axis=0)) + eps)

    sort_idx  = np.argsort(F_norm[:, 0])
    F_sorted  = F_norm[sort_idx]
    p1, p2    = F_sorted[0], F_sorted[-1]
    line_vec  = p2 - p1
    line_len  = np.linalg.norm(line_vec) + eps

    perp_dist = np.abs(
        np.cross(line_vec, p1 - F_sorted)
    ) / line_len                           # vectorised perpendicular distances

    return int(sort_idx[np.argmax(perp_dist)])


# ─────────────────────────────────────────────────────────────────────────────
# 6.  BASELINE: standard K-Means
# ─────────────────────────────────────────────────────────────────────────────

def kmeans_baseline(X: np.ndarray, labels_gt: np.ndarray, K: int = 4) -> dict:
    """Fit K-Means and collect all evaluation metrics for comparison."""
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
# 7.  VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

_GT_COLOURS  = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
_FAULT_NAMES = ["B007 (0.007\")", "B014 (0.014\")", "B021 (0.021\")", "B028 (0.028\")"]


def plot_results(
    F_pareto, knee_idx,
    X_scaled, labels_moc, labels_gt,
    metrics_moc, metrics_km,
    k_vals_pareto,
    save_dir=SAVE_DIR,
) -> str:
    """
    Generate a 2×3 diagnostic figure:

    Row 0 ┌─────────────────────────────────┬─────────────────┐
          │  Plot 1: Pareto Front           │  Plot 3: Table  │
          │  (TWCV vs Connectivity)         │  (MOC vs KM)    │
    Row 1 ├─────────────────┬───────────────┴─────────────────┤
          │  Plot 2a: PCA   │  Plot 2b: PCA    Plot 2c: t-SNE │
          │  MOC clusters   │  Ground truth    Ground truth   │
          └─────────────────┴───────────────────────────────── ┘
    """
    fig = plt.figure(figsize=(20, 13))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

    K_knee = metrics_moc["K"]

    # ── Plot 1: Pareto Front ─────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    sc  = ax1.scatter(
        F_pareto[:, 0], F_pareto[:, 1],
        c=k_vals_pareto, cmap="viridis",
        s=70, alpha=0.85, zorder=3,
    )
    cbar = plt.colorbar(sc, ax=ax1, pad=0.01)
    cbar.set_label("K (# clusters)", fontsize=9)

    ax1.scatter(
        F_pareto[knee_idx, 0], F_pareto[knee_idx, 1],
        c="red", s=260, marker="*", zorder=6,
        label=f"Knee Point  K={K_knee}",
    )
    ax1.set_xlabel("Objective 1 — Compactness (TWCV ↓)", fontsize=10)
    ax1.set_ylabel("Objective 2 — Connectivity Penalty (↓)", fontsize=10)
    ax1.set_title(
        "NSGA-II Pareto Front  —  MOC for CWRU Ball Bearing Fault Diagnosis\n"
        "← More compact clusters  |  Better topology preservation →",
        fontsize=11,
    )
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.25)

    # Annotation: engineering decision zones
    x_lo = F_pareto[:, 0].min()
    y_hi = F_pareto[:, 1].max()
    x_hi = F_pareto[:, 0].max()
    y_lo = F_pareto[:, 1].min()

    ax1.annotate(
        "◀  Coarse clusters\n    routine maintenance",
        xy=(x_lo + (x_hi - x_lo) * 0.02, y_hi * 0.88),
        fontsize=8, color="#1a5276",
        bbox=dict(boxstyle="round,pad=0.35", fc="#d6eaf8", alpha=0.85),
    )
    ax1.annotate(
        "Fine-grained clusters  ▶\nroot-cause analysis",
        xy=(x_hi * 0.70, y_lo + (y_hi - y_lo) * 0.04),
        fontsize=8, color="#1e8449",
        bbox=dict(boxstyle="round,pad=0.35", fc="#d5f5e3", alpha=0.85),
    )

    # ── Plot 2a: PCA — MOC knee solution ────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    pca   = PCA(n_components=2, random_state=42)
    X_2d  = pca.fit_transform(X_scaled)
    ev    = pca.explained_variance_ratio_
    cmap_moc = plt.cm.get_cmap("tab10", K_knee)

    for k in range(K_knee):
        mask = labels_moc == k
        ax2.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            color=cmap_moc(k), s=12, alpha=0.55, label=f"C{k}",
        )
    ax2.set_title(
        f"MOC — Knee Solution  (K={K_knee})\n"
        f"PCA  PC1={ev[0]*100:.1f}%  PC2={ev[1]*100:.1f}%",
        fontsize=9,
    )
    ax2.set_xlabel("PC1"); ax2.set_ylabel("PC2")
    ax2.legend(fontsize=7, markerscale=2, ncol=2)
    ax2.grid(True, alpha=0.2)

    # ── Plot 2b: PCA — Ground Truth ──────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    for lbl, (col, name) in enumerate(zip(_GT_COLOURS, _FAULT_NAMES)):
        mask = labels_gt == lbl
        ax3.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            color=col, s=12, alpha=0.55, label=name,
        )
    ax3.set_title("Ground Truth  (Fault Severity)", fontsize=9)
    ax3.set_xlabel("PC1"); ax3.set_ylabel("PC2")
    ax3.legend(fontsize=7, markerscale=2)
    ax3.grid(True, alpha=0.2)

    # ── Plot 2c: t-SNE — Ground Truth ────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    n_tsne  = min(2_000, len(X_scaled))
    rng_idx = np.random.choice(len(X_scaled), n_tsne, replace=False)
    tsne    = TSNE(n_components=2, random_state=42, perplexity=30,
                   max_iter=1_000, init="pca")
    X_tsne  = tsne.fit_transform(X_scaled[rng_idx])

    for lbl, (col, name) in enumerate(zip(_GT_COLOURS, _FAULT_NAMES)):
        mask = labels_gt[rng_idx] == lbl
        ax5.scatter(
            X_tsne[mask, 0], X_tsne[mask, 1],
            color=col, s=12, alpha=0.55, label=name,
        )
    ax5.set_title("t-SNE Projection  (Ground Truth)", fontsize=9)
    ax5.set_xlabel("t-SNE 1"); ax5.set_ylabel("t-SNE 2")
    ax5.legend(fontsize=7, markerscale=2)
    ax5.grid(True, alpha=0.2)

    # ── Plot 3: Metrics table ────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[0, 2])
    ax4.axis("off")

    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    header = ["Metric", "MOC (Knee)", "K-Means (K=4)"]
    rows   = [
        ["Silhouette ↑",    fmt(metrics_moc["sil"]),  fmt(metrics_km["sil"])],
        ["Davies-Bouldin ↓", fmt(metrics_moc["db"]),  fmt(metrics_km["db"])],
        ["ARI ↑",            fmt(metrics_moc["ari"]), fmt(metrics_km["ari"])],
        ["TWCV ↓",           fmt(metrics_moc["twcv"]),fmt(metrics_km["twcv"])],
        ["K  (#clusters)",   str(metrics_moc["K"]),   "4"],
    ]

    tbl = ax4.table(
        cellText=rows, colLabels=header,
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1.15, 2.1)

    # Header styling
    for col in range(3):
        tbl[(0, col)].set_facecolor("#2c3e50")
        tbl[(0, col)].set_text_props(color="white", fontweight="bold")

    # Highlight better value in each metric row
    higher_better = [True, False, True, False, None]
    for r, hb in enumerate(higher_better, start=1):
        if hb is None:
            continue
        try:
            v_moc = float(rows[r - 1][1])
            v_km  = float(rows[r - 1][2])
            good_col = 1 if (hb and v_moc >= v_km) or (not hb and v_moc <= v_km) else 2
            tbl[(r, good_col)].set_facecolor("#d5f5e3")
        except ValueError:
            pass

    ax4.set_title(
        "Evaluation Metrics\nMOC (Knee) vs K-Means Baseline",
        fontsize=10, pad=16,
    )

    fig.suptitle(
        "Multi-Objective Clustering (MOC) — CWRU Ball Bearing Fault Diagnosis\n"
        "NSGA-II  ·  Compactness & Connectivity Objectives  ·  12k Drive End",
        fontsize=13, fontweight="bold", y=1.005,
    )

    out_path = os.path.join(save_dir, "moc_bearing_diagnosis.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[PLOT] Saved → {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import time

    print("=" * 68)
    print("  Multi-Objective Clustering for CWRU Bearing Fault Diagnosis")
    print("=" * 68)

    # ── 1. Load & segment ────────────────────────────────────────────────────
    print("\n[1/6] Loading CWRU data ...")
    records  = load_cwru_b(BASE_DIR)
    segs, labels_gt = segment_signals(records)
    print(f"      Segments: {segs.shape}   unique labels: {np.unique(labels_gt)}")

    # ── 2. Feature extraction ────────────────────────────────────────────────
    print("\n[2/6] Extracting features ...")
    t0 = time.time()
    feats = extract_features(segs)
    X, scaler = preprocess(feats)
    print(f"      Feature matrix: {X.shape}  ({time.time()-t0:.1f}s)")
    print(f"      Class distribution: { {k: int((labels_gt==k).sum()) for k in np.unique(labels_gt)} }")

    # ── 3. Problem & algorithm setup ─────────────────────────────────────────
    print("\n[3/6] Building MOCProblem & NSGA-II ...")
    problem   = MOCProblem(X, knn_k=KNN_K)
    sampling  = SmartSampling(X)
    crossover = SBX(prob=0.9, eta=15, vtype=float)
    mutation  = StructuralMutation(X, prob_struct=0.10, eta=20.0)

    algorithm = NSGA2(
        pop_size=POP_SIZE,
        sampling=sampling,
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=True,
    )
    termination = get_termination("n_gen", N_GEN)

    # ── 4. Optimise ──────────────────────────────────────────────────────────
    print(f"\n[4/6] Running NSGA-II  (pop={POP_SIZE}, gen={N_GEN}) ...")
    print("      Progress displayed every 100 generations ...\n")
    t1  = time.time()
    res = minimize(
        problem, algorithm, termination,
        seed=42, verbose=True, save_history=False,
    )
    elapsed = time.time() - t1
    print(f"\n      Optimisation complete in {elapsed/60:.1f} min")

    # ── 5. Pareto analysis ───────────────────────────────────────────────────
    print("\n[5/6] Analysing Pareto front ...")
    F_pareto = res.F
    X_pareto = res.X

    k_vals = np.array([
        int(np.clip(round(x[0]), K_MIN, K_MAX)) for x in X_pareto
    ])
    print(f"      Pareto front size : {len(F_pareto)}")
    print(f"      K values present  : {sorted(set(k_vals.tolist()))}")

    knee_idx    = find_knee_point(F_pareto)
    K_knee, centres_knee = problem.decode(X_pareto[knee_idx])
    labels_moc  = problem.assign(centres_knee)

    print(f"      Knee point: K={K_knee} | "
          f"TWCV={F_pareto[knee_idx,0]:.2f} | "
          f"Conn={F_pareto[knee_idx,1]:.2f}")

    # ── 6. Metrics & plots ───────────────────────────────────────────────────
    print("\n[6/6] Evaluating and plotting ...")
    metrics_moc = {
        "sil":  silhouette_score(X, labels_moc),
        "db":   davies_bouldin_score(X, labels_moc),
        "ari":  adjusted_rand_score(labels_gt, labels_moc),
        "twcv": float(F_pareto[knee_idx, 0]),
        "K":    K_knee,
    }
    metrics_km = kmeans_baseline(X, labels_gt, K=4)

    # Pretty-print comparison table
    print("\n  ┌───────────────────┬──────────────┬─────────────────┐")
    print("  │ Metric            │  MOC (Knee)  │  K-Means (K=4)  │")
    print("  ├───────────────────┼──────────────┼─────────────────┤")
    for name, k, m in [
        ("Silhouette  ↑",    "sil",  "sil"),
        ("Davies-Bouldin ↓", "db",   "db"),
        ("ARI  ↑",           "ari",  "ari"),
        ("TWCV  ↓",          "twcv", "twcv"),
    ]:
        print(f"  │ {name:<17s} │ {metrics_moc[k]:>12.4f} │ {metrics_km[m]:>15.4f} │")
    print(f"  │ {'K (#clusters)':<17s} │ {metrics_moc['K']:>12} │ {'4':>15} │")
    print("  └───────────────────┴──────────────┴─────────────────┘")

    plot_results(
        F_pareto, knee_idx,
        X, labels_moc, labels_gt,
        metrics_moc, metrics_km,
        k_vals, save_dir=SAVE_DIR,
    )

    print("\n[DONE]  moc_bearing_diagnosis.png saved to", SAVE_DIR)
    return res, metrics_moc, metrics_km


if __name__ == "__main__":
    main()
