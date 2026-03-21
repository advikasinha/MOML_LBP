"""
MOC-FS: Multi-Objective Clustering with Feature Sparsity
for Bearing Fault Diagnosis using the CWRU Dataset

Pipeline:
  1. Data loading & feature extraction (12k Drive End)
  2. NSGA-II with variable-K chromosome encoding
  3. Hybrid initialization (KMeans / data points / random)
  4. Custom SBX + point-crossover / polynomial + bit-flip mutation
  5. Pareto front analysis, knee-point detection
  6. ARI / Silhouette validation + 5-panel visualisation

Author : Senior Research Engineer, Predictive Maintenance
Dataset: CWRU Bearing Dataset – 12 kHz Drive End accelerometer
GPU    : NVIDIA RTX 5090 (vast.ai)
"""

# ── stdlib ──────────────────────────────────────────────────────────────────
import os
import glob
import time
import warnings
import argparse
import json
import logging

# ── numeric / ML ────────────────────────────────────────────────────────────
import numpy as np
import scipy.io
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

# ── pymoo ────────────────────────────────────────────────────────────────────
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination

# ── visualisation ────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")          # headless – safe for GPU server
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3-D projection)

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════════════════
# GLOBAL CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════
WINDOW_SIZE  = 1024          # samples per segment
FS           = 12_000        # sampling rate (Hz) for 12k Drive End data
D            = 7             # number of hand-crafted features
MAX_K        = 10            # upper bound on cluster count
MIN_K        = 2             # lower bound on cluster count
KNN_K        = 10            # neighbours for connectedness penalty
POP_SIZE     = 150           # NSGA-II population size
N_GEN        = 100           # number of generations
RANDOM_SEED  = 42
RESULTS_DIR  = "moc_results"

FEATURE_NAMES = np.array([
    "RMS", "Kurtosis", "Skewness",
    "Crest Factor", "Peak-to-Peak", "Std Dev", "Spectral Centroid",
])
CLASS_NAMES = ["Normal", "Inner Race", "Ball", "Outer Race"]

# Chromosome layout  (total = N_VAR = 1 + MAX_K*D + D = 78)
#   x[0]               : K_real  ∈ [MIN_K, MAX_K+0.999] → K = int(x[0])
#   x[1 : 1+MAX_K*D]   : flattened cluster centres in standardised space
#   x[1+MAX_K*D :]     : feature mask (real ∈ [0,1]; ≥0.5 → active)
N_VAR     = 1 + MAX_K * D + D         # 78
CTR_START = 1
CTR_END   = 1 + MAX_K * D             # 71
MSK_START = CTR_END                   # 71
MSK_END   = N_VAR                     # 78

# Centre bounds in standardised space (data ≈ N(0,1) → [-5, 5] covers 5σ)
CTR_LO, CTR_HI = -5.0, 5.0

# ════════════════════════════════════════════════════════════════════════════
# LOGGING
# ════════════════════════════════════════════════════════════════════════════
os.makedirs(RESULTS_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(RESULTS_DIR, "run.log"), mode="w"),
    ],
)
log = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# 1.  DATA ENGINEERING & FEATURE EXTRACTION
# ════════════════════════════════════════════════════════════════════════════

def _find_de_key(mat_dict: dict) -> str | None:
    """Return the 12 kHz Drive-End accelerometer key from a CWRU .mat dict."""
    for key in mat_dict:
        if key.startswith("__"):
            continue
        kl = key.lower()
        if "de_time" in kl:
            return key
    # Fallback: first non-meta key that holds a numeric array
    for key in mat_dict:
        if not key.startswith("__") and hasattr(mat_dict[key], "shape"):
            return key
    return None


def extract_features(segment: np.ndarray) -> np.ndarray:
    """
    Extract 7 time/frequency-domain features from a 1-D signal window.

    Features
    --------
    0  RMS              – energy level
    1  Kurtosis         – impulsiveness (Pearson; 3 for Gaussian)
    2  Skewness         – signal asymmetry
    3  Crest Factor     – peak / RMS, sensitive to impulse faults
    4  Peak-to-Peak     – dynamic range
    5  Std Dev          – dispersion
    6  Spectral Centroid– centre of mass of the power spectrum (Hz)
    """
    seg = segment.ravel().astype(np.float64)
    rms    = np.sqrt(np.mean(seg ** 2))
    kurt   = stats.kurtosis(seg, fisher=False)   # Pearson kurtosis
    skew   = stats.skew(seg)
    peak   = np.max(np.abs(seg))
    crest  = peak / (rms + 1e-12)
    p2p    = np.max(seg) - np.min(seg)
    std    = np.std(seg)

    freqs  = np.fft.rfftfreq(len(seg), d=1.0 / FS)
    mag    = np.abs(np.fft.rfft(seg))
    sc     = np.sum(freqs * mag) / (np.sum(mag) + 1e-12)

    return np.array([rms, kurt, skew, crest, p2p, std, sc], dtype=np.float64)


def load_mat_file(path: str, label: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Segment a single .mat file and return (feature_matrix, labels)."""
    try:
        mat = scipy.io.loadmat(path)
    except Exception as exc:
        log.warning("  Cannot load %s: %s", path, exc)
        return None, None

    key = _find_de_key(mat)
    if key is None:
        log.warning("  No DE key in %s – skipping.", path)
        return None, None

    signal   = mat[key].ravel().astype(np.float64)
    n_segs   = len(signal) // WINDOW_SIZE
    if n_segs == 0:
        return None, None

    feats = np.vstack([
        extract_features(signal[i * WINDOW_SIZE : (i + 1) * WINDOW_SIZE])
        for i in range(n_segs)
    ])
    labels = np.full(n_segs, label, dtype=np.int32)
    return feats, labels


def load_cwru_dataset(base_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the entire CWRU 12 kHz Drive-End dataset.

    Label map
    ---------
    0 → Normal
    1 → Inner Race (IR)
    2 → Ball (B)
    3 → Outer Race (OR)
    """
    all_X, all_y = [], []
    total_files  = 0

    # ── Normal ──────────────────────────────────────────────────────────────
    normal_dir = os.path.join(base_dir, "Normal")
    for path in sorted(glob.glob(os.path.join(normal_dir, "*.mat"))):
        X, y = load_mat_file(path, label=0)
        if X is not None:
            all_X.append(X); all_y.append(y)
            log.info("  [Normal] %-30s  %5d segments", os.path.basename(path), len(X))
            total_files += 1

    # ── 12k Drive-End faults ────────────────────────────────────────────────
    fault_dir = os.path.join(base_dir, "12k_Drive_End_Bearing_Fault_Data")
    fault_map = {"IR": 1, "B": 2, "OR": 3}
    for fault_type, label in fault_map.items():
        pattern = os.path.join(fault_dir, fault_type, "**", "*.mat")
        for path in sorted(glob.glob(pattern, recursive=True)):
            X, y = load_mat_file(path, label=label)
            if X is not None:
                all_X.append(X); all_y.append(y)
                log.info("  [%-2s]     %-30s  %5d segments",
                         fault_type, os.path.basename(path), len(X))
                total_files += 1

    if not all_X:
        raise RuntimeError(f"No data found under {base_dir!r}.")

    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)
    log.info("Loaded %d files → %d segments × %d raw features.",
             total_files, len(X_all), X_all.shape[1])
    return X_all, y_all


# ════════════════════════════════════════════════════════════════════════════
# CHROMOSOME HELPERS
# ════════════════════════════════════════════════════════════════════════════

def decode_chromosome(x: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    """
    Decode a chromosome into (K, centres, mask).

    K        : number of active clusters (int, MIN_K … MAX_K)
    centres  : (K, D) array in standardised feature space
    mask     : length-D boolean array of active features
    """
    K       = int(np.clip(x[0], MIN_K, MAX_K + 0.999))
    centres = x[CTR_START : CTR_START + K * D].reshape(K, D)
    raw_msk = x[MSK_START : MSK_END]
    mask    = raw_msk >= 0.5
    if not mask.any():                          # guarantee ≥1 active feature
        mask[np.argmax(raw_msk)] = True
    return K, centres, mask


def assign_clusters(X_act: np.ndarray, C_act: np.ndarray) -> np.ndarray:
    """
    Assign N points to the nearest of K centres (L2 distance).

    Parameters
    ----------
    X_act : (N, d) feature matrix restricted to active features
    C_act : (K, d) cluster centres restricted to active features

    Returns
    -------
    labels : (N,) int array of cluster indices
    """
    # Squared distances via broadcasting  (N, K)
    diff = X_act[:, None, :] - C_act[None, :, :]    # (N, K, d)
    dist = np.einsum("nkd,nkd->nk", diff, diff)
    return np.argmin(dist, axis=1)


# ════════════════════════════════════════════════════════════════════════════
# 2.  MULTI-OBJECTIVE PROBLEM
# ════════════════════════════════════════════════════════════════════════════

class MOCProblem(Problem):
    """
    Three-objective clustering problem.

    Objectives (all minimised)
    --------------------------
    f1  Compactness    : normalised Total Within-Cluster Sum of Squares
    f2  Connectedness  : fraction of kNN edges that cross cluster boundaries
    f3  Simplicity     : number of active features (L0 norm of mask)
    """

    def __init__(self, X_scaled: np.ndarray, knn_graph: np.ndarray):
        self.X  = X_scaled          # (N, D)
        self.G  = knn_graph         # (N, KNN_K) — neighbour indices
        self.N  = X_scaled.shape[0]

        xl = np.empty(N_VAR)
        xu = np.empty(N_VAR)
        xl[0]             = float(MIN_K)
        xu[0]             = float(MAX_K) + 0.999
        xl[CTR_START:CTR_END] = CTR_LO
        xu[CTR_START:CTR_END] = CTR_HI
        xl[MSK_START:MSK_END] = 0.0
        xu[MSK_START:MSK_END] = 1.0

        super().__init__(n_var=N_VAR, n_obj=3, n_ieq_constr=0, xl=xl, xu=xu)

    def _evaluate(self, X_pop: np.ndarray, out: dict, *args, **kwargs):
        n_pop = X_pop.shape[0]
        F     = np.full((n_pop, 3), 1e9)

        for i in range(n_pop):
            x          = X_pop[i]
            K, C, mask = decode_chromosome(x)

            if not mask.any() or K < MIN_K:
                continue

            X_act = self.X[:, mask]          # (N, d_active)
            C_act = C[:, mask]               # (K, d_active)
            labels = assign_clusters(X_act, C_act)

            # ── f1: Compactness (TWCSS / N / d_active) ──────────────────
            # Normalise by the number of active features so the scale of f1
            # is invariant to feature count. This creates a genuine 3-way
            # trade-off: using more features can yield tighter per-dimension
            # clusters (lower f1) at the cost of higher f3.
            twcss = 0.0
            for k in range(K):
                pts = X_act[labels == k]
                if len(pts):
                    delta = pts - C_act[k]
                    twcss += np.einsum("nd,nd->", delta, delta)
            d_act = float(mask.sum())
            f1 = twcss / (self.N * d_act)

            # ── f2: Connectedness (kNN cross-cluster penalty / N*K) ──────
            # For each point, count how many of its KNN_K neighbours are
            # in a different cluster.
            # Vectorised: (N, KNN_K) → bool array of mismatches
            neigh_labels = labels[self.G]                 # (N, KNN_K)
            mismatch     = labels[:, None] != neigh_labels  # (N, KNN_K)
            f2 = mismatch.sum() / (self.N * KNN_K)

            # ── f3: Simplicity (active feature count) ────────────────────
            f3 = float(mask.sum())

            F[i] = [f1, f2, f3]

        out["F"] = F


# ════════════════════════════════════════════════════════════════════════════
# 3.  HYBRID INITIALISATION
# ════════════════════════════════════════════════════════════════════════════

class HybridSampling(Sampling):
    """
    Initial population (size = POP_SIZE) split into three equal thirds:

    1/3  KMeans-seeded : run K-Means for varying K; use resulting centroids.
    1/3  Data-point    : randomly pick K existing samples as initial centres.
    1/3  Pure random   : uniformly sample centres within the feature bounds.
    """

    def __init__(self, X_scaled: np.ndarray):
        super().__init__()
        self.X = X_scaled

    def _random_mask(self, rng: np.random.Generator, min_active: int = 2) -> np.ndarray:
        while True:
            m = (rng.random(D) > 0.5).astype(float)
            if m.sum() >= min_active:
                return m

    def _make_chromosome(
        self, rng: np.random.Generator, K: int, centres: np.ndarray
    ) -> np.ndarray:
        """Pack K, centres, and a random mask into a chromosome vector."""
        x = np.zeros(N_VAR)
        x[0]                    = float(K) + rng.random() * 0.999
        # Store first K centres; fill remaining MAX_K-K slots randomly
        x[CTR_START : CTR_START + K * D]       = np.clip(centres, CTR_LO, CTR_HI).ravel()
        x[CTR_START + K * D : CTR_END]         = rng.uniform(CTR_LO, CTR_HI, (MAX_K - K) * D)
        x[MSK_START : MSK_END]                 = self._random_mask(rng, min_active=2)
        return x

    def _do(self, problem, n_samples: int, **kwargs) -> np.ndarray:
        rng  = np.random.default_rng(RANDOM_SEED)
        n1   = n_samples // 3
        n2   = n_samples // 3
        n3   = n_samples - n1 - n2
        pop  = np.zeros((n_samples, N_VAR))
        idx  = 0

        # ── Segment 1: KMeans-seeded ─────────────────────────────────────
        log.info("  Hybrid init: KMeans-seeded (%d chromosomes)…", n1)
        for _ in range(n1):
            K  = int(rng.integers(MIN_K, MAX_K + 1))
            km = KMeans(n_clusters=K, n_init=3, max_iter=50,
                        random_state=int(rng.integers(1_000_000)))
            km.fit(self.X)
            pop[idx] = self._make_chromosome(rng, K, km.cluster_centers_)
            idx += 1

        # ── Segment 2: Random data points as centres ─────────────────────
        log.info("  Hybrid init: data-point seeded (%d chromosomes)…", n2)
        for _ in range(n2):
            K       = int(rng.integers(MIN_K, MAX_K + 1))
            chosen  = rng.choice(len(self.X), size=K, replace=False)
            centres = self.X[chosen]
            pop[idx] = self._make_chromosome(rng, K, centres)
            idx += 1

        # ── Segment 3: Pure random ───────────────────────────────────────
        log.info("  Hybrid init: pure random (%d chromosomes)…", n3)
        for _ in range(n3):
            K       = int(rng.integers(MIN_K, MAX_K + 1))
            centres = rng.uniform(CTR_LO, CTR_HI, (K, D))
            pop[idx] = self._make_chromosome(rng, K, centres)
            idx += 1

        return pop


# ════════════════════════════════════════════════════════════════════════════
# 4.  GENETIC OPERATORS
# ════════════════════════════════════════════════════════════════════════════

# ── SBX helper (scalar, called per variable) ─────────────────────────────────
def _sbx_pair(y1: float, y2: float, yl: float, yu: float, eta: float,
               rng_val: float) -> tuple[float, float]:
    """
    Simulated Binary Crossover for a single variable pair.
    Returns two offspring values (c1, c2).
    """
    if abs(y1 - y2) < 1e-10:
        return y1, y2
    a, b   = min(y1, y2), max(y1, y2)
    spread = b - a

    # beta-Q for c1
    beta1  = 1.0 + 2.0 * (a - yl) / spread
    alpha1 = 2.0 - beta1 ** (-(eta + 1))
    if rng_val <= 1.0 / alpha1:
        betaq1 = (rng_val * alpha1) ** (1.0 / (eta + 1))
    else:
        betaq1 = (1.0 / (2.0 - rng_val * alpha1)) ** (1.0 / (eta + 1))

    # beta-Q for c2
    beta2  = 1.0 + 2.0 * (yu - b) / spread
    alpha2 = 2.0 - beta2 ** (-(eta + 1))
    if rng_val <= 1.0 / alpha2:
        betaq2 = (rng_val * alpha2) ** (1.0 / (eta + 1))
    else:
        betaq2 = (1.0 / (2.0 - rng_val * alpha2)) ** (1.0 / (eta + 1))

    c1 = np.clip(0.5 * (a + b - betaq1 * spread), yl, yu)
    c2 = np.clip(0.5 * (a + b + betaq2 * spread), yl, yu)
    return c1, c2


def _point_crossover(m1: np.ndarray, m2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """1-point crossover on binary feature masks."""
    if len(m1) < 2:
        return m1.copy(), m2.copy()
    pt = np.random.randint(1, len(m1))
    c1 = np.concatenate([m1[:pt], m2[pt:]])
    c2 = np.concatenate([m2[:pt], m1[pt:]])
    return c1, c2


class MOCCrossover(Crossover):
    """
    Composite crossover operator:
      •  SBX on the K-value and cluster centres (continuous part).
      •  1-point crossover on the binary feature mask.
    """

    def __init__(self, eta: float = 15.0, prob: float = 0.9):
        super().__init__(2, 2)          # 2 parents → 2 offspring
        self.eta  = eta
        self.prob = prob

    def _do(self, problem, X: np.ndarray, **kwargs) -> np.ndarray:
        # X  : (2, n_matings, n_var)
        _, n_matings, n_var = X.shape
        Y = X.copy()

        xl = problem.xl
        xu = problem.xu

        for i in range(n_matings):
            if np.random.rand() > self.prob:
                continue

            p1, p2 = X[0, i].copy(), X[1, i].copy()
            o1, o2 = p1.copy(), p2.copy()

            # ── SBX on continuous part (K + all centres) ─────────────────
            for j in range(CTR_END):
                r = np.random.rand()
                if r <= 0.5:        # perform crossover on this variable
                    o1[j], o2[j] = _sbx_pair(
                        p1[j], p2[j], xl[j], xu[j], self.eta, np.random.rand()
                    )

            # ── 1-point crossover on feature mask ────────────────────────
            o1[MSK_START:], o2[MSK_START:] = _point_crossover(
                p1[MSK_START:], p2[MSK_START:]
            )

            Y[0, i] = o1
            Y[1, i] = o2

        return Y


class MOCMutation(Mutation):
    """
    Composite mutation operator:
      •  Polynomial mutation on K and cluster centres.
      •  Bit-flip mutation on the binary feature mask.
      •  Structural mutation: randomly increment / decrement K (prob_struct).
    """

    def __init__(self, eta: float = 20.0, prob_var: float = 1.0 / N_VAR,
                 prob_flip: float = 0.15, prob_struct: float = 0.05):
        super().__init__()
        self.eta         = eta
        self.prob_var    = prob_var
        self.prob_flip   = prob_flip
        self.prob_struct = prob_struct

    def _do(self, problem, X: np.ndarray, **kwargs) -> np.ndarray:
        Y   = X.copy()
        xl  = problem.xl
        xu  = problem.xu

        for i in range(len(Y)):
            # ── Polynomial mutation on continuous part ────────────────────
            for j in range(CTR_END):
                if np.random.rand() >= self.prob_var:
                    continue
                y   = Y[i, j]
                yl  = xl[j]; yu = xu[j]
                rng = np.random.rand()
                eta = self.eta
                if rng < 0.5:
                    xy      = 1.0 - (y - yl) / (yu - yl + 1e-12)
                    val     = 2.0 * rng + (1.0 - 2.0 * rng) * xy ** (eta + 1)
                    delta_q = val ** (1.0 / (eta + 1)) - 1.0
                else:
                    xy      = 1.0 - (yu - y) / (yu - yl + 1e-12)
                    val     = 2.0 * (1.0 - rng) + 2.0 * (rng - 0.5) * xy ** (eta + 1)
                    delta_q = 1.0 - val ** (1.0 / (eta + 1))
                Y[i, j] = np.clip(y + delta_q * (yu - yl), yl, yu)

            # ── Bit-flip on feature mask ──────────────────────────────────
            for j in range(MSK_START, N_VAR):
                if np.random.rand() < self.prob_flip:
                    Y[i, j] = 1.0 - Y[i, j]

            # Guarantee at least one active feature
            if (Y[i, MSK_START:] < 0.5).all():
                Y[i, MSK_START + np.random.randint(D)] = 1.0

            # ── Structural mutation (add / remove a cluster) ──────────────
            if np.random.rand() < self.prob_struct:
                K_cur  = int(np.clip(Y[i, 0], MIN_K, MAX_K))
                K_new  = int(np.clip(K_cur + np.random.choice([-1, 1]),
                                     MIN_K, MAX_K))
                Y[i, 0] = float(K_new) + np.random.rand() * 0.999

        return Y


# ════════════════════════════════════════════════════════════════════════════
# 5.  PARETO FRONT ANALYSIS
# ════════════════════════════════════════════════════════════════════════════

def find_knee_point(F: np.ndarray) -> int:
    """
    Knee-point detection via minimum distance to the ideal point
    in normalised objective space (after min-max scaling per objective).
    Equivalent to the Chebyshev minimum-achievement-scalarisation
    with equal weights.
    """
    F_min = F.min(axis=0)
    F_max = F.max(axis=0)
    denom = F_max - F_min + 1e-12
    F_norm = (F - F_min) / denom           # ∈ [0,1]^3
    dist   = np.linalg.norm(F_norm, axis=1)
    return int(np.argmin(dist))


def count_active(x: np.ndarray) -> int:
    return int((x[MSK_START:MSK_END] >= 0.5).sum() or 1)


# ════════════════════════════════════════════════════════════════════════════
# 6.  VALIDATION
# ════════════════════════════════════════════════════════════════════════════

def evaluate_solution(
    x: np.ndarray, X_scaled: np.ndarray, y_true: np.ndarray
) -> tuple[float, float, np.ndarray]:
    """
    Compute ARI and Silhouette Score for one Pareto solution.

    Returns
    -------
    ari   : Adjusted Rand Index against ground-truth bearing labels
    sil   : Silhouette Score (sampled if N > 5 000)
    mask  : boolean array of active features
    """
    K, C, mask = decode_chromosome(x)
    X_act      = X_scaled[:, mask]
    C_act      = C[:, mask]

    if X_act.shape[1] == 0:
        return 0.0, -1.0, mask

    labels = assign_clusters(X_act, C_act)
    if len(np.unique(labels)) < 2:
        return 0.0, -1.0, mask

    ari = adjusted_rand_score(y_true, labels)

    N = len(X_act)
    if N > 5_000:
        idx = np.random.choice(N, 5_000, replace=False)
        sil = silhouette_score(X_act[idx], labels[idx])
    else:
        sil = silhouette_score(X_act, labels)

    return ari, sil, mask


# ════════════════════════════════════════════════════════════════════════════
# 7.  VISUALISATION
# ════════════════════════════════════════════════════════════════════════════

def _save(fig: plt.Figure, name: str):
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved: %s", path)


def plot_pareto_3d(F: np.ndarray, knee_idx: int):
    fig = plt.figure(figsize=(12, 9))
    ax  = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        F[:, 0], F[:, 1], F[:, 2],
        c=F[:, 2], cmap="viridis_r", s=55, alpha=0.75, edgecolors="none",
    )
    ax.scatter(
        F[knee_idx, 0], F[knee_idx, 1], F[knee_idx, 2],
        color="red", s=280, marker="*", zorder=10, label="Knee Point",
    )

    plt.colorbar(sc, ax=ax, shrink=0.6, label="Simplicity (# active features)")
    ax.set_xlabel("Compactness (TWCSS/N)", fontsize=11, labelpad=8)
    ax.set_ylabel("Connectedness", fontsize=11, labelpad=8)
    ax.set_zlabel("Simplicity", fontsize=11, labelpad=8)
    ax.set_title(
        "3-D Pareto Front — MOC-FS Bearing Fault Diagnosis",
        fontsize=14, fontweight="bold",
    )
    ax.legend(fontsize=11)
    fig.tight_layout()
    _save(fig, "01_pareto_3d.png")


def plot_feature_importance(pareto_X: np.ndarray):
    counts = np.zeros(D)
    for x in pareto_X:
        mask = x[MSK_START:MSK_END] >= 0.5
        if not mask.any():
            mask[np.argmax(x[MSK_START:MSK_END])] = True
        counts += mask.astype(float)

    freq   = counts / len(pareto_X) * 100
    colours = cm.RdYlGn(freq / 100)

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(FEATURE_NAMES, freq, color=colours, edgecolor="black", linewidth=0.8)
    for bar, f in zip(bars, freq):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{f:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold",
        )
    ax.axhline(50, color="grey", linestyle="--", linewidth=0.9, label="50 % threshold")
    ax.set_ylim(0, 118)
    ax.set_xlabel("Feature", fontsize=12)
    ax.set_ylabel("Activation frequency (%)", fontsize=12)
    ax.set_title(
        "Feature Importance across Pareto-Optimal Solutions",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10)
    fig.tight_layout()
    _save(fig, "02_feature_importance.png")


def _project_2d(X_act: np.ndarray, method: str = "pca") -> np.ndarray:
    """PCA or t-SNE 2-D projection (subsample for t-SNE)."""
    if method == "tsne":
        max_pts = 3_000
        if len(X_act) > max_pts:
            idx   = np.random.choice(len(X_act), max_pts, replace=False)
            X_sub = X_act[idx]
        else:
            idx   = np.arange(len(X_act))
            X_sub = X_act
        # PCA pre-reduction to speed up t-SNE
        n_comp = min(50, X_sub.shape[1])
        if n_comp > 1:
            X_sub = PCA(n_components=n_comp).fit_transform(X_sub)
        proj = TSNE(n_components=2, random_state=RANDOM_SEED,
                    perplexity=30, max_iter=1_000).fit_transform(X_sub)
        return proj, idx
    else:
        proj = PCA(n_components=2).fit_transform(X_act)
        return proj, np.arange(len(X_act))


def plot_cluster_map(
    X_scaled: np.ndarray, labels: np.ndarray, y_true: np.ndarray,
    title: str, fname: str, method: str = "pca",
):
    proj, idx = _project_2d(X_scaled, method)
    labels_   = labels[idx]
    y_        = y_true[idx]

    fig, axes = plt.subplots(1, 2, figsize=(17, 7))

    sc1 = axes[0].scatter(proj[:, 0], proj[:, 1], c=labels_,
                          cmap="tab10", s=4, alpha=0.55)
    axes[0].set_title("Predicted Clusters", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Component 1"); axes[0].set_ylabel("Component 2")
    plt.colorbar(sc1, ax=axes[0], label="Cluster ID")

    sc2 = axes[1].scatter(proj[:, 0], proj[:, 1], c=y_,
                          cmap="Set1", vmin=0, vmax=3, s=4, alpha=0.55)
    axes[1].set_title("Ground-Truth Classes", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Component 1"); axes[1].set_ylabel("Component 2")
    cbar = plt.colorbar(sc2, ax=axes[1], ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(CLASS_NAMES, fontsize=9)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, fname)


def plot_decision_advantage(info: dict):
    """
    Side-by-side bar chart contrasting the 'Simple' and 'Compact' solutions
    across ARI, Silhouette, feature count, and cluster count.
    """
    metrics = ["ARI", "Silhouette", "# Features", "# Clusters"]
    colours = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, (name, sol) in zip(axes, info.items()):
        vals = [sol["ari"], max(sol["sil"], 0.0), sol["n_feat"], sol["K"]]
        bars = ax.barh(metrics, vals, color=colours, edgecolor="black", height=0.45)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}" if isinstance(v, float) else str(v),
                va="center", fontsize=11,
            )
        ax.set_xlim(0, max(vals) * 1.30 + 0.05)
        ax.set_title(f"'{name}' Solution", fontsize=13, fontweight="bold")
        ax.set_xlabel("Value", fontsize=11)

    feat_str = {
        k: [FEATURE_NAMES[i] for i in range(D) if info[k]["mask"][i]]
        for k in info
    }
    fig.suptitle(
        "Decision-Making Advantage: Simple (low-cost sensor) vs Compact (root-cause analysis)\n"
        f"Simple active: {feat_str['Simple']}     "
        f"Compact active: {feat_str['Compact']}",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, "05_decision_advantage.png")


def plot_convergence(gen_stats: list[dict]):
    """Plot best/mean/worst Pareto hypervolume proxy across generations."""
    if not gen_stats:
        return
    gens = [s["gen"] for s in gen_stats]
    bst  = [s["best_f1"] for s in gen_stats]
    mn   = [s["mean_f1"] for s in gen_stats]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gens, bst, label="Best Compactness", linewidth=2)
    ax.plot(gens, mn,  label="Mean Compactness", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Compactness (f1)", fontsize=12)
    ax.set_title("NSGA-II Convergence — Compactness Objective", fontsize=13,
                 fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.35)
    fig.tight_layout()
    _save(fig, "00_convergence.png")


# ════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ════════════════════════════════════════════════════════════════════════════

def main(base_dir: str):
    t_start = time.time()
    np.random.seed(RANDOM_SEED)

    log.info("=" * 70)
    log.info("  MOC-FS  — Multi-Objective Clustering with Feature Sparsity")
    log.info("  CWRU Bearing Dataset | 12 kHz Drive End")
    log.info("  pymoo NSGA-II  pop=%d  gen=%d", POP_SIZE, N_GEN)
    log.info("=" * 70)

    # ── 1. Load & extract ────────────────────────────────────────────────────
    log.info("\n[1] Loading CWRU dataset and extracting features…")
    X_raw, y_true = load_cwru_dataset(base_dir)

    unique, counts = np.unique(y_true, return_counts=True)
    for u, c in zip(unique, counts):
        log.info("  Class %d (%s): %d segments", u, CLASS_NAMES[u], c)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    log.info("Standardised feature matrix: %s", X_scaled.shape)

    # ── 2. kNN graph ─────────────────────────────────────────────────────────
    log.info("\n[2] Building kNN graph (k=%d)…", KNN_K)
    t_knn = time.time()
    nn = NearestNeighbors(n_neighbors=KNN_K + 1, algorithm="ball_tree", n_jobs=-1)
    nn.fit(X_scaled)
    _, knn_idx = nn.kneighbors(X_scaled)
    knn_graph  = knn_idx[:, 1:]            # exclude self  → (N, KNN_K)
    log.info("  kNN graph built in %.2f s.", time.time() - t_knn)

    # ── 3. NSGA-II optimisation ──────────────────────────────────────────────
    log.info("\n[3] Running NSGA-II optimisation…")
    problem   = MOCProblem(X_scaled, knn_graph)
    sampling  = HybridSampling(X_scaled)
    crossover = MOCCrossover(eta=15.0, prob=0.9)
    mutation  = MOCMutation(eta=20.0, prob_var=2.0 / N_VAR,
                            prob_flip=0.15, prob_struct=0.05)

    algorithm = NSGA2(
        pop_size=POP_SIZE,
        sampling=sampling,
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=True,
    )
    termination = get_termination("n_gen", N_GEN)

    t_opt = time.time()
    res = minimize(
        problem, algorithm, termination,
        seed=RANDOM_SEED, verbose=True, save_history=False,
    )
    opt_time = time.time() - t_opt
    log.info("Optimisation finished in %.1f s (%.1f min).", opt_time, opt_time / 60)

    # ── 4. Pareto front analysis ─────────────────────────────────────────────
    log.info("\n[4] Analysing Pareto front…")
    pareto_F = res.F         # (n_pareto, 3)
    pareto_X = res.X         # (n_pareto, N_VAR)
    n_pareto = len(pareto_F)
    log.info("  Pareto-optimal solutions : %d", n_pareto)
    log.info("  f1 Compactness   : [%.4f, %.4f]", pareto_F[:, 0].min(), pareto_F[:, 0].max())
    log.info("  f2 Connectedness : [%.4f, %.4f]", pareto_F[:, 1].min(), pareto_F[:, 1].max())
    log.info("  f3 Simplicity    : [%.1f, %.1f]",  pareto_F[:, 2].min(), pareto_F[:, 2].max())

    knee_idx = find_knee_point(pareto_F)
    K_knee, C_knee, mask_knee = decode_chromosome(pareto_X[knee_idx])
    log.info("  Knee point [idx=%d]: K=%d, active=%s",
             knee_idx, K_knee, [str(f) for f in FEATURE_NAMES[mask_knee]])

    # ── 5. Validation metrics ────────────────────────────────────────────────
    log.info("\n[5] Computing ARI / Silhouette for all %d Pareto solutions…", n_pareto)
    ari_arr = np.zeros(n_pareto)
    sil_arr = np.zeros(n_pareto)
    for i, x in enumerate(pareto_X):
        ari_arr[i], sil_arr[i], _ = evaluate_solution(x, X_scaled, y_true)

    log.info("  ARI   — mean %.4f | max %.4f | knee %.4f",
             ari_arr.mean(), ari_arr.max(), ari_arr[knee_idx])
    log.info("  Sil   — mean %.4f | max %.4f | knee %.4f",
             sil_arr.mean(), sil_arr.max(), sil_arr[knee_idx])

    # ── Identify 'Simple' and 'Compact' representative solutions ─────────────
    n_active = np.array([count_active(x) for x in pareto_X])

    # 'Simple': fewest active features; break ties by highest ARI
    simple_cands = np.where(n_active <= 3)[0]
    if len(simple_cands) == 0:
        simple_cands = np.argsort(n_active)[:max(1, n_pareto // 5)]
    simple_idx = simple_cands[np.argmax(ari_arr[simple_cands])]

    # 'Compact': lowest f1 (compactness objective)
    compact_idx = int(np.argmin(pareto_F[:, 0]))

    K_sim, _, mask_sim = decode_chromosome(pareto_X[simple_idx])
    K_cmp, _, mask_cmp = decode_chromosome(pareto_X[compact_idx])

    log.info("\n  'Simple'  solution [idx=%d]: K=%d, n_feat=%d, "
             "ARI=%.4f, active=%s",
             simple_idx, K_sim, n_active[simple_idx],
             ari_arr[simple_idx], [str(f) for f in FEATURE_NAMES[mask_sim]])
    log.info("  'Compact' solution [idx=%d]: K=%d, n_feat=%d, "
             "ARI=%.4f, active=%s",
             compact_idx, K_cmp, n_active[compact_idx],
             ari_arr[compact_idx], [str(f) for f in FEATURE_NAMES[mask_cmp]])

    # ── 6. Visualisations ────────────────────────────────────────────────────
    log.info("\n[6] Generating visualisations…")

    # 6a. 3-D Pareto plot
    plot_pareto_3d(pareto_F, knee_idx)

    # 6b. Feature importance bar chart
    plot_feature_importance(pareto_X)

    # 6c. Cluster map — Simple solution (PCA)
    K_s, C_s, mask_s = decode_chromosome(pareto_X[simple_idx])
    labels_s = assign_clusters(X_scaled[:, mask_s], C_s[:, mask_s])
    plot_cluster_map(
        X_scaled, labels_s, y_true,
        title=(f"Simple Solution  (K={K_s}, features={[str(f) for f in FEATURE_NAMES[mask_s]]})\n"
               f"ARI={ari_arr[simple_idx]:.4f}   Sil={sil_arr[simple_idx]:.4f}"),
        fname="03_cluster_map_simple_pca.png",
        method="pca",
    )

    # 6d. Cluster map — Compact solution (t-SNE)
    K_c, C_c, mask_c = decode_chromosome(pareto_X[compact_idx])
    labels_c = assign_clusters(X_scaled[:, mask_c], C_c[:, mask_c])
    plot_cluster_map(
        X_scaled, labels_c, y_true,
        title=(f"Compact Solution  (K={K_c}, n_feat={n_active[compact_idx]})\n"
               f"ARI={ari_arr[compact_idx]:.4f}   Sil={sil_arr[compact_idx]:.4f}"),
        fname="04_cluster_map_compact_tsne.png",
        method="tsne",
    )

    # 6e. Decision-making advantage panel
    da_info = {
        "Simple": {
            "ari": ari_arr[simple_idx], "sil": sil_arr[simple_idx],
            "n_feat": int(n_active[simple_idx]), "K": K_s, "mask": mask_s,
        },
        "Compact": {
            "ari": ari_arr[compact_idx], "sil": sil_arr[compact_idx],
            "n_feat": int(n_active[compact_idx]), "K": K_c, "mask": mask_c,
        },
    }
    plot_decision_advantage(da_info)

    # ── 7. Save results JSON ─────────────────────────────────────────────────
    summary = {
        "dataset": {"n_segments": int(len(X_raw)), "n_features": D,
                    "class_dist": {CLASS_NAMES[u]: int(c)
                                   for u, c in zip(unique, counts)}},
        "nsga2": {"pop_size": POP_SIZE, "n_gen": N_GEN,
                  "n_pareto": n_pareto, "opt_time_s": round(opt_time, 2)},
        "metrics": {
            "ari_mean": round(float(ari_arr.mean()), 4),
            "ari_max":  round(float(ari_arr.max()),  4),
            "sil_mean": round(float(sil_arr.mean()), 4),
            "sil_max":  round(float(sil_arr.max()),  4),
        },
        "knee":    {"K": K_knee, "ari": round(float(ari_arr[knee_idx]), 4),
                    "sil": round(float(sil_arr[knee_idx]), 4),
                    "active": [str(f) for f in FEATURE_NAMES[mask_knee]]},
        "simple":  {"K": K_s, "n_feat": int(n_active[simple_idx]),
                    "ari": round(float(ari_arr[simple_idx]), 4),
                    "sil": round(float(sil_arr[simple_idx]), 4),
                    "active": [str(f) for f in FEATURE_NAMES[mask_s]]},
        "compact": {"K": K_c, "n_feat": int(n_active[compact_idx]),
                    "ari": round(float(ari_arr[compact_idx]), 4),
                    "sil": round(float(sil_arr[compact_idx]), 4),
                    "active": [str(f) for f in FEATURE_NAMES[mask_c]]},
        "total_time_s": round(time.time() - t_start, 2),
    }
    json_path = os.path.join(RESULTS_DIR, "summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("  Summary saved: %s", json_path)

    # ── Save Pareto arrays for interactive visualisation ──────────────────
    np.save(os.path.join(RESULTS_DIR, "pareto_F.npy"),   pareto_F)
    np.save(os.path.join(RESULTS_DIR, "pareto_X.npy"),   pareto_X)
    np.save(os.path.join(RESULTS_DIR, "pareto_ari.npy"), ari_arr)
    np.save(os.path.join(RESULTS_DIR, "pareto_sil.npy"), sil_arr)
    np.save(os.path.join(RESULTS_DIR, "n_active.npy"),   n_active)
    log.info("  Pareto arrays saved to %s/", RESULTS_DIR)

    # ── Final print ──────────────────────────────────────────────────────────
    total = time.time() - t_start
    log.info("\n%s", "=" * 70)
    log.info("  FINAL RESULTS")
    log.info("=" * 70)
    log.info("  Segments  : %d × %d features", len(X_raw), D)
    log.info("  Pareto    : %d solutions", n_pareto)
    log.info("  Best ARI  : %.4f", ari_arr.max())
    log.info("  Best Sil  : %.4f", sil_arr.max())
    log.info("  Knee ARI  : %.4f  active=%s", ari_arr[knee_idx],
             [str(f) for f in FEATURE_NAMES[mask_knee]])
    log.info("")
    log.info("  DECISION-MAKING ADVANTAGE")
    log.info("  ┌─────────────────────────────────────────────────────────")
    log.info("  │  Simple  (%d feat)  ARI=%.4f  → %s",
             n_active[simple_idx], ari_arr[simple_idx],
             [str(f) for f in FEATURE_NAMES[mask_s]])
    log.info("  │  Use for: low-cost sensor deployments, edge devices")
    log.info("  ├─────────────────────────────────────────────────────────")
    log.info("  │  Compact (%d feat)  ARI=%.4f  → %s",
             n_active[compact_idx], ari_arr[compact_idx],
             [str(f) for f in FEATURE_NAMES[mask_c]])
    log.info("  │  Use for: deep root-cause analysis, full-sensor rigs")
    log.info("  └─────────────────────────────────────────────────────────")
    log.info("  Total runtime : %.1f s (%.1f min)", total, total / 60)
    log.info("  Results dir   : %s", os.path.abspath(RESULTS_DIR))
    log.info("=" * 70)

    return summary


# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MOC-FS Bearing Fault Diagnosis")
    parser.add_argument(
        "--data-dir", type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "CWRU-dataset"),
        help="Path to the CWRU-dataset root directory",
    )
    args = parser.parse_args()
    main(args.data_dir)
