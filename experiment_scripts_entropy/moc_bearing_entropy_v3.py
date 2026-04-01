"""
moc_bearing_v2.py — FIXED & ENHANCED pipeline
===============================================
Fixes applied vs v1
--------------------
FIX-1  Per-load-condition normalisation
        StandardScaler is fit *within* each load condition (0/1/2/3 HP)
        before merging.  This removes amplitude/speed artefacts that cause
        Normal@3HP to appear similar to faults@0HP.

FIX-2  Class balancing
        Outer-Race fault data (~3 324 segments) is randomly under-sampled to
        match the mean fault-class count (~1 860 segments).

FIX-3  Fault-severity tracking
        A 16-class label vector (fault_type × fault_size) is built so we can
        compute a severity-aware ARI that reveals sub-cluster structure.

ENHANCEMENTS
------------
ENH-1  Population 150→200, Generations 100→200, for a denser Pareto front.
ENH-2  PopulationHistoryCallback: snapshots at key generations → used for
        the population-evolution animation in make_presentation.py.
ENH-3  Per-generation convergence stats saved (gen_stats.json).
ENH-4  All arrays needed by make_presentation.py are written to disk.
"""

import os, glob, time, warnings, argparse, json, logging
import sys
import numpy as np
import scipy.io
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.callback import Callback
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.operators.selection.rnd import RandomSelection
from pymoo.util.ref_dirs import get_reference_directions
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Import LHFiD implementation from sibling repository folder.
LHFID_REPO_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "LHFiD"
)
if LHFID_REPO_DIR not in sys.path:
    sys.path.append(LHFID_REPO_DIR)

from LHFiD import LHFID

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════
WINDOW_SIZE = 1024
FS          = 12_000
D           = 7
MAX_K       = 10
MIN_K       = 2
KNN_K       = 10
POP_SIZE    = 200          # ENH-1
N_GEN       = 200          # ENH-1
RANDOM_SEED = 42
RESULTS_DIR = "moc_results_entropy"

# Generations at which to save full population snapshots (for animation)
SNAPSHOT_GENS = {1, 5, 10, 20, 35, 50, 75, 100, 150, 200}  # ENH-2

FEATURE_NAMES = np.array([
    "RMS", "Kurtosis", "Skewness",
    "Crest Factor", "Peak-to-Peak", "Std Dev", "Spectral Centroid",
])
CLASS_NAMES = ["Normal", "Inner Race", "Ball", "Outer Race"]

# Chromosome layout  (78 variables, same encoding as v1)
N_VAR     = 1 + MAX_K * D + D   # 78
CTR_START = 1
CTR_END   = 1 + MAX_K * D       # 71
MSK_START = CTR_END
MSK_END   = N_VAR
CTR_LO, CTR_HI = -5.0, 5.0

# ════════════════════════════════════════════════════════════════════════════
# LOGGING
# ════════════════════════════════════════════════════════════════════════════
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "history"), exist_ok=True)
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
# 1.  DATA ENGINEERING  (FIX-1, FIX-2, FIX-3)
# ════════════════════════════════════════════════════════════════════════════

def _find_de_key(mat_dict: dict):
    for key in mat_dict:
        if not key.startswith("__") and "de_time" in key.lower():
            return key
    for key in mat_dict:
        if not key.startswith("__") and hasattr(mat_dict[key], "shape"):
            return key
    return None


def extract_features(segment: np.ndarray) -> np.ndarray:
    seg  = segment.ravel().astype(np.float64)
    rms  = np.sqrt(np.mean(seg ** 2))
    kurt = stats.kurtosis(seg, fisher=False)
    skew = stats.skew(seg)
    peak = np.max(np.abs(seg))
    cf   = peak / (rms + 1e-12)
    p2p  = seg.max() - seg.min()
    std  = seg.std()
    freqs = np.fft.rfftfreq(len(seg), d=1.0 / FS)
    mag   = np.abs(np.fft.rfft(seg))
    sc    = np.sum(freqs * mag) / (np.sum(mag) + 1e-12)
    return np.array([rms, kurt, skew, cf, p2p, std, sc], dtype=np.float64)


SEVERITY_DIR_MAP = {"007": 1, "014": 2, "021": 3, "028": 4}


def _load_one(path: str, fault_type: int, severity: int, load_id: int):
    """Load one .mat → (feats, y_type, y_severity, y_load) or Nones."""
    try:
        mat = scipy.io.loadmat(path)
    except Exception as e:
        log.warning("Cannot load %s: %s", path, e)
        return None, None, None, None
    key = _find_de_key(mat)
    if key is None:
        return None, None, None, None
    sig    = mat[key].ravel().astype(np.float64)
    n_segs = len(sig) // WINDOW_SIZE
    if n_segs == 0:
        return None, None, None, None
    feats = np.vstack([
        extract_features(sig[i * WINDOW_SIZE:(i + 1) * WINDOW_SIZE])
        for i in range(n_segs)
    ])
    return (feats,
            np.full(n_segs, fault_type,  dtype=np.int32),
            np.full(n_segs, severity,    dtype=np.int32),
            np.full(n_segs, load_id,     dtype=np.int32))


def _load_id_from_filename(path: str) -> int:
    """Extract load condition (0/1/2/3) from filename suffix like '118_0.mat'."""
    try:
        return int(os.path.splitext(os.path.basename(path))[0].rsplit("_", 1)[-1])
    except (ValueError, IndexError):
        return -1


def load_cwru_dataset_v2(base_dir: str):
    """
    Returns
    -------
    X_raw      : (N, D) float64 — raw (not yet scaled) features
    y_type     : (N,)   int32   — 0=Normal, 1=IR, 2=Ball, 3=OR
    y_severity : (N,)   int32   — 0=no-fault, 1=7mil, 2=14mil, 3=21mil, 4=28mil
    y_load     : (N,)   int32   — 0/1/2/3 HP load condition
    """
    Xs, yt, ys, yl = [], [], [], []

    # ── Normal ──────────────────────────────────────────────────────────────
    normal_dir = os.path.join(base_dir, "Normal")
    for path in sorted(glob.glob(os.path.join(normal_dir, "*.mat"))):
        load_id = _load_id_from_filename(path)
        f, t, v, l = _load_one(path, 0, 0, load_id)
        if f is not None:
            Xs.append(f); yt.append(t); ys.append(v); yl.append(l)
            log.info("  [Normal]  %-28s  load=%d  %5d seg",
                     os.path.basename(path), load_id, len(f))

    # ── 12k Drive-End faults ────────────────────────────────────────────────
    fault_dir = os.path.join(base_dir, "12k_Drive_End_Bearing_Fault_Data")
    fault_map = {"IR": 1, "B": 2, "OR": 3}
    for fname, fault_type in fault_map.items():
        for sev_dir in sorted(glob.glob(
                os.path.join(fault_dir, fname, "*"))):
            sev_key  = os.path.basename(sev_dir)
            severity = SEVERITY_DIR_MAP.get(sev_key, 0)
            for path in sorted(glob.glob(os.path.join(sev_dir, "**", "*.mat"), recursive=True)):
                load_id = _load_id_from_filename(path)
                f, t, v, l = _load_one(path, fault_type, severity, load_id)
                if f is not None:
                    Xs.append(f); yt.append(t); ys.append(v); yl.append(l)
                    log.info("  [%-2s %s]  %-28s  load=%d  %5d seg",
                             fname, sev_key, os.path.basename(path),
                             load_id, len(f))

    X_raw      = np.vstack(Xs)
    y_type     = np.concatenate(yt)
    y_severity = np.concatenate(ys)
    y_load     = np.concatenate(yl)
    log.info("Total raw: %d segments × %d features.", len(X_raw), D)
    return X_raw, y_type, y_severity, y_load


# ── FIX-1: Per-load normalisation ────────────────────────────────────────────
def normalize_by_load(X_raw: np.ndarray, y_load: np.ndarray):
    """
    Fit a separate StandardScaler within each load condition, then merge.
    Removes load-induced amplitude shifts so the scaler does NOT absorb
    the variance caused by different operating conditions.
    """
    X_norm = np.empty_like(X_raw, dtype=np.float64)
    for lc in np.unique(y_load):
        if lc < 0:
            continue
        mask = y_load == lc
        X_norm[mask] = StandardScaler().fit_transform(X_raw[mask])
    # Any leftover rows (load=-1) use global scaler
    bad = y_load < 0
    if bad.any():
        X_norm[bad] = StandardScaler().fit_transform(X_raw[bad])
    return X_norm


# ── FIX-2: Class balancing (random undersampling of OR) ──────────────────────
def balance_classes(X: np.ndarray, y: np.ndarray, y_sev: np.ndarray,
                    y_load: np.ndarray, rng: np.random.Generator):
    """Under-sample the dominant class to the median class size."""
    classes, counts = np.unique(y, return_counts=True)
    target = int(np.median(counts))
    keep   = []
    for c, n in zip(classes, counts):
        idx = np.where(y == c)[0]
        if n > target:
            idx = rng.choice(idx, size=target, replace=False)
        keep.append(idx)
    keep = np.sort(np.concatenate(keep))
    log.info("After balancing: %d segments (target per class ≈ %d)", len(keep), target)
    for c in classes:
        log.info("  Class %d (%s): %d", c, CLASS_NAMES[c], (y[keep] == c).sum())
    return X[keep], y[keep], y_sev[keep], y_load[keep]


# ════════════════════════════════════════════════════════════════════════════
# CHROMOSOME HELPERS  (identical to v1)
# ════════════════════════════════════════════════════════════════════════════

def decode_chromosome(x: np.ndarray):
    K       = int(np.clip(x[0], MIN_K, MAX_K + 0.999))
    centres = x[CTR_START: CTR_START + K * D].reshape(K, D)
    raw_msk = x[MSK_START:MSK_END]
    mask    = raw_msk >= 0.5
    if not mask.any():
        mask[np.argmax(raw_msk)] = True
    return K, centres, mask


def assign_clusters(X_act: np.ndarray, C_act: np.ndarray) -> np.ndarray:
    diff = X_act[:, None, :] - C_act[None, :, :]
    dist = np.einsum("nkd,nkd->nk", diff, diff)
    return np.argmin(dist, axis=1)


def count_active(x: np.ndarray) -> int:
    return int((x[MSK_START:MSK_END] >= 0.5).sum() or 1)


# ════════════════════════════════════════════════════════════════════════════
# 2.  MULTI-OBJECTIVE PROBLEM  (identical to v1)
# ════════════════════════════════════════════════════════════════════════════

class MOCProblem(Problem):
    def __init__(self, X_scaled: np.ndarray, knn_graph: np.ndarray):
        self.X = X_scaled
        self.G = knn_graph
        self.N = X_scaled.shape[0]
        xl = np.empty(N_VAR); xu = np.empty(N_VAR)
        xl[0] = float(MIN_K); xu[0] = float(MAX_K) + 0.999
        xl[CTR_START:CTR_END] = CTR_LO; xu[CTR_START:CTR_END] = CTR_HI
        xl[MSK_START:MSK_END] = 0.0;    xu[MSK_START:MSK_END] = 1.0
        super().__init__(n_var=N_VAR, n_obj=4, n_ieq_constr=0, xl=xl, xu=xu)

    def _evaluate(self, X_pop: np.ndarray, out: dict, *args, **kwargs):
        n_pop = X_pop.shape[0]
        F     = np.full((n_pop, 4), 1e9)
        for i in range(n_pop):
            x = X_pop[i]
            K, C, mask = decode_chromosome(x)
            if not mask.any() or K < MIN_K:
                continue
            X_act  = self.X[:, mask]
            C_act  = C[:, mask]
            labels = assign_clusters(X_act, C_act)
            # f1: Compactness
            twcss = 0.0
            for k in range(K):
                pts = X_act[labels == k]
                if len(pts):
                    d = pts - C_act[k]
                    twcss += np.einsum("nd,nd->", d, d)
            f1 = twcss / (self.N * float(mask.sum()))
            # f2: Connectedness
            neigh_labels = labels[self.G]
            f2 = float((labels[:, None] != neigh_labels).sum()) / (self.N * KNN_K)
            # f3: Simplicity
            f3 = float(mask.sum())
            # f4: Partition Entropy / Gini Impurity
            dists = np.einsum("nkd,nkd->nk", X_act[:, None, :] - C_act[None, :, :], X_act[:, None, :] - C_act[None, :, :])
            inv_dists = 1.0 / (dists + 1e-12)
            probs = inv_dists / inv_dists.sum(axis=1, keepdims=True)
            gini = 1.0 - np.sum(probs**2, axis=1)
            f4 = float(np.mean(gini))
            F[i] = [f1, f2, f3, f4]
        out["F"] = F


# ════════════════════════════════════════════════════════════════════════════
# 3.  HYBRID INITIALISATION  (identical to v1, slightly wider mask spread)
# ════════════════════════════════════════════════════════════════════════════

class HybridSampling(Sampling):
    def __init__(self, X_scaled: np.ndarray):
        super().__init__()
        self.X = X_scaled

    def _random_mask(self, rng: np.random.Generator, min_active: int = 2) -> np.ndarray:
        while True:
            m = (rng.random(D) > 0.5).astype(float)
            if m.sum() >= min_active:
                return m

    def _make_chromosome(self, rng, K, centres) -> np.ndarray:
        x = np.zeros(N_VAR)
        x[0] = float(K) + rng.random() * 0.999
        x[CTR_START: CTR_START + K * D]  = np.clip(centres, CTR_LO, CTR_HI).ravel()
        x[CTR_START + K * D: CTR_END]    = rng.uniform(CTR_LO, CTR_HI, (MAX_K - K) * D)
        x[MSK_START: MSK_END]            = self._random_mask(rng, min_active=2)
        return x

    def _do(self, problem, n_samples: int, **kwargs) -> np.ndarray:
        rng = np.random.default_rng(RANDOM_SEED)
        n1  = n_samples // 3
        n2  = n_samples // 3
        n3  = n_samples - n1 - n2
        pop = np.zeros((n_samples, N_VAR))
        idx = 0

        log.info("  Hybrid init: KMeans-seeded (%d)…", n1)
        for _ in range(n1):
            K  = int(rng.integers(MIN_K, MAX_K + 1))
            km = KMeans(n_clusters=K, n_init=3, max_iter=50,
                        random_state=int(rng.integers(1_000_000)))
            km.fit(self.X)
            pop[idx] = self._make_chromosome(rng, K, km.cluster_centers_)
            idx += 1

        log.info("  Hybrid init: data-point seeded (%d)…", n2)
        for _ in range(n2):
            K       = int(rng.integers(MIN_K, MAX_K + 1))
            chosen  = rng.choice(len(self.X), size=K, replace=False)
            pop[idx] = self._make_chromosome(rng, K, self.X[chosen])
            idx += 1

        log.info("  Hybrid init: pure random (%d)…", n3)
        for _ in range(n3):
            K = int(rng.integers(MIN_K, MAX_K + 1))
            pop[idx] = self._make_chromosome(
                rng, K, rng.uniform(CTR_LO, CTR_HI, (K, D)))
            idx += 1

        return pop


# ════════════════════════════════════════════════════════════════════════════
# 4.  GENETIC OPERATORS  (identical to v1)
# ════════════════════════════════════════════════════════════════════════════

def _sbx_pair(y1, y2, yl, yu, eta, rv):
    if abs(y1 - y2) < 1e-10: return y1, y2
    a, b   = min(y1, y2), max(y1, y2)
    spread = b - a
    beta1  = 1.0 + 2.0 * (a - yl) / spread
    alpha1 = 2.0 - beta1 ** (-(eta + 1))
    betaq1 = ((rv * alpha1) ** (1.0/(eta+1)) if rv <= 1.0/alpha1
              else (1.0/(2.0 - rv*alpha1)) ** (1.0/(eta+1)))
    beta2  = 1.0 + 2.0 * (yu - b) / spread
    alpha2 = 2.0 - beta2 ** (-(eta + 1))
    betaq2 = ((rv * alpha2) ** (1.0/(eta+1)) if rv <= 1.0/alpha2
              else (1.0/(2.0 - rv*alpha2)) ** (1.0/(eta+1)))
    c1 = np.clip(0.5*(a+b-betaq1*spread), yl, yu)
    c2 = np.clip(0.5*(a+b+betaq2*spread), yl, yu)
    return c1, c2


def _point_crossover(m1, m2):
    if len(m1) < 2: return m1.copy(), m2.copy()
    pt = np.random.randint(1, len(m1))
    return np.concatenate([m1[:pt], m2[pt:]]), np.concatenate([m2[:pt], m1[pt:]])


class MOCCrossover(Crossover):
    def __init__(self, eta=15.0, prob=0.9):
        super().__init__(2, 2)
        self.eta = eta; self.prob = prob

    def _do(self, problem, X, **kwargs):
        _, n_matings, _ = X.shape
        Y  = X.copy()
        xl = problem.xl; xu = problem.xu
        for i in range(n_matings):
            if np.random.rand() > self.prob: continue
            p1, p2 = X[0,i].copy(), X[1,i].copy()
            o1, o2 = p1.copy(), p2.copy()
            for j in range(CTR_END):
                if np.random.rand() <= 0.5:
                    o1[j], o2[j] = _sbx_pair(p1[j], p2[j],
                                              xl[j], xu[j], self.eta,
                                              np.random.rand())
            o1[MSK_START:], o2[MSK_START:] = _point_crossover(
                p1[MSK_START:], p2[MSK_START:])
            Y[0,i] = o1; Y[1,i] = o2
        return Y


class MOCMutation(Mutation):
    def __init__(self, eta=20.0, prob_var=2.0/N_VAR, prob_flip=0.15, prob_struct=0.05):
        super().__init__()
        self.eta = eta; self.prob_var = prob_var
        self.prob_flip = prob_flip; self.prob_struct = prob_struct

    def _do(self, problem, X, **kwargs):
        Y  = X.copy()
        xl = problem.xl; xu = problem.xu
        for i in range(len(Y)):
            for j in range(CTR_END):
                if np.random.rand() >= self.prob_var: continue
                y = Y[i,j]; yl = xl[j]; yu = xu[j]; r = np.random.rand()
                eta = self.eta
                if r < 0.5:
                    xy = 1.0 - (y-yl)/(yu-yl+1e-12)
                    dq = (2.0*r+(1.0-2.0*r)*xy**(eta+1))**(1.0/(eta+1))-1.0
                else:
                    xy = 1.0 - (yu-y)/(yu-yl+1e-12)
                    dq = 1.0-(2.0*(1.0-r)+2.0*(r-0.5)*xy**(eta+1))**(1.0/(eta+1))
                Y[i,j] = np.clip(y+dq*(yu-yl), yl, yu)
            for j in range(MSK_START, N_VAR):
                if np.random.rand() < self.prob_flip:
                    Y[i,j] = 1.0 - Y[i,j]
            if (Y[i, MSK_START:] < 0.5).all():
                Y[i, MSK_START + np.random.randint(D)] = 1.0
            if np.random.rand() < self.prob_struct:
                K_cur = int(np.clip(Y[i,0], MIN_K, MAX_K))
                K_new = int(np.clip(K_cur+np.random.choice([-1,1]), MIN_K, MAX_K))
                Y[i,0] = float(K_new)+np.random.rand()*0.999
        return Y


# ════════════════════════════════════════════════════════════════════════════
# ENH-2: POPULATION HISTORY CALLBACK
# ════════════════════════════════════════════════════════════════════════════

class PopulationHistoryCallback(Callback):
    """
    Saves full population (F + X) at SNAPSHOT_GENS to disk.
    Captures per-generation convergence stats in self.gen_stats.
    """
    def __init__(self, save_dir: str, snapshot_gens: set):
        super().__init__()
        self.save_dir      = save_dir
        self.snapshot_gens = snapshot_gens
        self.gen_stats     = []

    def notify(self, algorithm):
        n   = algorithm.n_gen
        pop = algorithm.pop
        F   = pop.get("F")
        X   = pop.get("X")

        # Per-generation stats
        valid = ~np.any(np.isnan(F), axis=1)
        Fv = F[valid]
        if len(Fv):
            self.gen_stats.append({
                "gen":        n,
                "n_valid":    int(valid.sum()),
                "best_f1":    float(Fv[:, 0].min()),
                "mean_f1":    float(Fv[:, 0].mean()),
                "best_f2":    float(Fv[:, 1].min()),
                "mean_f2":    float(Fv[:, 1].mean()),
                "best_f3":    float(Fv[:, 2].min()),
                "mean_f3":    float(Fv[:, 2].mean()),
                "best_f4":    float(Fv[:, 3].min()),
                "mean_f4":    float(Fv[:, 3].mean()),
                "n_pareto_approx": int((Fv[:, 0] < Fv[:, 0].mean()).sum()),
            })

        # Save snapshot
        if n in self.snapshot_gens:
            path = os.path.join(self.save_dir, f"gen_{n:04d}.npz")
            np.savez_compressed(path, F=F[valid], X=X[valid], gen=np.array(n))
            log.info("  [Gen %3d] Snapshot saved (%d valid solutions).", n, valid.sum())


# ════════════════════════════════════════════════════════════════════════════
# 5.  PARETO ANALYSIS & VALIDATION  (identical to v1)
# ════════════════════════════════════════════════════════════════════════════

def find_knee_point(F: np.ndarray) -> int:
    denom  = F.max(0) - F.min(0) + 1e-12
    F_norm = (F - F.min(0)) / denom
    return int(np.argmin(np.linalg.norm(F_norm, axis=1)))


def evaluate_solution(x, X_scaled, y_true):
    K, C, mask = decode_chromosome(x)
    X_act      = X_scaled[:, mask]
    C_act      = C[:, mask]
    if X_act.shape[1] == 0:
        return 0.0, -1.0, mask
    labels = assign_clusters(X_act, C_act)
    if len(np.unique(labels)) < 2:
        return 0.0, -1.0, mask
    ari = adjusted_rand_score(y_true, labels)
    N   = len(X_act)
    idx = np.random.choice(N, 5_000, replace=False) if N > 5_000 else np.arange(N)
    sil = silhouette_score(X_act[idx], labels[idx])
    return ari, sil, mask


# ════════════════════════════════════════════════════════════════════════════
# 6.  QUICK VISUALISATIONS  (Pareto + feature importance, same as v1)
# ════════════════════════════════════════════════════════════════════════════

def _save(fig, name):
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved: %s", path)


def plot_pareto_3d(F, knee_idx):
    fig = plt.figure(figsize=(12, 9))
    ax  = fig.add_subplot(111, projection="3d")
    sc  = ax.scatter(F[:,0], F[:,1], F[:,2],
                     c=F[:,2], cmap="viridis_r", s=55, alpha=0.75, edgecolors="none")
    ax.scatter(F[knee_idx,0], F[knee_idx,1], F[knee_idx,2],
               color="red", s=280, marker="*", zorder=10, label="Knee")
    plt.colorbar(sc, ax=ax, shrink=0.6, label="# active features")
    ax.set_xlabel("Compactness"); ax.set_ylabel("Connectedness")
    ax.set_zlabel("Simplicity")
    ax.set_title("3-D Pareto Front v2 (load-normalised + balanced)",
                 fontsize=13, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    _save(fig, "01_pareto_3d_v2.png")


def plot_feature_importance(pareto_X):
    counts = np.zeros(D)
    for x in pareto_X:
        mask = x[MSK_START:MSK_END] >= 0.5
        if not mask.any(): mask[np.argmax(x[MSK_START:MSK_END])] = True
        counts += mask.astype(float)
    freq    = counts / len(pareto_X) * 100
    colours = cm.RdYlGn(freq / 100)
    fig, ax = plt.subplots(figsize=(11, 6))
    bars    = ax.bar(FEATURE_NAMES, freq, color=colours, edgecolor="black")
    for bar, f in zip(bars, freq):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5,
                f"{f:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax.axhline(50, color="grey", linestyle="--", linewidth=0.9)
    ax.set_ylim(0, 118); ax.set_ylabel("Activation frequency (%)")
    ax.set_title("Feature Importance — v2 Pareto solutions", fontweight="bold")
    fig.tight_layout()
    _save(fig, "02_feature_importance_v2.png")


def plot_cluster_map(X_scaled, labels, y_true, title, fname, method="pca"):
    if method == "tsne":
        n   = min(3_000, len(X_scaled))
        idx = np.random.choice(len(X_scaled), n, replace=False)
        Xs  = X_scaled[idx]
        npc = min(50, Xs.shape[1])
        if npc > 1: Xs = PCA(n_components=npc).fit_transform(Xs)
        proj = TSNE(n_components=2, random_state=RANDOM_SEED,
                    perplexity=30, max_iter=1_000).fit_transform(Xs)
    else:
        idx  = np.arange(len(X_scaled))
        proj = PCA(n_components=2).fit_transform(X_scaled)
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    sc1 = axes[0].scatter(proj[:,0], proj[:,1], c=labels[idx],
                          cmap="tab10", s=4, alpha=0.55)
    axes[0].set_title("Predicted Clusters", fontweight="bold")
    plt.colorbar(sc1, ax=axes[0])
    sc2 = axes[1].scatter(proj[:,0], proj[:,1], c=y_true[idx],
                          cmap="Set1", vmin=0, vmax=3, s=4, alpha=0.55)
    axes[1].set_title("Ground-Truth Classes", fontweight="bold")
    cbar = plt.colorbar(sc2, ax=axes[1], ticks=[0,1,2,3])
    cbar.ax.set_yticklabels(CLASS_NAMES, fontsize=9)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    _save(fig, fname)


def plot_convergence(gen_stats):
    if not gen_stats: return
    gens = [s["gen"]    for s in gen_stats]
    bf1  = [s["best_f1"] for s in gen_stats]
    bf2  = [s["best_f2"] for s in gen_stats]
    bf3  = [s["best_f3"] for s in gen_stats]
    bf4  = [s["best_f4"] for s in gen_stats]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, vals, lbl, col in zip(
            axes, [bf1, bf2, bf3, bf4],
            ["Best Compactness", "Best Connectedness", "Best Simplicity", "Best Partition Entropy"],
            ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]):
        ax.plot(gens, vals, color=col, linewidth=2)
        ax.set_xlabel("Generation"); ax.set_ylabel(lbl)
        ax.set_title(lbl, fontweight="bold"); ax.grid(True, alpha=0.3)
    fig.suptitle("Convergence (Entropy 4-Obj)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "00_convergence_v2.png")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main(base_dir: str, optimizer: str = "lhfid", pop_size: int = POP_SIZE,
         n_gen: int = N_GEN):
    t0 = time.time()
    rng = np.random.default_rng(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    log.info("=" * 70)
    log.info("  MOC-FS v2  |  optimizer=%s  pop=%d  gen=%d  fixes: load-norm + balance",
             optimizer.upper(), pop_size, n_gen)
    log.info("=" * 70)

    # ── 1. Load with load/severity metadata ─────────────────────────────────
    log.info("\n[1] Loading CWRU dataset (v2)…")
    X_raw, y_type, y_severity, y_load = load_cwru_dataset_v2(base_dir)
    unique, counts = np.unique(y_type, return_counts=True)
    for u, c in zip(unique, counts):
        log.info("  Class %d (%s): %d segs", u, CLASS_NAMES[u], c)

    # ── FIX-1: Per-load normalisation ────────────────────────────────────────
    log.info("\n[FIX-1] Per-load normalisation…")
    X_scaled = normalize_by_load(X_raw, y_load)
    log.info("  Done. Shape: %s", X_scaled.shape)

    # ── FIX-2: Class balancing ───────────────────────────────────────────────
    log.info("\n[FIX-2] Balancing classes…")
    X_scaled, y_type, y_severity, y_load = balance_classes(
        X_scaled, y_type, y_severity, y_load, rng)

    # ── FIX-3: 16-class severity label ──────────────────────────────────────
    # Compact id: Normal=0, IR_7mil=1 … IR_28mil=4, Ball_7mil=5 … OR_21mil=12
    sev_offset = {0: 0, 1: 0, 2: 1, 3: 2, 4: 3}   # severity index → offset
    fault_base = {0: 0, 1: 1, 2: 5, 3: 9}           # fault_type → first class id
    y_16cls = np.array([
        fault_base[t] + sev_offset.get(s, 0)
        for t, s in zip(y_type, y_severity)
    ], dtype=np.int32)
    n_unique_16 = len(np.unique(y_16cls))
    log.info("[FIX-3] 16-class label: %d distinct severity-fault classes.", n_unique_16)

    # ── 2. kNN graph ─────────────────────────────────────────────────────────
    log.info("\n[2] Building kNN graph (k=%d)…", KNN_K)
    t_knn = time.time()
    nn = NearestNeighbors(n_neighbors=KNN_K+1, algorithm="ball_tree", n_jobs=-1)
    nn.fit(X_scaled)
    _, knn_idx  = nn.kneighbors(X_scaled)
    knn_graph   = knn_idx[:, 1:]
    log.info("  kNN built in %.2f s.", time.time()-t_knn)

    # ── 3. Optimisation (NSGA-II / LHFiD) ──────────────────────────────────
    log.info("\n[3] Running %s  (pop=%d, gen=%d)…",
             optimizer.upper(), pop_size, n_gen)
    problem   = MOCProblem(X_scaled, knn_graph)
    sampling  = HybridSampling(X_scaled)
    crossover = MOCCrossover(eta=15.0, prob=0.9)
    mutation  = MOCMutation(eta=20.0, prob_var=2.0/N_VAR,
                            prob_flip=0.15, prob_struct=0.05)
    snapshot_gens = {g for g in SNAPSHOT_GENS if g <= n_gen}
    if n_gen not in snapshot_gens:
        snapshot_gens.add(n_gen)
    callback  = PopulationHistoryCallback(
        os.path.join(RESULTS_DIR, "history"), snapshot_gens)

    if optimizer == "lhfid":
        # For 4 objectives we keep partitions modest (8 → 165 refs) to avoid exploding evaluations.
        ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=8) # 165 vectors
        algorithm = LHFID(
            pop_size=pop_size,
            ref_dirs=ref_dirs,
            sampling=sampling,
            crossover=crossover,
            selection=RandomSelection(),
            mutation=mutation,
            eliminate_duplicates=True,
        )
        termination = ("n_gen", max(n_gen, 10_000))
    else:
        algorithm = NSGA2(pop_size=pop_size, sampling=sampling,
                          crossover=crossover, mutation=mutation,
                          eliminate_duplicates=True)
        termination = get_termination("n_gen", n_gen)

    t_opt = time.time()
    res   = minimize(problem, algorithm,
                     termination,
                     seed=RANDOM_SEED, verbose=True,
                     callback=callback, save_history=False)
    opt_t = time.time() - t_opt
    log.info("Optimisation done in %.1f s (%.1f min).", opt_t, opt_t/60)

    # ── 4. Pareto front ──────────────────────────────────────────────────────
    log.info("\n[4] Pareto front analysis…")
    pareto_F = res.F
    pareto_X = res.X
    n_pareto = len(pareto_F)
    log.info("  %d Pareto solutions", n_pareto)
    for j, nm in enumerate(["Compactness","Connectedness","Simplicity","Partition Entropy"]):
        log.info("  f%d %s: [%.4f, %.4f]",
                 j+1, nm, pareto_F[:,j].min(), pareto_F[:,j].max())

    knee_idx       = find_knee_point(pareto_F)
    K_kn, _, mk_kn = decode_chromosome(pareto_X[knee_idx])
    log.info("  Knee: K=%d  active=%s  idx=%d",
             K_kn, list(FEATURE_NAMES[mk_kn]), knee_idx)

    # ── 5. Validation ────────────────────────────────────────────────────────
    log.info("\n[5] Validation metrics…")
    n_active = np.array([count_active(x) for x in pareto_X])
    ari_arr  = np.zeros(n_pareto)
    sil_arr  = np.zeros(n_pareto)
    ari16_arr= np.zeros(n_pareto)

    for i, x in enumerate(pareto_X):
        ari_arr[i],  sil_arr[i],  _ = evaluate_solution(x, X_scaled, y_type)
        ari16_arr[i], _, _          = evaluate_solution(x, X_scaled, y_16cls)

    log.info("  4-class ARI — mean %.4f | max %.4f | knee %.4f",
             ari_arr.mean(), ari_arr.max(), ari_arr[knee_idx])
    log.info("  16-class ARI— mean %.4f | max %.4f",
             ari16_arr.mean(), ari16_arr.max())
    log.info("  Silhouette  — mean %.4f | max %.4f",
             sil_arr.mean(), sil_arr.max())

    # ── Representative solutions ─────────────────────────────────────────────
    simple_cands = np.where(n_active <= 3)[0]
    if len(simple_cands) == 0:
        simple_cands = np.argsort(n_active)[:max(1, n_pareto//5)]
    simple_idx  = simple_cands[np.argmax(ari_arr[simple_cands])]
    compact_idx = int(np.argmin(pareto_F[:,0]))

    K_s, _, ms = decode_chromosome(pareto_X[simple_idx])
    K_c, _, mc = decode_chromosome(pareto_X[compact_idx])
    log.info("\n  Simple  [%d feat, K=%d]: ARI=%.4f → %s",
             n_active[simple_idx], K_s, ari_arr[simple_idx],
             list(FEATURE_NAMES[ms]))
    log.info("  Compact [%d feat, K=%d]: ARI=%.4f → %s",
             n_active[compact_idx], K_c, ari_arr[compact_idx],
             list(FEATURE_NAMES[mc]))

    # ── 6. Visualisations ────────────────────────────────────────────────────
    log.info("\n[6] Generating visualisations…")
    plot_pareto_3d(pareto_F, knee_idx)
    plot_feature_importance(pareto_X)
    plot_convergence(callback.gen_stats)

    # Cluster maps
    K_s2, C_s, ms2 = decode_chromosome(pareto_X[simple_idx])
    labels_s = assign_clusters(X_scaled[:,ms2], C_s[:,ms2])
    plot_cluster_map(X_scaled, labels_s, y_type,
        f"Simple (K={K_s2}, {list(FEATURE_NAMES[ms2])})  ARI={ari_arr[simple_idx]:.4f}",
        "03_cluster_simple_v2.png", method="pca")

    K_c2, C_c, mc2 = decode_chromosome(pareto_X[compact_idx])
    labels_c = assign_clusters(X_scaled[:,mc2], C_c[:,mc2])
    plot_cluster_map(X_scaled, labels_c, y_type,
        f"Compact (K={K_c2}, {list(FEATURE_NAMES[mc2])})  ARI={ari_arr[compact_idx]:.4f}",
        "04_cluster_compact_v2.png", method="tsne")

    # ── 7. Save all arrays ───────────────────────────────────────────────────
    log.info("\n[7] Saving arrays to %s/…", RESULTS_DIR)
    save = lambda name, arr: np.save(os.path.join(RESULTS_DIR, name), arr)
    save("pareto_F.npy",    pareto_F)
    save("pareto_X.npy",    pareto_X)
    save("pareto_ari.npy",  ari_arr)
    save("pareto_ari16.npy",ari16_arr)
    save("pareto_sil.npy",  sil_arr)
    save("n_active.npy",    n_active)
    save("X_scaled.npy",    X_scaled)
    save("y_type.npy",      y_type)
    save("y_severity.npy",  y_severity)
    save("y_load.npy",      y_load)
    save("y_16cls.npy",     y_16cls)

    # Raw un-normalised features (needed for waveform physics plots)
    X_raw2, y_t2, y_s2, y_l2 = load_cwru_dataset_v2(base_dir)
    save("X_raw.npy", X_raw2)

    # Save convergence stats
    json.dump(callback.gen_stats,
              open(os.path.join(RESULTS_DIR,"gen_stats.json"),"w"), indent=2)

    summary = {
        "v": 2,
        "fixes": ["per-load-normalisation", "class-balancing", "severity-labels"],
        "dataset": {"n_balanced": int(len(X_scaled)),
                    "n_classes_16": int(n_unique_16)},
        "optimizer": {
            "name": optimizer,
            "pop": pop_size,
            "gen": n_gen,
            "time_s": round(opt_t, 2),
        },
        "metrics": {
            "ari4_max":  round(float(ari_arr.max()),4),
            "ari16_max": round(float(ari16_arr.max()),4),
            "sil_max":   round(float(sil_arr.max()),4),
        },
        "pareto": {"n": n_pareto,
                   "f3_range": [int(pareto_F[:,2].min()), int(pareto_F[:,2].max())]},
        "simple":  {"K":K_s,"nfeat":int(n_active[simple_idx]),
                    "ari4":round(float(ari_arr[simple_idx]),4),
                    "active":list(FEATURE_NAMES[ms])},
        "compact": {"K":K_c,"nfeat":int(n_active[compact_idx]),
                    "ari4":round(float(ari_arr[compact_idx]),4),
                    "active":list(FEATURE_NAMES[mc])},
        "total_s": round(time.time()-t0, 2),
    }
    json.dump(summary, open(os.path.join(RESULTS_DIR,"summary_v2.json"),"w"), indent=2)
    log.info("  summary_v2.json written.")

    total = time.time() - t0
    log.info("\n" + "="*70)
    log.info("  v2 RESULTS")
    log.info("="*70)
    log.info("  Balanced dataset : %d segments", len(X_scaled))
    log.info("  Pareto solutions : %d", n_pareto)
    log.info("  4-class ARI max  : %.4f  (v1: 0.3014)", ari_arr.max())
    log.info("  16-class ARI max : %.4f  (reveals severity sub-clusters)",
             ari16_arr.max())
    log.info("  Silhouette max   : %.4f", sil_arr.max())
    log.info("  Simple  solution : %d feat, ARI=%.4f → %s",
             n_active[simple_idx], ari_arr[simple_idx], list(FEATURE_NAMES[ms]))
    log.info("  Compact solution : %d feat, ARI=%.4f → %s",
             n_active[compact_idx], ari_arr[compact_idx], list(FEATURE_NAMES[mc]))
    log.info("  Total time       : %.1f s (%.1f min)", total, total/60)
    log.info("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "CWRU-dataset"))
    parser.add_argument("--optimizer", choices=["nsga2", "lhfid"], default="lhfid")
    parser.add_argument("--pop-size", type=int, default=POP_SIZE)
    parser.add_argument("--n-gen", type=int, default=N_GEN)
    args = parser.parse_args()
    main(args.data_dir, optimizer=args.optimizer,
         pop_size=args.pop_size, n_gen=args.n_gen)
