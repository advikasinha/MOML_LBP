"""
make_mechanical.py
==================
Generates mechanical-context visualisation assets for the MOC-FS bearing
fault diagnosis project.  All figures are saved to  moc_results_v2/mechanical/

Panels produced
---------------
  M1_bearing_schematic.png   — annotated SKF 6205-2RS cross-section diagram
  M2_fault_frequencies.png   — characteristic fault frequencies vs RPM table + bar chart
  M3_fault_waveforms.png     — sample time-domain waveforms per fault class (from raw data)
  M4_fault_fft.png           — FFT of each fault class with frequency markers
  M5_feature_physics.png     — feature→fault mapping (which features catch which faults)
  M6_tsne_classes.png        — t-SNE of X_scaled coloured by 16-class label
  M7_ari_silhouette.png      — ARI / Silhouette comparison across v1 and v2 runs
  M8_pareto_knee_detail.png  — 2-D projections of Pareto front with knee highlighted

Run after moc_bearing_v2.py has completed:
  python make_mechanical.py [--results-dir moc_results_v2]
"""

import os
import argparse
import warnings
import textwrap

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, Arc, Circle, FancyBboxPatch
from matplotlib.colors import ListedColormap

warnings.filterwarnings("ignore")

# ── Constants (must match moc_bearing_v2.py) ─────────────────────────────────

FEATURE_NAMES = ["RMS", "Kurtosis", "Skewness",
                 "Crest Factor", "Peak-to-Peak", "Std Dev", "Spectral Centroid"]
D = len(FEATURE_NAMES)

# SKF 6205-2RS geometry (all in mm)
BEARING = dict(
    Pd   = 38.50,   # pitch diameter
    Bd   = 7.94,    # ball diameter
    n    = 9,       # number of balls
    phi  = 0.0,     # contact angle (rad) — 0 for radial
)

# Shaft speeds to tabulate
RPM_LIST = [1797, 1772, 1750, 1730]
RPM_LABEL = ["0 HP\n(1797 rpm)", "1 HP\n(1772 rpm)", "2 HP\n(1750 rpm)", "3 HP\n(1730 rpm)"]

FAULT_COLORS = {
    "Normal":   "#4daf4a",
    "IR Fault": "#e41a1c",
    "Ball":     "#ff7f00",
    "OR Fault": "#377eb8",
}

FAULT_ABBREV = {0: "Normal", 1: "IR Fault", 2: "Ball", 3: "OR Fault"}
FAULT_SHORT  = {0: "N", 1: "IR", 2: "B", 3: "OR"}


# ── Fault frequency maths ─────────────────────────────────────────────────────

def fault_freqs(rpm):
    fr   = rpm / 60.0
    Pd, Bd, n, phi = BEARING["Pd"], BEARING["Bd"], BEARING["n"], BEARING["phi"]
    bpfi = (n / 2) * fr * (1 + (Bd / Pd) * np.cos(phi))
    bpfo = (n / 2) * fr * (1 - (Bd / Pd) * np.cos(phi))
    bsf  = (Pd / (2 * Bd)) * fr * (1 - ((Bd / Pd) * np.cos(phi)) ** 2)
    ftf  = (fr / 2) * (1 - (Bd / Pd) * np.cos(phi))
    return dict(BPFI=bpfi, BPFO=bpfo, BSF=bsf, FTF=ftf)


# ── Helpers ───────────────────────────────────────────────────────────────────

def save(fig, path, dpi=150):
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")


def make_outdir(results_dir):
    d = os.path.join(results_dir, "mechanical")
    os.makedirs(d, exist_ok=True)
    return d


# ═════════════════════════════════════════════════════════════════════════════
# M1 — Bearing schematic
# ═════════════════════════════════════════════════════════════════════════════

def m1_bearing_schematic(out_dir):
    fig, ax = plt.subplots(figsize=(9, 9), facecolor="#f8f9fa")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-70, 70)
    ax.set_ylim(-70, 70)

    # Outer race
    outer = Circle((0, 0), 55, fill=False, linewidth=4, color="#1a1a2e")
    outer_fill = Circle((0, 0), 55, fill=True, color="#e8e8e8", zorder=0)
    inner_race_outer = Circle((0, 0), 47, fill=True, color="#f8f9fa", zorder=1)
    ax.add_patch(outer_fill)
    ax.add_patch(outer)
    ax.add_patch(inner_race_outer)

    # Inner race
    inner_o = Circle((0, 0), 27, fill=False, linewidth=4, color="#1a1a2e", zorder=2)
    inner_i = Circle((0, 0), 20, fill=True, color="#e8e8e8", zorder=2)
    inner_i2 = Circle((0, 0), 20, fill=False, linewidth=3, color="#1a1a2e", zorder=2)
    ax.add_patch(inner_i)
    ax.add_patch(inner_o)
    ax.add_patch(inner_i2)

    # Shaft hole
    shaft = Circle((0, 0), 13, fill=True, color="#c0c0c0", zorder=3)
    shaft_b = Circle((0, 0), 13, fill=False, linewidth=2, color="#555", zorder=3)
    ax.add_patch(shaft)
    ax.add_patch(shaft_b)

    # Balls (n=9 evenly spaced on pitch circle r = Pd/2)
    r_pitch = BEARING["Pd"] / 2   # 19.25 mm (scaled to plot units: *1)
    r_ball  = BEARING["Bd"] / 2   # 3.97 mm
    scale   = 1.9                 # scale mm → plot units
    n_balls = BEARING["n"]
    fault_ball_idx = 4            # highlight one ball as "ball fault"

    for k in range(n_balls):
        angle = 2 * np.pi * k / n_balls
        cx    = r_pitch * scale * np.cos(angle)
        cy    = r_pitch * scale * np.sin(angle)
        color = "#e41a1c" if k == fault_ball_idx else "#555577"
        lw    = 2.5       if k == fault_ball_idx else 1.2
        ball  = Circle((cx, cy), r_ball * scale,
                       fill=True, color="#aabbcc", zorder=4,
                       linewidth=lw, edgecolor=color)
        ax.add_patch(ball)
        if k == fault_ball_idx:
            ax.annotate("", xy=(cx + r_ball * scale * 1.5, cy),
                        xytext=(cx + r_ball * scale * 3, cy),
                        arrowprops=dict(arrowstyle="->", color="#e41a1c", lw=1.5))

    # Dimension lines
    ax.annotate("", xy=(0, -52 * 1.0), xytext=(0, -27 * 1.0),
                arrowprops=dict(arrowstyle="<->", color="#444", lw=1.2))
    ax.text(2, -40, "Pd = 38.5 mm", fontsize=8, color="#333", va="center")

    # Labels — outer race fault zone
    ax.text(0, 60, "Outer Race (OR)\nBPFO = 3.585 × fr", fontsize=10,
            ha="center", va="bottom", color="#377eb8", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#377eb8", alpha=0.9))
    ax.annotate("", xy=(0, 47), xytext=(0, 58),
                arrowprops=dict(arrowstyle="->", color="#377eb8", lw=1.5))

    # Labels — inner race fault zone
    ax.text(-62, 15, "Inner Race (IR)\nBPFI = 5.415 × fr", fontsize=10,
            ha="right", va="center", color="#e41a1c", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#e41a1c", alpha=0.9))
    ax.annotate("", xy=(-27, 7), xytext=(-44, 14),
                arrowprops=dict(arrowstyle="->", color="#e41a1c", lw=1.5))

    # Labels — ball fault
    ax.text(50, -25, "Rolling Element\nBSF = 2.357 × fr", fontsize=10,
            ha="left", va="center", color="#ff7f00", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#ff7f00", alpha=0.9))
    r_b_cx = r_pitch * scale * np.cos(2 * np.pi * fault_ball_idx / n_balls)
    r_b_cy = r_pitch * scale * np.sin(2 * np.pi * fault_ball_idx / n_balls)
    ax.annotate("", xy=(r_b_cx + r_ball * scale * 1.5, r_b_cy),
                xytext=(44, -20),
                arrowprops=dict(arrowstyle="->", color="#ff7f00", lw=1.5))

    # Labels — cage / FTF
    ax.text(-55, -40, "Cage (FTF)\n0.398 × fr", fontsize=10,
            ha="center", va="center", color="#555",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#999", alpha=0.9))

    # Shaft label
    ax.text(0, 0, "Shaft", fontsize=9, ha="center", va="center",
            color="#444", fontweight="bold")

    # 9 balls label
    ax.text(-2, 32, "9 balls", fontsize=8, color="#555577", ha="center",
            style="italic")

    ax.set_title("SKF 6205-2RS Drive-End Bearing — Fault Zone Map\n"
                 "fr = shaft rotation frequency  |  CWRU dataset @ 12 kHz",
                 fontsize=13, fontweight="bold", pad=15)

    save(fig, os.path.join(out_dir, "M1_bearing_schematic.png"))


# ═════════════════════════════════════════════════════════════════════════════
# M2 — Fault frequency table + bar chart
# ═════════════════════════════════════════════════════════════════════════════

def m2_fault_frequencies(out_dir):
    fig = plt.figure(figsize=(14, 7), facecolor="white")
    gs  = GridSpec(1, 2, figure=fig, width_ratios=[1, 1.4], wspace=0.35)

    ax_tbl = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1])

    # ── Table ──────────────────────────────────────────────────────────────
    ax_tbl.axis("off")
    headers = ["Load (HP)", "RPM", "BPFI (Hz)", "BPFO (Hz)", "BSF (Hz)", "FTF (Hz)"]
    rows    = []
    for rpm in RPM_LIST:
        ff = fault_freqs(rpm)
        rows.append([
            str(RPM_LIST.index(rpm)),
            str(rpm),
            f"{ff['BPFI']:.1f}",
            f"{ff['BPFO']:.1f}",
            f"{ff['BSF']:.1f}",
            f"{ff['FTF']:.1f}",
        ])

    tbl = ax_tbl.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.1, 2.0)

    # Header colouring
    header_colors = ["#cccccc", "#cccccc", "#ffcccc", "#cce5ff", "#ffe8cc", "#e8e8e8"]
    for j, color in enumerate(header_colors):
        tbl[(0, j)].set_facecolor(color)
        tbl[(0, j)].set_text_props(fontweight="bold")

    ax_tbl.set_title("Characteristic Fault Frequencies\n(SKF 6205-2RS, Drive End)",
                     fontsize=12, fontweight="bold", pad=10)

    # ── Bar chart ──────────────────────────────────────────────────────────
    rpm_nom   = 1797
    ff_nom    = fault_freqs(rpm_nom)
    freq_keys = ["BPFI", "BPFO", "BSF", "FTF"]
    freq_vals = [ff_nom[k] for k in freq_keys]
    colors    = ["#e41a1c", "#377eb8", "#ff7f00", "#888888"]
    bars = ax_bar.barh(freq_keys, freq_vals, color=colors, edgecolor="white",
                       linewidth=0.8, height=0.55)

    for bar, val in zip(bars, freq_vals):
        ax_bar.text(val + 2, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f} Hz", va="center", ha="left", fontsize=11, fontweight="bold")

    # Harmonics (dashed lines at 2× and 3×)
    for val, color in zip(freq_vals, colors):
        for mult in [2, 3]:
            ax_bar.axvline(val * mult, color=color, linestyle="--", alpha=0.3, linewidth=0.8)

    ax_bar.set_xlabel("Frequency (Hz)", fontsize=11)
    ax_bar.set_title(f"Fault Frequencies @ {rpm_nom} RPM (0 HP)\nDashed = 2nd and 3rd harmonics",
                     fontsize=12, fontweight="bold")
    ax_bar.set_xlim(0, 550)
    ax_bar.spines[["top", "right"]].set_visible(False)
    ax_bar.tick_params(axis="y", labelsize=12)

    # Multiplier annotations
    multipliers = dict(BPFI=5.415, BPFO=3.585, BSF=2.357, FTF=0.398)
    for i, key in enumerate(freq_keys):
        ax_bar.text(5, i, f"  {multipliers[key]}×fr", va="center", ha="left",
                    fontsize=9, color="white", fontweight="bold")

    fig.suptitle("Bearing Fault Characteristic Frequencies — CWRU SKF 6205-2RS",
                 fontsize=13, fontweight="bold", y=1.01)

    save(fig, os.path.join(out_dir, "M2_fault_frequencies.png"))


# ═════════════════════════════════════════════════════════════════════════════
# M3 — Fault waveforms  (synthetic if raw data unavailable)
# ═════════════════════════════════════════════════════════════════════════════

def _make_synthetic_signal(fault_type, n=2048, fs=12000):
    """Generate physically-motivated synthetic vibration signal for each fault."""
    t  = np.arange(n) / fs
    fr = 1797 / 60
    ff = fault_freqs(1797)
    # Base shaft rotation
    sig = 0.3 * np.sin(2 * np.pi * fr * t)
    np.random.seed(42 + fault_type)
    noise = 0.05 * np.random.randn(n)

    if fault_type == 0:   # Normal
        sig += noise

    elif fault_type == 1: # IR fault — impulsive at BPFI
        bpfi = ff["BPFI"]
        for h in [1, 2, 3]:
            sig += (0.8 / h) * np.sin(2 * np.pi * bpfi * h * t) * (
                1 + 0.5 * np.sin(2 * np.pi * fr * t))   # amplitude modulation
        # Add impulsive component
        T_fi = fs / bpfi
        for start in np.arange(0, n, T_fi, dtype=int):
            end = min(start + 15, n)
            decay = np.exp(-np.arange(end - start) * 500 / fs)
            sig[start:end] += 1.5 * decay * np.sin(2 * np.pi * 3000 * t[start:end])
        sig += noise

    elif fault_type == 2: # Ball fault — BSF, modulated by FTF
        bsf = ff["BSF"]; ftf = ff["FTF"]
        for h in [1, 2]:
            sig += (0.6 / h) * np.sin(2 * np.pi * bsf * h * t) * (
                0.8 + 0.4 * np.sin(2 * np.pi * ftf * t))
        T_bsf = fs / bsf
        for start in np.arange(0, n, T_bsf, dtype=int):
            end = min(start + 12, n)
            decay = np.exp(-np.arange(end - start) * 600 / fs)
            sig[start:end] += 1.2 * decay * np.sin(2 * np.pi * 2500 * t[start:end])
        sig += 0.08 * np.random.randn(n)

    elif fault_type == 3: # OR fault — impulsive at BPFO (stationary, no AM)
        bpfo = ff["BPFO"]
        for h in [1, 2, 3]:
            sig += (1.0 / h) * np.sin(2 * np.pi * bpfo * h * t)
        T_fo = fs / bpfo
        for start in np.arange(0, n, T_fo, dtype=int):
            end = min(start + 10, n)
            decay = np.exp(-np.arange(end - start) * 700 / fs)
            sig[start:end] += 2.0 * decay * np.sin(2 * np.pi * 3500 * t[start:end])
        sig += noise

    return t, sig


def m3_fault_waveforms(out_dir, X_raw=None, y_type=None):
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), facecolor="white",
                             sharex=False, constrained_layout=True)

    fs = 12000
    for ft in range(4):
        ax = axes[ft]

        if X_raw is not None and y_type is not None:
            idx = np.where(y_type == ft)[0]
            if len(idx):
                raw = X_raw[idx[0]]
                t   = np.arange(len(raw)) / fs
                sig = raw
            else:
                t, sig = _make_synthetic_signal(ft)
        else:
            t, sig = _make_synthetic_signal(ft)

        show = min(len(t), int(0.05 * fs))  # 50 ms
        color = list(FAULT_COLORS.values())[ft]
        ax.plot(t[:show] * 1000, sig[:show], color=color, linewidth=0.9)
        ax.set_ylabel("Amplitude (g)", fontsize=9)
        ax.set_title(f"{list(FAULT_COLORS.keys())[ft]}", fontsize=11,
                     fontweight="bold", color=color, loc="left", pad=3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, alpha=0.25)

        # Annotate key waveform stats
        rms = np.sqrt(np.mean(sig ** 2))
        kurt = float(np.mean((sig - sig.mean()) ** 4) / (np.std(sig) ** 4 + 1e-12))
        ax.text(0.98, 0.88, f"RMS={rms:.3f}  Kurt={kurt:.2f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, bbox=dict(boxstyle="round", facecolor="white",
                                      edgecolor="#ccc", alpha=0.85))

    axes[-1].set_xlabel("Time (ms)", fontsize=10)
    fig.suptitle("Vibration Waveforms — 50 ms window @ 12 kHz\n"
                 "CWRU Drive-End Bearing Fault Types", fontsize=13, fontweight="bold")

    save(fig, os.path.join(out_dir, "M3_fault_waveforms.png"))


# ═════════════════════════════════════════════════════════════════════════════
# M4 — Fault FFT with frequency markers
# ═════════════════════════════════════════════════════════════════════════════

def m4_fault_fft(out_dir, X_raw=None, y_type=None):
    fs    = 12000
    ff    = fault_freqs(1797)

    fig, axes = plt.subplots(4, 1, figsize=(14, 11), facecolor="white",
                             sharex=True, constrained_layout=True)

    freq_marker_cfg = {
        0: [],  # Normal — no fault lines
        1: [(ff["BPFI"] * h, f"BPFI×{h}" if h > 1 else "BPFI", "#e41a1c") for h in [1, 2, 3]],
        2: [(ff["BSF"]  * h, f"BSF×{h}"  if h > 1 else "BSF",  "#ff7f00") for h in [1, 2]] +
           [(ff["FTF"],  "FTF", "#bb6600")],
        3: [(ff["BPFO"] * h, f"BPFO×{h}" if h > 1 else "BPFO", "#377eb8") for h in [1, 2, 3]],
    }

    for ft in range(4):
        ax = axes[ft]

        if X_raw is not None and y_type is not None:
            idx = np.where(y_type == ft)[0]
            if len(idx):
                sig = X_raw[idx[0]]
            else:
                _, sig = _make_synthetic_signal(ft)
        else:
            _, sig = _make_synthetic_signal(ft)

        N    = len(sig)
        freq = np.fft.rfftfreq(N, 1.0 / fs)
        mag  = np.abs(np.fft.rfft(sig * np.hanning(N))) / N

        color = list(FAULT_COLORS.values())[ft]
        ax.semilogy(freq, mag + 1e-8, color=color, linewidth=0.7, alpha=0.85)
        ax.set_ylabel("|FFT| (log)", fontsize=9)
        ax.set_title(list(FAULT_COLORS.keys())[ft], fontsize=11,
                     fontweight="bold", color=color, loc="left", pad=2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, alpha=0.2, which="both")
        ax.set_xlim(0, 5000)

        for (fq, label, lc) in freq_marker_cfg[ft]:
            ax.axvline(fq, color=lc, linestyle="--", linewidth=1.2, alpha=0.9)
            ax.text(fq + 8, ax.get_ylim()[1] * 0.3, label, fontsize=7.5,
                    color=lc, rotation=90, va="top", fontweight="bold")

        # Shaft harmonics (grey)
        for h in range(1, 6):
            f_sh = 1797 / 60 * h
            ax.axvline(f_sh, color="#aaa", linestyle=":", linewidth=0.8, alpha=0.5)

    axes[-1].set_xlabel("Frequency (Hz)", fontsize=10)
    fig.suptitle("FFT Spectra — Bearing Fault Types\n"
                 "Dashed = fault harmonics  |  Dotted = shaft harmonics",
                 fontsize=13, fontweight="bold")

    save(fig, os.path.join(out_dir, "M4_fault_fft.png"))


# ═════════════════════════════════════════════════════════════════════════════
# M5 — Feature → fault physics mapping
# ═════════════════════════════════════════════════════════════════════════════

def m5_feature_physics(out_dir):
    # Relevance matrix: rows=features, cols=fault types
    # Values: 0=none, 1=low, 2=medium, 3=high (based on bearing fault literature)
    relevance = np.array([
        # Norm  IR   Ball  OR
        [1,     2,   1,    2],   # RMS
        [1,     3,   2,    3],   # Kurtosis
        [1,     2,   1,    2],   # Skewness
        [1,     3,   2,    3],   # Crest Factor
        [1,     2,   2,    3],   # Peak-to-Peak
        [1,     2,   1,    2],   # Std Dev
        [1,     2,   3,    2],   # Spectral Centroid
    ], dtype=float)

    fault_labels = ["Normal", "IR", "Ball", "OR"]
    cmap = plt.cm.YlOrRd

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="white",
                             gridspec_kw={"width_ratios": [1.5, 1]})

    # ── Heatmap ────────────────────────────────────────────────────────────
    ax = axes[0]
    im = ax.imshow(relevance, cmap=cmap, aspect="auto", vmin=0, vmax=3)
    ax.set_xticks(range(4)); ax.set_xticklabels(fault_labels, fontsize=11)
    ax.set_yticks(range(D)); ax.set_yticklabels(FEATURE_NAMES, fontsize=11)
    ax.set_title("Feature Sensitivity to Fault Type\n(Literature-based, 0=none → 3=high)",
                 fontsize=12, fontweight="bold")

    text_labels = ["", "Low", "Med", "High"]
    for i in range(D):
        for j in range(4):
            val = int(relevance[i, j])
            color = "white" if val >= 2 else "black"
            ax.text(j, i, text_labels[val], ha="center", va="center",
                    fontsize=10, color=color, fontweight="bold")

    fig.colorbar(im, ax=ax, shrink=0.7, label="Sensitivity level")

    # ── Physics explanation panel ──────────────────────────────────────────
    ax2 = axes[1]
    ax2.axis("off")

    explanations = [
        ("RMS / Std Dev", "#e07070",
         "Overall energy level.\nElevated when ANY fault\ncauses increased vibration."),
        ("Kurtosis / Crest Factor", "#e07070",
         "Impulsiveness (4th moment ratio).\nHigh for IR and OR because\nball-raceway impacts = sharp spikes."),
        ("Skewness", "#ffaa55",
         "Waveform asymmetry.\nSlightly elevated for directional\nload asymmetry in faults."),
        ("Peak-to-Peak", "#ffaa55",
         "Max excursion range.\nSensitive to large impulsive\nevents (OR > IR due to load zone)."),
        ("Spectral Centroid", "#77aaff",
         "Frequency centre of mass.\nBall faults shift energy to BSF\nregion (2–3 kHz band)."),
    ]

    y = 0.95
    for name, color, desc in explanations:
        ax2.text(0.0, y, f"■ {name}", transform=ax2.transAxes,
                 fontsize=10, fontweight="bold", color=color, va="top")
        wrapped = textwrap.fill(desc, 38)
        ax2.text(0.05, y - 0.04, wrapped, transform=ax2.transAxes,
                 fontsize=9, color="#333", va="top", linespacing=1.4)
        y -= 0.19

    ax2.set_title("Physics behind each feature", fontsize=12, fontweight="bold", pad=10)

    fig.suptitle("Feature–Fault Sensitivity Map  |  SKF 6205-2RS",
                 fontsize=13, fontweight="bold")

    save(fig, os.path.join(out_dir, "M5_feature_physics.png"))


# ═════════════════════════════════════════════════════════════════════════════
# M6 — t-SNE of scaled features, coloured by 16-class label
# ═════════════════════════════════════════════════════════════════════════════

def m6_tsne_classes(out_dir, X_scaled, y_type, y_severity, y_16cls):
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("  scikit-learn not available — skipping M6"); return

    print("  Running t-SNE (this may take ~30s)…")
    rng  = np.random.RandomState(42)
    idx  = rng.choice(len(X_scaled), min(3000, len(X_scaled)), replace=False)
    X_s  = X_scaled[idx]
    yt   = y_type[idx]
    ys   = y_severity[idx]
    y16  = y_16cls[idx]

    tsne = TSNE(n_components=2, perplexity=40, random_state=42, max_iter=1000)
    Z    = tsne.fit_transform(X_s)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7), facecolor="white")

    # ── Panel 1: coloured by fault type ────────────────────────────────────
    ax = axes[0]
    cmap4 = ListedColormap(list(FAULT_COLORS.values()))
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=yt, cmap=cmap4, vmin=0, vmax=3,
                    s=8, alpha=0.7, linewidths=0)
    ax.set_title("t-SNE coloured by Fault Type", fontsize=12, fontweight="bold")
    ax.axis("off")
    handles = [mpatches.Patch(color=c, label=l)
               for l, c in FAULT_COLORS.items()]
    ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.9)

    # ── Panel 2: coloured by 16-class label ────────────────────────────────
    ax2 = axes[1]
    n_cls = len(np.unique(y16))
    cmap16 = plt.cm.get_cmap("tab20", n_cls)
    sc2 = ax2.scatter(Z[:, 0], Z[:, 1], c=y16, cmap=cmap16,
                      s=8, alpha=0.7, linewidths=0)
    ax2.set_title(f"t-SNE coloured by 16-class Label\n(fault type × severity size)",
                  fontsize=12, fontweight="bold")
    ax2.axis("off")
    cb = fig.colorbar(sc2, ax=ax2, shrink=0.8, pad=0.01)
    cb.set_label("16-class index", fontsize=9)

    fig.suptitle("t-SNE Embedding of Normalised Features  |  CWRU Dataset v2\n"
                 "(per-load normalised, class-balanced, n≈3000 random sample)",
                 fontsize=12, fontweight="bold")

    save(fig, os.path.join(out_dir, "M6_tsne_classes.png"), dpi=150)


# ═════════════════════════════════════════════════════════════════════════════
# M7 — ARI / Silhouette comparison bar chart (v1 vs v2)
# ═════════════════════════════════════════════════════════════════════════════

def m7_ari_silhouette(out_dir, ari, sil, ari16=None):
    # Published v1 values (from moc_results/summary.json)
    v1 = dict(ari4=0.3014, sil=0.6100, ari16=None)
    v2 = dict(ari4=float(ari.max()), sil=float(sil.max()),
               ari16=float(ari16.max()) if ari16 is not None else None)

    fig, axes = plt.subplots(1, 3, figsize=(13, 5), facecolor="white")

    metrics_cfg = [
        ("ARI (4-class)", "ari4",   "#e41a1c", 0, 1),
        ("ARI (16-class)", "ari16", "#ff7f00", 0, 1),
        ("Silhouette",    "sil",    "#377eb8", -1, 1),
    ]

    for ax, (title, key, color, ymin, ymax) in zip(axes, metrics_cfg):
        v1_val = v1.get(key)
        v2_val = v2.get(key)
        vals   = []
        labels = []
        colors = []
        if v1_val is not None:
            vals.append(v1_val); labels.append("v1\n(buggy)"); colors.append("#bbbbbb")
        if v2_val is not None:
            vals.append(v2_val); labels.append("v2\n(fixed)"); colors.append(color)

        bars = ax.bar(labels, vals, color=colors, edgecolor="white",
                      linewidth=1.2, width=0.45)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                    f"{val:.4f}", ha="center", va="bottom",
                    fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylim(0, min(ymax * 1.25, 1.1))
        ax.spines[["top", "right"]].set_visible(False)
        ax.axhline(1.0, color="#ccc", linestyle="--", linewidth=0.8)

        # Improvement arrow
        if v1_val is not None and v2_val is not None:
            delta = v2_val - v1_val
            sign  = "+" if delta >= 0 else ""
            ax.annotate("", xy=(1, v2_val - 0.01), xytext=(0, v1_val + 0.01),
                        arrowprops=dict(arrowstyle="->", color="#00aa00", lw=1.5,
                                        connectionstyle="arc3,rad=-0.25"))
            ax.text(0.5, (v1_val + v2_val) / 2, f"{sign}{delta:.4f}",
                    ha="center", va="bottom", fontsize=9, color="#007700",
                    fontweight="bold", transform=ax.get_xaxis_transform())

    fig.suptitle("MOC-FS Performance: v1 (OR bug) vs v2 (fixed)\n"
                 "Higher ARI & Silhouette = better fault cluster alignment",
                 fontsize=13, fontweight="bold")

    save(fig, os.path.join(out_dir, "M7_ari_silhouette.png"))


# ═════════════════════════════════════════════════════════════════════════════
# M8 — 2-D Pareto projections with knee point
# ═════════════════════════════════════════════════════════════════════════════

def _knee_point(F):
    F_norm = (F - F.min(0)) / (F.max(0) - F.min(0) + 1e-12)
    return int(np.argmin(np.linalg.norm(F_norm, axis=1)))


def m8_pareto_projections(out_dir, F, ari, sil):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="white")
    ki = _knee_point(F)
    bi = int(np.argmax(ari))

    pairs = [(0, 1, "f1 Compactness", "f2 Connectedness"),
             (0, 2, "f1 Compactness", "f3 Simplicity (# feat)"),
             (1, 2, "f2 Connectedness", "f3 Simplicity (# feat)")]

    norm_ari = (ari - ari.min()) / (ari.max() - ari.min() + 1e-12)

    for ax, (i, j, xl, yl) in zip(axes, pairs):
        sc = ax.scatter(F[:, i], F[:, j], c=norm_ari, cmap="RdYlGn",
                        s=35, alpha=0.8, edgecolors="white", linewidths=0.4)
        ax.scatter(F[ki, i], F[ki, j], color="red",  s=200, marker="D",
                   zorder=10, label=f"Knee  (ARI={ari[ki]:.3f})", linewidths=1.5,
                   edgecolors="darkred")
        ax.scatter(F[bi, i], F[bi, j], color="gold", s=200, marker="s",
                   zorder=11, label=f"Best ARI={ari[bi]:.3f}", linewidths=1.5,
                   edgecolors="darkorange")
        ax.set_xlabel(xl, fontsize=10)
        ax.set_ylabel(yl, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.2)

    cb = fig.colorbar(sc, ax=axes[-1], shrink=0.85, pad=0.03)
    cb.set_label("ARI (normalised)", fontsize=9)

    fig.suptitle("Pareto Front Projections — MOC-FS v2\n"
                 "Red ◆ = Knee point  |  Gold ■ = Best ARI  |  Colour = ARI",
                 fontsize=13, fontweight="bold")

    save(fig, os.path.join(out_dir, "M8_pareto_projections.png"))


# ═════════════════════════════════════════════════════════════════════════════
# main
# ═════════════════════════════════════════════════════════════════════════════

def main(results_dir: str):
    out_dir = make_outdir(results_dir)
    print(f"Output directory: {out_dir}")

    # ── Load Pareto arrays ────────────────────────────────────────────────
    F   = np.load(os.path.join(results_dir, "pareto_F.npy"))
    X   = np.load(os.path.join(results_dir, "pareto_X.npy"))
    ari = np.load(os.path.join(results_dir, "pareto_ari.npy"))
    sil = np.load(os.path.join(results_dir, "pareto_sil.npy"))
    print(f"  Pareto: {len(F)} solutions  |  ARI max={ari.max():.4f}  Sil max={sil.max():.4f}")

    # ── Optional arrays (may not exist in v1 results) ─────────────────────
    def _try_load(fname):
        p = os.path.join(results_dir, fname)
        if os.path.isfile(p):
            return np.load(p)
        return None

    ari16   = _try_load("pareto_ari16.npy")
    X_scaled = _try_load("X_scaled.npy")
    X_raw    = _try_load("X_raw.npy")
    y_type   = _try_load("y_type.npy")
    y_sev    = _try_load("y_severity.npy")
    y_load   = _try_load("y_load.npy")
    y_16cls  = _try_load("y_16cls.npy")

    # ── Generate all panels ───────────────────────────────────────────────
    print("\n[M1] Bearing schematic…")
    m1_bearing_schematic(out_dir)

    print("[M2] Fault frequency table…")
    m2_fault_frequencies(out_dir)

    print("[M3] Fault waveforms…")
    m3_fault_waveforms(out_dir, X_raw, y_type)

    print("[M4] Fault FFT…")
    m4_fault_fft(out_dir, X_raw, y_type)

    print("[M5] Feature physics map…")
    m5_feature_physics(out_dir)

    if X_scaled is not None and y_type is not None and y_16cls is not None:
        print("[M6] t-SNE (3000-sample)…")
        m6_tsne_classes(out_dir, X_scaled, y_type, y_sev, y_16cls)
    else:
        print("[M6] Skipped — X_scaled / y_type / y_16cls not found in results dir")

    print("[M7] ARI/Silhouette comparison…")
    m7_ari_silhouette(out_dir, ari, sil, ari16)

    print("[M8] Pareto 2-D projections…")
    m8_pareto_projections(out_dir, F, ari, sil)

    print(f"\nAll mechanical visualisations saved to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir", type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "moc_results_v2"),
    )
    args = parser.parse_args()
    main(args.results_dir)
