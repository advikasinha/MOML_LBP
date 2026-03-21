"""
make_3d_pareto.py
=================
Standalone script that reads the saved Pareto arrays from moc_bearing.py and
produces two interactive outputs:

  1. pareto_3d_interactive.html  — fully rotatable Plotly 3-D scatter
                                   (open in any browser, no server needed)
  2. pareto_3d_rotating.gif      — 360° animated GIF (matplotlib)

Run after moc_bearing.py has completed:
  python3 make_3d_pareto.py [--results-dir moc_results]
"""

import os
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ── constants (must match moc_bearing.py) ────────────────────────────────────
D             = 7
MAX_K         = 10
MIN_K         = 2
MSK_START     = 1 + MAX_K * D    # 71
FEATURE_NAMES = [
    "RMS", "Kurtosis", "Skewness",
    "Crest Factor", "Peak-to-Peak", "Std Dev", "Spectral Centroid",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def active_features(x: np.ndarray) -> list[str]:
    mask = x[MSK_START : MSK_START + D] >= 0.5
    if not mask.any():
        mask[np.argmax(x[MSK_START : MSK_START + D])] = True
    return [FEATURE_NAMES[i] for i in range(D) if mask[i]]


def knee_point(F: np.ndarray) -> int:
    F_norm = (F - F.min(0)) / (F.max(0) - F.min(0) + 1e-12)
    return int(np.argmin(np.linalg.norm(F_norm, axis=1)))


# ── 1. Plotly interactive HTML ────────────────────────────────────────────────

def make_plotly_html(F, X, ari, sil, n_active, out_path: str):
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except ImportError:
        print("  plotly not installed — skipping HTML. Run: pip install plotly")
        return

    ki = knee_point(F)
    best_ari_i = int(np.argmax(ari))

    # ── Colour by ARI ────────────────────────────────────────────────────────
    hover_texts = []
    for i in range(len(F)):
        feats = active_features(X[i])
        K_val = int(np.clip(X[i, 0], MIN_K, MAX_K))
        hover_texts.append(
            f"<b>Solution #{i}</b><br>"
            f"f1 Compactness: {F[i,0]:.5f}<br>"
            f"f2 Connectedness: {F[i,1]:.5f}<br>"
            f"f3 Simplicity: {int(F[i,2])} feat<br>"
            f"ARI: {ari[i]:.4f}<br>"
            f"Silhouette: {sil[i]:.4f}<br>"
            f"K: {K_val}<br>"
            f"Features: {', '.join(feats)}"
        )

    # Main scatter
    scatter = go.Scatter3d(
        x=F[:, 0], y=F[:, 1], z=F[:, 2],
        mode="markers",
        marker=dict(
            size=6,
            color=ari,
            colorscale="RdYlGn",
            cmin=0, cmax=max(ari.max(), 0.01),
            colorbar=dict(title="ARI", thickness=18, len=0.7),
            opacity=0.85,
            line=dict(width=0.5, color="white"),
        ),
        text=hover_texts,
        hovertemplate="%{text}<extra></extra>",
        name="Pareto solutions",
    )

    # Knee point marker
    knee_marker = go.Scatter3d(
        x=[F[ki, 0]], y=[F[ki, 1]], z=[F[ki, 2]],
        mode="markers+text",
        marker=dict(size=14, color="red", symbol="diamond",
                    line=dict(width=2, color="darkred")),
        text=["Knee"],
        textposition="top center",
        textfont=dict(size=13, color="red"),
        hovertemplate=(
            f"<b>Knee Point</b><br>"
            f"f1: {F[ki,0]:.5f}<br>f2: {F[ki,1]:.5f}<br>f3: {int(F[ki,2])}<br>"
            f"ARI: {ari[ki]:.4f}<br>Sil: {sil[ki]:.4f}"
            "<extra></extra>"
        ),
        name="Knee point",
    )

    # Best ARI marker
    best_marker = go.Scatter3d(
        x=[F[best_ari_i, 0]], y=[F[best_ari_i, 1]], z=[F[best_ari_i, 2]],
        mode="markers+text",
        marker=dict(size=14, color="gold", symbol="square",
                    line=dict(width=2, color="darkorange")),
        text=["Best ARI"],
        textposition="top center",
        textfont=dict(size=13, color="darkorange"),
        hovertemplate=(
            f"<b>Best ARI Solution</b><br>"
            f"f1: {F[best_ari_i,0]:.5f}<br>f2: {F[best_ari_i,1]:.5f}<br>"
            f"f3: {int(F[best_ari_i,2])}<br>"
            f"ARI: {ari[best_ari_i]:.4f}<br>Sil: {sil[best_ari_i]:.4f}<br>"
            f"Features: {', '.join(active_features(X[best_ari_i]))}"
            "<extra></extra>"
        ),
        name="Best ARI",
    )

    fig = go.Figure(data=[scatter, knee_marker, best_marker])

    fig.update_layout(
        title=dict(
            text="MOC-FS: 3-D Pareto Front — CWRU Bearing Fault Diagnosis<br>"
                 "<sup>Colour = ARI (higher = better fault alignment) | "
                 "Hover for details | Drag to rotate</sup>",
            x=0.5, xanchor="center", font=dict(size=16),
        ),
        scene=dict(
            xaxis=dict(title="f1 Compactness (TWCSS/N/d)", backgroundcolor="rgb(240,240,255)"),
            yaxis=dict(title="f2 Connectedness (kNN penalty)", backgroundcolor="rgb(240,255,240)"),
            zaxis=dict(title="f3 Simplicity (# active features)", backgroundcolor="rgb(255,245,240)",
                       tickvals=[1, 2, 3, 4, 5, 6, 7]),
            camera=dict(eye=dict(x=1.6, y=1.6, z=0.9)),
            bgcolor="white",
        ),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="lightgrey", borderwidth=1),
        margin=dict(l=0, r=0, b=20, t=80),
        width=1100, height=750,
        paper_bgcolor="white",
    )

    # Add annotation panel in the corner
    n_pareto = len(F)
    fig.add_annotation(
        text=(
            f"<b>Summary</b><br>"
            f"Pareto solutions: {n_pareto}<br>"
            f"Best ARI: {ari.max():.4f}<br>"
            f"Best Sil: {sil.max():.4f}<br>"
            f"f3 range: [{int(F[:,2].min())} – {int(F[:,2].max())}] feat"
        ),
        xref="paper", yref="paper",
        x=0.99, y=0.01,
        xanchor="right", yanchor="bottom",
        showarrow=False,
        font=dict(size=11),
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="lightgrey",
        borderwidth=1,
        borderpad=6,
    )

    pio.write_html(fig, file=out_path, auto_open=False,
                   include_plotlyjs="cdn", full_html=True)
    print(f"  Saved interactive HTML: {out_path}")
    print(f"  → Open in any browser. Drag to rotate, scroll to zoom, hover for details.")


# ── 2. Matplotlib rotating GIF ────────────────────────────────────────────────

def make_rotating_gif(F, X, ari, sil, n_active, out_path: str,
                      n_frames: int = 120, fps: int = 24):
    """
    360° rotation GIF.  Each frame steps the azimuth by 3°.
    Points are coloured by ARI; knee (red ◆) and best-ARI (gold ■) are marked.
    Colorbar is created ONCE before the animation loop to avoid frame corruption.
    """
    ki     = knee_point(F)
    best_i = int(np.argmax(ari))

    norm_ari = (ari - ari.min()) / (ari.max() - ari.min() + 1e-12)
    colours  = cm.RdYlGn(norm_ari)
    sizes    = 25 + 15 * (n_active - n_active.min())

    # Fixed axis limits so the view doesn't jump between frames
    xlim = (F[:, 0].min(), F[:, 0].max())
    ylim = (F[:, 1].min(), F[:, 1].max())
    zlim = (F[:, 2].min(), F[:, 2].max())

    fig = plt.figure(figsize=(10, 8), facecolor="white")
    ax  = fig.add_subplot(111, projection="3d", facecolor="white")

    # ── Colorbar created ONCE outside the animation loop ─────────────────────
    sm = plt.cm.ScalarMappable(
        cmap="RdYlGn",
        norm=plt.Normalize(vmin=float(ari.min()), vmax=float(ari.max())),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.45, pad=0.08, aspect=20)
    cbar.set_label("ARI", fontsize=9)

    azimuths = np.linspace(0, 360, n_frames, endpoint=False)

    def draw_frame(frame):
        ax.cla()
        ax.set_facecolor("white")

        ax.scatter(F[:, 0], F[:, 1], F[:, 2],
                   c=colours, s=sizes, alpha=0.80,
                   edgecolors="none", depthshade=True)

        ax.scatter(F[ki, 0], F[ki, 1], F[ki, 2],
                   color="red", s=220, marker="D", zorder=10,
                   label=f"Knee (ARI={ari[ki]:.3f})")

        ax.scatter(F[best_i, 0], F[best_i, 1], F[best_i, 2],
                   color="gold", s=220, marker="s", zorder=11,
                   edgecolors="darkorange", linewidths=1.2,
                   label=f"Best ARI={ari[best_i]:.3f}")

        ax.set_xlabel("f1  Compactness", fontsize=9, labelpad=6)
        ax.set_ylabel("f2  Connectedness", fontsize=9, labelpad=6)
        ax.set_zlabel("f3  Simplicity", fontsize=9, labelpad=6)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
        ax.set_title(
            "MOC-FS Pareto Front — CWRU Bearing Faults\n"
            "(colour = ARI, size = # active features)",
            fontsize=11, fontweight="bold",
        )
        ax.legend(loc="upper left", fontsize=8, framealpha=0.85)
        ax.view_init(elev=22, azim=azimuths[frame])

    anim = FuncAnimation(fig, draw_frame, frames=n_frames,
                         interval=1000 / fps, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps), dpi=110)
    plt.close(fig)
    print(f"  Saved rotating GIF : {out_path}  ({n_frames} frames @ {fps} fps)")


# ── main ──────────────────────────────────────────────────────────────────────

def main(results_dir: str):
    print(f"Loading Pareto arrays from {results_dir}/…")

    F        = np.load(os.path.join(results_dir, "pareto_F.npy"))
    X        = np.load(os.path.join(results_dir, "pareto_X.npy"))
    ari      = np.load(os.path.join(results_dir, "pareto_ari.npy"))
    sil      = np.load(os.path.join(results_dir, "pareto_sil.npy"))
    n_active = np.load(os.path.join(results_dir, "n_active.npy"))

    print(f"  {len(F)} Pareto solutions loaded.")
    print(f"  f3 range: {int(F[:,2].min())} – {int(F[:,2].max())} features")
    print(f"  ARI  max: {ari.max():.4f}  |  Sil max: {sil.max():.4f}")

    html_path = os.path.join(results_dir, "pareto_3d_interactive.html")
    gif_path  = os.path.join(results_dir, "pareto_3d_rotating.gif")

    print("\n[1] Generating interactive Plotly HTML…")
    make_plotly_html(F, X, ari, sil, n_active, html_path)

    print("\n[2] Generating rotating GIF (120 frames)…")
    make_rotating_gif(F, X, ari, sil, n_active, gif_path)

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir", type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "moc_results"),
    )
    args = parser.parse_args()
    main(args.results_dir)
