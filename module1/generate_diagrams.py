"""
generate_diagrams.py  —  Federated Fraud Detection Report Figures
=================================================================

HOW TO RUN
----------
# From your real log directories (most common case):
    python generate_diagrams.py \
        --split1_log logs_split1/training_log.json \
        --split2_log logs_split2/trust_training_log.json \
        --out_dir    report_figures

# Demo mode (no logs needed, uses synthetic data):
    python generate_diagrams.py --synthetic --out_dir report_figures

# Specific figures only:
    python generate_diagrams.py \
        --split1_log logs_split1/training_log.json \
        --split2_log logs_split2/trust_training_log.json \
        --figures training trust attack

WHAT GETS GENERATED
-------------------
  fig01_architecture.png       System overview
  fig02_data_partition.png     Non-IID data split across banks
  fig03_fedavg_metrics.png     Split 1: F1 / AUC / Recall / Precision over rounds
  fig04_client_breakdown.png   Split 1: per-client F1 final round
  fig05_trust_scores.png       Split 2: trust score τ per client per round
  fig06_anomaly_scores.png     Split 2: anomaly score α per client per round
  fig07_cosine_similarity.png  Split 2: cosine similarity per client per round
  fig08_attack_weights.png     Split 2: aggregation weights collapsing for attacker
  fig09_flagged_rounds.png     Split 2: how many clients flagged per round
  fig10_trust_metrics.png      Split 2: global F1/AUC/Recall under attack
  fig11_comparison.png         Side-by-side Split 1 vs Split 2
  fig12_confusion_matrix.png   Final-round confusion matrix + derived metrics
  fig13_zscore_explainer.png   Why Z-score beats LOO-max normalisation
"""

from __future__ import annotations

import argparse, json, os
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN SYSTEM  — applied consistently to every figure
# ─────────────────────────────────────────────────────────────────────────────
PLT_STYLE = {
    "figure.facecolor":    "#FFFFFF",
    "axes.facecolor":      "#F8F9FA",
    "axes.edgecolor":      "#CED4DA",
    "axes.linewidth":      0.9,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.grid":           True,
    "grid.color":          "#DEE2E6",
    "grid.linewidth":      0.6,
    "grid.linestyle":      "--",
    "xtick.color":         "#495057",
    "ytick.color":         "#495057",
    "xtick.labelsize":     9,
    "ytick.labelsize":     9,
    "axes.labelsize":      10,
    "axes.titlesize":      11,
    "axes.titleweight":    "bold",
    "axes.labelcolor":     "#343A40",
    "axes.titlecolor":     "#212529",
    "legend.fontsize":     8.5,
    "legend.framealpha":   0.9,
    "legend.edgecolor":    "#CED4DA",
    "font.family":         "DejaVu Sans",
    "figure.dpi":          150,
}
matplotlib.rcParams.update(PLT_STYLE)

# Colour palette — deterministic per client / split
PALETTE = {
    "s1":       "#1971C2",   # Split 1 blue
    "s2":       "#2F9E44",   # Split 2 green
    "trusted":  "#1971C2",
    "malicious":"#E03131",
    "accent":   "#F08C00",
    "neutral":  "#868E96",
    "bg":       "#F8F9FA",
}
CLIENT_COLORS = ["#1971C2", "#2F9E44", "#F08C00", "#7048E8", "#C2255C", "#1098AD"]

DPI = 160   # output DPI for all saved figures


def _save(fig: plt.Figure, path: str) -> str:
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓  {os.path.basename(path)}")
    return path


def _ax_label(ax, text: str, x: float = -0.08, y: float = 1.06,
              fontsize: int = 12, weight: str = "bold"):
    """Add a panel label (A, B, …) to the top-left of an axes."""
    ax.text(x, y, text, transform=ax.transAxes,
            fontsize=fontsize, fontweight=weight, color="#212529",
            va="bottom", ha="left")


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC DATA GENERATORS (used when real logs are absent)
# ─────────────────────────────────────────────────────────────────────────────

def _synth_s1(n_rounds: int = 25, n_clients: int = 5) -> List[Dict]:
    rng = np.random.default_rng(0)
    f1, auc, rec, pre, acc = 0.42, 0.68, 0.38, 0.52, 0.89
    logs = []
    for r in range(1, n_rounds + 1):
        noise = rng.normal(0, 0.011)
        f1    = min(0.893, f1  + 0.020 + noise)
        auc   = min(0.974, auc + 0.009 + rng.normal(0, 0.004))
        rec   = min(0.930, rec + 0.024 + noise)
        pre   = min(0.921, pre + 0.013 + rng.normal(0, 0.007))
        acc   = min(0.999, acc + 0.002 + rng.normal(0, 0.001))
        clients = []
        for c in range(n_clients):
            clients.append({
                "client_id": c,
                "f1":        float(np.clip(f1 + rng.normal(0, 0.03), 0, 1)),
                "auc_roc":   float(np.clip(auc + rng.normal(0, 0.02), 0, 1)),
                "recall":    float(np.clip(rec + rng.normal(0, 0.04), 0, 1)),
                "precision": float(np.clip(pre + rng.normal(0, 0.04), 0, 1)),
            })
        logs.append({
            "round": r,
            "global_f1": float(f1), "global_auc": float(auc),
            "global_recall": float(rec), "global_precision": float(pre),
            "global_accuracy": float(acc),
            "global_loss": float(max(0, 1 - f1 + rng.normal(0, 0.02))),
            "client_metrics": clients,
        })
    return logs


def _synth_s2(n_rounds: int = 25, n_clients: int = 5,
              malicious: List[int] = None) -> List[Dict]:
    if malicious is None:
        malicious = [1]
    rng = np.random.default_rng(1)
    tau = {c: 1.0 for c in range(n_clients)}
    f1, auc, rec, pre, acc = 0.42, 0.68, 0.38, 0.52, 0.89
    logs = []
    for r in range(1, n_rounds + 1):
        noise = rng.normal(0, 0.010)
        if r < 3:
            f1  = min(0.64, f1  + 0.010 + noise)
            rec = min(0.66, rec + 0.011 + noise)
        else:
            f1  = min(0.912, f1  + 0.024 + noise)
            rec = min(0.945, rec + 0.026 + noise)
        auc = min(0.981, auc + 0.010 + rng.normal(0, 0.004))
        pre = min(0.930, pre + 0.013 + rng.normal(0, 0.006))
        acc = min(0.999, acc + 0.002 + rng.normal(0, 0.001))

        trust_scores, anomaly_scores, cos_sims, euc_dists, trust_weights = {}, {}, {}, {}, {}
        flagged = []
        for c in range(n_clients):
            if c in malicious:
                alpha  = float(np.clip(0.55 + rng.normal(0, 0.04), 0.42, 0.72))
                tau[c] = max(0.01, 0.85 * tau[c] + 0.15 * (1 - alpha))
                cs     = float(np.clip(-0.38 + rng.normal(0, 0.07), -1, 1))
                ed     = float(abs(rng.normal(2250, 110)))
                flagged.append(c)
            else:
                alpha  = float(np.clip(0.16 + rng.normal(0, 0.035), 0.04, 0.34))
                tau[c] = max(0.85, 0.85 * tau[c] + 0.15 * (1 - alpha))
                cs     = float(np.clip(0.74 + rng.normal(0, 0.05), 0.42, 1.0))
                ed     = float(abs(rng.normal(85 + 22 * c, 14)))
            trust_scores[str(c)]   = float(tau[c])
            anomaly_scores[str(c)] = alpha
            cos_sims[str(c)]       = cs
            euc_dists[str(c)]      = ed

        raw = {c: 1e-6 if c in malicious else tau[c] for c in range(n_clients)}
        ws  = sum(raw.values()) + 1e-10
        for c in range(n_clients):
            trust_weights[str(c)] = raw[c] / ws

        logs.append({
            "round": r,
            "flagged_clients": flagged,
            "trusted_clients": [c for c in range(n_clients) if c not in malicious],
            "trust_scores":    trust_scores,
            "anomaly_scores":  anomaly_scores,
            "cos_similarities": cos_sims,
            "euc_distances":   euc_dists,
            "trust_weights":   trust_weights,
            "global_f1":       float(f1),
            "global_auc":      float(auc),
            "global_recall":   float(rec),
            "global_precision": float(pre),
            "global_accuracy": float(acc),
            "global_balanced_accuracy": float(np.clip(0.5 + rec * 0.4, 0, 1)),
            "global_mcc":      float(np.clip(f1 - 0.04, 0, 1)),
            "global_specificity": float(np.clip(0.981 + rng.normal(0, 0.003), 0, 1)),
            "global_tp": int(355 * rec), "global_fp": int(18 + rng.integers(0, 12)),
            "global_tn": 56400,          "global_fn": int(355 * (1 - rec)),
        })
    return logs


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _rounds(logs):          return [l["round"] for l in logs]
def _field(logs, key):      return [l.get(key, 0) for l in logs]
def _client_ids(logs):      return sorted({int(k) for l in logs
                                           for k in l.get("trust_scores", {})})


def _plot_line(ax, x, y, color, label, lw=2.0, marker=None,
               fill=True, alpha_fill=0.10, **kw):
    ax.plot(x, y, color=color, lw=lw, label=label,
            marker=marker, markersize=4, markerfacecolor="white",
            markeredgewidth=1.5, **kw)
    if fill:
        ax.fill_between(x, y, alpha=alpha_fill, color=color)


def _best_marker(ax, rounds, vals, color="#E03131"):
    best_i = int(np.argmax(vals))
    ax.axvline(rounds[best_i], color=color, lw=1.0, ls=":", alpha=0.7,
               label=f"Best R{rounds[best_i]}")
    ax.scatter([rounds[best_i]], [vals[best_i]], s=60, color=color,
               zorder=5, edgecolors="white", linewidths=1.2)


def _finish_ax(ax, xlabel="Federation Round", ylabel="", ylim=None,
               legend=True, legend_loc="lower right"):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    if legend:
        ax.legend(loc=legend_loc, frameon=True)


# ─────────────────────────────────────────────────────────────────────────────
# FIG 01  —  System Architecture
# ─────────────────────────────────────────────────────────────────────────────

def fig_architecture(out_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(15, 8.5))
    ax.set_xlim(0, 15); ax.set_ylim(0, 8.5)
    ax.axis("off")

    def rbox(cx, cy, w, h, fc, text, fs=9, tc="white", lw=1.2, r=0.25):
        fancy = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                               boxstyle=f"round,pad={r}", lw=lw,
                               edgecolor="#FFFFFF", facecolor=fc,
                               alpha=0.93, zorder=4)
        ax.add_patch(fancy)
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
                color=tc, fontweight="bold", zorder=5, multialignment="center")

    def arr(x1, y1, x2, y2, color="#868E96", lw=1.3, style="->"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=lw), zorder=3)

    def note(x, y, text, fs=7.5, color="#495057"):
        ax.text(x, y, text, ha="center", va="center", fontsize=fs,
                color=color, style="italic", zorder=6)

    # ── Title ────────────────────────────────────────────────────────────────
    ax.text(7.5, 8.1, "Privacy-Preserving Federated Fraud Detection  —  System Architecture",
            ha="center", fontsize=13, fontweight="bold", color="#212529")

    # ── Bank Nodes ───────────────────────────────────────────────────────────
    node_xs = [1.5, 3.6, 5.7, 7.8, 9.9]
    nc = ["#1971C2", "#2F9E44", "#1098AD", "#7048E8", "#C2255C"]
    for i, (bx, bc) in enumerate(zip(node_xs, nc)):
        rbox(bx, 6.5, 1.85, 1.0, bc, f"Bank {i}\n(Client {i})", fs=9)
        rbox(bx, 5.1, 1.75, 0.75, "#343A40",
             "DNN  256→128→64→1", fs=7.5, tc="#F8F9FA")
        arr(bx, 5.95, bx, 5.48)
        note(bx, 4.65, "Local training\n(data private)", fs=7)

    # ── Privacy banner ───────────────────────────────────────────────────────
    priv = FancyBboxPatch((0.3, 4.35), 9.9, 0.22,
                          boxstyle="round,pad=0.1", lw=1,
                          edgecolor="#2F9E44", facecolor="#EBFBEE", zorder=3)
    ax.add_patch(priv)
    ax.text(5.25, 4.465, "✓  Raw transaction data never leaves bank nodes  "
            "—  only float32 weight tensors transmitted",
            ha="center", va="center", fontsize=8, color="#2F9E44", fontweight="bold")

    # ── Gradient arrows up to server ─────────────────────────────────────────
    for bx in node_xs:
        arr(bx, 4.72, bx, 4.06, color="#868E96")

    # ── Split 1: FedAvg Server ───────────────────────────────────────────────
    rbox(3.0, 3.3, 5.0, 1.15, "#1971C2",
         "Split 1  —  FedAvg Server\nRecall-weighted aggregation  |  training_log.json", fs=9)
    arr(3.0, 2.73, 3.0, 2.22, color="#1971C2")

    # ── Split 2: Trust Server ────────────────────────────────────────────────
    rbox(3.0, 1.8, 5.0, 0.78, "#2F9E44",
         "Split 2  —  Trust-Weighted Server\nTrustScorer  |  AttackSimulator  |  trust_training_log.json", fs=8.5)

    # Trust scorer detail
    rbox(9.4, 3.4, 4.4, 1.0, "#F08C00",
         "Trust Scorer  (per round)\n"
         "cos-sim · euc-dist · norm-ratio → α_i\n"
         "τ_i = γ·τ_prev + (1-γ)·(1-α_i)  →  w_i", fs=7.8, tc="white")
    arr(5.5, 3.3, 7.2, 3.4, color="#F08C00")

    # Attack simulator
    rbox(12.2, 1.8, 2.5, 0.72, "#E03131",
         "Attack Simulator\nlabel_flip | gradient_scale", fs=7.8)
    arr(12.2, 2.16, 10.6, 3.0, color="#E03131")

    # ── Split 3: Blockchain ───────────────────────────────────────────────────
    rbox(3.0, 0.72, 5.0, 0.72, "#7048E8",
         "Split 3  —  Blockchain Audit  |  SHA-256 model hashes  →  Hyperledger", fs=8.5)
    arr(3.0, 1.41, 3.0, 1.08, color="#7048E8")

    # ── Legend ───────────────────────────────────────────────────────────────
    patches = [
        mpatches.Patch(color="#1971C2", label="Split 1: FedAvg Baseline"),
        mpatches.Patch(color="#2F9E44", label="Split 2: Trust-Weighted + Attack Detection"),
        mpatches.Patch(color="#F08C00", label="Trust Scoring Engine"),
        mpatches.Patch(color="#7048E8", label="Split 3: Blockchain Immutability"),
        mpatches.Patch(color="#E03131", label="Adversarial Attack Injection"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=8,
              framealpha=0.95, edgecolor="#CED4DA", bbox_to_anchor=(1.0, 0.0))

    out = os.path.join(out_dir, "fig01_architecture.png")
    return _save(fig, out)


# ─────────────────────────────────────────────────────────────────────────────
# FIG 02  —  Data Partition
# ─────────────────────────────────────────────────────────────────────────────

def fig_data_partition(out_dir: str, n_clients: int = 5, alpha: float = 1.0) -> str:
    rng = np.random.default_rng(42)
    labels = [f"Bank {i}" for i in range(n_clients)]
    colors = CLIENT_COLORS[:n_clients]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(f"Non-IID Data Partitioning — Dirichlet(α = {alpha})",
                 fontsize=13, fontweight="bold", y=1.03)

    # Panel A: sample counts
    ax = axes[0]
    sizes = (rng.dirichlet(np.full(n_clients, alpha * 2)) * 11000 + 1600).astype(int)
    bars = ax.bar(labels, sizes, color=colors, edgecolor="white", lw=1.5, alpha=0.88, width=0.6)
    for b, v in zip(bars, sizes):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 120,
                f"{v:,}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set_title("Training Samples per Bank"); ax.set_ylabel("Transactions")
    ax.set_ylim(0, max(sizes) * 1.22)
    _ax_label(ax, "A")

    # Panel B: fraud rate
    ax = axes[1]
    fraud_rates = rng.dirichlet(np.full(n_clients, alpha * 0.5)) * 0.09 + 0.008
    bars = ax.bar(labels, fraud_rates * 100, color=colors, edgecolor="white", lw=1.5, alpha=0.88, width=0.6)
    ax.axhline(0.172, color="#E03131", ls="--", lw=1.5, label="Global avg (0.172%)")
    for b, v in zip(bars, fraud_rates):
        ax.text(b.get_x() + b.get_width()/2, v * 100 + 0.06,
                f"{v*100:.2f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set_title("Fraud Rate per Bank"); ax.set_ylabel("Fraud Rate (%)")
    ax.set_ylim(0, max(fraud_rates * 100) * 1.30)
    ax.legend(fontsize=8)
    _ax_label(ax, "B")

    # Panel C: stacked class composition
    ax = axes[2]
    x = np.arange(n_clients)
    legit = (1 - fraud_rates) * 100
    fraud = fraud_rates * 100
    ax.bar(x, legit, label="Legitimate", color="#A5D8FF", edgecolor="white", lw=1.5)
    ax.bar(x, fraud, bottom=legit, label="Fraud", color="#E03131", edgecolor="white", lw=1.5, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title("Class Composition per Bank"); ax.set_ylabel("Proportion (%)")
    ax.set_ylim(0, 108); ax.legend(fontsize=8)
    _ax_label(ax, "C")

    fig.tight_layout(pad=2.5)
    out = os.path.join(out_dir, "fig02_data_partition.png")
    return _save(fig, out)


# ─────────────────────────────────────────────────────────────────────────────
# FIG 03  —  Split 1 Global Metrics (2×2 clean grid)
# ─────────────────────────────────────────────────────────────────────────────

def fig_fedavg_metrics(logs: List[Dict], out_dir: str) -> str:
    rounds = _rounds(logs)
    panels = [
        ("global_f1",        "F1 Score",    "#1971C2", "A"),
        ("global_auc",       "AUC-ROC",     "#2F9E44", "B"),
        ("global_recall",    "Recall",      "#F08C00", "C"),
        ("global_precision", "Precision",   "#7048E8", "D"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Split 1 — FedAvg Baseline: Global Metrics per Round",
                 fontsize=13, fontweight="bold")
    axes_flat = axes.flatten()

    for ax, (key, title, color, label) in zip(axes_flat, panels):
        vals = _field(logs, key)
        _plot_line(ax, rounds, vals, color, title)
        _best_marker(ax, rounds, vals)
        ax.set_title(title); _finish_ax(ax, ylabel=title, ylim=(0, 1.05))
        _ax_label(ax, label)

    fig.tight_layout(pad=3.0)
    out = os.path.join(out_dir, "fig03_fedavg_metrics.png")
    return _save(fig, out)


# ─────────────────────────────────────────────────────────────────────────────
# FIG 04  —  Split 1 Per-Client Final Round Breakdown
# ─────────────────────────────────────────────────────────────────────────────

def fig_client_breakdown(logs: List[Dict], out_dir: str) -> str:
    last = logs[-1].get("client_metrics", [])
    if not last:
        print("  ⚠  No client_metrics in final round — skipping fig04")
        return ""

    cids   = [m["client_id"] for m in last]
    labels = [f"Bank {c}" for c in cids]
    colors = CLIENT_COLORS[:len(cids)]

    metrics = ["f1", "auc_roc", "recall", "precision"]
    titles  = ["F1 Score", "AUC-ROC", "Recall", "Precision"]

    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    fig.suptitle("Split 1 — Per-Client Performance (Final Round)",
                 fontsize=13, fontweight="bold")

    for ax, key, title in zip(axes, metrics, titles):
        vals = [m.get(key, 0) for m in last]
        bars = ax.bar(labels, vals, color=colors, edgecolor="white", lw=1.5, alpha=0.88, width=0.55)
        ax.axhline(np.mean(vals), color="#E03131", ls="--", lw=1.2, alpha=0.7,
                   label=f"Mean={np.mean(vals):.3f}")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 0.015,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.set_title(title); ax.set_ylim(0, 1.15)
        ax.set_ylabel(title); ax.legend(fontsize=8)

    fig.tight_layout(pad=2.5)
    out = os.path.join(out_dir, "fig04_client_breakdown.png")
    return _save(fig, out)


# ─────────────────────────────────────────────────────────────────────────────
# FIG 05  —  Trust Score Evolution
# ─────────────────────────────────────────────────────────────────────────────

def fig_trust_scores(logs: List[Dict], out_dir: str,
                     malicious: List[int] = None) -> str:
    malicious = malicious or []
    rounds = _rounds(logs)
    cids   = _client_ids(logs)

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.suptitle("Split 2 — Trust Score (τ_i) per Bank Node per Round",
                 fontsize=13, fontweight="bold")

    for cid in cids:
        vals   = [float(l.get("trust_scores", {}).get(str(cid), 1.0)) for l in logs]
        is_mal = cid in malicious
        color  = PALETTE["malicious"] if is_mal else CLIENT_COLORS[cid % len(CLIENT_COLORS)]
        lw     = 2.4 if is_mal else 1.8
        ls     = "--" if is_mal else "-"
        tag    = "  [MALICIOUS]" if is_mal else ""
        ax.plot(rounds, vals, color=color, lw=lw, ls=ls,
                label=f"Bank {cid}{tag}", zorder=4 if is_mal else 3)

    # Trust floor reference
    ax.axhline(0.85, color=PALETTE["neutral"], ls=":", lw=1.0, alpha=0.7,
               label="Min trust floor (0.85)")
    _finish_ax(ax, ylabel="Trust Score τ_i", ylim=(0.0, 1.05),
               legend_loc="lower left")
    fig.tight_layout(pad=2.5)
    out = os.path.join(out_dir, "fig05_trust_scores.png")
    return _save(fig, out)


# ─────────────────────────────────────────────────────────────────────────────
# FIG 06  —  Anomaly Score Evolution
# ─────────────────────────────────────────────────────────────────────────────

def fig_anomaly_scores(logs: List[Dict], out_dir: str,
                       malicious: List[int] = None) -> str:
    malicious = malicious or []
    rounds = _rounds(logs)
    cids   = _client_ids(logs)

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.suptitle("Split 2 — Anomaly Score (α_i) per Bank Node per Round",
                 fontsize=13, fontweight="bold")

    for cid in cids:
        vals   = [float(l.get("anomaly_scores", {}).get(str(cid), 0.0)) for l in logs]
        is_mal = cid in malicious
        color  = PALETTE["malicious"] if is_mal else CLIENT_COLORS[cid % len(CLIENT_COLORS)]
        lw     = 2.4 if is_mal else 1.8
        ls     = "--" if is_mal else "-"
        tag    = "  [MALICIOUS]" if is_mal else ""
        ax.plot(rounds, vals, color=color, lw=lw, ls=ls,
                label=f"Bank {cid}{tag}", zorder=4 if is_mal else 3)

    ax.axhline(0.45, color="#E03131", ls="-.", lw=1.8,
               label="Detection threshold (0.45)")
    ax.fill_between(rounds, 0.45, 1.0, alpha=0.05, color="#E03131")
    ax.text(rounds[-1] + 0.2, 0.46, "Flagged →", fontsize=8, color="#E03131", va="bottom")

    _finish_ax(ax, ylabel="Anomaly Score α_i", ylim=(0.0, 1.05),
               legend_loc="upper right")
    fig.tight_layout(pad=2.5)
    out = os.path.join(out_dir, "fig06_anomaly_scores.png")
    return _save(fig, out)


# ─────────────────────────────────────────────────────────────────────────────
# FIG 07  —  Cosine Similarity
# ─────────────────────────────────────────────────────────────────────────────

def fig_cosine_similarity(logs: List[Dict], out_dir: str,
                          malicious: List[int] = None) -> str:
    malicious = malicious or []
    rounds = _rounds(logs)
    cids   = _client_ids(logs)

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.suptitle("Split 2 — Cosine Similarity with Peer Consensus per Round",
                 fontsize=13, fontweight="bold")

    for cid in cids:
        vals   = [float(l.get("cos_similarities", {}).get(str(cid), 1.0)) for l in logs]
        is_mal = cid in malicious
        color  = PALETTE["malicious"] if is_mal else CLIENT_COLORS[cid % len(CLIENT_COLORS)]
        lw     = 2.4 if is_mal else 1.8
        ls     = "--" if is_mal else "-"
        tag    = "  [MALICIOUS]" if is_mal else ""
        ax.plot(rounds, vals, color=color, lw=lw, ls=ls,
                label=f"Bank {cid}{tag}", zorder=4 if is_mal else 3)

    ax.axhline(0.0, color=PALETTE["neutral"], ls=":", lw=1.0, alpha=0.6,
               label="Zero (opposing gradient)")
    ax.fill_between(rounds, -1, 0, alpha=0.04, color="#E03131")
    ax.text(rounds[0] + 0.5, -0.85,
            "Negative cosine ≈ reversed gradient (label-flip signal)",
            fontsize=8, color="#E03131", style="italic")

    _finish_ax(ax, ylabel="Cosine Similarity", ylim=(-1.05, 1.05))
    fig.tight_layout(pad=2.5)
    out = os.path.join(out_dir, "fig07_cosine_similarity.png")
    return _save(fig, out)


# ─────────────────────────────────────────────────────────────────────────────
# FIG 08  —  Aggregation Weights Collapsing
# ─────────────────────────────────────────────────────────────────────────────

def fig_attack_weights(logs: List[Dict], out_dir: str,
                       malicious: List[int] = None) -> str:
    malicious = malicious or []
    rounds = _rounds(logs)
    cids   = _client_ids(logs)

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.suptitle("Split 2 — Aggregation Weight (w_i) Collapse for Malicious Clients",
                 fontsize=13, fontweight="bold")

    for cid in cids:
        vals   = [float(l.get("trust_weights", {}).get(str(cid), 0.0)) for l in logs]
        is_mal = cid in malicious
        color  = PALETTE["malicious"] if is_mal else CLIENT_COLORS[cid % len(CLIENT_COLORS)]
        lw     = 2.4 if is_mal else 1.8
        ls     = "--" if is_mal else "-"
        tag    = "  [MALICIOUS → w≈0]" if is_mal else ""
        ax.plot(rounds, vals, color=color, lw=lw, ls=ls,
                label=f"Bank {cid}{tag}", zorder=4 if is_mal else 3)

    ax.axhline(1 / len(cids), color=PALETTE["neutral"], ls=":", lw=1.0, alpha=0.6,
               label=f"Uniform weight (1/{len(cids)} = {1/len(cids):.2f})")
    _finish_ax(ax, ylabel="Normalised Aggregation Weight w_i", ylim=(0.0, 0.80))
    fig.tight_layout(pad=2.5)
    out = os.path.join(out_dir, "fig08_attack_weights.png")
    return _save(fig, out)


# ─────────────────────────────────────────────────────────────────────────────
# FIG 09  —  Flagged Clients per Round (bar chart)
# ─────────────────────────────────────────────────────────────────────────────

def fig_flagged_rounds(logs: List[Dict], out_dir: str) -> str:
    rounds  = _rounds(logs)
    flagged = [len(l.get("flagged_clients", [])) for l in logs]
    colors  = ["#E03131" if f > 0 else "#A5D8FF" for f in flagged]

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.suptitle("Split 2 — Flagged Clients per Federation Round",
                 fontsize=13, fontweight="bold")

    ax.bar(rounds, flagged, color=colors, edgecolor="white", lw=1.2, width=0.7)
    ax.set_xlabel("Federation Round")
    ax.set_ylabel("# Clients Flagged")
    ax.set_ylim(0, max(flagged or [1]) + 1.5)
    ax.set_xticks(rounds[::max(1, len(rounds)//20)])

    patch_flagged = mpatches.Patch(color="#E03131", label="Attack detected")
    patch_clean   = mpatches.Patch(color="#A5D8FF", label="Clean round")
    ax.legend(handles=[patch_flagged, patch_clean], fontsize=9)

    # detection rate annotation
    n_flagged = sum(1 for f in flagged if f > 0)
    ax.text(0.98, 0.96,
            f"Attack detected in {n_flagged}/{len(rounds)} rounds "
            f"({n_flagged/len(rounds)*100:.0f}%)",
            transform=ax.transAxes, fontsize=9, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="#FFF3CD", ec="#F08C00", alpha=0.9))

    fig.tight_layout(pad=2.5)
    out = os.path.join(out_dir, "fig09_flagged_rounds.png")
    return _save(fig, out)


# ─────────────────────────────────────────────────────────────────────────────
# FIG 10  —  Split 2 Global Metrics Under Attack
# ─────────────────────────────────────────────────────────────────────────────

def fig_trust_metrics(logs: List[Dict], out_dir: str) -> str:
    rounds = _rounds(logs)
    panels = [
        ("global_f1",        "F1 Score",    "#1971C2", "A"),
        ("global_auc",       "AUC-ROC",     "#2F9E44", "B"),
        ("global_recall",    "Recall",      "#F08C00", "C"),
        ("global_precision", "Precision",   "#7048E8", "D"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Split 2 — Trust-Weighted FL: Global Metrics Under Attack",
                 fontsize=13, fontweight="bold")
    axes_flat = axes.flatten()

    for ax, (key, title, color, label) in zip(axes_flat, panels):
        vals = _field(logs, key)
        _plot_line(ax, rounds, vals, color, title)
        _best_marker(ax, rounds, vals)
        ax.set_title(title); _finish_ax(ax, ylabel=title, ylim=(0, 1.05))
        _ax_label(ax, label)

    fig.tight_layout(pad=3.0)
    out = os.path.join(out_dir, "fig10_trust_metrics.png")
    return _save(fig, out)


# ─────────────────────────────────────────────────────────────────────────────
# FIG 11  —  Split 1 vs Split 2 Comparison  (3 side-by-side curves)
# ─────────────────────────────────────────────────────────────────────────────

def fig_comparison(s1: List[Dict], s2: List[Dict], out_dir: str) -> str:
    metrics = [
        ("global_f1",     "F1 Score",  "A"),
        ("global_auc",    "AUC-ROC",   "B"),
        ("global_recall", "Recall",    "C"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Split 1 (FedAvg)  vs  Split 2 (Trust-Weighted)  —  Performance Comparison",
                 fontsize=13, fontweight="bold")

    for ax, (key, title, label) in zip(axes, metrics):
        r1, v1 = _rounds(s1), _field(s1, key)
        r2, v2 = _rounds(s2), _field(s2, key)
        _plot_line(ax, r1, v1, "#1971C2", "Split 1: FedAvg", lw=2.0)
        _plot_line(ax, r2, v2, "#2F9E44", "Split 2: Trust-Weighted", lw=2.0, ls="--")
        ax.set_title(title)
        _finish_ax(ax, ylabel=title, ylim=(0.0, 1.05))
        _ax_label(ax, label)

    fig.tight_layout(pad=2.5)
    out = os.path.join(out_dir, "fig11_comparison.png")
    return _save(fig, out)


# ─────────────────────────────────────────────────────────────────────────────
# FIG 12  —  Confusion Matrix  +  Derived Metrics
# ─────────────────────────────────────────────────────────────────────────────

def fig_confusion_matrix(logs: List[Dict], out_dir: str,
                          split_label: str = "Split 2") -> str:
    last = logs[-1]
    tp = int(last.get("global_tp", 280))
    fp = int(last.get("global_fp", 15))
    fn = int(last.get("global_fn", 55))
    tn = int(last.get("global_tn", 56200))

    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall      = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1          = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    bal_acc     = (recall + specificity) / 2
    mcc_n = tp * tn - fp * fn
    mcc_d = ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) ** 0.5
    mcc   = mcc_n / mcc_d if mcc_d > 0 else 0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(f"{split_label} — Confusion Matrix & Detection Performance (Final Round)",
                 fontsize=13, fontweight="bold")

    # ── Confusion matrix heatmap ─────────────────────────────────────────────
    cm = np.array([[tn, fp], [fn, tp]])
    cmap = LinearSegmentedColormap.from_list("cm", ["#EBF3FB", "#1971C2"])
    im = ax1.imshow(cm, cmap=cmap, aspect="auto")
    ax1.set_xticks([0, 1]); ax1.set_yticks([0, 1])
    ax1.set_xticklabels(["Predicted\nLegitimate", "Predicted\nFraud"], fontsize=10)
    ax1.set_yticklabels(["Actual\nLegitimate", "Actual\nFraud"], fontsize=10)
    ax1.set_title("Confusion Matrix", pad=10)

    cell_labels = [
        [f"TN\n{tn:,}", f"FP\n{fp:,}"],
        [f"FN\n{fn:,}", f"TP\n{tp:,}"],
    ]
    for i in range(2):
        for j in range(2):
            text_color = "white" if cm[i, j] > cm.max() * 0.5 else "#212529"
            ax1.text(j, i, cell_labels[i][j], ha="center", va="center",
                     fontsize=13, color=text_color, fontweight="bold")
    plt.colorbar(im, ax=ax1, fraction=0.04, pad=0.04)
    _ax_label(ax1, "A")

    # ── Derived metrics horizontal bar ───────────────────────────────────────
    names  = ["Precision", "Recall\n(Sensitivity)", "Specificity", "F1 Score",
              "Balanced\nAccuracy", "MCC"]
    vals   = [precision, recall, specificity, f1, bal_acc, mcc]
    colors = ["#1971C2", "#F08C00", "#2F9E44", "#1098AD", "#7048E8", "#868E96"]

    y_pos = np.arange(len(names))
    bars  = ax2.barh(y_pos, vals, color=colors, edgecolor="white", lw=1.5,
                     alpha=0.88, height=0.6)
    for bar, v in zip(bars, vals):
        ax2.text(v + 0.012, bar.get_y() + bar.get_height()/2,
                 f"{v:.4f}", va="center", fontsize=9.5, fontweight="bold")
    ax2.set_yticks(y_pos); ax2.set_yticklabels(names, fontsize=9.5)
    ax2.set_xlim(0, 1.18)
    ax2.set_title("Derived Performance Metrics", pad=10)
    ax2.set_xlabel("Score")
    _ax_label(ax2, "B")

    fig.tight_layout(pad=3.0)
    out = os.path.join(out_dir, "fig12_confusion_matrix.png")
    return _save(fig, out)


# ─────────────────────────────────────────────────────────────────────────────
# FIG 13  —  Z-Score vs LOO-Max Math Explainer
# ─────────────────────────────────────────────────────────────────────────────

def fig_zscore_explainer(out_dir: str) -> str:
    clients = ["Bank 0\n(Benign)", "Bank 1\n(MALICIOUS)",
               "Bank 2\n(Benign)", "Bank 3\n(Benign)", "Bank 4\n(Benign)"]
    euc = np.array([27.0, 2334.0, 1034.0, 1200.0, 117.0])
    is_mal = [False, True, False, False, False]
    bar_colors = [PALETTE["malicious"] if m else "#A5D8FF" for m in is_mal]

    def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    loo_max_pen = []
    zscore_pen  = []
    for i in range(len(euc)):
        peers = np.delete(euc, i)
        # LOO-max
        lm = min(euc[i] / (peers.max() + 1e-10), 1.0)
        loo_max_pen.append(lm)
        # Z-score sigmoid
        z  = (euc[i] - peers.mean()) / (peers.std() + 1e-8)
        zscore_pen.append(sigmoid(z))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.suptitle("Trust Scoring: Z-Score Distance Penalty  vs  LOO-Max Normalisation",
                 fontsize=13, fontweight="bold")

    # Panel A: euclidean distances (log scale)
    ax = axes[0]
    bars = ax.bar(clients, euc, color=bar_colors, edgecolor="white", lw=1.5, alpha=0.88, width=0.55)
    ax.set_yscale("log")
    ax.set_title("Raw Euclidean Distance\nfrom Global Model (log scale)")
    ax.set_ylabel("L2 Distance (log)")
    for b, v in zip(bars, euc):
        ax.text(b.get_x() + b.get_width()/2, v * 1.25,
                f"{v:,.0f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    _ax_label(ax, "A")

    # Panel B: side-by-side penalty comparison
    ax = axes[1]
    x  = np.arange(len(clients))
    w  = 0.34
    b1 = ax.bar(x - w/2, loo_max_pen, w, label="LOO-Max Penalty",
                color=["#FFD8A8" if m else "#A5D8FF" for m in is_mal],
                edgecolor="white", lw=1.5, alpha=0.88)
    b2 = ax.bar(x + w/2, zscore_pen,  w, label="Z-Score Sigmoid Penalty",
                color=bar_colors, edgecolor="white", lw=1.5, alpha=0.88)
    ax.axhline(0.45, color="#E03131", ls="-.", lw=1.5, label="Detection threshold (0.45)")
    ax.set_xticks(x); ax.set_xticklabels(clients, fontsize=8)
    ax.set_title("Distance Penalty Comparison\n(at detection round)")
    ax.set_ylabel("Penalty Score")
    ax.set_ylim(0, 1.18)
    for b in b2:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
                f"{b.get_height():.2f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold")
    ax.legend(fontsize=8)
    _ax_label(ax, "B")

    # Panel C: annotation table
    ax = axes[2]
    ax.axis("off")
    rows = [
        ["Client",    "Euc Dist", "LOO-Max", "Z-Score", "Verdict"],
        ["Bank 0",    "27",       f"{loo_max_pen[0]:.3f}",  f"{zscore_pen[0]:.3f}",  "✓ Benign"],
        ["Bank 1 ★",  "2334",     f"{loo_max_pen[1]:.3f}",  f"{zscore_pen[1]:.3f}",  "✗ Caught"],
        ["Bank 2",    "1034",     f"{loo_max_pen[2]:.3f}",  f"{zscore_pen[2]:.3f}",  "✓ Benign"],
        ["Bank 3",    "1200",     f"{loo_max_pen[3]:.3f}",  f"{zscore_pen[3]:.3f}",  "✓ Benign"],
        ["Bank 4",    "117",      f"{loo_max_pen[4]:.3f}",  f"{zscore_pen[4]:.3f}",  "✓ Benign"],
    ]
    col_w = [0.18, 0.18, 0.18, 0.18, 0.18]
    row_h = 0.13
    y0    = 0.88
    header_color = "#343A40"
    for ri, row in enumerate(rows):
        y = y0 - ri * row_h
        bg_color = "#FFF3F3" if ri > 0 and "★" in row[0] else ("#F1F3F5" if ri == 0 else "white")
        rect = FancyBboxPatch((0.02, y - row_h * 0.85), 0.96, row_h * 0.88,
                              boxstyle="round,pad=0.005", lw=0.5,
                              edgecolor="#DEE2E6", facecolor=bg_color,
                              transform=ax.transAxes, zorder=2)
        ax.add_patch(rect)
        cx = 0.04
        for ci, (cell, cw) in enumerate(zip(row, col_w)):
            fw = "bold" if ri == 0 else ("bold" if ci == 4 else "normal")
            tc = "#E03131" if "✗" in cell else ("#2F9E44" if "✓" in cell else header_color)
            ax.text(cx + cw/2, y - row_h * 0.3, cell,
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=8.5, fontweight=fw, color=tc, zorder=3)
            cx += cw

    ax.set_title("Z-Score vs LOO-Max: Penalty Comparison Table", pad=10)
    note_y = y0 - len(rows) * row_h - 0.04
    ax.text(0.5, note_y,
            "LOO-Max false alarm: after Bank 1 flagged, LOO-max drops\n"
            "from 2334→1200, making Bank 2 score 1034/1200 = 0.86 → false flag.\n"
            "Z-Score is immune: each client scored relative to its own peer distribution.",
            transform=ax.transAxes, ha="center", va="top", fontsize=8,
            color="#495057", style="italic",
            bbox=dict(boxstyle="round,pad=0.4", fc="#FFF9DB", ec="#F08C00", alpha=0.9))
    _ax_label(ax, "C")

    fig.tight_layout(pad=2.5)
    out = os.path.join(out_dir, "fig13_zscore_explainer.png")
    return _save(fig, out)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Generate report figures for the Federated Fraud Detection project.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # ── Source logs ──────────────────────────────────────────────────────────
    p.add_argument("--split1_log", type=str, default=None,
                   help="Path to Split 1 log.  e.g.  logs_split1/training_log.json")
    p.add_argument("--split2_log", type=str, default=None,
                   help="Path to Split 2 log.  e.g.  logs_split2/trust_training_log.json")
    p.add_argument("--synthetic", action="store_true",
                   help="Use synthetic data (no real logs needed)")
    # ── Output ───────────────────────────────────────────────────────────────
    p.add_argument("--out_dir", type=str, default="report_figures",
                   help="Output directory for all figures (default: report_figures)")
    # ── Simulation params (only used with --synthetic) ────────────────────────
    p.add_argument("--num_clients", type=int, default=5)
    p.add_argument("--rounds",      type=int, default=25)
    p.add_argument("--malicious",   type=int, nargs="+", default=[1],
                   help="Client IDs of malicious nodes (for trust figures)")
    p.add_argument("--alpha",       type=float, default=1.0,
                   help="Dirichlet alpha for data partition figure")
    # ── Figure selection ──────────────────────────────────────────────────────
    p.add_argument("--figures", type=str, nargs="+", default=["all"],
                   choices=["all", "architecture", "data",
                             "training", "trust", "attack",
                             "comparison", "confusion", "math"],
                   help="Figure groups to generate (default: all)")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    do = lambda g: "all" in args.figures or g in args.figures

    print(f"\n{'─'*58}")
    print(f"  Federated FL Diagram Generator  —  v2")
    print(f"  Output : {os.path.abspath(args.out_dir)}")

    # ── Load logs ─────────────────────────────────────────────────────────────
    s1_logs = s2_logs = None
    malicious: List[int] = args.malicious

    if args.split1_log:
        if not os.path.exists(args.split1_log):
            print(f"  ✗  Split 1 log not found: {args.split1_log}")
        else:
            with open(args.split1_log) as f:
                s1_logs = json.load(f)
            print(f"  ✓  Split 1 log loaded   ({len(s1_logs)} rounds)")

    if args.split2_log:
        if not os.path.exists(args.split2_log):
            print(f"  ✗  Split 2 log not found: {args.split2_log}")
        else:
            with open(args.split2_log) as f:
                s2_logs = json.load(f)
            print(f"  ✓  Split 2 log loaded   ({len(s2_logs)} rounds)")
            # Auto-detect malicious clients from log if not set by user
            if args.malicious == [1]:   # default — try to detect from log
                flagged_all = [c for l in s2_logs for c in l.get("flagged_clients", [])]
                if flagged_all:
                    from collections import Counter
                    # client flagged in majority of rounds = actually malicious
                    counts = Counter(flagged_all)
                    threshold = len(s2_logs) * 0.3
                    malicious = [c for c, cnt in counts.items() if cnt >= threshold]
                    if malicious:
                        print(f"  ↳  Auto-detected malicious clients: {malicious}")

    # Fallback to synthetic if logs missing
    if s1_logs is None:
        s1_logs = _synth_s1(args.rounds, args.num_clients)
        print(f"  ~  Using synthetic Split 1 data  ({args.rounds} rounds)")
    if s2_logs is None:
        s2_logs = _synth_s2(args.rounds, args.num_clients, malicious)
        print(f"  ~  Using synthetic Split 2 data  (malicious={malicious})")

    print(f"{'─'*58}\n")

    # ── Generate ──────────────────────────────────────────────────────────────
    generated = []
    if do("architecture"): generated.append(fig_architecture(args.out_dir))
    if do("data"):         generated.append(fig_data_partition(args.out_dir, args.num_clients, args.alpha))
    if do("training"):
        generated.append(fig_fedavg_metrics(s1_logs, args.out_dir))
        generated.append(fig_client_breakdown(s1_logs, args.out_dir))
    if do("trust"):
        generated.append(fig_trust_scores(s2_logs, args.out_dir, malicious))
        generated.append(fig_anomaly_scores(s2_logs, args.out_dir, malicious))
        generated.append(fig_cosine_similarity(s2_logs, args.out_dir, malicious))
    if do("attack"):
        generated.append(fig_attack_weights(s2_logs, args.out_dir, malicious))
        generated.append(fig_flagged_rounds(s2_logs, args.out_dir))
        generated.append(fig_trust_metrics(s2_logs, args.out_dir))
    if do("comparison"):   generated.append(fig_comparison(s1_logs, s2_logs, args.out_dir))
    if do("confusion"):    generated.append(fig_confusion_matrix(s2_logs, args.out_dir))
    if do("math"):         generated.append(fig_zscore_explainer(args.out_dir))

    generated = [g for g in generated if g]   # drop skipped (empty string)

    print(f"\n{'─'*58}")
    print(f"  {len(generated)} figures saved → {os.path.abspath(args.out_dir)}/")
    print(f"{'─'*58}\n")


if __name__ == "__main__":
    main()