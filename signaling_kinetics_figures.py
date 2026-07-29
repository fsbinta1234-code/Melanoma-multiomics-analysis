# -*- coding: utf-8 -*-
"""
signaling_kinetics_figures.py
==============================================================================
Generates the figures that directly mirror the signaling-transition-kinetics
model (docs/phase2/): the three-phase response of a BRAF V600E melanoma cell to
combined BRAFi + MEKi.

Two diagram-matching panels, both from REAL data:

  1. Phase 1 — "KSEA scores drop"  (phase1_ksea_dropping_bars.png)
     Bar chart of KSEA z-scores for the MAPK core under BRAFi (PXD013923),
     reproducing the diagram's dropping KSEA bars = acute MAPK shutdown.

  2. Phase 2/3 — pathway-score trajectories  (signaling_kinetics_pathway_trajectories.png)
     Pathway activity over the BRAFi time course (GSE110054, M229/M397),
     expressed as Δlog2 vs the DMSO baseline of each cell line:
       • MAPK output (ERK target genes) — drops acutely, then rebounds
         (the transcriptional readout of MAPK/phosphoproteomic rebound);
       • PI3K/AKT/mTOR — rises above baseline (parallel-route activation);
       • RTK (AXL, PDGFR, EGFR, MET, ...) — rises progressively;
       • EMT / invasion — rises (phenotypic switching).
     This reproduces the diagram's rising PI3K/AKT/mTOR and RTK panels and the
     rebounding-kinase-activity concept.

Run:  python3 signaling_kinetics_figures.py
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pipeline_config as cfg

FPKM_FILE = "datas/GEO_temporal/GSE110054_FPKM.txt.gz"
KSEA_FILE = "results/outputs/KSEA_PXD013923_kinase_zscores.csv"

# Time points (days) per cell line
CELLS = {
    "M229": {"M229_DMSO": 0, "M229_3d": 3, "M229_21d": 21, "M229_2mo": 60, "M229_90d": 90},
    "M397": {"M397_DMSO": 0, "M397_3d": 3, "M397_11d": 11, "M397_21d": 21, "M397_73d": 73},
}

# Pathway gene sets
PATHWAYS = {
    "MAPK output (ERK targets)": ["DUSP4", "DUSP6", "SPRY2", "SPRY4", "ETV4",
                                  "ETV5", "PHLDA1", "FOS", "EGR1", "DUSP5"],
    "PI3K / AKT / mTOR": ["AKT1", "AKT3", "MTOR", "RPS6KB1", "PIK3CA", "PIK3CB",
                          "IRS1", "IRS2", "GAB1", "RICTOR"],
    "RTK (AXL, PDGFR, EGFR, MET)": ["AXL", "PDGFRB", "PDGFRA", "EGFR", "MET",
                                    "FGFR1", "ERBB3", "IGF1R"],
    "EMT / invasion": ["VIM", "ZEB1", "ZEB2", "TWIST1", "SNAI2", "FN1",
                       "CDH2", "MMP2"],
}
PATHWAY_COLORS = {
    "MAPK output (ERK targets)": cfg.COLOR_RESIST,
    "PI3K / AKT / mTOR": cfg.COLOR_ACCENT,
    "RTK (AXL, PDGFR, EGFR, MET)": cfg.COLOR_CONTROL,
    "EMT / invasion": "#9b59b6",
}

# MAPK core kinases for the Phase-1 KSEA bar chart
MAPK_CORE = ["BRAF", "RAF1", "ARAF", "MAP2K1", "MAP2K2", "MAPK1", "MAPK3",
             "KSR1", "RPS6KA1", "RPS6KA3"]


# ---------------------------------------------------------------------------
# Phase 1 — KSEA dropping bars
# ---------------------------------------------------------------------------
def plot_phase1_ksea_bars():
    if not os.path.exists(KSEA_FILE):
        print(f"[warn] {KSEA_FILE} not found — skipping Phase-1 bar chart")
        return
    k = pd.read_csv(KSEA_FILE)
    kcol = "kinase" if "kinase" in k.columns else k.columns[1]
    k = k.set_index(kcol)
    present = [g for g in MAPK_CORE if g in k.index]
    z = k.loc[present, "BRAFi"].sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [cfg.COLOR_RESIST if v < 0 else cfg.COLOR_CONTROL for v in z.values]
    ax.barh(range(len(z)), z.values, color=colors, edgecolor="#333", lw=0.5)
    ax.set_yticks(range(len(z)))
    ax.set_yticklabels(z.index)
    ax.axvline(0, color="dimgrey", lw=0.8)
    ax.set_xlabel("KSEA z-score under BRAFi (30 min, A375)")
    ax.set_title("Phase 1 — Acute Suppression: KSEA scores drop\n"
                 "MAPK-core kinase activity is shut down (target phosphosites)")
    for i, v in enumerate(z.values):
        ax.text(v - 0.15 if v < 0 else v + 0.15, i, f"{v:+.1f}",
                va="center", ha="right" if v < 0 else "left", fontsize=8)
    ax.set_xlim(min(z.values) - 1.5, 1.0)
    fig.tight_layout()
    cfg.save_figure(fig, "phase1_ksea_dropping_bars")


# ---------------------------------------------------------------------------
# Phase 2/3 — pathway-score trajectories (Δlog2 vs DMSO)
# ---------------------------------------------------------------------------
def compute_trajectories():
    df = pd.read_csv(FPKM_FILE, sep="\t", index_col=0)
    with np.errstate(invalid="ignore"):
        log2 = np.log2(df + 1)

    records = []
    for cell, tmap in CELLS.items():
        cols = [c for c in tmap if c in log2.columns]
        dmso = cols[0]
        for pw, genes in PATHWAYS.items():
            present = [g for g in genes if g in log2.index]
            delta = log2.loc[present, cols].sub(log2.loc[present, dmso], axis=0).mean(axis=0)
            for c in cols:
                records.append({"cell_line": cell, "pathway": pw,
                                "day": tmap[c], "delta_log2_vs_DMSO": float(delta[c]),
                                "n_genes": len(present)})
    return pd.DataFrame(records)


def plot_trajectories(traj):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()
    markers = {"M229": ("o", "-"), "M397": ("s", "--")}

    for ax, pw in zip(axes, PATHWAYS):
        color = PATHWAY_COLORS[pw]
        for cell, (mk, ls) in markers.items():
            sub = traj[(traj["pathway"] == pw) & (traj["cell_line"] == cell)].sort_values("day")
            ax.plot(sub["day"], sub["delta_log2_vs_DMSO"], marker=mk, ls=ls,
                    color=color, lw=2, ms=7, label=cell,
                    markeredgecolor="#333", markeredgewidth=0.5)
        ax.axhline(0, color="dimgrey", lw=0.8, ls=":")
        ax.set_title(pw, fontsize=11, fontweight="bold", color=color)
        ax.set_xlabel("Days under BRAFi")
        ax.set_ylabel("Δ log2 vs DMSO baseline")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Signaling-transition kinetics — pathway activity over the BRAFi time course\n"
                 "MAPK output drops then rebounds; PI3K/AKT/mTOR, RTK and EMT rise "
                 "(GSE110054, M229 & M397)",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    cfg.save_figure(fig, "signaling_kinetics_pathway_trajectories")


def run():
    cfg.apply_style()
    print("=" * 60)
    print("Signaling-transition-kinetics figures (phase2 model)")
    print("=" * 60)

    print("[Phase 1] KSEA dropping bars...")
    plot_phase1_ksea_bars()

    print("[Phase 2/3] Pathway-score trajectories...")
    traj = compute_trajectories()
    plot_trajectories(traj)
    cfg.save_table(traj, "signaling_kinetics_trajectory_scores.csv")

    # Print the key directions for the log
    print("\nPathway change (Δlog2 vs DMSO) at the last time point:")
    for cell in CELLS:
        last_day = max(CELLS[cell].values())
        print(f"  {cell} (day {last_day}):")
        for pw in PATHWAYS:
            sub = traj[(traj.pathway == pw) & (traj.cell_line == cell)].sort_values("day")
            print(f"     {pw:28s}: {sub['delta_log2_vs_DMSO'].iloc[-1]:+.2f}"
                  f"   (min over course: {sub['delta_log2_vs_DMSO'].min():+.2f})")
    print("\nDone.")


def main():
    run()


if __name__ == "__main__":
    main()
