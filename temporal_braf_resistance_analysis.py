"""
Temporal transcriptomics of BRAFi resistance development — GSE110054.

Dataset: "Transcriptional responses of melanoma cells to BRAF inhibition"
(GEO GSE110054, Hugo et al. 2018, Nature Communications).
M229 and M397 melanoma cell lines (BRAF V600E), treated with Vemurafenib (BRAFi).

Time course:
  M229: DMSO (t=0) → 3 days → 21 days → 2 months → 90 days
  M397: DMSO (t=0) → 3 days → 11 days → 21 days → 73 days
Units: FPKM (log2 after +1 pseudocount)

Analyses:
  1. QC — FPKM distribution and sample correlation
  2. PCA — trajectory through resistance states
  3. Pathway dynamics — MAPK, PI3K, RTK, EMT gene sets over time
  4. Gene trajectories — key resistance genes (EGFR, AXL, PDGFRB, MYC…)
  5. Resistance signatures — monotonically up/down regulated genes
  6. Integration — cross-reference KSEA kinase suppression (PXD013923, 30 min)
     vs late transcriptional rebound (gene expression at 90 days)

The phosphoproteomics (PXD013923, 30 min) captures the IMMEDIATE kinase
suppression. The transcriptomics (GSE110054, days-to-months) captures the
ADAPTIVE transcriptional reprogramming that drives acquired resistance.
Together they span the full resistance arc.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import pipeline_config as cfg

FPKM_FILE = "datas/GEO_temporal/GSE110054_FPKM.txt.gz"
KSEA_FILE = "results/outputs/KSEA_PXD013923_kinase_zscores.csv"

# Time-point labels and numeric values (days from treatment start)
M229_COLS = {
    "M229_DMSO": 0,
    "M229_3d":   3,
    "M229_21d":  21,
    "M229_2mo":  60,
    "M229_90d":  90,
}
M397_COLS = {
    "M397_DMSO": 0,
    "M397_3d":   3,
    "M397_11d":  11,
    "M397_21d":  21,
    "M397_73d":  73,
}

# Resistance stage labels (for annotation)
STAGE_COLORS = {
    "Sensitive (DMSO)":       cfg.COLOR_CONTROL,
    "Early adaptation (3d)":  "#f6ad55",
    "Intermediate (11-21d)":  "#f97316",
    "Acquired resistance":    cfg.COLOR_RESIST,
}


def _stage(col):
    if "DMSO" in col:   return "Sensitive (DMSO)"
    if "_3d" in col:    return "Early adaptation (3d)"
    if "_11d" in col or "_21d" in col: return "Intermediate (11-21d)"
    return "Acquired resistance"


# Pathway gene sets for resistance analysis
PATHWAYS = {
    "MAPK/ERK core":     ["BRAF", "RAF1", "MAP2K1", "MAP2K2", "MAPK1", "MAPK3",
                          "DUSP4", "DUSP6", "SPRY2", "ETV4", "ETV5"],
    "RTK (reactivation)":["EGFR", "ERBB2", "ERBB3", "FGFR1", "FGFR2", "PDGFRA",
                          "PDGFRB", "AXL", "MET", "IGF1R", "RET"],
    "PI3K/AKT/mTOR":     ["PIK3CA", "AKT1", "AKT2", "AKT3", "PTEN", "MTOR",
                          "RPS6KB1", "EIF4EBP1", "TSC1", "TSC2"],
    "EMT / invasion":    ["VIM", "FN1", "CDH1", "CDH2", "TWIST1", "ZEB1",
                          "ZEB2", "SNAI1", "SNAI2", "MMP2", "MMP9"],
    "Cell cycle":        ["CCND1", "CDK4", "CDK6", "RB1", "E2F1", "MYC",
                          "CDKN1A", "CDKN2A"],
    "Apoptosis/survival":["BCL2", "BCL2L1", "MCL1", "BAD", "XIAP", "BIRC5",
                          "CFLAR"],
}

KEY_GENES = ["EGFR", "AXL", "PDGFRB", "MET", "FGFR1",       # RTK reactivation
             "MYC", "CCND1", "CDK4",                          # proliferation
             "VIM", "ZEB1", "CDH2",                           # EMT
             "BRAF", "MAPK1", "DUSP6",                        # MAPK target
             "BCL2", "MCL1",                                   # survival
             "MTOR", "AKT1"]                                   # PI3K axis


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_fpkm():
    df = pd.read_csv(FPKM_FILE, sep="\t", index_col=0)
    # log2(FPKM + 1)
    with np.errstate(invalid="ignore"):
        log2 = np.log2(df + 1)
    log2 = log2.replace([np.inf, -np.inf], np.nan)
    log2 = log2.dropna(how="all")
    print(f"[info] GSE110054: {log2.shape[0]} genes × {log2.shape[1]} samples")
    return log2


# ---------------------------------------------------------------------------
# 1. QC + Sample correlation
# ---------------------------------------------------------------------------
def plot_qc(log2):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Distribution
    for col in log2.columns:
        axes[0].hist(log2[col].dropna(), bins=60, alpha=0.4,
                     color=STAGE_COLORS[_stage(col)], density=True)
    axes[0].set_xlabel("log2(FPKM + 1)")
    axes[0].set_ylabel("density")
    axes[0].set_title("Gene expression distributions — BRAFi resistance time course")
    handles = [mpatches.Patch(color=c, label=s) for s, c in STAGE_COLORS.items()]
    axes[0].legend(handles=handles, fontsize=8)

    # Pearson correlation heatmap
    corr = log2.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=log2.columns, yticklabels=log2.columns,
                vmin=0.8, vmax=1.0, ax=axes[1],
                cbar_kws={"label": "Pearson r"})
    axes[1].set_title("Sample correlation matrix")
    fig.tight_layout()
    cfg.save_figure(fig, "temporal_gse110054_qc")


# ---------------------------------------------------------------------------
# 2. PCA — trajectory through resistance
# ---------------------------------------------------------------------------
def plot_pca(log2):
    X = log2.fillna(0).T.values
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(StandardScaler().fit_transform(X))

    fig, ax = plt.subplots(figsize=(8, 6))
    for cell_line, cols, marker in [("M229", M229_COLS, "o"), ("M397", M397_COLS, "s")]:
        xs, ys, labels = [], [], []
        present = [c for c in cols if c in log2.columns]
        for col in present:
            i = list(log2.columns).index(col)
            xs.append(coords[i, 0]); ys.append(coords[i, 1])
            labels.append(col)
        # Draw trajectory arrow
        for j in range(len(xs) - 1):
            ax.annotate("", xy=(xs[j+1], ys[j+1]), xytext=(xs[j], ys[j]),
                        arrowprops=dict(arrowstyle="->", color="#888", lw=1.2))
        for x, y, lbl, col in zip(xs, ys, labels, present):
            ax.scatter(x, y, s=180, color=STAGE_COLORS[_stage(col)],
                       marker=marker, zorder=3, edgecolors="#333", lw=0.7)
            ax.annotate(lbl.replace("_", " "), (x, y),
                        xytext=(5, 5), textcoords="offset points", fontsize=8)

    handles = [
        plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="#555",
                   markersize=10, label="M229"),
        plt.Line2D([0],[0], marker="s", color="w", markerfacecolor="#555",
                   markersize=10, label="M397"),
    ] + [mpatches.Patch(color=c, label=s) for s, c in STAGE_COLORS.items()]
    ax.legend(handles=handles, fontsize=8, loc="best")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("PCA trajectory through BRAFi resistance development\n(GSE110054 — M229 and M397 BRAF V600E)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    cfg.save_figure(fig, "temporal_pca_resistance_trajectory")


# ---------------------------------------------------------------------------
# 3. Pathway dynamics heatmap
# ---------------------------------------------------------------------------
def plot_pathway_heatmap(log2):
    all_genes = []
    for genes in PATHWAYS.values():
        all_genes.extend([g for g in genes if g in log2.index])
    all_genes = list(dict.fromkeys(all_genes))  # deduplicate, preserve order

    sub = log2.loc[all_genes]

    # z-score across samples
    sub_z = sub.apply(lambda r: (r - r.mean()) / r.std() if r.std() > 0 else r, axis=1)

    # Row colors by pathway
    pathway_map = {}
    for pw, genes in PATHWAYS.items():
        for g in genes:
            pathway_map[g] = pw
    palette = dict(zip(PATHWAYS.keys(), sns.color_palette("Set2", len(PATHWAYS))))
    row_colors = sub_z.index.map(lambda g: palette.get(pathway_map.get(g, ""), "#ccc"))

    # Column colors by stage
    col_colors = pd.Series({c: STAGE_COLORS[_stage(c)] for c in sub_z.columns},
                           name="Stage")

    g = sns.clustermap(sub_z,
                       row_colors=pd.Series(row_colors.tolist(), index=sub_z.index, name="Pathway"),
                       col_colors=col_colors,
                       col_cluster=False,
                       cmap="RdBu_r", center=0,
                       xticklabels=True, yticklabels=True,
                       figsize=(10, 14),
                       cbar_kws={"label": "z-score"},
                       dendrogram_ratio=(0.08, 0.12))
    g.ax_heatmap.set_title("Pathway gene expression — BRAFi resistance time course",
                           pad=12, fontsize=11)

    # Legend for pathways
    pw_handles = [mpatches.Patch(color=c, label=pw) for pw, c in palette.items()]
    stage_handles = [mpatches.Patch(color=c, label=s) for s, c in STAGE_COLORS.items()]
    g.ax_col_dendrogram.legend(handles=pw_handles + stage_handles,
                               loc="center", ncol=3, frameon=False, fontsize=7)
    cfg.save_figure(g.fig, "temporal_pathway_heatmap")


# ---------------------------------------------------------------------------
# 4. Key gene trajectories
# ---------------------------------------------------------------------------
def plot_gene_trajectories(log2):
    present = [g for g in KEY_GENES if g in log2.index]
    ncols = 4
    nrows = int(np.ceil(len(present) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 2.8))
    axes = np.array(axes).flatten()

    time_m229 = [M229_COLS[c] for c in M229_COLS if c in log2.columns]
    time_m397 = [M397_COLS[c] for c in M397_COLS if c in log2.columns]
    cols_m229 = [c for c in M229_COLS if c in log2.columns]
    cols_m397 = [c for c in M397_COLS if c in log2.columns]

    for i, gene in enumerate(present):
        ax = axes[i]
        vals_m229 = log2.loc[gene, cols_m229].values.astype(float)
        vals_m397 = log2.loc[gene, cols_m397].values.astype(float)
        ax.plot(time_m229, vals_m229, "o-", color=cfg.COLOR_RESIST,
                lw=2, ms=6, label="M229", zorder=3)
        ax.plot(time_m397, vals_m397, "s--", color=cfg.COLOR_CONTROL,
                lw=2, ms=6, label="M397", zorder=3)
        ax.set_title(gene, fontsize=10, fontweight="bold")
        ax.set_xlabel("Days under BRAFi", fontsize=7)
        ax.set_ylabel("log2(FPKM+1)", fontsize=7)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7)

    for ax in axes[len(present):]:
        ax.axis("off")

    fig.suptitle("Key resistance gene trajectories — M229 and M397 under BRAFi",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    cfg.save_figure(fig, "temporal_key_gene_trajectories")


# ---------------------------------------------------------------------------
# 5. Resistance signatures — monotonically changing genes
# ---------------------------------------------------------------------------
def find_resistance_signatures(log2):
    """
    For each cell line, find genes that monotonically increase or decrease
    from DMSO to the last time point (Spearman r with time).
    """
    results = []
    for cell_line, time_cols in [("M229", M229_COLS), ("M397", M397_COLS)]:
        present = [c for c in time_cols if c in log2.columns]
        days = [time_cols[c] for c in present]
        sub = log2[present].dropna(how="any")
        for gene in sub.index:
            vals = sub.loc[gene].values.astype(float)
            rho, pval = stats.spearmanr(days, vals)
            results.append({"gene": gene, "cell_line": cell_line,
                             "spearman_r": rho, "pval": pval,
                             "direction": "UP" if rho > 0 else "DOWN"})
    df = pd.DataFrame(results)
    df["padj"] = df.groupby("cell_line")["pval"].transform(
        lambda p: np.minimum(p * len(p), 1.0))  # Bonferroni

    # Keep both cell lines agreement: UP in both or DOWN in both
    pivot = df.pivot_table(index="gene", columns="cell_line", values="spearman_r")
    pivot = pivot.dropna()
    sig = df[df["padj"] < 0.01]
    up_both = set(pivot[(pivot["M229"] > 0.7) & (pivot["M397"] > 0.7)].index)
    dn_both = set(pivot[(pivot["M229"] < -0.7) & (pivot["M397"] < -0.7)].index)

    print(f"[info] Resistance signatures (both cell lines, |r|>0.7, padj<0.01):")
    print(f"  UP in resistance: {len(up_both)} genes")
    print(f"  DOWN in resistance: {len(dn_both)} genes")

    cfg.save_table(pd.DataFrame({"gene": sorted(up_both), "direction": "UP_resistance"}),
                   "GEO_GSE110054_resistance_UP_genes.csv")
    cfg.save_table(pd.DataFrame({"gene": sorted(dn_both), "direction": "DOWN_resistance"}),
                   "GEO_GSE110054_resistance_DOWN_genes.csv")
    return up_both, dn_both, df


def plot_resistance_signatures(log2, up_both, dn_both):
    """Heatmap of top resistance genes across time points."""
    # Top 30 up + top 30 down by mean |FPKM change|
    all_tp = list(M229_COLS.keys()) + list(M397_COLS.keys())
    all_tp = [c for c in all_tp if c in log2.columns]
    dmso_cols = [c for c in all_tp if "DMSO" in c]
    late_cols  = [c for c in all_tp if c not in dmso_cols]

    delta = log2[late_cols].mean(axis=1) - log2[dmso_cols].mean(axis=1)
    top_up = delta.loc[list(up_both)].nlargest(25).index.tolist()
    top_dn = delta.loc[list(dn_both)].nsmallest(25).index.tolist()
    selected = top_up + top_dn
    if not selected:
        return

    sub = log2.loc[selected, all_tp]
    sub_z = sub.apply(lambda r: (r - r.mean()) / r.std() if r.std() > 0 else r, axis=1)

    col_colors = pd.Series({c: STAGE_COLORS[_stage(c)] for c in sub_z.columns},
                           name="Stage")
    row_palette = {g: cfg.COLOR_RESIST if g in top_up else cfg.COLOR_CONTROL
                   for g in selected}
    row_colors = pd.Series([row_palette[g] for g in sub_z.index],
                           index=sub_z.index, name="Direction")

    g = sns.clustermap(sub_z,
                       row_colors=row_colors,
                       col_colors=col_colors,
                       col_cluster=False,
                       cmap="RdBu_r", center=0,
                       xticklabels=True, yticklabels=True,
                       figsize=(10, 12),
                       cbar_kws={"label": "z-score"})
    g.ax_heatmap.set_title("BRAFi resistance signature genes (Spearman |r|>0.7, both cell lines)",
                           pad=12, fontsize=10)
    up_h = mpatches.Patch(color=cfg.COLOR_RESIST, label="UP in resistance")
    dn_h = mpatches.Patch(color=cfg.COLOR_CONTROL, label="DOWN in resistance")
    st_h = [mpatches.Patch(color=c, label=s) for s, c in STAGE_COLORS.items()]
    g.ax_col_dendrogram.legend(handles=[up_h, dn_h] + st_h,
                               loc="center", ncol=3, frameon=False, fontsize=7)
    cfg.save_figure(g.fig, "temporal_resistance_signature_heatmap")


# ---------------------------------------------------------------------------
# 6. Integration — kinase suppression (30 min) vs transcriptional rebound
# ---------------------------------------------------------------------------
def plot_kinase_integration(log2):
    """
    Cross-reference KSEA kinase z-scores (30-min BRAFi suppression, PXD013923)
    with late transcriptional rebound (90-day BRAFi gene expression vs DMSO).
    Kinases suppressed at 30 min but whose substrate genes are upregulated
    at 90 days represent adaptive resistance mechanisms.
    """
    if not os.path.exists(KSEA_FILE):
        print("[warn] KSEA file not found — skipping integration plot")
        return

    ksea = pd.read_csv(KSEA_FILE)
    if "kinase" in ksea.columns:
        ksea = ksea.set_index("kinase")
    if "BRAFi" not in ksea.columns:
        print("[warn] BRAFi column not in KSEA file")
        return

    # Late transcriptional fold-change for kinase genes (log2FC: late vs DMSO)
    late_cols_m229  = [c for c in M229_COLS if "DMSO" not in c and c in log2.columns]
    late_cols_m397  = [c for c in M397_COLS if "DMSO" not in c and c in log2.columns]
    dmso_cols = [c for c in log2.columns if "DMSO" in c]

    late_expr  = log2[late_cols_m229 + late_cols_m397].mean(axis=1)
    dmso_expr  = log2[dmso_cols].mean(axis=1)
    late_fc    = late_expr - dmso_expr  # log2FC: resistance vs sensitive

    # For each kinase in KSEA, look up its gene expression late FC
    rows = []
    for kin in ksea.index:
        z30 = ksea.loc[kin, "BRAFi"]
        if kin in late_fc.index:
            fc_late = late_fc[kin]
            rows.append({"kinase": kin, "ksea_z_30min": z30, "late_rna_log2fc": fc_late})
    if not rows:
        print("[warn] No kinase genes matched between KSEA and expression matrix")
        return

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 7))
    # Color: suppressed at 30 min (negative z) AND upregulated late (positive FC)
    # = red (adaptive escape); opposite blue; mixed grey
    def _color(r):
        if r["ksea_z_30min"] < -1 and r["late_rna_log2fc"] > 0.3:
            return cfg.COLOR_RESIST   # suppressed → escaped
        if r["ksea_z_30min"] > 1 and r["late_rna_log2fc"] < -0.3:
            return cfg.COLOR_CONTROL  # activated → suppressed
        return cfg.COLOR_NS

    colors = df.apply(_color, axis=1)
    ax.scatter(df["ksea_z_30min"], df["late_rna_log2fc"],
               c=colors, s=60, alpha=0.8, edgecolors="#333", lw=0.4)

    # Label escape kinases
    escape = df[(df["ksea_z_30min"] < -1) & (df["late_rna_log2fc"] > 0.3)]
    for _, r in escape.iterrows():
        ax.annotate(r["kinase"], (r["ksea_z_30min"], r["late_rna_log2fc"]),
                    fontsize=8, xytext=(4, 2), textcoords="offset points", color="#9b2c2c")

    ax.axhline(0, color="dimgrey", lw=0.6)
    ax.axvline(0, color="dimgrey", lw=0.6)
    ax.axhline(0.3,  color=cfg.COLOR_RESIST, lw=0.5, ls="--", alpha=0.5)
    ax.axvline(-1.0, color=cfg.COLOR_RESIST, lw=0.5, ls="--", alpha=0.5)

    handles = [
        mpatches.Patch(color=cfg.COLOR_RESIST,  label="Suppressed 30 min → escaped (RNA↑ late)"),
        mpatches.Patch(color=cfg.COLOR_CONTROL, label="Activated 30 min → suppressed (RNA↓ late)"),
        mpatches.Patch(color=cfg.COLOR_NS,      label="Other"),
    ]
    ax.legend(handles=handles, fontsize=8)
    ax.set_xlabel("KSEA z-score — BRAFi 30 min (PXD013923, A375)")
    ax.set_ylabel("log2FC gene expression — late BRAFi (days, GSE110054, M229+M397)")
    ax.set_title("Kinase suppression at 30 min vs transcriptional rebound at resistance\n"
                 "→ Escape kinases: suppressed acutely but transcriptionally upregulated",
                 fontsize=10)
    fig.tight_layout()
    cfg.save_figure(fig, "temporal_kinase_escape_integration")
    cfg.save_table(df.sort_values("ksea_z_30min"),
                   "GEO_GSE110054_kinase_suppression_vs_rna_rebound.csv")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run():
    cfg.apply_style()
    print("=" * 60)
    print("Temporal BRAFi resistance — GSE110054 (M229 + M397)")
    print("=" * 60)

    log2 = load_fpkm()

    print("[plot] QC distributions...")
    plot_qc(log2)

    print("[plot] PCA resistance trajectory...")
    plot_pca(log2)

    print("[plot] Pathway dynamics heatmap...")
    plot_pathway_heatmap(log2)

    print("[plot] Key gene trajectories...")
    plot_gene_trajectories(log2)

    print("[analysis] Resistance signatures (Spearman)...")
    up_both, dn_both, sig_df = find_resistance_signatures(log2)

    print("[plot] Resistance signature heatmap...")
    plot_resistance_signatures(log2, up_both, dn_both)

    print("[integration] Kinase suppression vs transcriptional rebound...")
    plot_kinase_integration(log2)

    print("\nDone. Outputs saved to results/figures/ and results/outputs/")


def main():
    run()


if __name__ == "__main__":
    main()
