"""
Real-data analysis — PXD022992 (directDIA phosphoproteome across 6 melanoma cell lines).

Dataset: "Data-independent Acquisition-based Proteome and Phosphoproteome Profiling
across Six Melanoma Cell Lines" (PRIDE PXD022992).
Spectronaut report (TSV), 55,943 phosphopeptide precursors, 6 cell lines × 2 replicates.

Cell lines and mutation status:
  A375  — BRAF V600E (BRAFi-sensitive prototype)
  SH4   — BRAF V600E
  SK    — SK-MEL-28, BRAF V600E
  7951  — RPMI-7951, BRAF V600E
  G361  — NRAS Q61R (BRAFi-resistant via RAS)
  HTB69 — SK-MEL-31, NRAS mutant / BRAF WT

Analyses:
  1. QC — missing value rates per cell line, intensity distributions
  2. Phosphosite matrix — replicate average → log2 → Perseus imputation
  3. Variance-based selection → heatmap + hierarchical clustering
  4. PCA — cell lines coloured by mutation status
  5. BRAF V600E vs NRAS — differential phosphorylation (t-test, volcano)
  6. KSEA-style — known kinase substrates (MAPK/AKT/mTOR) enrichment by cell line
  7. Cross-cell-line kinase scores — bar chart + heatmap
"""
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import pipeline_config as cfg

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DIA_FILE = ("datas/PXD022992/"
            "20201020_083553_SPnew_Melanoma_directDIA_phosLocalized000"
            "_20201020_Report.xls")

# ---------------------------------------------------------------------------
# Cell-line metadata
# ---------------------------------------------------------------------------
# Map token → (cell_line_label, mutation_group)
#   tokens extracted from column names: SK, HTB69, G361, SH4, A375, 7951
CELL_LINE_META = {
    "A375":  ("A375",    "BRAF_V600E"),
    "SH4":   ("SH-4",    "BRAF_V600E"),
    "SK":    ("SK-MEL-28","BRAF_V600E"),
    "7951":  ("RPMI-7951","BRAF_V600E"),
    "G361":  ("G361",    "NRAS_mut"),
    "HTB69": ("SK-MEL-31","NRAS_mut"),
}
MUTATION_COLORS = {
    "BRAF_V600E": cfg.COLOR_RESIST,
    "NRAS_mut":   cfg.COLOR_CONTROL,
}

# Kinase-substrate sets for KSEA
KSEA_SETS = {
    "ERK (MAPK1/3)": {
        "MAPK1","MAPK3","MAP2K1","MAP2K2","RPS6KA1","RPS6KA2","RPS6KA3",
        "ELK1","ELK4","ETV4","ETV5","STMN1","DUSP4","DUSP6","KSR1","FOS",
        "FOSL1","JUN","EIF4EBP1","MKNK1","MKNK2","MAPKAPK2",
    },
    "AKT": {
        "AKT1","AKT2","AKT3","PRAS40","GSK3B","FOXO1","FOXO3","TSC2",
        "MDM2","BAD","IRS1","BRAF","RAF1","PIK3R1","PIK3R2",
    },
    "mTOR": {
        "MTOR","RPS6KB1","RPS6KB2","RPS6","EIF4EBP1","ULK1","ULK2",
        "4EBP1","S6K1","S6K2","LARP1","TIF1B",
    },
    "CDK (cell-cycle)": {
        "CDK1","CDK2","CDK4","CDK6","CDK7","RB1","CCND1","CCNE1",
        "E2F1","E2F4","CDKN1A","CDKN1B","TP53",
    },
    "FAK/SRC": {
        "PTK2","SRC","FYN","YES1","LCK","BCAR1","PXN","CTTN","CRKL",
        "CRK","NEDD9","TLN1",
    },
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"\[(\d+)\]\s+\S+_(SK|HTB69|G361|SH4|A375|7951)_phos_(\d)\.raw")


def _parse_sample_columns(columns):
    """Return dict: token → list of column names (replicates)."""
    groups: dict[str, list[str]] = {}
    for col in columns:
        m = _TOKEN_RE.search(col)
        if m:
            cell = m.group(2)
            groups.setdefault(cell, []).append(col)
    return groups


def _make_site_id(row):
    """Gene_AA-position from ModifiedSequence + ProteinPTMLocations."""
    gene = str(row["PG.Genes"]).split(";")[0].strip()
    ptm = str(row["EG.ProteinPTMLocations"]).split(";")[0].strip()
    # e.g. "(S435)" → "S435"; "(C148,S159)" → "S159" (last phospho)
    pts = re.findall(r"[STY]\d+", ptm)
    site = pts[-1] if pts else "unk"
    return f"{gene}_{site}"


def _make_unique(labels):
    seen, out = {}, []
    for x in labels:
        if x in seen:
            seen[x] += 1; out.append(f"{x}.{seen[x]}")
        else:
            seen[x] = 0; out.append(x)
    return pd.Index(out)


def load_dia_matrix():
    """
    Return (matrix, meta):
      matrix — log2 intensity, rows=phosphosites, cols=cell-line tokens
      meta   — gene, site_id, modification
    """
    df = pd.read_csv(DIA_FILE, sep="\t", low_memory=False)
    df = df.replace("Filtered", np.nan)

    groups = _parse_sample_columns(df.columns)
    print(f"[info] cell lines detected: {sorted(groups.keys())}")

    # Build replicate-averaged matrix (log2 first, then average)
    avg_cols = {}
    for cell, cols in groups.items():
        nums = df[cols].apply(pd.to_numeric, errors="coerce")
        with np.errstate(invalid="ignore"):
            log2 = np.log2(nums.replace(0, np.nan))
        avg_cols[cell] = log2.mean(axis=1).values

    matrix = pd.DataFrame(avg_cols)

    # Site identifiers
    site_raw = df.apply(_make_site_id, axis=1).values
    matrix.index = _make_unique(site_raw)
    meta = pd.DataFrame({
        "gene": df["PG.Genes"].str.split(";").str[0].values,
        "modified_seq": df["EG.ModifiedSequence"].values,
        "ptm_location": df["EG.ProteinPTMLocations"].values,
    }, index=matrix.index)

    # Drop sites missing in ALL cell lines
    before = len(matrix)
    matrix = matrix.dropna(how="all")
    meta = meta.loc[matrix.index]
    print(f"[info] sites before={before} | after drop-all-NA={len(matrix)}")

    # Impute remaining NaN (Perseus-style: mean - 1.8*std per column)
    matrix = _impute(matrix)
    return matrix, meta


def _impute(df):
    out = df.copy()
    for col in out.columns:
        vals = out[col].dropna()
        if len(vals) == 0:
            continue
        fill = vals.mean() - 1.8 * vals.std()
        out[col] = out[col].fillna(fill)
    return out


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------
BRAF_LINES = [k for k, (_, g) in CELL_LINE_META.items() if g == "BRAF_V600E"]
NRAS_LINES = [k for k, (_, g) in CELL_LINE_META.items() if g == "NRAS_mut"]

# Filter to cell lines present in matrix
def _braf(mat): return [c for c in BRAF_LINES if c in mat.columns]
def _nras(mat): return [c for c in NRAS_LINES  if c in mat.columns]


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_qc(matrix_raw_log2, meta):
    """Missing value rates and intensity distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Missing rate per cell line (from raw before imputation)
    df = pd.read_csv(DIA_FILE, sep="\t", low_memory=False).replace("Filtered", np.nan)
    groups = _parse_sample_columns(df.columns)
    miss = {}
    for cell, cols in sorted(groups.items()):
        nums = df[cols].apply(pd.to_numeric, errors="coerce")
        miss[CELL_LINE_META.get(cell, (cell, ""))[0]] = nums.isna().mean(axis=None) * 100
    miss_s = pd.Series(miss).sort_values()
    colors_bar = [MUTATION_COLORS[CELL_LINE_META[k][1]] for k in sorted(groups.keys())]
    # reorder colors to match sorted miss_s
    order_keys = [k for k, v in sorted(CELL_LINE_META.items(), key=lambda x: miss.get(x[1][0], 0))]
    colors_bar2 = [MUTATION_COLORS[CELL_LINE_META[k][1]] for k in order_keys if CELL_LINE_META[k][0] in miss_s.index]

    axes[0].barh(range(len(miss_s)), miss_s.values, color=cfg.COLOR_CONTROL, edgecolor="#333", linewidth=0.5)
    axes[0].set_yticks(range(len(miss_s))); axes[0].set_yticklabels(miss_s.index)
    axes[0].set_xlabel("Missing values (%)")
    axes[0].set_title("Missing value rate per cell line")
    axes[0].axvline(30, color=cfg.COLOR_RESIST, ls="--", lw=0.8, label="30% threshold")
    axes[0].legend(fontsize=8)

    # Intensity distribution after imputation
    for cell in matrix_raw_log2.columns:
        label, mut = CELL_LINE_META.get(cell, (cell, "BRAF_V600E"))
        axes[1].hist(matrix_raw_log2[cell].dropna(), bins=80, alpha=0.5,
                     color=MUTATION_COLORS[mut], label=label, density=True)
    axes[1].set_xlabel("log2 intensity")
    axes[1].set_ylabel("density")
    axes[1].set_title("Phosphopeptide intensity distributions (log2, imputed)")
    handles = [mpatches.Patch(color=MUTATION_COLORS["BRAF_V600E"], label="BRAF V600E"),
               mpatches.Patch(color=MUTATION_COLORS["NRAS_mut"],   label="NRAS mut")]
    axes[1].legend(handles=handles)

    fig.tight_layout()
    cfg.save_figure(fig, "pxd022992_qc")


def plot_heatmap(matrix, meta, n_sites=500):
    """Top-variance sites heatmap with cell-line clustering."""
    var = matrix.var(axis=1).nlargest(n_sites).index
    sub = matrix.loc[var]

    # column annotation colors
    col_colors = pd.Series(
        {c: MUTATION_COLORS[CELL_LINE_META.get(c, (c, "BRAF_V600E"))[1]] for c in sub.columns},
        name="Mutation"
    )
    labels = {c: CELL_LINE_META.get(c, (c, ""))[0] for c in sub.columns}
    sub_labeled = sub.rename(columns=labels)
    col_colors_labeled = col_colors.rename(index=labels)

    g = sns.clustermap(sub_labeled, cmap="RdBu_r", center=0,
                       col_colors=col_colors_labeled,
                       xticklabels=True, yticklabels=False,
                       figsize=(9, 10),
                       cbar_kws={"label": "log2 intensity (z-scored across cell lines)"},
                       z_score=0)
    g.ax_heatmap.set_title(f"Top {n_sites} most variable phosphosites", pad=12, fontsize=12)
    handles = [mpatches.Patch(color=MUTATION_COLORS["BRAF_V600E"], label="BRAF V600E"),
               mpatches.Patch(color=MUTATION_COLORS["NRAS_mut"],   label="NRAS mut")]
    g.ax_col_dendrogram.legend(handles=handles, loc="center", ncol=2, frameon=False)
    cfg.save_figure(g.fig, "pxd022992_heatmap_top_variance")


def plot_pca(matrix):
    """PCA of cell lines; rows=phosphosites, cols=cell lines → transpose."""
    X = matrix.T.values
    sc = StandardScaler()
    Xz = sc.fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(Xz)

    fig, ax = plt.subplots(figsize=(7, 6))
    for i, cell in enumerate(matrix.columns):
        label, mut = CELL_LINE_META.get(cell, (cell, "BRAF_V600E"))
        ax.scatter(coords[i, 0], coords[i, 1],
                   color=MUTATION_COLORS[mut], s=200, zorder=3,
                   edgecolors="#333", linewidths=0.8)
        ax.annotate(label, (coords[i, 0], coords[i, 1]),
                    textcoords="offset points", xytext=(7, 4), fontsize=9)

    handles = [mpatches.Patch(color=MUTATION_COLORS["BRAF_V600E"], label="BRAF V600E"),
               mpatches.Patch(color=MUTATION_COLORS["NRAS_mut"],   label="NRAS mut")]
    ax.legend(handles=handles)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.axhline(0, color="lightgrey", lw=0.6); ax.axvline(0, color="lightgrey", lw=0.6)
    ax.set_title("PCA of 6 melanoma cell lines — directDIA phosphoproteome")
    cfg.save_figure(fig, "pxd022992_pca_cell_lines")


def plot_volcano(matrix, meta):
    """Volcano: BRAF V600E vs NRAS — mean difference and t-test."""
    braf_c = _braf(matrix)
    nras_c = _nras(matrix)
    if not braf_c or not nras_c:
        print("[warn] volcano: not enough groups"); return

    rows = []
    for site in matrix.index:
        b = matrix.loc[site, braf_c].values.astype(float)
        n = matrix.loc[site, nras_c].values.astype(float)
        if np.std(b) == 0 and np.std(n) == 0:
            continue
        fc = np.mean(b) - np.mean(n)
        if len(b) > 1 and len(n) > 1:
            _, pv = stats.ttest_ind(b, n, equal_var=False)
        else:
            pv = 1.0
        rows.append({"site": site, "gene": meta.loc[site, "gene"],
                     "log2FC_BRAFvsNRAS": fc, "pval": pv})

    res = pd.DataFrame(rows).dropna()
    res["-log10p"] = -np.log10(res["pval"].clip(1e-300))
    P_THRESH = 0.05; FC_THRESH = 1.0

    sig = res[(res["pval"] < P_THRESH) & (res["log2FC_BRAFvsNRAS"].abs() > FC_THRESH)]
    up_braf = sig[sig["log2FC_BRAFvsNRAS"] > 0]
    up_nras = sig[sig["log2FC_BRAFvsNRAS"] < 0]

    fig, ax = plt.subplots(figsize=(9, 6))
    ns = res[~res.index.isin(sig.index)]
    ax.scatter(ns["log2FC_BRAFvsNRAS"], ns["-log10p"],
               s=6, alpha=0.25, c=cfg.COLOR_NS, edgecolors="none", label="NS")
    ax.scatter(up_braf["log2FC_BRAFvsNRAS"], up_braf["-log10p"],
               s=15, alpha=0.7, c=cfg.COLOR_RESIST, edgecolors="none",
               label=f"Higher BRAF V600E (n={len(up_braf)})")
    ax.scatter(up_nras["log2FC_BRAFvsNRAS"], up_nras["-log10p"],
               s=15, alpha=0.7, c=cfg.COLOR_CONTROL, edgecolors="none",
               label=f"Higher NRAS (n={len(up_nras)})")
    # label top 10 per direction
    for sub in [up_braf.nlargest(5, "log2FC_BRAFvsNRAS"),
                up_nras.nsmallest(5, "log2FC_BRAFvsNRAS")]:
        for _, row in sub.iterrows():
            ax.annotate(row["gene"], (row["log2FC_BRAFvsNRAS"], row["-log10p"]),
                        fontsize=7, textcoords="offset points", xytext=(3, 2),
                        color="#333")

    ax.axhline(-np.log10(P_THRESH), color="dimgrey", ls="--", lw=0.8)
    ax.axvline(FC_THRESH, color="dimgrey",  ls=":",  lw=0.8)
    ax.axvline(-FC_THRESH, color="dimgrey", ls=":",  lw=0.8)
    ax.set_xlabel("log2FC (BRAF V600E − NRAS)")
    ax.set_ylabel("−log10(p-value)")
    ax.set_title("Differential phosphorylation: BRAF V600E vs NRAS mutant cell lines")
    ax.legend(fontsize=9)
    fig.tight_layout()
    cfg.save_figure(fig, "pxd022992_volcano_braf_vs_nras")
    cfg.save_table(res.sort_values("pval"), "PXD022992_differential_phospho_BRAFvsNRAS.csv")
    print(f"[info] volcano: {len(up_braf)} higher in BRAF V600E, {len(up_nras)} higher in NRAS")
    return res


def plot_ksea(matrix, meta):
    """
    KSEA-style: for each kinase set, compute mean phospho-intensity per cell line
    (sites belonging to the kinase's known substrates), z-scored across cell lines.
    """
    gene_to_sites = {}
    for site, g in meta["gene"].items():
        if not isinstance(g, str) or not g:
            continue
        gene_to_sites.setdefault(g.upper(), []).append(site)

    # Build kinase score matrix
    scores = {}
    sizes = {}
    for kinase, substrate_genes in KSEA_SETS.items():
        sites = []
        for g in substrate_genes:
            sites.extend(gene_to_sites.get(g.upper(), []))
        sites = [s for s in sites if s in matrix.index]
        sizes[kinase] = len(sites)
        if len(sites) == 0:
            continue
        scores[kinase] = matrix.loc[sites].mean(axis=0)

    if not scores:
        print("[warn] KSEA: no substrates matched"); return

    ksea_mat = pd.DataFrame(scores).T  # rows=kinase sets, cols=cell lines
    # z-score across cell lines for each kinase set
    ksea_z = ksea_mat.apply(lambda r: (r - r.mean()) / r.std() if r.std() > 0 else r - r.mean(), axis=1)
    # rename columns to pretty labels
    ksea_z.columns = [CELL_LINE_META.get(c, (c, ""))[0] for c in ksea_z.columns]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    # Heatmap
    sns.heatmap(ksea_z, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                linewidths=0.5, ax=axes[0], cbar_kws={"label": "z-score"})
    axes[0].set_title("Kinase pathway activity (substrate mean intensity, z-scored)",
                      fontsize=10)
    axes[0].set_xlabel("")

    # Bar chart — A375 vs G361 as BRAF vs NRAS prototype
    braf_key = "A375"; nras_key = "G361"
    braf_label = CELL_LINE_META.get(braf_key, (braf_key, ""))[0]
    nras_label  = CELL_LINE_META.get(nras_key,  (nras_key, ""))[0]
    if braf_label in ksea_z.columns and nras_label in ksea_z.columns:
        x = np.arange(len(ksea_z))
        w = 0.35
        axes[1].bar(x - w/2, ksea_z[braf_label], w, label=braf_label,
                    color=cfg.COLOR_RESIST, edgecolor="#333", lw=0.5)
        axes[1].bar(x + w/2, ksea_z[nras_label], w, label=nras_label,
                    color=cfg.COLOR_CONTROL, edgecolor="#333", lw=0.5)
        axes[1].set_xticks(x); axes[1].set_xticklabels(ksea_z.index, rotation=20, ha="right")
        axes[1].axhline(0, color="dimgrey", lw=0.6)
        axes[1].set_ylabel("z-score kinase activity")
        axes[1].set_title(f"Kinase activity: {braf_label} (BRAF V600E) vs {nras_label} (NRAS)")
        axes[1].legend()

    fig.tight_layout()
    cfg.save_figure(fig, "pxd022992_ksea_kinase_activity")

    # Export substrate counts + scores
    ksea_out = ksea_mat.copy()
    ksea_out.columns = [CELL_LINE_META.get(c, (c, ""))[0] for c in ksea_out.columns]
    ksea_out.insert(0, "n_substrates", pd.Series(sizes))
    cfg.save_table(ksea_out, "PXD022992_ksea_kinase_scores.csv")
    print(f"[info] KSEA: {len(scores)} kinase sets, substrate counts: "
          + ", ".join(f"{k}={v}" for k, v in sizes.items()))


def export_tables(matrix, meta):
    """Save full matrix and top-variant sites table."""
    full = meta.join(matrix)
    cfg.save_table(full, "PXD022992_phosphosite_matrix.csv")

    # Top 100 most variable sites
    top = matrix.var(axis=1).nlargest(100).index
    top_out = meta.loc[top].join(matrix.loc[top])
    cfg.save_table(top_out, "PXD022992_top100_variable_sites.csv")

    print(f"[info] exported: full matrix {matrix.shape}, top-100 variable sites")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run():
    cfg.apply_style()
    print("=" * 60)
    print("PXD022992 — directDIA phosphoproteome (6 melanoma cell lines)")
    print("=" * 60)

    matrix, meta = load_dia_matrix()

    # Summary stats per cell line
    for cell in matrix.columns:
        label, mut = CELL_LINE_META.get(cell, (cell, "?"))
        print(f"  {label:12s} ({mut:12s}): "
              f"median={matrix[cell].median():+.2f}  "
              f"sites={matrix[cell].notna().sum()}")

    print("\n[plots] QC figures...")
    plot_qc(matrix, meta)

    print("[plots] Heatmap top-500 variable sites...")
    plot_heatmap(matrix, meta, n_sites=500)

    print("[plots] PCA cell lines...")
    plot_pca(matrix)

    print("[plots] Volcano BRAF V600E vs NRAS...")
    plot_volcano(matrix, meta)

    print("[plots] KSEA kinase activity...")
    plot_ksea(matrix, meta)

    print("[tables] Exporting tables...")
    export_tables(matrix, meta)

    print("\nDone. Outputs saved to results/figures/ and results/outputs/")


def main():
    run()


if __name__ == "__main__":
    main()
