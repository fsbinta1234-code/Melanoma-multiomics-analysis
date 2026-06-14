"""
TCGA-SKCM analysis — molecular subtypes, mutation landscape, survival,
and integration with phosphoproteomics signatures.

Requires datas/TCGA-SKCM/ produced by tcga_skcm_download.py:
  clinical.csv               — patient demographics + survival
  driver_mutations.csv       — BRAF/NRAS/NF1/… somatic mutations
  rnaseq_subset_manifest.csv — manifest of downloaded RNA-seq files
  rnaseq_subset/             — STAR-Counts TSV files (one per sample)

Analyses:
  1. Mutation landscape — frequency bar chart for driver genes
  2. Molecular subtypes — BRAF V600E / NRAS / NF1 / Triple-WT classification
  3. Kaplan-Meier survival by subtype
  4. MAPK pathway expression by subtype (RNA-seq, subset of samples)
  5. Integration: cross-reference PXD022992 BRAF V600E vs NRAS differential
     phospho sites against TCGA subtype-specific gene expression
"""
import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
from scipy import stats

import pipeline_config as cfg

DATA_DIR = "datas/TCGA-SKCM"
RNASEQ_DIR = os.path.join(DATA_DIR, "rnaseq_subset")

# Molecular subtype colors
SUBTYPE_COLORS = {
    "BRAF_V600E": cfg.COLOR_RESIST,
    "NRAS_mut":   cfg.COLOR_CONTROL,
    "NF1_mut":    cfg.COLOR_ACCENT,
    "Triple_WT":  cfg.COLOR_NS,
}

MAPK_GENES = ["BRAF", "MAP2K1", "MAP2K2", "MAPK1", "MAPK3", "RAF1", "ARAF",
              "NRAS", "KRAS", "HRAS", "NF1",
              "RPS6KA1", "RPS6KA3", "ELK1", "DUSP4", "DUSP6", "SPRY2"]

PI3K_GENES = ["PIK3CA", "PIK3CB", "PIK3R1", "AKT1", "AKT2", "AKT3",
              "PTEN", "MTOR", "RPS6KB1", "EIF4EBP1"]


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_data():
    clinical = pd.read_csv(os.path.join(DATA_DIR, "clinical.csv"))
    mutations = pd.read_csv(os.path.join(DATA_DIR, "driver_mutations.csv"))
    return clinical, mutations


def assign_subtypes(clinical, mutations):
    """Assign primary molecular subtype per patient (priority: BRAF > NRAS > NF1 > WT)."""
    braf = set(mutations[(mutations["gene"] == "BRAF") &
                          (mutations["aa_change"].str.startswith("V600", na=False))]["case_id"])
    nras = set(mutations[mutations["gene"] == "NRAS"]["case_id"])
    nf1  = set(mutations[mutations["gene"] == "NF1"]["case_id"])

    def subtype(cid):
        if cid in braf: return "BRAF_V600E"
        if cid in nras: return "NRAS_mut"
        if cid in nf1:  return "NF1_mut"
        return "Triple_WT"

    clinical = clinical.copy()
    clinical["subtype"] = clinical["submitter_id"].map(subtype).fillna("Triple_WT")
    return clinical, {"BRAF_V600E": len(braf), "NRAS_mut": len(nras), "NF1_mut": len(nf1)}


# ---------------------------------------------------------------------------
# 1. Mutation landscape
# ---------------------------------------------------------------------------
def plot_mutation_landscape(mutations, clinical_st):
    """Frequency and type of driver mutations."""
    n_patients = len(clinical_st)

    # Mutation frequency per gene
    freq = (mutations.groupby("gene")["case_id"]
            .nunique()
            .sort_values(ascending=True)
            .tail(15))
    pct = (freq / n_patients * 100).round(1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Bar chart
    colors = [cfg.COLOR_RESIST if g == "BRAF" else
              cfg.COLOR_CONTROL if g == "NRAS" else
              cfg.COLOR_ACCENT  if g == "NF1"  else cfg.COLOR_NS
              for g in pct.index]
    axes[0].barh(range(len(pct)), pct.values, color=colors, edgecolor="#333", lw=0.5)
    axes[0].set_yticks(range(len(pct)))
    axes[0].set_yticklabels(pct.index)
    axes[0].set_xlabel("Patients mutated (%)")
    axes[0].set_title("TCGA-SKCM driver gene mutation frequency")
    for i, v in enumerate(pct.values):
        axes[0].text(v + 0.3, i, f"{v}%", va="center", fontsize=8)

    # Subtype pie
    sub_counts = clinical_st["subtype"].value_counts()
    pie_colors = [SUBTYPE_COLORS.get(s, "#aaa") for s in sub_counts.index]
    wedges, texts, autotexts = axes[1].pie(
        sub_counts.values, labels=sub_counts.index,
        colors=pie_colors, autopct="%1.1f%%",
        startangle=90, pctdistance=0.75,
        textprops={"fontsize": 9},
    )
    axes[1].set_title(f"Molecular subtypes (n={n_patients} patients)")

    fig.tight_layout()
    cfg.save_figure(fig, "tcga_skcm_mutation_landscape")

    # Export
    mut_summary = pd.DataFrame({"gene": freq.index, "n_patients": freq.values,
                                 "pct": pct.values}).sort_values("pct", ascending=False)
    cfg.save_table(mut_summary, "TCGA_SKCM_mutation_frequency.csv")
    cfg.save_table(clinical_st[["submitter_id", "subtype", "vital_status",
                                "days_to_death", "days_to_last_followup",
                                "tumor_stage"]],
                   "TCGA_SKCM_clinical_with_subtypes.csv")
    print(f"[info] Subtypes: {sub_counts.to_dict()}")


# ---------------------------------------------------------------------------
# 2. Kaplan-Meier survival by subtype
# ---------------------------------------------------------------------------
def _km_curve(ax, times, events, label, color, linestyle="-"):
    """Simple Kaplan-Meier estimator (no external lifelines dependency)."""
    times = np.array(times, dtype=float)
    events = np.array(events, dtype=float)
    mask = ~np.isnan(times)
    times, events = times[mask], events[mask]
    if len(times) == 0:
        return

    order = np.argsort(times)
    t_sorted = times[order]
    e_sorted = events[order]
    unique_t = np.unique(t_sorted)

    n_risk = len(t_sorted)
    km = 1.0
    km_vals = [1.0]
    t_vals = [0.0]
    for t in unique_t:
        idx = t_sorted == t
        d = e_sorted[idx].sum()        # deaths at t
        km *= (1 - d / n_risk)
        n_risk -= idx.sum()
        km_vals.append(km)
        t_vals.append(t / 30.44)       # days → months

    ax.step(t_vals, km_vals, where="post", label=f"{label} (n={len(times)})",
            color=color, linestyle=linestyle, linewidth=2)


def plot_survival(clinical_st):
    """Kaplan-Meier by molecular subtype."""
    df = clinical_st.copy()
    df["is_dead"] = (df["vital_status"] == "Dead").astype(float)
    df["os_days"] = df["days_to_death"].combine_first(df["days_to_last_followup"])
    df = df.dropna(subset=["os_days"])

    fig, ax = plt.subplots(figsize=(9, 6))
    for subtype, style in [("BRAF_V600E", "-"), ("NRAS_mut", "--"),
                           ("NF1_mut", ":"), ("Triple_WT", "-.")]:
        sub = df[df["subtype"] == subtype]
        if len(sub) < 5:
            continue
        _km_curve(ax, sub["os_days"], sub["is_dead"],
                  label=subtype.replace("_", " "),
                  color=SUBTYPE_COLORS[subtype],
                  linestyle=style)

    ax.set_xlabel("Overall survival (months)")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0)
    ax.axhline(0.5, color="lightgrey", lw=0.7, ls="--")
    ax.legend(fontsize=9)
    ax.set_title("Kaplan-Meier overall survival — TCGA-SKCM molecular subtypes")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    cfg.save_figure(fig, "tcga_skcm_survival_km")


# ---------------------------------------------------------------------------
# 3. RNA-seq — pathway expression by subtype
# ---------------------------------------------------------------------------
def load_rnaseq_subset():
    """Load STAR-Counts TSV files; return TPM-like matrix (genes × samples)."""
    manifest = pd.read_csv(os.path.join(DATA_DIR, "rnaseq_subset_manifest.csv"))
    files = glob.glob(os.path.join(RNASEQ_DIR, "*.tsv"))
    if not files:
        print("[warn] No RNA-seq TSV files found — skipping expression analysis")
        return None, None

    dfs = {}
    for path in files:
        fname = os.path.basename(path)
        row = manifest[manifest["file_name"] == fname]
        if row.empty:
            continue
        case_id = row.iloc[0]["case_submitter_id"]
        subtype = row.iloc[0].get("subtype", "Unknown")
        try:
            tsv = pd.read_csv(path, sep="\t", comment="#", index_col=0)
            # STAR augmented output: column 'tpm_unstranded' or 'fpkm_unstranded'
            tpm_col = next((c for c in tsv.columns
                           if "tpm" in c.lower() or "fpkm" in c.lower()), None)
            if tpm_col:
                dfs[f"{case_id}|{subtype}"] = tsv[tpm_col]
        except Exception:
            continue

    if not dfs:
        return None, None

    mat = pd.DataFrame(dfs)
    # Keep only ENSG rows, drop version suffix
    mat.index = mat.index.str.split(".").str[0]
    mat = mat[~mat.index.str.startswith("N_")]

    # Build gene symbol index (column 'gene_name' or use Ensembl IDs)
    # STAR output has Ensembl IDs — try to find gene_name column in one file
    gene_names = None
    for path in files[:1]:
        try:
            tmp = pd.read_csv(path, sep="\t", comment="#", index_col=0)
            if "gene_name" in tmp.columns:
                gene_names = tmp["gene_name"].str.split(".").str[0]
                gene_names.index = tmp.index.str.split(".").str[0]
        except Exception:
            pass

    return mat, gene_names


def plot_rnaseq_pathway(mat, gene_names, clinical_st):
    """Heatmap of MAPK/PI3K pathway genes by molecular subtype."""
    if mat is None:
        return

    # Map gene symbols to ENSG IDs
    if gene_names is not None:
        symbol_to_ensg = {v: k for k, v in gene_names.items()}
    else:
        symbol_to_ensg = {}

    pathway_genes = MAPK_GENES + PI3K_GENES
    ensg_ids = [symbol_to_ensg[g] for g in pathway_genes if g in symbol_to_ensg]
    if not ensg_ids:
        print("[warn] No pathway gene ENSG IDs found — skipping pathway heatmap")
        return

    sub_mat = mat.loc[[e for e in ensg_ids if e in mat.index]]
    if sub_mat.empty:
        return

    # Reverse map for labels
    ensg_to_symbol = {v: k for k, v in symbol_to_ensg.items()}
    sub_mat.index = sub_mat.index.map(lambda x: ensg_to_symbol.get(x, x))

    # log2(TPM+1); replace inf/NaN with row mean for clustering
    with np.errstate(invalid="ignore"):
        sub_log = np.log2(sub_mat.apply(pd.to_numeric, errors="coerce") + 1)
    sub_log = sub_log.replace([np.inf, -np.inf], np.nan)
    sub_log = sub_log.apply(lambda row: row.fillna(row.mean()), axis=1)
    sub_log = sub_log.dropna(how="all")

    # Sort columns by subtype
    col_subtypes = [c.split("|")[1] for c in sub_log.columns]
    sort_order = sorted(range(len(col_subtypes)),
                        key=lambda i: list(SUBTYPE_COLORS.keys()).index(
                            col_subtypes[i]) if col_subtypes[i] in SUBTYPE_COLORS else 99)
    sub_log = sub_log.iloc[:, sort_order]
    col_subtypes_sorted = [col_subtypes[i] for i in sort_order]

    col_colors = pd.Series([SUBTYPE_COLORS.get(s, "#aaa") for s in col_subtypes_sorted],
                           index=sub_log.columns, name="Subtype")

    g = sns.clustermap(sub_log, cmap="RdBu_r", center=sub_log.values.mean(),
                       col_colors=col_colors,
                       col_cluster=False,
                       xticklabels=False, yticklabels=True,
                       figsize=(12, 8),
                       cbar_kws={"label": "log2(TPM+1)"},
                       z_score=0)
    g.ax_heatmap.set_title("MAPK / PI3K pathway expression by subtype\n(TCGA-SKCM RNA-seq subset)",
                           pad=12, fontsize=11)
    handles = [mpatches.Patch(color=c, label=s) for s, c in SUBTYPE_COLORS.items()]
    g.ax_col_dendrogram.legend(handles=handles, loc="center", ncol=4, frameon=False)
    cfg.save_figure(g.fig, "tcga_skcm_pathway_expression")


# ---------------------------------------------------------------------------
# 4. Integration — top differential phospho vs RNA-seq expression
# ---------------------------------------------------------------------------
def plot_integration(clinical_st):
    """Cross-reference PXD022992 BRAF vs NRAS differential phospho genes
    with TCGA-SKCM subtype expression. Only descriptive (no paired samples).
    """
    diff_path = "results/outputs/PXD022992_differential_phospho_BRAFvsNRAS.csv"
    if not os.path.exists(diff_path):
        print("[warn] PXD022992 differential table not found — skipping integration")
        return

    diff = pd.read_csv(diff_path)
    # Top 20 higher in BRAF V600E and top 20 higher in NRAS
    up_braf = diff[diff["log2FC_BRAFvsNRAS"] > 1].nlargest(20, "log2FC_BRAFvsNRAS")
    up_nras = diff[diff["log2FC_BRAFvsNRAS"] < -1].nsmallest(20, "log2FC_BRAFvsNRAS")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, subset, title, color in [
        (axes[0], up_braf, "Top phosphosites higher in BRAF V600E\n(PXD022992 cell lines)",
         cfg.COLOR_RESIST),
        (axes[1], up_nras, "Top phosphosites higher in NRAS\n(PXD022992 cell lines)",
         cfg.COLOR_CONTROL),
    ]:
        if subset.empty:
            ax.axis("off"); continue
        y = range(len(subset))
        ax.barh(list(y), subset["log2FC_BRAFvsNRAS"].abs().values,
                color=color, edgecolor="#333", lw=0.4, alpha=0.8)
        ax.set_yticks(list(y))
        ax.set_yticklabels(subset["gene"].values, fontsize=8)
        ax.set_xlabel("|log2FC|")
        ax.set_title(title, fontsize=9)
        ax.axvline(1, color="dimgrey", ls="--", lw=0.7)

    fig.suptitle("Phosphoproteomics–genomics integration: BRAF V600E vs NRAS differential genes",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    cfg.save_figure(fig, "tcga_skcm_phospho_integration")

    cfg.save_table(pd.concat([
        up_braf.assign(direction="higher_BRAF"),
        up_nras.assign(direction="higher_NRAS"),
    ]), "TCGA_SKCM_phospho_integration_genes.csv")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run():
    cfg.apply_style()
    print("=" * 60)
    print("TCGA-SKCM analysis")
    print("=" * 60)

    # Check data availability
    for fname in ["clinical.csv", "driver_mutations.csv"]:
        if not os.path.exists(os.path.join(DATA_DIR, fname)):
            print(f"[ERROR] {os.path.join(DATA_DIR, fname)} not found.")
            print("        Run tcga_skcm_download.py first.")
            return

    clinical, mutations = load_data()
    clinical_st, subtype_counts = assign_subtypes(clinical, mutations)

    print(f"[info] Patients: {len(clinical_st)} | Subtypes: {subtype_counts}")

    print("[plot] Mutation landscape...")
    plot_mutation_landscape(mutations, clinical_st)

    print("[plot] Kaplan-Meier survival...")
    plot_survival(clinical_st)

    print("[plot] Integration phospho vs genomics...")
    plot_integration(clinical_st)

    print("[rnaseq] Loading RNA-seq subset...")
    mat, gene_names = load_rnaseq_subset()
    if mat is not None:
        print(f"[info] RNA-seq: {mat.shape[0]} genes × {mat.shape[1]} samples")
        plot_rnaseq_pathway(mat, gene_names, clinical_st)
    else:
        print("[info] RNA-seq files not yet available — run tcga_skcm_download.py")

    print("\nDone. Outputs saved to results/figures/ and results/outputs/")


def main():
    run()


if __name__ == "__main__":
    main()
