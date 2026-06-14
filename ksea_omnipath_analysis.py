"""
Kinase-Substrate Enrichment Analysis (KSEA) — OmniPath network.

Uses the real kinase-substrate phosphorylation network from OmniPath
(cached at datas/omnipath/enzsub_phospho_genesymbols.tsv) to compute
rigorous kinase activity scores from two real datasets:

  • PXD013923  — BRAFi/MEKi/ERKi acute inhibition (SILAC, A375, 30 min)
  • PXD022992  — DIA phosphoproteome across 6 melanoma cell lines (baseline)

KSEA algorithm (Casado et al. 2013 / Hernandez-Armenta et al. 2017):
  For each kinase k with substrate set S_k:
    score(k) = (mean_S_k - mean_all) / (σ_all / √|S_k|)
  This is a t-like z-score; positive = kinase more active, negative = suppressed.
  Significance via permutation test (n=1000 random substrate sets of same size).

Only kinases with ≥ 5 measured substrates are scored.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

import pipeline_config as cfg

OMNIPATH_FILE = "datas/omnipath/enzsub_phospho_genesymbols.tsv"
PXD013923_MATRIX = "results/outputs/PXD013923_inhibitor_log2_matrix.csv"
PXD022992_MATRIX = "results/outputs/PXD022992_phosphosite_matrix.csv"

MIN_SUBSTRATES = 5
N_PERMS = 1000
SEED = 42

# Kinases of primary interest in melanoma BRAFi/MEKi resistance
FOCAL_KINASES = {
    "MAPK1", "MAPK3", "MAP2K1", "MAP2K2", "BRAF", "RAF1",
    "AKT1", "AKT2", "AKT3",
    "MTOR", "RPS6KB1", "RPS6KA1", "RPS6KA3",
    "CDK1", "CDK2", "CDK4", "CDK6",
    "PTK2", "SRC",
    "EGFR", "IGF1R", "MET",
    "PRKCA", "PRKCB", "PRKCD",
    "GSK3B",
}


# ---------------------------------------------------------------------------
# OmniPath network
# ---------------------------------------------------------------------------
def load_omnipath():
    """Return dict: kinase_gene → set of phosphosite strings (GENE_Rpos)."""
    df = pd.read_csv(OMNIPATH_FILE, sep="\t")
    # Build site key matching our matrix format: GENE_Apos e.g. MAPK1_T185
    df["site_key"] = (df["substrate_genesymbol"].str.strip() + "_" +
                      df["residue_type"].str.strip() +
                      df["residue_offset"].astype(str))
    kin2sites = {}
    for kin, grp in df.groupby("enzyme_genesymbol"):
        kin2sites[kin] = set(grp["site_key"])
    print(f"[info] OmniPath: {len(df)} records | {len(kin2sites)} kinases")
    return kin2sites


# ---------------------------------------------------------------------------
# KSEA core
# ---------------------------------------------------------------------------
def ksea_score(values: pd.Series, kin2sites: dict, n_perms: int = N_PERMS):
    """
    Compute KSEA z-score per kinase for one condition (Series: index=site_keys).
    Returns DataFrame: kinase, n_substrates, mean_sub, z_score, p_emp.
    """
    rng = np.random.default_rng(SEED)
    all_vals = values.dropna().values
    global_mean = all_vals.mean()
    global_std = all_vals.std() if all_vals.std() > 0 else 1.0

    rows = []
    for kin, sites in kin2sites.items():
        matched = values.index.intersection(sites)
        n = len(matched)
        if n < MIN_SUBSTRATES:
            continue
        sub_vals = values.loc[matched].dropna().values
        if len(sub_vals) < MIN_SUBSTRATES:
            continue
        obs_mean = sub_vals.mean()
        z = (obs_mean - global_mean) / (global_std / np.sqrt(len(sub_vals)))

        # Empirical p-value (permutation)
        null_z = np.array([
            ((rng.choice(all_vals, len(sub_vals), replace=False).mean() - global_mean)
             / (global_std / np.sqrt(len(sub_vals))))
            for _ in range(n_perms)
        ])
        p_emp = (np.abs(null_z) >= np.abs(z)).mean()
        rows.append({"kinase": kin, "n_substrates": n, "mean_substrates": obs_mean,
                     "z_score": z, "p_emp": p_emp})

    return pd.DataFrame(rows).sort_values("z_score", ascending=False)


# ---------------------------------------------------------------------------
# Apply KSEA to both datasets
# ---------------------------------------------------------------------------
def run_ksea_pxd013923(kin2sites):
    """KSEA on PXD013923 — one score per inhibitor condition."""
    df = pd.read_csv(PXD013923_MATRIX, index_col=0)
    inhibitors = ["BRAFi", "MEKi", "ERKi"]
    inhibitors = [c for c in inhibitors if c in df.columns]

    results = {}
    for inh in inhibitors:
        vals = df[inh].dropna()
        print(f"  PXD013923 {inh}: {len(vals)} sites with values")
        results[inh] = ksea_score(vals, kin2sites)

    return results


def run_ksea_pxd022992(kin2sites):
    """KSEA on PXD022992 — one score per cell line."""
    df = pd.read_csv(PXD022992_MATRIX, index_col=0)
    meta_cols = {"gene", "modified_seq", "ptm_location"}
    sample_cols = [c for c in df.columns if c not in meta_cols]

    results = {}
    for cell in sample_cols:
        vals = df[cell].dropna()
        print(f"  PXD022992 {cell}: {len(vals)} sites")
        results[cell] = ksea_score(vals, kin2sites)

    return results


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _focal_top(df_res: pd.DataFrame, n: int = 30):
    """Filter to focal kinases + top/bottom n by z-score."""
    focal = df_res[df_res["kinase"].isin(FOCAL_KINASES)]
    top = df_res.nlargest(n, "z_score")
    bot = df_res.nsmallest(n, "z_score")
    return pd.concat([focal, top, bot]).drop_duplicates("kinase")


def plot_pxd013923_ksea(results):
    """Heatmap: z-score across BRAFi/MEKi/ERKi for focal kinases."""
    inhibitors = list(results.keys())
    # collect all kinases passing threshold
    all_kins = set()
    for df in results.values():
        sig = df[(df["p_emp"] < 0.05) | df["kinase"].isin(FOCAL_KINASES)]
        all_kins.update(sig["kinase"].tolist())

    mat = {}
    for inh, df in results.items():
        row = df.set_index("kinase")["z_score"]
        mat[inh] = row
    mat_df = pd.DataFrame(mat).loc[list(all_kins)].dropna(how="all")
    mat_df = mat_df.reindex(
        mat_df.abs().max(axis=1).nlargest(min(40, len(mat_df))).index
    )

    fig, ax = plt.subplots(figsize=(8, max(6, len(mat_df) * 0.3)))
    sns.heatmap(mat_df, cmap="RdBu_r", center=0, annot=True, fmt=".2f",
                linewidths=0.4, ax=ax, cbar_kws={"label": "KSEA z-score"})
    ax.set_title("Kinase activity (KSEA / OmniPath) — BRAFi, MEKi, ERKi\n"
                 "A375, 30 min, PXD013923", fontsize=11)
    ax.set_xlabel(""); ax.set_ylabel("Kinase")
    fig.tight_layout()
    cfg.save_figure(fig, "ksea_pxd013923_inhibitors")
    cfg.save_table(mat_df.reset_index().rename(columns={"index": "kinase"}),
                   "KSEA_PXD013923_kinase_zscores.csv")


def plot_pxd022992_ksea(results):
    """Heatmap: z-score across 6 cell lines for focal/significant kinases."""
    cell_lines = list(results.keys())
    all_kins = set(FOCAL_KINASES)
    for df in results.values():
        all_kins.update(df[df["p_emp"] < 0.05]["kinase"].tolist())

    mat = {}
    for cell, df in results.items():
        mat[cell] = df.set_index("kinase")["z_score"]
    mat_df = pd.DataFrame(mat)
    valid_kins = [k for k in all_kins if k in mat_df.index]
    mat_df = mat_df.loc[valid_kins].dropna(how="all")
    mat_df = mat_df.reindex(
        mat_df.var(axis=1).nlargest(min(40, len(mat_df))).index
    )

    fig, ax = plt.subplots(figsize=(10, max(6, len(mat_df) * 0.3)))
    sns.heatmap(mat_df, cmap="RdBu_r", center=0, annot=True, fmt=".2f",
                linewidths=0.4, ax=ax, cbar_kws={"label": "KSEA z-score"})
    ax.set_title("Kinase activity (KSEA / OmniPath) — 6 melanoma cell lines\n"
                 "PXD022992 directDIA", fontsize=11)
    ax.set_xlabel("Cell line"); ax.set_ylabel("Kinase")
    fig.tight_layout()
    cfg.save_figure(fig, "ksea_pxd022992_cell_lines")
    cfg.save_table(mat_df.reset_index().rename(columns={"index": "kinase"}),
                   "KSEA_PXD022992_kinase_zscores.csv")


def plot_combined_volcano(results_013923, results_022992):
    """
    Two-panel: PXD013923 (BRAFi) kinase z-score vs −log10(p_emp),
              PXD022992 (A375 vs G361) kinase z-score difference.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A — PXD013923 BRAFi
    df_brafi = results_013923.get("BRAFi", pd.DataFrame())
    if not df_brafi.empty:
        df_brafi["-log10p"] = -np.log10(df_brafi["p_emp"].clip(1e-6))
        sig = df_brafi[df_brafi["p_emp"] < 0.05]
        ns  = df_brafi[df_brafi["p_emp"] >= 0.05]
        axes[0].scatter(ns["z_score"], ns["-log10p"], s=20, alpha=0.3,
                        c=cfg.COLOR_NS, edgecolors="none")
        up = sig[sig["z_score"] > 0]
        dn = sig[sig["z_score"] < 0]
        axes[0].scatter(up["z_score"], up["-log10p"], s=40, c=cfg.COLOR_RESIST,
                        alpha=0.8, edgecolors="none", label=f"Active (n={len(up)})")
        axes[0].scatter(dn["z_score"], dn["-log10p"], s=40, c=cfg.COLOR_CONTROL,
                        alpha=0.8, edgecolors="none", label=f"Suppressed (n={len(dn)})")
        for _, r in pd.concat([up.nlargest(5, "z_score"),
                               dn.nsmallest(5, "z_score")]).iterrows():
            axes[0].annotate(r["kinase"], (r["z_score"], r["-log10p"]),
                             fontsize=7, xytext=(3, 2), textcoords="offset points")
        axes[0].axhline(-np.log10(0.05), ls="--", color="dimgrey", lw=0.8)
        axes[0].axvline(0, color="dimgrey", lw=0.6)
        axes[0].set_xlabel("KSEA z-score"); axes[0].set_ylabel("−log10(p_emp)")
        axes[0].set_title("PXD013923 BRAFi — kinase activity volcano")
        axes[0].legend(fontsize=8)

    # Panel B — PXD022992 A375 vs G361 (BRAF V600E vs NRAS)
    df_a375 = results_022992.get("A375", pd.DataFrame())
    df_g361 = results_022992.get("G361", pd.DataFrame())
    if not df_a375.empty and not df_g361.empty:
        merged = df_a375.set_index("kinase")[["z_score"]].rename(
            columns={"z_score": "A375"}).join(
            df_g361.set_index("kinase")[["z_score", "p_emp"]].rename(
                columns={"z_score": "G361"}), how="inner")
        merged["diff"] = merged["A375"] - merged["G361"]
        merged["-log10p"] = -np.log10(df_a375.set_index("kinase")["p_emp"].reindex(merged.index).clip(1e-6))
        sig_m = merged[merged["-log10p"] > -np.log10(0.05)]
        ns_m  = merged[merged["-log10p"] <= -np.log10(0.05)]
        axes[1].scatter(ns_m["diff"], ns_m["-log10p"], s=20, alpha=0.3,
                        c=cfg.COLOR_NS, edgecolors="none")
        up_m = sig_m[sig_m["diff"] > 0]
        dn_m = sig_m[sig_m["diff"] < 0]
        axes[1].scatter(up_m["diff"], up_m["-log10p"], s=40, c=cfg.COLOR_RESIST,
                        alpha=0.8, edgecolors="none",
                        label=f"Higher A375 (BRAF, n={len(up_m)})")
        axes[1].scatter(dn_m["diff"], dn_m["-log10p"], s=40, c=cfg.COLOR_CONTROL,
                        alpha=0.8, edgecolors="none",
                        label=f"Higher G361 (NRAS, n={len(dn_m)})")
        for _, r in pd.concat([up_m.nlargest(5, "diff"),
                               dn_m.nsmallest(5, "diff")]).iterrows():
            axes[1].annotate(r.name, (r["diff"], r["-log10p"]),
                             fontsize=7, xytext=(3, 2), textcoords="offset points")
        axes[1].axhline(-np.log10(0.05), ls="--", color="dimgrey", lw=0.8)
        axes[1].axvline(0, color="dimgrey", lw=0.6)
        axes[1].set_xlabel("KSEA z-score  (A375 − G361)")
        axes[1].set_ylabel("−log10(p_emp, A375)")
        axes[1].set_title("PXD022992: kinase activity difference\nA375 (BRAF V600E) vs G361 (NRAS)")
        axes[1].legend(fontsize=8)

    fig.suptitle("Kinase-Substrate Enrichment Analysis — OmniPath network",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    cfg.save_figure(fig, "ksea_combined_volcano")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run():
    cfg.apply_style()
    print("=" * 60)
    print("KSEA — OmniPath kinase-substrate network")
    print("=" * 60)

    kin2sites = load_omnipath()

    print("\n[PXD013923] BRAFi/MEKi/ERKi...")
    res_013923 = run_ksea_pxd013923(kin2sites)
    plot_pxd013923_ksea(res_013923)

    print("\n[PXD022992] 6 cell lines...")
    res_022992 = run_ksea_pxd022992(kin2sites)
    plot_pxd022992_ksea(res_022992)

    print("\n[combined] Volcano panels...")
    plot_combined_volcano(res_013923, res_022992)

    # Print focal kinase summary for BRAFi
    brafi = res_013923.get("BRAFi", pd.DataFrame())
    if not brafi.empty:
        focal = brafi[brafi["kinase"].isin(FOCAL_KINASES)].sort_values("z_score")
        print("\nFocal kinases — BRAFi z-scores:")
        for _, r in focal.iterrows():
            print(f"  {r['kinase']:12s}  z={r['z_score']:+.2f}  n={int(r['n_substrates'])}  p={r['p_emp']:.3f}")

    print("\nDone.")


def main():
    run()


if __name__ == "__main__":
    main()
