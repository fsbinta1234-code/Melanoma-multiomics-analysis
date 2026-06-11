"""
Phase 4 — Differential phosphorylation analysis (ARoe/resistant vs LacZ/control).

Computes log2FC and a t-test per phosphosite, applies FDR correction
(Benjamini-Hochberg) and produces an annotated volcano plot and a heatmap
(clustermap) of the most significant sites.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from scipy import stats
from statsmodels.stats.multitest import multipletests

import pipeline_config as cfg
from cleanDatas import CleanDatas


class DifferentialPhosphorylationAnalysis:
    FC_THRESHOLD = 1.0    # minimum |log2FC|
    ADJ_P_CUTOFF = 0.05   # FDR
    TOP_N_HEATMAP = 40
    TOP_N_LABELS = 12

    @staticmethod
    def compute_results() -> pd.DataFrame:
        """Compute log2FC, p-value and FDR per phosphosite. Returns the full table."""
        phospho, meta = CleanDatas.clean_phospho_sty_sites(return_meta=True)
        ctrl_cols, res_cols = cfg.maxquant_groups(phospho)

        ctrl = phospho[ctrl_cols].to_numpy()
        res = phospho[res_cols].to_numpy()

        log2_fc = res.mean(axis=1) - ctrl.mean(axis=1)
        _, p_values = stats.ttest_ind(res, ctrl, axis=1)
        p_values = np.nan_to_num(p_values, nan=1.0)
        adj_p = multipletests(p_values, method="fdr_bh")[1]

        results = pd.DataFrame(
            {
                "Gene": meta["Gene"].values,
                "Log2FC": log2_fc,
                "PValue": p_values,
                "AdjPValue": adj_p,
            },
            index=phospho.index,
        )
        sig = (results["AdjPValue"] < DifferentialPhosphorylationAnalysis.ADJ_P_CUTOFF) & (
            results["Log2FC"].abs() > DifferentialPhosphorylationAnalysis.FC_THRESHOLD
        )
        results["Significant"] = sig
        results["Direction"] = np.where(
            ~sig, "ns", np.where(results["Log2FC"] > 0, "Up (resistant)", "Down (resistant)")
        )
        return results

    @staticmethod
    def run_analysis() -> pd.DataFrame:
        cfg.apply_style()
        results = DifferentialPhosphorylationAnalysis.compute_results()
        significant = results[results["Significant"]].copy()

        n_up = (significant["Log2FC"] > 0).sum()
        n_down = (significant["Log2FC"] < 0).sum()
        print(f"Phosphosites tested    : {len(results)}")
        print(f"Significant (FDR<{DifferentialPhosphorylationAnalysis.ADJ_P_CUTOFF}, "
              f"|log2FC|>{DifferentialPhosphorylationAnalysis.FC_THRESHOLD}) : "
              f"{len(significant)}  (↑{n_up} resistant, ↓{n_down} control)")
        if len(significant):
            print("Top 10 by FDR:")
            print(significant.nsmallest(10, "AdjPValue")[["Gene", "Log2FC", "AdjPValue"]].to_string())

        DifferentialPhosphorylationAnalysis._plot_volcano(results)
        if len(significant):
            DifferentialPhosphorylationAnalysis._plot_heatmap(significant)
        else:
            print("  (no significant sites — heatmap skipped)")

        return significant

    @staticmethod
    def _plot_volcano(results: pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(11, 7.5))
        neglogp = -np.log10(results["PValue"].clip(lower=1e-300))

        palette = {"ns": cfg.COLOR_NS, "Up (resistant)": cfg.COLOR_UP, "Down (resistant)": cfg.COLOR_DOWN}
        for direction, color in palette.items():
            m = results["Direction"] == direction
            ax.scatter(results.loc[m, "Log2FC"], neglogp[m], c=color, s=18,
                       alpha=0.55 if direction == "ns" else 0.85,
                       edgecolors="none", label=f"{direction} ({m.sum()})")

        fc = DifferentialPhosphorylationAnalysis.FC_THRESHOLD
        ax.axvline(fc, ls="--", color="dimgrey", lw=0.8)
        ax.axvline(-fc, ls="--", color="dimgrey", lw=0.8)
        ax.axhline(-np.log10(DifferentialPhosphorylationAnalysis.ADJ_P_CUTOFF),
                   ls=":", color="steelblue", lw=0.9)

        # Label the most extreme sites (highest -log10 p among significant)
        sig = results[results["Significant"]]
        top = sig.reindex(neglogp[sig.index].sort_values(ascending=False).index).head(
            DifferentialPhosphorylationAnalysis.TOP_N_LABELS)
        for site, row in top.iterrows():
            ax.annotate(site, (row["Log2FC"], -np.log10(max(row["PValue"], 1e-300))),
                        fontsize=7.5, xytext=(4, 3), textcoords="offset points", color="#222222")

        ax.set_xlabel("Log2 Fold Change  (ARoe resistant / LacZ control)")
        ax.set_ylabel("-Log10 (p-value)")
        ax.set_title("Volcano Plot — Differential Phosphorylation in BRAFi/MEKi Resistance")
        ax.legend(title="Regulation", loc="upper left", fontsize=9)
        cfg.save_figure(fig, "04_differential_phosphorylation_volcano")

    @staticmethod
    def _plot_heatmap(significant: pd.DataFrame) -> None:
        phospho = CleanDatas.clean_phospho_sty_sites()
        top_sites = significant.nsmallest(
            DifferentialPhosphorylationAnalysis.TOP_N_HEATMAP, "AdjPValue").index
        subset = phospho.loc[top_sites]

        ctrl_cols, res_cols = cfg.maxquant_groups(phospho)
        col_colors = pd.Series(
            {c: (cfg.COLOR_RESIST if c in set(res_cols) else cfg.COLOR_CONTROL)
             for c in phospho.columns}, name="Group")

        g = sns.clustermap(
            subset, z_score=0, cmap=cfg.HEATMAP_CMAP, center=0,
            col_colors=col_colors, figsize=(12, 11),
            yticklabels=True, xticklabels=True,
            cbar_kws={"label": "z-score (log2 intensity)"},
            linewidths=0.2, linecolor="white",
        )
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=7)
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), fontsize=7, rotation=90)
        g.ax_heatmap.set_xlabel("Sample")
        handles = [Patch(facecolor=cfg.COLOR_CONTROL, label="LacZ (control)"),
                   Patch(facecolor=cfg.COLOR_RESIST, label="ARoe (resistant)")]
        g.ax_heatmap.legend(handles=handles, title="Group", loc="upper left",
                            bbox_to_anchor=(1.02, 1.18), fontsize=8)
        g.figure.suptitle(
            f"Top {DifferentialPhosphorylationAnalysis.TOP_N_HEATMAP} differential phosphosites (z-score)",
            y=1.01, fontsize=14, fontweight="bold")
        cfg.save_figure(g.figure, "04_differential_phosphorylation_heatmap")


def main():
    DifferentialPhosphorylationAnalysis.run_analysis()


if __name__ == "__main__":
    main()
