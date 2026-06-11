"""
Phase 7 — Differential proteomics analysis (ARoe/resistant vs LacZ/control).

Computes log2FC and a t-test per protein (LFQ) and identifies candidate adaptive
regulators. Produces a volcano plot and the fold-change distribution.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests

import pipeline_config as cfg
from cleanDatas import CleanDatas


class ProteomicsAnalysis:
    FC_THRESHOLD = 1.0
    ADJ_P_CUTOFF = 0.05

    @staticmethod
    def run_analysis() -> pd.DataFrame:
        cfg.apply_style()
        protein = CleanDatas.clean_protein_groups()
        ctrl_cols, res_cols = cfg.maxquant_groups(protein)

        ctrl = protein[ctrl_cols].to_numpy()
        res = protein[res_cols].to_numpy()
        fc = res.mean(axis=1) - ctrl.mean(axis=1)
        _, p = stats.ttest_ind(res, ctrl, axis=1)
        p = np.nan_to_num(p, nan=1.0)
        adj = multipletests(p, method="fdr_bh")[1]

        results = pd.DataFrame(
            {"ProteinFoldChange": fc, "PValue": p, "AdjPValue": adj},
            index=protein.index,
        )
        results["Significant"] = (results["AdjPValue"] < ProteomicsAnalysis.ADJ_P_CUTOFF) & (
            results["ProteinFoldChange"].abs() > ProteomicsAnalysis.FC_THRESHOLD)

        n_sig = int(results["Significant"].sum())
        print(f"Proteins analysed : {len(results)}  |  significant : {n_sig}")
        if n_sig:
            top = results[results["Significant"]].reindex(
                results["ProteinFoldChange"].abs().sort_values(ascending=False).index).head(10)
            print("Top 10 by |log2FC|:")
            print(top[["ProteinFoldChange", "AdjPValue"]].to_string())

        ProteomicsAnalysis._plot_volcano(results)
        ProteomicsAnalysis._plot_distribution(results)
        return results

    @staticmethod
    def _plot_volcano(results: pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(10, 7))
        neglogp = -np.log10(results["PValue"].clip(lower=1e-300))
        sig = results["Significant"]
        ax.scatter(results.loc[~sig, "ProteinFoldChange"], neglogp[~sig],
                   c=cfg.COLOR_NS, s=16, alpha=0.5, edgecolors="none",
                   label=f"Not significant ({int((~sig).sum())})")
        up = sig & (results["ProteinFoldChange"] > 0)
        dn = sig & (results["ProteinFoldChange"] < 0)
        ax.scatter(results.loc[up, "ProteinFoldChange"], neglogp[up], c=cfg.COLOR_UP,
                   s=22, alpha=0.85, edgecolors="none", label=f"Up resistant ({int(up.sum())})")
        ax.scatter(results.loc[dn, "ProteinFoldChange"], neglogp[dn], c=cfg.COLOR_DOWN,
                   s=22, alpha=0.85, edgecolors="none", label=f"Down resistant ({int(dn.sum())})")
        for prot, row in results[sig].reindex(neglogp[sig].sort_values(ascending=False).index).head(10).iterrows():
            ax.annotate(prot, (row["ProteinFoldChange"], -np.log10(max(row["PValue"], 1e-300))),
                        fontsize=7.5, xytext=(4, 3), textcoords="offset points")
        ax.axvline(ProteomicsAnalysis.FC_THRESHOLD, ls="--", color="dimgrey", lw=0.8)
        ax.axvline(-ProteomicsAnalysis.FC_THRESHOLD, ls="--", color="dimgrey", lw=0.8)
        ax.axhline(-np.log10(ProteomicsAnalysis.ADJ_P_CUTOFF), ls=":", color="steelblue", lw=0.9)
        ax.set_xlabel("Log2 Fold Change (ARoe resistant / LacZ control)")
        ax.set_ylabel("-Log10 (p-value)")
        ax.set_title("Volcano Plot — Differential Protein Abundance")
        ax.legend(loc="upper left", fontsize=9)
        cfg.save_figure(fig, "07_proteomics_volcano")

    @staticmethod
    def _plot_distribution(results: pd.DataFrame) -> None:
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(9, 5.5))
        sns.histplot(results["ProteinFoldChange"].dropna(), bins=70, kde=True,
                     color=cfg.COLOR_CONTROL, ax=ax)
        for x in (-1, 1):
            ax.axvline(x, ls="--", color="crimson", lw=0.9)
        ax.axvline(0, ls="-", color="dimgrey", lw=0.6)
        ax.set_xlabel("Log2 Fold Change (ARoe resistant / LacZ control)")
        ax.set_title("Distribution of Protein Abundance Fold Changes")
        cfg.save_figure(fig, "07_proteomics_fc_distribution")


def main():
    ProteomicsAnalysis.run_analysis()


if __name__ == "__main__":
    main()
