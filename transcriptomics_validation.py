"""
Phase 8 — Transcriptomics validation (GEO GSE199405).

Uses the REAL metadata mapping (ARoe = resistant, LacZ = control) to compute
log2FC and a t-test per probe, providing orthogonal evidence of the resistance
signatures. Expression values are already on a log2 (RMA) scale.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests

import pipeline_config as cfg
from cleanDatas import CleanDatas


class TranscriptomicsValidation:
    FC_THRESHOLD = 1.0
    ADJ_P_CUTOFF = 0.05

    @staticmethod
    def run_analysis() -> pd.DataFrame:
        cfg.apply_style()
        expr = CleanDatas.clean_geo_series_matrix()
        ctrl_cols, res_cols = cfg.geo_groups(expr)
        print(f"GEO groups — control (LacZ): {len(ctrl_cols)} | resistant (ARoe): {len(res_cols)}")

        ctrl = expr[ctrl_cols].to_numpy()
        res = expr[res_cols].to_numpy()
        fc = res.mean(axis=1) - ctrl.mean(axis=1)          # already log2
        _, p = stats.ttest_ind(res, ctrl, axis=1)
        p = np.nan_to_num(p, nan=1.0)
        adj = multipletests(p, method="fdr_bh")[1]

        results = pd.DataFrame(
            {"GeneFoldChange": fc, "PValue": p, "AdjPValue": adj}, index=expr.index)
        results["Significant"] = (results["AdjPValue"] < TranscriptomicsValidation.ADJ_P_CUTOFF) & (
            results["GeneFoldChange"].abs() > TranscriptomicsValidation.FC_THRESHOLD)

        print(f"Probes analysed : {len(results)}  |  significant : {int(results['Significant'].sum())}")

        TranscriptomicsValidation._plot_volcano(results)
        TranscriptomicsValidation._plot_distribution(results)
        return results

    @staticmethod
    def _plot_volcano(results: pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(10, 7))
        neglogp = -np.log10(results["PValue"].clip(lower=1e-300))
        sig = results["Significant"]
        ax.scatter(results.loc[~sig, "GeneFoldChange"], neglogp[~sig], c=cfg.COLOR_NS,
                   s=10, alpha=0.4, edgecolors="none", label=f"Not significant ({int((~sig).sum())})")
        up = sig & (results["GeneFoldChange"] > 0)
        dn = sig & (results["GeneFoldChange"] < 0)
        ax.scatter(results.loc[up, "GeneFoldChange"], neglogp[up], c=cfg.COLOR_UP,
                   s=16, alpha=0.8, edgecolors="none", label=f"Up resistant ({int(up.sum())})")
        ax.scatter(results.loc[dn, "GeneFoldChange"], neglogp[dn], c=cfg.COLOR_DOWN,
                   s=16, alpha=0.8, edgecolors="none", label=f"Down resistant ({int(dn.sum())})")
        ax.axvline(1, ls="--", color="dimgrey", lw=0.8)
        ax.axvline(-1, ls="--", color="dimgrey", lw=0.8)
        ax.axhline(-np.log10(TranscriptomicsValidation.ADJ_P_CUTOFF), ls=":", color="steelblue", lw=0.9)
        ax.set_xlabel("Log2 Fold Change (ARoe resistant / LacZ control)")
        ax.set_ylabel("-Log10 (p-value)")
        ax.set_title("Volcano Plot — Differential Gene Expression (GSE199405)")
        ax.legend(loc="upper left", fontsize=9)
        cfg.save_figure(fig, "08_transcriptomics_volcano")

    @staticmethod
    def _plot_distribution(results: pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        sns.histplot(results["GeneFoldChange"].dropna(), bins=90, kde=True,
                     color=cfg.COLOR_ACCENT, ax=ax)
        for x in (-1, 1):
            ax.axvline(x, ls="--", color="crimson", lw=0.9)
        ax.axvline(0, ls="-", color="dimgrey", lw=0.6)
        ax.set_xlabel("Log2 Fold Change (ARoe resistant / LacZ control)")
        ax.set_title("Distribution of Gene Expression Fold Changes")
        cfg.save_figure(fig, "08_transcriptomics_fc_distribution")


def main():
    TranscriptomicsValidation.run_analysis()


if __name__ == "__main__":
    main()
