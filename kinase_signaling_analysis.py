"""
Phase 6 — Signalling-pathway activity inference.

Instead of arbitrary weights, each pathway's activity score is the mean log2FC of
the phosphosites whose genes belong to that pathway (MAPK/ERK, PI3K-AKT, mTOR,
EMT), reflecting the signalling rewiring associated with BRAFi/MEKi resistance.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pipeline_config as cfg
from differential_phosphorylation_analysis import DifferentialPhosphorylationAnalysis


class KinaseSignalingAnalysis:

    @staticmethod
    def run_analysis() -> pd.DataFrame:
        cfg.apply_style()
        results = DifferentialPhosphorylationAnalysis.compute_results()

        # Gene set per site (a site may map to several genes separated by ';')
        gene_sets = results["Gene"].fillna("").str.upper().str.split(";")

        rows = []
        for pathway, genes in cfg.PATHWAY_GENES.items():
            mask = gene_sets.apply(lambda gl: any(g in genes for g in gl))
            subset = results[mask]
            n_sig = int(subset["Significant"].sum())
            rows.append({
                "Pathway": pathway,
                "Activity": subset["Log2FC"].mean() if len(subset) else np.nan,
                "n_sites": int(len(subset)),
                "n_significant": n_sig,
            })
        kinase_df = pd.DataFrame(rows)

        print(kinase_df.to_string(index=False))
        cfg.save_table(kinase_df, "Pathway_Activity_Scores.csv", index=False)
        KinaseSignalingAnalysis._plot(kinase_df)
        return kinase_df

    @staticmethod
    def _plot(kinase_df: pd.DataFrame) -> None:
        df = kinase_df.dropna(subset=["Activity"]).sort_values("Activity")
        fig, ax = plt.subplots(figsize=(9, 5.5))
        colors = [cfg.COLOR_UP if v > 0 else cfg.COLOR_DOWN for v in df["Activity"]]
        bars = ax.bar(df["Pathway"], df["Activity"], color=colors, edgecolor="#333333", alpha=0.9)
        for bar, n, ns in zip(bars, df["n_sites"], df["n_significant"]):
            ax.annotate(f"n={n}\n({ns} sig.)",
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center", va="bottom" if bar.get_height() >= 0 else "top",
                        fontsize=8, xytext=(0, 3 if bar.get_height() >= 0 else -3),
                        textcoords="offset points")
        ax.axhline(0, color="dimgrey", lw=0.8, ls="--")
        ax.set_ylabel("Activity (mean log2FC of pathway sites)")
        ax.set_title("Signalling-Pathway Activity in Resistance (ARoe vs LacZ)")
        cfg.save_figure(fig, "06_pathway_activity_scores")


def main():
    KinaseSignalingAnalysis.run_analysis()


if __name__ == "__main__":
    main()
