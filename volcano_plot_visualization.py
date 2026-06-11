"""
Phase 5 — Volcano plot of differential phosphorylation (publication quality).

Reuses the Phase 4 statistical computation and highlights the phosphosites
significantly associated with BRAFi/MEKi resistance.
"""
import numpy as np
import matplotlib.pyplot as plt

import pipeline_config as cfg
from differential_phosphorylation_analysis import DifferentialPhosphorylationAnalysis


class VolcanoPlotVisualization:

    @staticmethod
    def plot() -> None:
        cfg.apply_style()
        results = DifferentialPhosphorylationAnalysis.compute_results()

        fig, ax = plt.subplots(figsize=(11, 7.5))
        neglogp = -np.log10(results["PValue"].clip(lower=1e-300))
        palette = {"ns": cfg.COLOR_NS, "Up (resistant)": cfg.COLOR_UP, "Down (resistant)": cfg.COLOR_DOWN}
        for direction, color in palette.items():
            m = results["Direction"] == direction
            ax.scatter(results.loc[m, "Log2FC"], neglogp[m], c=color, s=18,
                       alpha=0.55 if direction == "ns" else 0.85, edgecolors="none",
                       label=f"{direction} ({int(m.sum())})")

        fc = DifferentialPhosphorylationAnalysis.FC_THRESHOLD
        ax.axvline(fc, ls="--", color="dimgrey", lw=0.8)
        ax.axvline(-fc, ls="--", color="dimgrey", lw=0.8)
        ax.axhline(-np.log10(DifferentialPhosphorylationAnalysis.ADJ_P_CUTOFF),
                   ls=":", color="steelblue", lw=0.9)
        ax.set_xlabel("Log2 Fold Change  (ARoe resistant / LacZ control)")
        ax.set_ylabel("-Log10 (p-value)")
        ax.set_title("Volcano Plot of Differential Phosphorylation")
        ax.legend(title="Regulation", loc="upper left", fontsize=9)
        cfg.save_figure(fig, "05_volcano_plot")

        n_sig = int(results["Significant"].sum())
        print(f"Significant sites : {n_sig} of {len(results)}")


def main():
    VolcanoPlotVisualization.plot()


if __name__ == "__main__":
    main()
