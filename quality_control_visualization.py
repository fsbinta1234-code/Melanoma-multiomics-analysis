"""
Phase 2 (QC) — Quality-control visualisations of the three omics layers.

Generates boxplots of per-sample intensity distributions (coloured by group),
allowing a check of normalisation and sample-to-sample consistency.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import pipeline_config as cfg
from cleanDatas import CleanDatas


class QualityControlVisualization:

    @staticmethod
    def _boxplot(ax, matrix, is_resistant, title) -> None:
        data = [matrix[c].values for c in matrix.columns]
        bp = ax.boxplot(data, patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", lw=1.1),
                        widths=0.65)
        for patch, col in zip(bp["boxes"], matrix.columns):
            patch.set_facecolor(cfg.COLOR_RESIST if is_resistant(col) else cfg.COLOR_CONTROL)
            patch.set_alpha(0.85)
            patch.set_edgecolor("#333333")
        ax.set_xticks(range(1, len(matrix.columns) + 1))
        ax.set_xticklabels(matrix.columns, rotation=90, fontsize=7)
        ax.set_ylabel("Intensity (log2)")
        ax.set_title(title)

    @staticmethod
    def run() -> None:
        cfg.apply_style()

        phospho = CleanDatas.clean_phospho_sty_sites()
        protein = CleanDatas.clean_protein_groups()
        geo = CleanDatas.clean_geo_series_matrix()

        _, res_p = cfg.maxquant_groups(phospho)
        _, res_pr = cfg.maxquant_groups(protein)
        _, res_g = cfg.geo_groups(geo)
        res_p, res_pr, res_g = set(res_p), set(res_pr), set(map(str, res_g))

        fig, axes = plt.subplots(3, 1, figsize=(14, 13))
        QualityControlVisualization._boxplot(
            axes[0], phospho, lambda c: c in res_p,
            f"Phosphoproteomics — {phospho.shape[0]} sites × {phospho.shape[1]} samples")
        QualityControlVisualization._boxplot(
            axes[1], protein, lambda c: c in res_pr,
            f"Proteomics — {protein.shape[0]} proteins × {protein.shape[1]} samples")
        QualityControlVisualization._boxplot(
            axes[2], geo, lambda c: str(c) in res_g,
            f"Transcriptomics (GEO) — {geo.shape[0]} probes × {geo.shape[1]} samples")

        handles = [Patch(facecolor=cfg.COLOR_CONTROL, label="LacZ (control)"),
                   Patch(facecolor=cfg.COLOR_RESIST, label="ARoe (resistant)")]
        axes[0].legend(handles=handles, loc="upper right", fontsize=9)
        fig.suptitle("Quality Control — Normalised Intensity Distributions",
                     fontsize=15, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        cfg.save_figure(fig, "02_quality_control_boxplots")


def main():
    QualityControlVisualization.run()


if __name__ == "__main__":
    main()
