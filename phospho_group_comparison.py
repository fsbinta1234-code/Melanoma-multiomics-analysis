"""
Phase 4 (variant) — Group comparison: control (LacZ) vs resistant (ARoe).

Identifies differentially phosphorylated sites by reusing the Phase 4 statistical
computation and summarises the significant findings. Produces a bar chart of the
top up/down-regulated sites.
"""
import matplotlib.pyplot as plt

import pipeline_config as cfg
from differential_phosphorylation_analysis import DifferentialPhosphorylationAnalysis


class PhosphoGroupComparison:

    TOP_N = 20

    @staticmethod
    def compare_control_vs_resistant():
        cfg.apply_style()
        results = DifferentialPhosphorylationAnalysis.compute_results()
        significant = results[results["Significant"]].copy()

        print(f"Significant sites: {len(significant)} "
              f"(↑{int((significant['Log2FC'] > 0).sum())} / "
              f"↓{int((significant['Log2FC'] < 0).sum())})")

        if significant.empty:
            print("  (no significant sites)")
            return significant

        # Top sites by |log2FC| (mix of up and down)
        top = significant.reindex(
            significant["Log2FC"].abs().sort_values(ascending=False).index
        ).head(PhosphoGroupComparison.TOP_N).sort_values("Log2FC")

        fig, ax = plt.subplots(figsize=(9, 8))
        colors = [cfg.COLOR_UP if v > 0 else cfg.COLOR_DOWN for v in top["Log2FC"]]
        ax.barh(top.index, top["Log2FC"], color=colors, edgecolor="#333333", alpha=0.9)
        ax.axvline(0, color="dimgrey", lw=0.8)
        ax.set_xlabel("Log2 Fold Change (ARoe resistant / LacZ control)")
        ax.set_title(f"Top {PhosphoGroupComparison.TOP_N} differential phosphosites by |log2FC|")
        ax.tick_params(axis="y", labelsize=8)
        cfg.save_figure(fig, "04_phospho_group_comparison_top_sites")

        print("Top 10:")
        print(top.tail(10)[["Gene", "Log2FC", "AdjPValue"]].to_string())
        return significant


def main():
    PhosphoGroupComparison.compare_control_vs_resistant()


if __name__ == "__main__":
    main()
