import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests
from cleanDatas import CleanDatas


class VolcanoPlotVisualization:
    """
    Produces publication-quality volcano plots from differential
    phosphorylation data, highlighting phosphosites significantly associated
    with BRAFi/MEKi resistance and adaptive signaling rewiring.

    Expected Output: significant phosphosite visualization and
    resistance-associated signaling patterns.
    """

    FC_THRESHOLD = 1.0
    ADJ_P_CUTOFF = 0.05

    @staticmethod
    def compute_results() -> pd.DataFrame:
        """
        Computes fold change and p-values for all phosphosites.

        Loads the cleaned log2-normalised intensity matrix, splits samples
        into control (columns 0–2) and resistant (columns 3–5) groups, and
        runs a per-site independent t-test followed by Benjamini-Hochberg
        FDR correction.

        Returns
        -------
        pd.DataFrame
            Full results table with columns Log2FC, PValue, AdjPValue,
            and Significant for every tested phosphosite.
        """
        phospho_log2 = CleanDatas.clean_phospho_sty_sites()

        control   = phospho_log2.iloc[:, 0:3]
        resistant = phospho_log2.iloc[:, 3:6]

        log2_fc = resistant.mean(axis=1) - control.mean(axis=1)

        p_values = [
            stats.ttest_ind(
                resistant.iloc[i],
                control.iloc[i],
                nan_policy='omit'
            ).pvalue
            for i in range(len(phospho_log2))
        ]

        _, adj_p_values, _, _ = multipletests(p_values, method='fdr_bh')

        results = pd.DataFrame(
            {
                'Log2FC':    log2_fc,
                'PValue':    p_values,
                'AdjPValue': adj_p_values,
            },
            index=phospho_log2.index,
        )

        results['Significant'] = (
            (results['AdjPValue'] < VolcanoPlotVisualization.ADJ_P_CUTOFF) &
            (results['Log2FC'].abs() > VolcanoPlotVisualization.FC_THRESHOLD)
        )

        return results

    @staticmethod
    def plot(results: pd.DataFrame = None) -> None:
        """
        Draws a volcano plot of log2 fold change vs -log10(p-value).

        Significant sites (passing both FDR and fold-change thresholds) are
        highlighted in red; all other sites are shown in grey.  Dashed
        vertical lines mark the |log2FC| threshold and a dotted horizontal
        line marks the adjusted p-value cutoff.

        Parameters
        ----------
        results : pd.DataFrame, optional
            Full results table with columns Log2FC, PValue, and Significant.
            When None, compute_results() is called internally.
        """
        if results is None:
            results = VolcanoPlotVisualization.compute_results()

        non_sig = results[~results['Significant']]
        sig      = results[ results['Significant']]

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.scatter(
            non_sig['Log2FC'],
            -np.log10(non_sig['PValue']),
            c='lightgrey', alpha=0.5, s=15, label='Not significant'
        )
        ax.scatter(
            sig['Log2FC'],
            -np.log10(sig['PValue']),
            c='crimson', alpha=0.8, s=20, label='Significant'
        )

        fc  = VolcanoPlotVisualization.FC_THRESHOLD
        adj = VolcanoPlotVisualization.ADJ_P_CUTOFF

        ax.axvline( fc, linestyle='--', color='dimgrey', linewidth=0.8)
        ax.axvline(-fc, linestyle='--', color='dimgrey', linewidth=0.8)
        ax.axhline(
            -np.log10(adj),
            linestyle=':', color='steelblue', linewidth=0.8
        )

        ax.set_xlabel('Log2 Fold Change', fontsize=12)
        ax.set_ylabel('-Log10 PValue', fontsize=12)
        ax.set_title('Volcano Plot of Differential Phosphorylation', fontsize=13)
        ax.legend(frameon=False)

        plt.tight_layout()
        plt.show()

        print(f"Significant sites plotted : {len(sig)}")
        print(f"Total sites               : {len(results)}")


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main():
    VolcanoPlotVisualization.plot()


if __name__ == "__main__":
    main()
