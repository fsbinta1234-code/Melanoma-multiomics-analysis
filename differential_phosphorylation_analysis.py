import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
from cleanDatas import CleanDatas


class DifferentialPhosphorylationAnalysis:
    """
    Performs differential phosphorylation analysis between control and
    resistant melanoma sample groups.

    Computes log2 fold change and t-test p-values per phosphosite, applies
    Benjamini-Hochberg FDR correction, and generates a volcano plot and a
    heatmap of the top significant sites.
    """

    FC_THRESHOLD  = 1.0   # |log2FC| cutoff for biological relevance
    ADJ_P_CUTOFF  = 0.05  # FDR-adjusted p-value cutoff
    TOP_N_HEATMAP = 50    # number of top sites shown in the heatmap

    @staticmethod
    def run_analysis() -> pd.DataFrame:
        """
        Runs the full differential phosphorylation pipeline.

        Loads the cleaned log2-normalised phospho intensity matrix, splits
        samples into control (columns 0–2) and resistant (columns 3–5) groups,
        and computes per-site statistics.  Significant sites are defined by
        |log2FC| > FC_THRESHOLD and adjusted p-value < ADJ_P_CUTOFF.

        After filtering, two figures are produced:
          1. Volcano plot  – log2FC vs -log10(p-value) for all sites.
          2. Heatmap       – z-score intensities of the top N significant sites.

        Returns
        -------
        pd.DataFrame
            Significant phosphosites with columns:
            Log2FC, PValue, AdjPValue, Significant.
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
                'Log2FC':      log2_fc,
                'PValue':      p_values,
                'AdjPValue':   adj_p_values,
            },
            index=phospho_log2.index,
        )

        results['Significant'] = (
            (results['AdjPValue'] < DifferentialPhosphorylationAnalysis.ADJ_P_CUTOFF) &
            (results['Log2FC'].abs() > DifferentialPhosphorylationAnalysis.FC_THRESHOLD)
        )

        significant = results[results['Significant']]

        print(f"Total phosphosites tested : {len(results)}")
        print(f"Significant sites found   : {len(significant)}")
        print(significant.head(10))

        DifferentialPhosphorylationAnalysis._plot_volcano(results)
        DifferentialPhosphorylationAnalysis._plot_heatmap(phospho_log2, significant)

        return significant

    @staticmethod
    def _plot_volcano(results: pd.DataFrame) -> None:
        """
        Draws a volcano plot for all tested phosphosites.

        Parameters
        ----------
        results : pd.DataFrame
            Full results table with Log2FC, PValue, and Significant columns.
        """
        fig, ax = plt.subplots(figsize=(10, 7))

        non_sig = results[~results['Significant']]
        sig      = results[ results['Significant']]

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

        fc_thresh = DifferentialPhosphorylationAnalysis.FC_THRESHOLD
        ax.axvline( fc_thresh, linestyle='--', color='dimgrey', linewidth=0.8)
        ax.axvline(-fc_thresh, linestyle='--', color='dimgrey', linewidth=0.8)
        ax.axhline(
            -np.log10(DifferentialPhosphorylationAnalysis.ADJ_P_CUTOFF),
            linestyle=':', color='steelblue', linewidth=0.8
        )

        ax.set_xlabel('Log2 Fold Change (Resistant / Control)', fontsize=12)
        ax.set_ylabel('-Log10 P-Value', fontsize=12)
        ax.set_title('Volcano Plot — Differential Phosphorylation', fontsize=13)
        ax.legend(frameon=False)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def _plot_heatmap(phospho_log2: pd.DataFrame, significant: pd.DataFrame) -> None:
        """
        Draws a heatmap of z-score intensities for the top significant sites.

        Sites are ranked by adjusted p-value and the top N are displayed.

        Parameters
        ----------
        phospho_log2 : pd.DataFrame
            Full log2-normalised intensity matrix (all sites x all samples).
        significant : pd.DataFrame
            Filtered results table produced by run_analysis().
        """
        top_n = DifferentialPhosphorylationAnalysis.TOP_N_HEATMAP
        top_sites = significant.nsmallest(top_n, 'AdjPValue').index
        subset    = phospho_log2.loc[phospho_log2.index.isin(top_sites)]

        z_scored = subset.apply(
            lambda row: (row - row.mean()) / row.std() if row.std() > 0 else row,
            axis=1
        )

        fig, ax = plt.subplots(figsize=(10, max(6, len(subset) // 4)))
        sns.heatmap(
            z_scored,
            cmap='RdBu_r',
            center=0,
            linewidths=0.3,
            linecolor='white',
            yticklabels=False,
            ax=ax,
        )
        ax.set_title(
            f'Top {top_n} Differentially Phosphorylated Sites (z-score)',
            fontsize=13
        )
        ax.set_xlabel('Sample')
        plt.tight_layout()
        plt.show()


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main():
    DifferentialPhosphorylationAnalysis.run_analysis()


if __name__ == "__main__":
    main()
