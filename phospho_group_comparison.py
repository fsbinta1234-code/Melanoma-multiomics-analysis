import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests
from cleanDatas import CleanDatas


class PhosphoGroupComparison:
    """
    Compares phosphorylation intensities between control and resistant sample
    groups using fold change and independent t-tests.  Results are filtered
    to retain only statistically significant and biologically relevant sites.
    """

    @staticmethod
    def compare_control_vs_resistant() -> pd.DataFrame:
        """
        Identifies differentially phosphorylated sites between control and
        resistant melanoma sample groups.

        The cleaned log2-normalised phospho intensity matrix is split into two
        groups based on column position:
          - control   : first 3 columns (LacZ + DMSO conditions)
          - resistant : next  3 columns (ARoe + Dabrafenib conditions)

        For each phosphosite the method computes:
          - Log2 fold change  : mean(resistant) - mean(control)
          - P-value           : independent two-sample t-test
          - Adjusted p-value  : Benjamini-Hochberg FDR correction across all sites

        Significant sites are defined as those with |log2FC| > 1 and
        adjusted p-value < 0.05.

        Returns
        -------
        pd.DataFrame
            Significant phosphosites with columns:
            FoldChange, PValue, AdjPValue.
        """
        # Load the cleaned and log2-normalised phospho intensity matrix
        phospho_log2 = CleanDatas.clean_phospho_sty_sites()

        # Split samples into control and resistant groups by column index
        control   = phospho_log2.iloc[:, 0:3]
        resistant = phospho_log2.iloc[:, 3:6]

        # Compute log2 fold change per phosphosite (resistant minus control means)
        fold_change = resistant.mean(axis=1) - control.mean(axis=1)

        # Run an independent t-test for each phosphosite row
        p_values = []
        for i in range(len(phospho_log2)):
            p = stats.ttest_ind(
                resistant.iloc[i],
                control.iloc[i],
                nan_policy='omit'
            ).pvalue
            p_values.append(p)

        # Apply Benjamini-Hochberg correction to control false discovery rate
        # across the thousands of simultaneous tests performed above
        _, adj_p_values, _, _ = multipletests(p_values, method='fdr_bh')

        # Assemble one row per phosphosite with all statistical metrics
        results = pd.DataFrame({
            'FoldChange': fold_change,
            'PValue':     p_values,
            'AdjPValue':  adj_p_values,
        })

        # Retain only sites that pass both the effect size and significance thresholds
        significant = results[
            (abs(results['FoldChange']) > 1) &
            (results['AdjPValue'] < 0.05)
        ]

        print(significant.head())

        # Volcano plot: all sites plotted as fold change vs -log10(p-value)
        # Sites beyond the dashed vertical lines have |log2FC| > 1
        plt.figure(figsize=(10, 6))
        plt.scatter(
            results['FoldChange'],
            -np.log10(results['PValue']),
            alpha=0.6
        )
        plt.axvline(1,  linestyle='--')  # upper fold change threshold
        plt.axvline(-1, linestyle='--')  # lower fold change threshold
        plt.xlabel('Log2 Fold Change')
        plt.ylabel('-Log10 PValue')
        plt.title('Volcano Plot of Differential Phosphorylation')
        plt.tight_layout()
        plt.show()

        return significant


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main():
    PhosphoGroupComparison.compare_control_vs_resistant()


if __name__ == "__main__":
    main()
