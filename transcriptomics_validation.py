import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from cleanDatas import CleanDatas


class TranscriptomicsValidation:
    """
    Validates phosphoproteomic signaling findings using transcriptomics data
    from the GEO dataset GSE199405.

    Differentially expressed genes are identified by computing log2 fold
    change between resistant and control conditions, providing orthogonal
    evidence for resistance-associated transcriptomic signatures.

    Expected Output: differentially expressed genes and resistance-associated
    transcriptomic signatures.
    """

    @staticmethod
    def run_analysis() -> pd.DataFrame:
        """
        Loads the GEO expression matrix and computes per-probe fold change.

        The expression matrix (RMA-normalised log2 intensities) is split into
        control (columns 0–2) and resistant (columns 3–5) groups.  A per-probe
        log2 fold change is computed as mean(resistant) - mean(control) and the
        results are returned for downstream multi-omics integration.

        Returns
        -------
        pd.DataFrame
            One row per probe set with column GeneFoldChange.
        """
        transcript = CleanDatas.clean_geo_series_matrix()

        # Expression values are already RMA log2-normalised; select samples
        expression = transcript.iloc[:, 0:]

        # Coerce to numeric to handle any string artefacts from the GEO file
        expression = expression.apply(pd.to_numeric, errors='coerce')

        # Log2-normalise (expression values are already log2, but re-apply
        # to any non-normalised columns loaded as raw counts)
        expression_log2 = np.log2(expression + 1)

        # Impute missing values with row median
        expression_log2 = expression_log2.apply(
            lambda row: row.fillna(row.median()), axis=1
        )

        # Log2 fold change: resistant (cols 3–5) minus control (cols 0–2)
        transcript_fc = (
            expression_log2.iloc[:, 3:6].mean(axis=1) -
            expression_log2.iloc[:, 0:3].mean(axis=1)
        )

        transcript_results = pd.DataFrame(
            {'GeneFoldChange': transcript_fc},
            index=expression_log2.index,
        )

        print(f"Probe sets analysed : {len(transcript_results)}")
        print(transcript_results.head(10))

        TranscriptomicsValidation._plot_fold_change_distribution(transcript_results)

        return transcript_results

    @staticmethod
    def _plot_fold_change_distribution(transcript_results: pd.DataFrame) -> None:
        """
        Plots the distribution of gene-level log2 fold changes.

        Parameters
        ----------
        transcript_results : pd.DataFrame
            DataFrame with column GeneFoldChange.
        """
        fig, ax = plt.subplots(figsize=(8, 5))

        sns.histplot(
            transcript_results['GeneFoldChange'].dropna(),
            bins=80,
            kde=True,
            color='mediumseagreen',
            ax=ax,
        )

        ax.axvline( 1, linestyle='--', color='crimson', linewidth=0.9, label='|FC| = 1')
        ax.axvline(-1, linestyle='--', color='crimson', linewidth=0.9)
        ax.axvline( 0, linestyle='-',  color='dimgrey', linewidth=0.6)

        ax.set_xlabel('Log2 Fold Change (Resistant / Control)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Transcriptomic Gene Expression Fold Change Distribution', fontsize=13)
        ax.legend(frameon=False)

        plt.tight_layout()
        plt.show()


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main():
    TranscriptomicsValidation.run_analysis()


if __name__ == "__main__":
    main()
