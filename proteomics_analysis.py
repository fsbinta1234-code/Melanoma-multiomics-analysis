import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from cleanDatas import CleanDatas


class ProteomicsAnalysis:
    """
    Analyzes protein abundance profiles to identify adaptive resistance
    regulators and differentially expressed proteins between control and
    resistant melanoma conditions.

    Expected Output: differential protein abundance, adaptive signaling
    proteins, and therapeutic target candidates.
    """

    @staticmethod
    def run_analysis() -> pd.DataFrame:
        """
        Cleans proteinGroups data and computes per-protein log2 fold change.

        The cleaned protein groups table is filtered to remove contaminants
        and reverse-database hits.  LFQ intensity columns are extracted,
        log2-normalised, and split into control (columns 0–2) and resistant
        (columns 3–5) groups for fold-change calculation.

        Returns
        -------
        pd.DataFrame
            One row per protein group with column ProteinFoldChange.
        """
        protein = CleanDatas.clean_protein_groups()

        # Remove contaminants and reverse hits
        protein_clean = protein[
            (protein['Reverse'] != '+') &
            (protein['Potential contaminant'] != '+')
        ]

        # Extract LFQ intensity columns
        lfq_cols = [
            col for col in protein_clean.columns
            if 'LFQ intensity' in col
        ]

        protein_lfq  = protein_clean[lfq_cols]
        protein_log2 = np.log2(protein_lfq + 1)

        # Impute missing values with column median
        protein_log2 = protein_log2.fillna(protein_log2.median())

        # Log2 fold change: resistant (cols 3–5) minus control (cols 0–2)
        protein_fc = (
            protein_log2.iloc[:, 3:6].mean(axis=1) -
            protein_log2.iloc[:, 0:3].mean(axis=1)
        )

        protein_results = pd.DataFrame(
            {'ProteinFoldChange': protein_fc},
            index=protein_log2.index,
        )

        print(f"Proteins analysed : {len(protein_results)}")
        print(protein_results.head(10))

        ProteomicsAnalysis._plot_fold_change_distribution(protein_results)

        return protein_results

    @staticmethod
    def _plot_fold_change_distribution(protein_results: pd.DataFrame) -> None:
        """
        Plots the distribution of protein-level log2 fold changes.

        Parameters
        ----------
        protein_results : pd.DataFrame
            DataFrame with column ProteinFoldChange.
        """
        fig, ax = plt.subplots(figsize=(8, 5))

        sns.histplot(
            protein_results['ProteinFoldChange'].dropna(),
            bins=60,
            kde=True,
            color='steelblue',
            ax=ax,
        )

        ax.axvline( 1, linestyle='--', color='crimson',  linewidth=0.9, label='|FC| = 1')
        ax.axvline(-1, linestyle='--', color='crimson',  linewidth=0.9)
        ax.axvline( 0, linestyle='-',  color='dimgrey', linewidth=0.6)

        ax.set_xlabel('Log2 Fold Change (Resistant / Control)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Protein Abundance Fold Change Distribution', fontsize=13)
        ax.legend(frameon=False)

        plt.tight_layout()
        plt.show()


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main():
    ProteomicsAnalysis.run_analysis()


if __name__ == "__main__":
    main()
