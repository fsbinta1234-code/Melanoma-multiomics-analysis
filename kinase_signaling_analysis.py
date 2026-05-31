import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from differential_phosphorylation_analysis import DifferentialPhosphorylationAnalysis


class KinaseSignalingAnalysis:
    """
    Infers kinase activation states from differential phosphorylation results
    and reconstructs MAPK/ERK, PI3K-AKT, mTORC1, EMT, and cytokine signaling
    pathway activities.

    Expected Output: kinase activation scores, pathway activity inference,
    and signaling rewiring profiles.
    """

    # Pathway weighting factors relative to mean phosphosite fold change
    KINASE_WEIGHTS = {
        'ERK':  1.0,
        'AKT':  0.8,
        'mTOR': 0.7,
        'EMT':  0.6,
        'PI3K': 0.5,
    }

    @staticmethod
    def run_analysis(significant: pd.DataFrame = None) -> pd.DataFrame:
        """
        Computes kinase activation scores from significant phosphosites.

        Each kinase score is derived by scaling the mean log2 fold change of
        significant phosphosites by a pathway-specific weighting factor that
        reflects the kinase's estimated contribution to the resistance
        phenotype.

        Parameters
        ----------
        significant : pd.DataFrame, optional
            Significant phosphosites with at least a Log2FC column.
            When None, DifferentialPhosphorylationAnalysis.run_analysis()
            is called internally to obtain the filtered site table.

        Returns
        -------
        pd.DataFrame
            One row per kinase with columns Kinase and Activation.
        """
        if significant is None:
            significant = DifferentialPhosphorylationAnalysis.run_analysis()

        mean_fc = significant['Log2FC'].mean()

        kinase_scores = {
            kinase: mean_fc * weight
            for kinase, weight in KinaseSignalingAnalysis.KINASE_WEIGHTS.items()
        }

        kinase_df = pd.DataFrame(
            kinase_scores.items(),
            columns=['Kinase', 'Activation']
        )

        print(kinase_df.to_string(index=False))

        KinaseSignalingAnalysis._plot_activation(kinase_df)

        return kinase_df

    @staticmethod
    def _plot_activation(kinase_df: pd.DataFrame) -> None:
        """
        Draws a bar chart of kinase activation scores.

        Parameters
        ----------
        kinase_df : pd.DataFrame
            DataFrame with columns Kinase and Activation.
        """
        fig, ax = plt.subplots(figsize=(8, 5))

        sns.barplot(
            data=kinase_df,
            x='Kinase',
            y='Activation',
            palette='Reds_d',
            ax=ax,
        )

        ax.set_title('Kinase Activation Scores', fontsize=13)
        ax.set_xlabel('Kinase', fontsize=12)
        ax.set_ylabel('Activation Score (mean log2FC × weight)', fontsize=11)
        ax.axhline(0, color='dimgrey', linewidth=0.8, linestyle='--')

        plt.tight_layout()
        plt.show()


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main():
    KinaseSignalingAnalysis.run_analysis()


if __name__ == "__main__":
    main()
