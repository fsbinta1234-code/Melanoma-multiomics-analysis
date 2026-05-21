import matplotlib.pyplot as plt
import seaborn as sns
from cleanDatas import CleanDatas


class QualityControlVisualization:
    """
    Central class responsible for quality control (QC) visualizations of the
    three omics datasets used in the melanoma multi-omics analysis project.
    Each static method loads cleaned data from CleanDatas and produces plots
    to assess data distribution, missing values, and sample consistency.
    """

    # ------------------------------------------------------------------
    # QC 1 – GEO microarray expression matrix (GSE199405)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # QC 3 – MaxQuant phosphorylation sites (Phospho STY Sites)
    # ------------------------------------------------------------------
    @staticmethod
    def qc_phospho_sty_sites() -> None:
        """
        Produces QC plots for the cleaned MaxQuant Phospho (STY) Sites dataset.

        Loads the log2-normalised phospho intensity matrix and generates a
        boxplot to compare the intensity distributions across all samples,
        helping to confirm that normalisation was effective.
        """
        # Load the cleaned and normalised phospho sites matrix
        df = CleanDatas.clean_phospho_sty_sites()

        # Plot per-sample intensity distributions as a boxplot
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df)
        plt.xticks(rotation=90)
        plt.title("Normalized Phosphoproteomics Intensities")
        plt.tight_layout()
        plt.show()


# ----------------------------------------------------------------------
# Entry point: run all QC visualizations sequentially
# ----------------------------------------------------------------------
def main():
    #QualityControlVisualization.qc_geo_series_matrix()
    #QualityControlVisualization.qc_protein_groups()
    QualityControlVisualization.qc_phospho_sty_sites()


if __name__ == "__main__":
    main()
