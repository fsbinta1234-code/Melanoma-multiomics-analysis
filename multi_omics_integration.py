import pandas as pd
from differential_phosphorylation_analysis import DifferentialPhosphorylationAnalysis
from proteomics_analysis import ProteomicsAnalysis
from transcriptomics_validation import TranscriptomicsValidation


class MultiOmicsIntegration:
    """
    Integrates phosphoproteomics, proteomics, and transcriptomics results to
    reconstruct melanoma resistance biology.

    All three omics layers are concatenated column-wise into a unified
    multi-omics DataFrame that captures cross-layer resistance signals.

    Expected Output: integrated resistance biomarkers, cross-omics signaling
    networks, and systems biology resistance signatures.
    """

    OUTPUT_FILE = 'Integrated_Multiomics.csv'

    @staticmethod
    def integrate(
        significant:        pd.DataFrame = None,
        protein_results:    pd.DataFrame = None,
        transcript_results: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        Concatenates phospho, protein, and transcript fold-change tables.

        Each omics layer is reset to a plain integer index before
        concatenation so rows from different datasets align by position
        rather than by a shared index.  The integrated table is also saved
        to disk as a CSV file for downstream systems biology analysis.

        Parameters
        ----------
        significant : pd.DataFrame, optional
            Significant phosphosites from Phase 4.  When None,
            DifferentialPhosphorylationAnalysis.run_analysis() is called.
        protein_results : pd.DataFrame, optional
            Protein fold-change table from Phase 7.  When None,
            ProteomicsAnalysis.run_analysis() is called.
        transcript_results : pd.DataFrame, optional
            Transcript fold-change table from Phase 8.  When None,
            TranscriptomicsValidation.run_analysis() is called.

        Returns
        -------
        pd.DataFrame
            Integrated multi-omics DataFrame with one column per omics layer.
        """
        if significant is None:
            significant = DifferentialPhosphorylationAnalysis.run_analysis()

        if protein_results is None:
            protein_results = ProteomicsAnalysis.run_analysis()

        if transcript_results is None:
            transcript_results = TranscriptomicsValidation.run_analysis()

        multiomics = pd.concat(
            [
                significant.reset_index(drop=True),
                protein_results.reset_index(drop=True),
                transcript_results.reset_index(drop=True),
            ],
            axis=1,
        )

        print(f"Integrated multi-omics shape : {multiomics.shape}")
        print(multiomics.head())

        multiomics.to_csv(
            MultiOmicsIntegration.OUTPUT_FILE,
            index=False,
        )
        print(f"Saved → {MultiOmicsIntegration.OUTPUT_FILE}")

        return multiomics


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main():
    MultiOmicsIntegration.integrate()


if __name__ == "__main__":
    main()
