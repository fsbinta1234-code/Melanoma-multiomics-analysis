import pandas as pd
from differential_phosphorylation_analysis import DifferentialPhosphorylationAnalysis
from proteomics_analysis import ProteomicsAnalysis
from transcriptomics_validation import TranscriptomicsValidation
from multi_omics_integration import MultiOmicsIntegration


class FinalResultExport:
    """
    Exports all computational outputs for downstream biological interpretation
    and publication-quality analysis.

    Each omics layer is saved as a separate CSV file and an integrated
    multi-omics summary is produced.  All files are written to the working
    directory and are ready for further processing in R, Excel, or Cytoscape.

    Expected Output: final phosphoproteomics results, final proteomics
    results, final transcriptomics results, and final integrated systems
    biology outputs.
    """

    PHOSPHO_FILE     = 'Differential_Phosphorylation.csv'
    PROTEOMICS_FILE  = 'Proteomics_Results.csv'
    TRANSCRIPT_FILE  = 'Transcriptomics_Results.csv'
    MULTIOMICS_FILE  = 'Integrated_Multiomics.csv'

    @staticmethod
    def export(
        significant:        pd.DataFrame = None,
        protein_results:    pd.DataFrame = None,
        transcript_results: pd.DataFrame = None,
        multiomics:         pd.DataFrame = None,
    ) -> None:
        """
        Saves all omics results to CSV files.

        Any omics layer that is not provided is computed by calling its
        upstream analysis module.  The multi-omics integration step is also
        run if not provided, producing the unified Integrated_Multiomics.csv.

        Parameters
        ----------
        significant : pd.DataFrame, optional
            Significant phosphosites from Phase 4.
        protein_results : pd.DataFrame, optional
            Protein fold-change table from Phase 7.
        transcript_results : pd.DataFrame, optional
            Transcript fold-change table from Phase 8.
        multiomics : pd.DataFrame, optional
            Integrated multi-omics table from Phase 9.
        """
        if significant is None:
            significant = DifferentialPhosphorylationAnalysis.run_analysis()

        if protein_results is None:
            protein_results = ProteomicsAnalysis.run_analysis()

        if transcript_results is None:
            transcript_results = TranscriptomicsValidation.run_analysis()

        if multiomics is None:
            multiomics = MultiOmicsIntegration.integrate(
                significant, protein_results, transcript_results
            )

        significant.to_csv(FinalResultExport.PHOSPHO_FILE,    index=False)
        protein_results.to_csv(FinalResultExport.PROTEOMICS_FILE, index=False)
        transcript_results.to_csv(FinalResultExport.TRANSCRIPT_FILE, index=False)
        multiomics.to_csv(FinalResultExport.MULTIOMICS_FILE,  index=False)

        print('Complete melanoma systems biology analysis finished.')
        print(f"  {FinalResultExport.PHOSPHO_FILE}")
        print(f"  {FinalResultExport.PROTEOMICS_FILE}")
        print(f"  {FinalResultExport.TRANSCRIPT_FILE}")
        print(f"  {FinalResultExport.MULTIOMICS_FILE}")


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main():
    FinalResultExport.export()


if __name__ == "__main__":
    main()
