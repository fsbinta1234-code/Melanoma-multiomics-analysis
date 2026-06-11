"""
Phase 14 — Final result export.

Runs the differential analyses and saves every table to results/outputs/,
reusing already-computed results to avoid recomputation.
"""
import pipeline_config as cfg
from differential_phosphorylation_analysis import DifferentialPhosphorylationAnalysis
from proteomics_analysis import ProteomicsAnalysis
from transcriptomics_validation import TranscriptomicsValidation
from multi_omics_integration import MultiOmicsIntegration


class FinalResultExport:

    @staticmethod
    def export() -> None:
        cfg.apply_style()

        phospho_results = DifferentialPhosphorylationAnalysis.compute_results()
        protein_results = ProteomicsAnalysis.run_analysis()
        transcript_results = TranscriptomicsValidation.run_analysis()
        integrated = MultiOmicsIntegration.integrate(
            phospho_results, protein_results, transcript_results)

        significant = phospho_results[phospho_results["Significant"]]

        # Save FULL tables (with the Significant column for filtering), plus a
        # file containing only the significant phosphosites.
        cfg.save_table(phospho_results, "Differential_Phosphorylation_all.csv")
        cfg.save_table(significant, "Differential_Phosphorylation_significant.csv")
        cfg.save_table(protein_results, "Proteomics_Results.csv")
        cfg.save_table(transcript_results, "Transcriptomics_Results.csv")
        cfg.save_table(integrated, "Integrated_Multiomics.csv", index=False)

        print("\nMelanoma systems-biology analysis complete.")
        print(f"  Significant phosphosites : {len(significant)}")
        print(f"  Significant proteins     : {int(protein_results['Significant'].sum())}")
        print(f"  Significant probes       : {int(transcript_results['Significant'].sum())}")
        print(f"  Integrated genes         : {len(integrated)}")


def main():
    FinalResultExport.export()


if __name__ == "__main__":
    main()
