"""
Phase 9 — Multi-omics integration.

Integrates phosphoproteomics and proteomics at the GENE level (merge by symbol),
which lets us cross-reference signalling (phosphorylation) with protein
abundance. The transcriptomics layer is kept as separate evidence (indexed by
probe, with no gene↔probe mapping in this dataset).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pipeline_config as cfg
from differential_phosphorylation_analysis import DifferentialPhosphorylationAnalysis
from proteomics_analysis import ProteomicsAnalysis
from transcriptomics_validation import TranscriptomicsValidation


class MultiOmicsIntegration:

    @staticmethod
    def integrate(phospho_results=None, protein_results=None, transcript_results=None) -> pd.DataFrame:
        cfg.apply_style()
        if phospho_results is None:
            phospho_results = DifferentialPhosphorylationAnalysis.compute_results()
        if protein_results is None:
            protein_results = ProteomicsAnalysis.run_analysis()
        if transcript_results is None:
            transcript_results = TranscriptomicsValidation.run_analysis()

        # Phospho aggregated per gene
        ph = phospho_results.copy()
        ph["GeneSym"] = ph["Gene"].fillna("NA").str.split(";").str[0]
        phospho_gene = ph.groupby("GeneSym").agg(
            PhosphoLog2FC=("Log2FC", "mean"),
            n_phosphosites=("Log2FC", "size"),
            n_sig_phospho=("Significant", "sum"),
        )

        # Protein per gene (index is already the gene symbol)
        prot = protein_results.copy()
        prot.index.name = "GeneSym"
        protein_gene = prot.groupby(level=0).agg(
            ProteinLog2FC=("ProteinFoldChange", "mean"),
            ProteinSig=("Significant", "max"),
        )

        integrated = phospho_gene.join(protein_gene, how="inner").reset_index()
        integrated = integrated.sort_values("PhosphoLog2FC", ascending=False)

        print(f"Integrated genes (phospho ∩ protein) : {len(integrated)}")
        if len(integrated) > 1:
            r = integrated[["PhosphoLog2FC", "ProteinLog2FC"]].corr().iloc[0, 1]
            print(f"Phospho×protein correlation (log2FC) : r = {r:.3f}")
        cfg.save_table(integrated, "Integrated_Multiomics.csv", index=False)

        MultiOmicsIntegration._plot_scatter(integrated)
        return integrated

    @staticmethod
    def _plot_scatter(integrated: pd.DataFrame) -> None:
        if integrated.empty:
            return
        fig, ax = plt.subplots(figsize=(8.5, 7.5))
        concordant = (np.sign(integrated["PhosphoLog2FC"]) == np.sign(integrated["ProteinLog2FC"]))
        ax.scatter(integrated.loc[concordant, "PhosphoLog2FC"], integrated.loc[concordant, "ProteinLog2FC"],
                   c=cfg.COLOR_ACCENT, s=22, alpha=0.7, edgecolors="none", label="Concordant")
        ax.scatter(integrated.loc[~concordant, "PhosphoLog2FC"], integrated.loc[~concordant, "ProteinLog2FC"],
                   c=cfg.COLOR_NS, s=18, alpha=0.5, edgecolors="none", label="Discordant")
        ax.axhline(0, color="dimgrey", lw=0.7)
        ax.axvline(0, color="dimgrey", lw=0.7)

        # Label genes with a strong signal in both layers
        strong = integrated[(integrated["PhosphoLog2FC"].abs() > 1) & (integrated["ProteinLog2FC"].abs() > 0.5)]
        for _, row in strong.head(15).iterrows():
            ax.annotate(row["GeneSym"], (row["PhosphoLog2FC"], row["ProteinLog2FC"]),
                        fontsize=8, xytext=(4, 3), textcoords="offset points")
        ax.set_xlabel("Phosphorylation — mean log2FC per gene")
        ax.set_ylabel("Protein abundance — log2FC")
        ax.set_title("Multi-omics Integration: Phosphorylation × Proteome (per gene)")
        ax.legend(loc="upper left", fontsize=9)
        cfg.save_figure(fig, "09_multiomics_phospho_vs_protein")


def main():
    MultiOmicsIntegration.integrate()


if __name__ == "__main__":
    main()
