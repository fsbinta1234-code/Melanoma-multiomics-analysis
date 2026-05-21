import pandas as pd
import os


class ReadData:
    """
    Central class responsible for loading the three omics datasets used in the
    melanoma multi-omics analysis project.  Each static method targets one
    specific file inside the 'datas/' directory and returns a pandas DataFrame
    ready for downstream analysis.
    """

    # ------------------------------------------------------------------
    # Dataset 1 – GEO microarray expression matrix (GSE199405)
    # ------------------------------------------------------------------
    @staticmethod
    def read_geo_series_matrix(path: str = "datas/GSE199405_series_matrix.txt") -> pd.DataFrame:
        """
        Reads the GEO Series Matrix file for accession GSE199405.

        File format details
        -------------------
        The file has two distinct sections:
          1. Metadata header  – lines that start with '!' (e.g. !Series_title,
             !Sample_title, etc.).  These lines are skipped by pandas when
             comment='!' is set.
          2. Expression table – delimited by the sentinel lines
             '!series_matrix_table_begin' and '!series_matrix_table_end'.
             The first row of this table is the header:
               ID_REF  GSM5972234  GSM5972235  ...  GSM5972245
             Subsequent rows hold normalised expression values (RMA) for each
             of the 135 750 Affymetrix Clariom D probe sets across 12 samples
             (3 melanoma cell lines × 2 vectors × 2 treatments).

        Returns
        -------
        pd.DataFrame
            Shape: (135 750, 12).
            Index : probe IDs (ID_REF column).
            Columns : GEO sample accession IDs (GSMxxxxxxx).
        """
        # Verify the file exists before attempting to read it
        if not os.path.exists(path):
            raise FileNotFoundError(f"GEO series matrix file not found: {path}")

        # pandas' comment parameter drops every line whose first non-whitespace
        # character matches the given string.  Because all metadata lines start
        # with '!', this cleanly skips them, including the
        # '!series_matrix_table_begin/end' sentinel lines.
        df = pd.read_csv(
            path,
            sep="\t",       # columns are tab-separated
            comment="!",    # ignore metadata lines starting with '!'
            index_col=0,    # use ID_REF as the row index
            quotechar='"',  # values are wrapped in double quotes
        )

        # Remove any rows whose index is NaN (can occur if pandas reads a blank
        # line between the sentinel and the header)
        df = df[df.index.notna()]

        return df

    # ------------------------------------------------------------------
    # Dataset 2 – MaxQuant protein groups (label-free proteomics)
    # ------------------------------------------------------------------
    @staticmethod
    def read_protein_groups(path: str = "datas/proteinGroups.csv") -> pd.DataFrame:
        """
        Reads the MaxQuant proteinGroups output file.

        File format details
        -------------------
        Tab-separated text exported by MaxQuant after label-free quantification
        (LFQ).  Key columns include:
          - 'Protein IDs'          : UniProt accession(s) per protein group.
          - 'Majority protein IDs' : Most abundant accession in the group.
          - 'Gene names'           : HGNC gene symbol(s).
          - 'Protein names'        : Full protein name(s).
          - 'LFQ intensity *'      : Normalised LFQ intensities per sample.
          - 'iBAQ *'               : Intensity-based absolute quantification.
          - 'Peptide counts *'     : Peptides identified per sample.
          - 'Razor + unique peptides', 'Unique peptides', etc.

        The file contains ~6 587 protein groups across all melanoma samples.

        Returns
        -------
        pd.DataFrame
            Shape: (~6 587 rows, many columns).
            Index : default integer index (row number).
            Columns : all MaxQuant output columns.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"proteinGroups file not found: {path}")

        # The file uses tabs as delimiter (MaxQuant default export format)
        df = pd.read_csv(
            path,
            sep="\t",
            low_memory=False,  # mixed-type columns require full scan to infer dtype
        )

        return df

    # ------------------------------------------------------------------
    # Dataset 3 – MaxQuant phosphorylation sites (Phospho STY Sites)
    # ------------------------------------------------------------------
    @staticmethod
    def read_phospho_sty_sites(path: str = "datas/Phospho__STY_Sites.csv") -> pd.DataFrame:
        """
        Reads the MaxQuant Phospho (STY) Sites output file.

        File format details
        -------------------
        Tab-separated text exported by MaxQuant containing all detected
        phosphorylation events on serine (S), threonine (T), and tyrosine (Y)
        residues.  Key columns include:
          - 'Proteins'               : UniProt accession(s) of the host protein.
          - 'Positions within proteins': Amino-acid position of the phosphosite.
          - 'Leading proteins'       : Most likely protein bearing the site.
          - 'Gene names'             : HGNC gene symbol.
          - 'Localization prob'      : PTM-Score localisation probability (0–1).
          - 'Score diff'             : Score difference for site localisation.
          - 'PEP'                    : Posterior error probability.
          - 'Localization prob *'    : Per-sample localisation probabilities.
          - 'Intensity *'            : Raw intensities per sample.
          - 'Ratio H/L *'            : Heavy-to-light ratios (SILAC, if used).

        The file contains ~22 369 phosphosite entries.

        Returns
        -------
        pd.DataFrame
            Shape: (~22 369 rows, many columns).
            Index : default integer index (row number).
            Columns : all MaxQuant phospho-site output columns.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Phospho (STY) Sites file not found: {path}")

        df = pd.read_csv(
            path,
            sep="\t",
            low_memory=False,
        )

        return df


# ----------------------------------------------------------------------
# Quick smoke-test: run each loader and print a brief summary
# ----------------------------------------------------------------------
if __name__ == "__main__":

    datasets = {
        "GEO Expression Matrix": ReadData.read_geo_series_matrix,
        "Protein Groups":        ReadData.read_protein_groups,
        "Phospho (STY) Sites":   ReadData.read_phospho_sty_sites,
    }

    for name, loader in datasets.items():
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        df = loader()
        print(f"  Shape   : {df.shape[0]} rows x {df.shape[1]} columns")
        print(f"  Columns : {list(df.columns[:5])} ...")
        print(f"  dtypes  :\n{df.dtypes.value_counts().to_string()}")
        print(f"  Missing : {df.isnull().sum().sum()} total NaN values")
