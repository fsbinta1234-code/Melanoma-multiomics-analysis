import pandas as pd
import numpy as np
from readDataset import ReadData


class CleanDatas:
    """
    Central class responsible for cleaning and preprocessing the three omics
    datasets used in the melanoma multi-omics analysis project.  Each static
    method loads its own raw data internally and returns a cleaned DataFrame
    ready for downstream analysis.
    """

    # ------------------------------------------------------------------
    # Cleaning 1 – GEO microarray expression matrix (GSE199405)
    # ------------------------------------------------------------------
    @staticmethod
    def clean_geo_series_matrix() -> pd.DataFrame:
        """
        Loads and cleans the GEO Series Matrix expression data (GSE199405).

        The raw file contains 135 750 Affymetrix Clariom D probe-set rows and
        12 sample columns (3 melanoma cell lines x 2 vectors x 2 treatments).
        Expression values are RMA-normalised log2 intensities.

        Returns
        -------
        pd.DataFrame
            Cleaned expression DataFrame.
        """
        # Load the raw expression matrix using the dedicated reader
        df = ReadData.read_geo_series_matrix()
        return df
    # ------------------------------------------------------------------
    # Cleaning 2 – MaxQuant protein groups (label-free proteomics)
    # ------------------------------------------------------------------
    @staticmethod
    def clean_protein_groups() -> pd.DataFrame:
        """
        Loads and cleans the MaxQuant proteinGroups dataset.

        The raw file contains ~6 587 protein groups across all melanoma samples
        with 418 columns covering peptide counts, LFQ intensities, iBAQ values,
        and quality-control flags produced by MaxQuant.

        Returns
        -------
        pd.DataFrame
            Cleaned protein groups DataFrame.
        """
        # Load the raw protein groups table using the dedicated reader
        df = ReadData.read_protein_groups()

        # --- cleaning steps go here ---

        return df

    # ------------------------------------------------------------------
    # Cleaning 3 – MaxQuant phosphorylation sites (Phospho STY Sites)
    # ------------------------------------------------------------------
    @staticmethod
    def clean_phospho_sty_sites() -> pd.DataFrame:
        """
        Loads and cleans the MaxQuant Phospho (STY) Sites dataset.

        The raw file contains ~22 369 phosphorylation events on serine (S),
        threonine (T), and tyrosine (Y) residues across all samples, with 651
        columns covering localisation probabilities, intensities, and scores.

        Returns
        -------
        pd.DataFrame
            Cleaned phospho sites DataFrame.
        """
        # Load the raw phospho sites table using the dedicated reader
        df = ReadData.read_phospho_sty_sites()

        # --- cleaning steps go here ---
        print(df.columns)
    
        # --- cleaning steps go here ---
        # REMOVE CONTAMINANTS
        phospho_clean = df[
        (df['Reverse'] != '+') &
        (df['Potential contaminant'] != '+')
        ]
        # FILTER PHOSPHOSITES
        phospho_clean = phospho_clean[
        phospho_clean['Localization prob'] >= 0.75
        ]
        # SELECT INTENSITY COLUMNS
        intensity_cols = [
        col for col in phospho_clean.columns
        if 'Intensity' in col
        ]
        phospho_intensity = phospho_clean[intensity_cols]
        # LOG2 NORMALIZATION
        phospho_log2 = np.log2(
        phospho_intensity + 1
        )
        # HANDLE MISSING VALUES
        phospho_log2 = phospho_log2.fillna(
        phospho_log2.median()
        )
        '''
        # QUALITY CONTROL VISUALIZATION
        plt.figure(figsize=(10,6))
        sns.boxplot(data=phospho_log2)
        plt.xticks(rotation=90)
        plt.title("Normalized Phosphoproteomics Intensities")
        plt.show()
        '''
        return phospho_log2

        return df


# ----------------------------------------------------------------------
# Quick smoke-test: run each cleaner and print a brief summary
# ----------------------------------------------------------------------
if __name__ == "__main__":

    cleaners = {
        "GEO Expression Matrix": CleanDatas.clean_geo_series_matrix,
        "Protein Groups":        CleanDatas.clean_protein_groups,
        "Phospho (STY) Sites":   CleanDatas.clean_phospho_sty_sites,
    }

    for name, cleaner in cleaners.items():
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        cleaned = cleaner()
        print(f"  Shape   : {cleaned.shape if cleaned is not None else 'not implemented yet'}")
