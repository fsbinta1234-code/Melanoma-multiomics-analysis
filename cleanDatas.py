"""
Phase 2 — Cleaning and quality control of the three omics datasets.

Each method loads the raw data via ReadData, removes contaminants/reverse hits,
selects only the real sample columns (1A..12D in MaxQuant), applies log2
normalisation and missing-value imputation, returning a sample-per-column matrix
ready for analysis.
"""
import numpy as np
import pandas as pd

import pipeline_config as cfg
from readDataset import ReadData


class CleanDatas:
    """Cleaning and pre-processing of the phospho, protein and transcript matrices."""

    LOCALIZATION_CUTOFF = 0.75  # minimum phosphosite localisation confidence

    # ------------------------------------------------------------------
    # GEO microarray (GSE199405) — transcriptomics
    # ------------------------------------------------------------------
    @staticmethod
    def clean_geo_series_matrix() -> pd.DataFrame:
        """GEO expression matrix (already log2-normalised by RMA).

        Only coerces to numeric and drops fully-empty probes. Does NOT re-apply
        log2 (the values are already on a log2 scale).
        """
        df = ReadData.read_geo_series_matrix()
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.dropna(how="all")
        return df

    # ------------------------------------------------------------------
    # MaxQuant proteinGroups — proteomics (LFQ)
    # ------------------------------------------------------------------
    @staticmethod
    def clean_protein_groups() -> pd.DataFrame:
        """Protein matrix: imputed log2(LFQ), columns = tokens 1A..12D.

        Index = gene symbol (or 'Majority protein IDs' as a fallback).
        """
        df = ReadData.read_protein_groups()

        # Remove contaminants and reverse-database hits
        for flag in ("Reverse", "Potential contaminant"):
            if flag in df.columns:
                df = df[df[flag] != "+"]

        # Keep only real LFQ sample columns (1A..12D)
        col_map = cfg.maxquant_clean_column_map(df.columns, "LFQ intensity ")
        lfq = df[list(col_map)].rename(columns=col_map)
        lfq = lfq[sorted(lfq.columns, key=lambda t: (cfg._token_number(t), t[-1]))]

        # Readable index: gene (1st symbol) or protein ID
        gene = df.get("Gene names", pd.Series(index=df.index, dtype="object"))
        majority = df.get("Majority protein IDs", pd.Series(index=df.index, dtype="object"))
        ids = gene.fillna("").astype(str).str.split(";").str[0]
        ids = ids.where(ids != "", majority.fillna("PG").astype(str).str.split(";").str[0])
        lfq.index = CleanDatas._make_unique(ids.values)

        # log2 + min-value imputation
        log2 = np.log2(lfq + 1)
        log2 = cfg.min_value_impute(log2)
        return log2

    # ------------------------------------------------------------------
    # MaxQuant Phospho (STY) Sites — phosphoproteomics
    # ------------------------------------------------------------------
    @staticmethod
    def clean_phospho_sty_sites(return_meta: bool = False):
        """Phosphosite matrix: imputed log2(Intensity), columns 1A..12D.

        Filters contaminants/reverse hits and requires Localization prob ≥ 0.75.
        Index = '<GENE>_<AA><pos>' (unique). With return_meta=True, also returns
        a Series with the full gene name(s) per site.
        """
        df = ReadData.read_phospho_sty_sites()

        for flag in ("Reverse", "Potential contaminant"):
            if flag in df.columns:
                df = df[df[flag] != "+"]
        if "Localization prob" in df.columns:
            df = df[df["Localization prob"] >= CleanDatas.LOCALIZATION_CUTOFF]
        df = df.reset_index(drop=True)

        # Keep only real sample intensity columns (1A..12D)
        col_map = cfg.maxquant_clean_column_map(df.columns, "Intensity ")
        intensity = df[list(col_map)].rename(columns=col_map)
        intensity = intensity[
            sorted(intensity.columns, key=lambda t: (cfg._token_number(t), t[-1]))
        ]

        # Readable site identifier: GENE_AApos
        gene_full = df.get("Gene names", pd.Series(index=df.index, dtype="object")).fillna("NA").astype(str)
        gene1 = gene_full.str.split(";").str[0].replace("", "NA")
        aa = df.get("Amino acid", pd.Series(index=df.index, dtype="object")).fillna("?").astype(str)
        pos = pd.to_numeric(df.get("Position", pd.Series(index=df.index)), errors="coerce").fillna(0).astype(int)
        site_ids = CleanDatas._make_unique((gene1 + "_" + aa + pos.astype(str)).values)
        intensity.index = site_ids

        # log2 + min-value imputation
        log2 = np.log2(intensity + 1)
        log2 = cfg.min_value_impute(log2)

        if return_meta:
            meta = pd.DataFrame({"Gene": gene_full.values}, index=site_ids)
            return log2, meta
        return log2

    # ------------------------------------------------------------------
    # helper: make labels unique (suffix .1, .2, ... for duplicates)
    # ------------------------------------------------------------------
    @staticmethod
    def _make_unique(labels) -> pd.Index:
        seen = {}
        out = []
        for lab in labels:
            if lab in seen:
                seen[lab] += 1
                out.append(f"{lab}.{seen[lab]}")
            else:
                seen[lab] = 0
                out.append(lab)
        return pd.Index(out)


# ----------------------------------------------------------------------
def main():
    print("=== Dataset cleaning ===")
    geo = CleanDatas.clean_geo_series_matrix()
    print(f"GEO transcriptomics : {geo.shape[0]} probes × {geo.shape[1]} samples")

    prot = CleanDatas.clean_protein_groups()
    ctrl, res = cfg.maxquant_groups(prot)
    print(f"Proteins            : {prot.shape[0]} × {prot.shape[1]}  "
          f"(control={len(ctrl)}, resistant={len(res)})")

    phos = CleanDatas.clean_phospho_sty_sites()
    ctrl, res = cfg.maxquant_groups(phos)
    print(f"Phosphosites        : {phos.shape[0]} × {phos.shape[1]}  "
          f"(control={len(ctrl)}, resistant={len(res)})")


if __name__ == "__main__":
    main()
