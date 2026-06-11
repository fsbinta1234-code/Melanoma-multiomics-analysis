"""
Phase 11 — PCA and clustering of the samples.

Reduces the phosphoproteomic profile to 2 principal components and applies
K-Means to investigate heterogeneity and resistance-associated grouping. There
are 48 samples (24 ARoe + 24 LacZ); labels and colours are derived from the
experimental design.
"""
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import pipeline_config as cfg
from cleanDatas import CleanDatas


class PcaAndClusteringAnalysis:
    N_CLUSTERS = 2
    RANDOM_STATE = 42

    @staticmethod
    def run_analysis() -> dict:
        cfg.apply_style()
        phospho = CleanDatas.clean_phospho_sty_sites()
        tokens = list(phospho.columns)
        labels = cfg.maxquant_labels(phospho)              # 1=ARoe, 0=LacZ

        X = StandardScaler().fit_transform(phospho.T.to_numpy())

        pca = PCA(n_components=2, random_state=PcaAndClusteringAnalysis.RANDOM_STATE)
        pcs = pca.fit_transform(X)
        var = pca.explained_variance_ratio_ * 100
        print(f"Explained variance: PC1={var[0]:.1f}%  PC2={var[1]:.1f}%")

        km = KMeans(n_clusters=PcaAndClusteringAnalysis.N_CLUSTERS,
                    random_state=PcaAndClusteringAnalysis.RANDOM_STATE, n_init=10)
        clusters = km.fit_predict(X)

        PcaAndClusteringAnalysis._plot(pcs, var, tokens, labels, clusters)
        return {"pcs": pcs, "clusters": clusters, "var": var}

    @staticmethod
    def _plot(pcs, var, tokens, labels, clusters) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # (a) Coloured by experimental group, marker by cell line
        markers = {"A375": "o", "M14": "s", "WM9": "^"}
        for i, tok in enumerate(tokens):
            color = cfg.COLOR_RESIST if labels[i] == 1 else cfg.COLOR_CONTROL
            axes[0].scatter(pcs[i, 0], pcs[i, 1], c=color,
                            marker=markers[cfg.maxquant_cell_line(tok)],
                            s=90, edgecolors="#333333", linewidths=0.5, zorder=3)
        from matplotlib.lines import Line2D
        legend1 = [Line2D([0], [0], marker="o", color="w", markerfacecolor=cfg.COLOR_RESIST,
                          markersize=10, label="ARoe (resistant)"),
                   Line2D([0], [0], marker="o", color="w", markerfacecolor=cfg.COLOR_CONTROL,
                          markersize=10, label="LacZ (control)")]
        legend2 = [Line2D([0], [0], marker=m, color="w", markerfacecolor="#888888",
                          markersize=10, label=cl) for cl, m in markers.items()]
        axes[0].legend(handles=legend1 + legend2, fontsize=9, loc="best")
        axes[0].set_title("PCA — coloured by group / cell line")

        # (b) Coloured by K-Means cluster
        sc = axes[1].scatter(pcs[:, 0], pcs[:, 1], c=clusters, cmap="Set1",
                             s=90, edgecolors="#333333", linewidths=0.5, zorder=3)
        axes[1].set_title(f"K-Means (k={PcaAndClusteringAnalysis.N_CLUSTERS})")
        plt.colorbar(sc, ax=axes[1], label="Cluster", fraction=0.046, pad=0.04)

        for ax in axes:
            ax.set_xlabel(f"PC1 ({var[0]:.1f}%)")
            ax.set_ylabel(f"PC2 ({var[1]:.1f}%)")
            ax.axhline(0, color="lightgrey", lw=0.6)
            ax.axvline(0, color="lightgrey", lw=0.6)

        fig.suptitle("PCA & Clustering of Melanoma Samples (phosphoproteomics)",
                     fontsize=15, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        cfg.save_figure(fig, "11_pca_clustering")


def main():
    PcaAndClusteringAnalysis.run_analysis()


if __name__ == "__main__":
    main()
