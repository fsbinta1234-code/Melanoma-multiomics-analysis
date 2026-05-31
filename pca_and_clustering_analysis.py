import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from cleanDatas import CleanDatas


class PcaAndClusteringAnalysis:
    """
    Investigates melanoma subtype heterogeneity and resistance-associated
    clustering patterns using PCA dimensionality reduction and K-Means
    unsupervised clustering on phosphoproteomic intensity profiles.

    Expected Output: PCA plots, tumor subtype clusters, and
    resistance-associated subgrouping.
    """

    N_COMPONENTS = 2
    N_CLUSTERS   = 3
    RANDOM_STATE = 42

    # Human-readable sample labels for the 6 columns in the intensity matrix
    SAMPLE_LABELS = [
        'Control_1', 'Control_2', 'Control_3',
        'Resistant_1', 'Resistant_2', 'Resistant_3',
    ]

    @staticmethod
    def run_analysis(phospho_log2: pd.DataFrame = None) -> dict:
        """
        Runs PCA and K-Means clustering on the phospho intensity matrix.

        The intensity matrix is transposed so each sample is a row and each
        phosphosite is a feature.  PCA reduces the feature space to two
        principal components for visualisation.  K-Means partitions the
        samples into N_CLUSTERS groups that may correspond to melanoma
        resistance subtypes.

        Parameters
        ----------
        phospho_log2 : pd.DataFrame, optional
            Log2-normalised phospho intensity matrix (sites × samples).
            When None, CleanDatas.clean_phospho_sty_sites() is called.

        Returns
        -------
        dict
            Dictionary with keys:
            - 'pca_result' : np.ndarray of shape (n_samples, 2)
            - 'clusters'   : np.ndarray of cluster labels (length n_samples)
            - 'pca'        : fitted PCA object
            - 'kmeans'     : fitted KMeans object
        """
        if phospho_log2 is None:
            phospho_log2 = CleanDatas.clean_phospho_sty_sites()

        # Transpose: rows = samples, columns = phosphosite features
        X = phospho_log2.T.values

        # ── PCA ──────────────────────────────────────────────────────────
        pca        = PCA(n_components=PcaAndClusteringAnalysis.N_COMPONENTS)
        pca_result = pca.fit_transform(X)

        explained = pca.explained_variance_ratio_ * 100
        print(
            f"PCA variance explained : "
            f"PC1={explained[0]:.1f}%  PC2={explained[1]:.1f}%"
        )

        PcaAndClusteringAnalysis._plot_pca(pca_result, explained)

        # ── K-Means ───────────────────────────────────────────────────────
        kmeans   = KMeans(
            n_clusters=PcaAndClusteringAnalysis.N_CLUSTERS,
            random_state=PcaAndClusteringAnalysis.RANDOM_STATE,
            n_init=10,
        )
        clusters = kmeans.fit_predict(X)

        print(f"Cluster assignments : {clusters}")

        PcaAndClusteringAnalysis._plot_clusters(pca_result, clusters)

        return {
            'pca_result': pca_result,
            'clusters':   clusters,
            'pca':        pca,
            'kmeans':     kmeans,
        }

    @staticmethod
    def _plot_pca(pca_result: np.ndarray, explained: np.ndarray) -> None:
        """
        Scatter plot of samples in PCA space, coloured by group.

        Parameters
        ----------
        pca_result : np.ndarray
            Array of shape (n_samples, 2) with PC coordinates.
        explained : np.ndarray
            Variance explained by each component (percentages).
        """
        colours = ['steelblue'] * 3 + ['crimson'] * 3
        labels  = PcaAndClusteringAnalysis.SAMPLE_LABELS

        fig, ax = plt.subplots(figsize=(8, 6))
        for i, (x, y) in enumerate(pca_result):
            ax.scatter(x, y, c=colours[i], s=120, zorder=3)
            ax.annotate(labels[i], (x, y), textcoords='offset points',
                        xytext=(6, 4), fontsize=9)

        ax.set_xlabel(f'PC1 ({explained[0]:.1f}%)', fontsize=12)
        ax.set_ylabel(f'PC2 ({explained[1]:.1f}%)', fontsize=12)
        ax.set_title('PCA of Melanoma Phosphoproteomic Samples', fontsize=13)
        ax.axhline(0, color='lightgrey', linewidth=0.6)
        ax.axvline(0, color='lightgrey', linewidth=0.6)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def _plot_clusters(pca_result: np.ndarray, clusters: np.ndarray) -> None:
        """
        Scatter plot of samples coloured by K-Means cluster assignment.

        Parameters
        ----------
        pca_result : np.ndarray
            Array of shape (n_samples, 2) with PC coordinates.
        clusters : np.ndarray
            Cluster label for each sample.
        """
        labels = PcaAndClusteringAnalysis.SAMPLE_LABELS

        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(
            pca_result[:, 0],
            pca_result[:, 1],
            c=clusters,
            cmap='Set1',
            s=120,
            zorder=3,
        )
        for i, label in enumerate(labels):
            ax.annotate(label, pca_result[i], textcoords='offset points',
                        xytext=(6, 4), fontsize=9)

        plt.colorbar(scatter, ax=ax, label='Cluster')
        ax.set_xlabel('PC1', fontsize=12)
        ax.set_ylabel('PC2', fontsize=12)
        ax.set_title(
            f'K-Means Clustering (k={PcaAndClusteringAnalysis.N_CLUSTERS}) '
            f'of Melanoma Samples',
            fontsize=13
        )
        ax.axhline(0, color='lightgrey', linewidth=0.6)
        ax.axvline(0, color='lightgrey', linewidth=0.6)

        plt.tight_layout()
        plt.show()


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main():
    PcaAndClusteringAnalysis.run_analysis()


if __name__ == "__main__":
    main()
