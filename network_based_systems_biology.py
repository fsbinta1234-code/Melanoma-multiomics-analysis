import matplotlib.pyplot as plt
import networkx as nx


class NetworkBasedSystemsBiology:
    """
    Reconstructs protein-protein interaction networks and kinase-substrate
    signaling systems for BRAFi/MEKi resistance in melanoma using
    graph-based systems biology approaches.

    Known pathway edges are encoded from the MAPK/ERK, PI3K-AKT, mTOR,
    and EMT resistance signaling literature.

    Expected Output: resistance signaling networks, kinase-substrate
    interaction maps, and systems biology communication networks.
    """

    # Directed edges representing known resistance signaling relationships
    NETWORK_EDGES = [
        ('RAS',   'BRAF'),
        ('BRAF',  'MEK'),
        ('MEK',   'ERK'),
        ('ERK',   'mTOR'),
        ('AKT',   'mTOR'),
        ('ERK',   'EMT'),
        ('PI3K',  'AKT'),
        ('AKT',   'ERK'),   # feedback reactivation in resistance
        ('mTOR',  'EMT'),
        ('EMT',   'AKT'),   # EMT-driven PI3K feedback
    ]

    @staticmethod
    def build_network() -> nx.DiGraph:
        """
        Constructs the melanoma resistance signaling network as a directed graph.

        Nodes represent signaling proteins and kinases.  Edges represent
        known activating or inhibitory relationships curated from the
        BRAFi/MEKi resistance literature.

        Returns
        -------
        nx.DiGraph
            Directed graph with all curated resistance signaling edges.
        """
        G = nx.DiGraph()
        G.add_edges_from(NetworkBasedSystemsBiology.NETWORK_EDGES)

        print(f"Nodes : {list(G.nodes)}")
        print(f"Edges : {G.number_of_edges()}")

        NetworkBasedSystemsBiology._plot_network(G)

        return G

    @staticmethod
    def _plot_network(G: nx.DiGraph) -> None:
        """
        Visualises the signaling network with a spring layout.

        Parameters
        ----------
        G : nx.DiGraph
            Directed melanoma resistance signaling graph.
        """
        pos = nx.spring_layout(G, seed=42)

        fig, ax = plt.subplots(figsize=(8, 6))

        nx.draw(
            G,
            pos,
            with_labels=True,
            node_size=3000,
            node_color='steelblue',
            font_color='white',
            font_size=10,
            font_weight='bold',
            edge_color='dimgrey',
            arrows=True,
            arrowsize=20,
            ax=ax,
        )

        ax.set_title('Melanoma Resistance Signaling Network', fontsize=13)
        plt.tight_layout()
        plt.show()


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main():
    NetworkBasedSystemsBiology.build_network()


if __name__ == "__main__":
    main()
