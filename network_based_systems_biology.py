"""
Phase 12 — Network-based systems biology.

Reconstructs the BRAFi/MEKi resistance signalling network (MAPK/ERK, PI3K-AKT,
mTOR, EMT) as a directed graph, with nodes coloured by pathway and sized by
centrality.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import networkx as nx

import pipeline_config as cfg


class NetworkBasedSystemsBiology:

    NETWORK_EDGES = [
        ("RAS", "BRAF"), ("BRAF", "MEK"), ("MEK", "ERK"), ("ERK", "mTOR"),
        ("AKT", "mTOR"), ("ERK", "EMT"), ("PI3K", "AKT"),
        ("AKT", "ERK"),    # feedback reactivation in resistance
        ("mTOR", "EMT"), ("EMT", "AKT"),
    ]

    NODE_PATHWAY = {
        "RAS": "MAPK/ERK", "BRAF": "MAPK/ERK", "MEK": "MAPK/ERK", "ERK": "MAPK/ERK",
        "PI3K": "PI3K-AKT", "AKT": "PI3K-AKT", "mTOR": "mTOR", "EMT": "EMT",
    }
    PATHWAY_COLOR = {
        "MAPK/ERK": cfg.COLOR_RESIST, "PI3K-AKT": cfg.COLOR_CONTROL,
        "mTOR": cfg.COLOR_ACCENT, "EMT": "#E69F00",
    }

    @staticmethod
    def build_network() -> nx.DiGraph:
        G = nx.DiGraph()
        G.add_edges_from(NetworkBasedSystemsBiology.NETWORK_EDGES)
        for node in G.nodes:
            G.nodes[node]["pathway"] = NetworkBasedSystemsBiology.NODE_PATHWAY.get(node, "other")
        print(f"Nodes: {G.number_of_nodes()}  |  Edges: {G.number_of_edges()}")
        NetworkBasedSystemsBiology._plot(G)
        return G

    @staticmethod
    def _plot(G: nx.DiGraph) -> None:
        cfg.apply_style()
        pos = nx.spring_layout(G, seed=42, k=1.2)
        centrality = nx.betweenness_centrality(G)
        node_colors = [NetworkBasedSystemsBiology.PATHWAY_COLOR.get(
            G.nodes[n]["pathway"], cfg.COLOR_NS) for n in G.nodes]
        node_sizes = [2200 + 6000 * centrality[n] for n in G.nodes]

        fig, ax = plt.subplots(figsize=(10, 8))
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#666666", arrows=True,
                               arrowsize=22, width=1.6, connectionstyle="arc3,rad=0.08",
                               node_size=node_sizes)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                               node_size=node_sizes, edgecolors="#222222", linewidths=1.2)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=11, font_weight="bold",
                                font_color="white")
        handles = [Patch(facecolor=c, label=p)
                   for p, c in NetworkBasedSystemsBiology.PATHWAY_COLOR.items()]
        ax.legend(handles=handles, title="Pathway", loc="upper left", fontsize=9)
        ax.set_title("BRAFi/MEKi Resistance Signalling Network in Melanoma")
        ax.axis("off")
        cfg.save_figure(fig, "12_resistance_signaling_network")


def main():
    NetworkBasedSystemsBiology.build_network()


if __name__ == "__main__":
    main()
