"""
Phase 13 — Cytoscape export.

Exports the signalling network in GraphML (the recommended format for Cytoscape),
preserving each node's pathway attribute.
"""
import os
import networkx as nx

import pipeline_config as cfg
from network_based_systems_biology import NetworkBasedSystemsBiology


class CytoscapeExport:

    OUTPUT_NAME = "Melanoma_Resistance_Network.graphml"

    @staticmethod
    def export(G: nx.DiGraph = None) -> str:
        if G is None:
            G = NetworkBasedSystemsBiology.build_network()
        cfg.ensure_dirs()
        path = os.path.join(cfg.OUT_DIR, CytoscapeExport.OUTPUT_NAME)
        nx.write_graphml(G, path)
        print(f"Network exported → {os.path.relpath(path, cfg.BASE_DIR)}")
        print(f"Nodes: {G.number_of_nodes()}  |  Edges: {G.number_of_edges()}")
        print("Open in Cytoscape: File → Import → Network from File")
        return path


def main():
    CytoscapeExport.export()


if __name__ == "__main__":
    main()
