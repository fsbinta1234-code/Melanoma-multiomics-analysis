import networkx as nx
from network_based_systems_biology import NetworkBasedSystemsBiology


class CytoscapeExport:
    """
    Exports the melanoma resistance signaling network to Cytoscape-compatible
    formats for advanced systems biology visualisation and pathway annotation.

    GraphML is the recommended import format for Cytoscape and preserves all
    node and edge attributes defined in the NetworkX graph.

    Expected Output: Cytoscape-compatible network files ready for import.
    """

    OUTPUT_FILE = 'Melanoma_Resistance_Network.graphml'

    @staticmethod
    def export(G: nx.DiGraph = None) -> None:
        """
        Writes the signaling network to a GraphML file for Cytoscape import.

        If no graph is provided, the melanoma resistance signaling network is
        built internally by NetworkBasedSystemsBiology.build_network().

        The resulting .graphml file can be opened directly in Cytoscape via
        File → Import → Network from File.

        Parameters
        ----------
        G : nx.DiGraph, optional
            Directed signaling graph to export.  When None,
            NetworkBasedSystemsBiology.build_network() is called.
        """
        if G is None:
            G = NetworkBasedSystemsBiology.build_network()

        nx.write_graphml(G, CytoscapeExport.OUTPUT_FILE)

        print(f"Network exported successfully → {CytoscapeExport.OUTPUT_FILE}")
        print(f"Nodes : {G.number_of_nodes()}")
        print(f"Edges : {G.number_of_edges()}")
        print("Open in Cytoscape via: File → Import → Network from File")


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main():
    CytoscapeExport.export()


if __name__ == "__main__":
    main()
