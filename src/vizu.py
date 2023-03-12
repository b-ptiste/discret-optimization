from pulp import *
import networkx as nx
import matplotlib.pyplot as plt


def affichage(g, s, t, trace):
    """
    Cette fonction permettra d'afficher un graphe networkx en coloriant en rouge un chemin
    **trace** dans le graphe, que l'on a spécifié à la fonction.
    """
    len(g.nodes())  # nombre de noeuds dans le graphe.
    pos = nx.circular_layout(
        g
    )  # ici c'est juste un argument pour faire un graphe un peu ordonné
    # ici on va définir les couleurs des arcs.
    for edg in g.edges():
        a = list(edg)
        # on va
        if edg in trace:
            g[a[0]][a[1]]["color"] = "red"
        else:
            g[a[0]][a[1]]["color"] = "grey"

    colors = [g[u][v]["color"] for u, v in g.edges()]
    [g[u][v]["weight"] for u, v in g.edges()]

    couleurs_sommets = ["yellow"] * g.number_of_nodes()

    sommets_a_voir = [
        s,
        t,
    ]  # ici on colorie juste la source et le puit d'une couleur différente.

    for i in sommets_a_voir:
        couleurs_sommets[i] = "red"  # source et puit en rouge.

    options = {
        "node_color": couleurs_sommets,
        "node_size": 550,
        "pos": pos,
        "edge_color": colors,
        "with_labels": True,
    }

    plt.figure()
    labels = nx.get_edge_attributes(g, "weight")
    nx.draw(g, **options)
    nx.draw_networkx_edge_labels(g, edge_labels=labels, pos=pos)
    plt.show()
