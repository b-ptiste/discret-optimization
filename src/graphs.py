from pulp import *
import networkx as nx
import numpy as np
from random import randint


def make_graphe(i, R, cout_max, b):
    # Créer le graphe avec des valeurs aléatoires
    G = nx.complete_graph(i).to_directed()
    for u, v in G.edges():
        L = [randint(1, cout_max) for i in range(R + 1)]
        G.add_weighted_edges_from([(u, v, L)])
    # Réaliser la matrice de coût pour la modèle PULP +
    # le dictionnaire de matrice de Ressources
    C = np.zeros((len(G.nodes), len(G.nodes)))
    # Coût
    for u, v in G.edges():
        C[u][v] = G[u][v]["weight"][0]

    # matrice d'adjacence :
    A = np.zeros((len(G.nodes()), len(G.nodes())))
    for u, v in G.edges():
        A[u, v] = 1

    # Ressources + Dictionnaires des contraintes
    dict_ressources = {}
    dict_b = {}
    for r in range(1, R + 1):
        dict_ressources[r] = np.zeros((len(G.nodes), len(G.nodes)))
        # Contrainte
        dict_b[r] = b[r - 1]
        for u, v in G.edges():
            dict_ressources[r][u][v] = G[u][v]["weight"][r]
    return G, C, A, dict_ressources, dict_b


def make_random_graphe(i, R, cout_max, b, p):
    # Créer le graphe avec des valeurs aléatoires
    G = nx.fast_gnp_random_graph(i, p).to_directed()
    for u, v in G.edges():
        L = [randint(1, cout_max) for i in range(R + 1)]
        G.add_weighted_edges_from([(u, v, L)])
    # Réaliser la matrice de coût pour la modèle
    # PULP + le dictionnaire de matrice de Ressources

    C = np.zeros((len(G.nodes), len(G.nodes)))
    # Coût
    for u, v in G.edges():
        C[u][v] = G[u][v]["weight"][0]
    # Ressources + Dictionnaires des contraintes
    dict_ressources = {}
    dict_b = {}

    # matrice d'adjacence du graphe.
    A = np.zeros((len(G.nodes()), len(G.nodes())))
    for u, v in G.edges():
        A[u, v] = 1

    for r in range(1, R + 1):
        dict_ressources[r] = np.zeros((len(G.nodes), len(G.nodes)))
        # Contrainte
        dict_b[r] = b[r - 1]
        for u, v in G.edges():
            dict_ressources[r][u][v] = G[u][v]["weight"][r]

    return G, C, A, dict_ressources, dict_b


def random_choose_node(N):
    return np.random.choice(N, 2, replace=False)
