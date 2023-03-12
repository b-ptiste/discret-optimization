from pulp import *
import networkx as nx
import numpy as np
import time


def Pareto(tab):
    """
    On commence par écrire une fonction qui donnée un ensemble d'Etiquette,
    nous renvoie simplement ce même ensemble mais en ayant supprimer les etiquettes
    qui sont dominées au sens de Pareto. Cette fonction nous servira pour mettre à jour
    les Etiquettes qui sont présentent sur chaque sommet à chaque itérations de l'algorithme
    de correction d'etiquette.
    """

    # filtre qui va nous permettre se savoir qu'elles etiquettes garder.
    # filtre[j] = True <=> on conserve l'etiquette j de tab

    # on initialise tout à True.
    filtre = [True] * len(tab)

    for i in range(len(tab)):
        # si filtre[i] = False, cela veut dire que i est dominé
        if filtre[i]:
            E = tab[i]  # E
            for j in range(i + 1, len(tab)):
                Ep = tab[j]  # E'

                # le but va être de comparer E et E'

                test = [E[r] - Ep[r] for r in range(len(E))]

                t1 = all(k <= 0 for k in test)
                t2 = all(k >= 0 for k in test)
                t3 = all(k == 0 for k in test)

                # E domine Ep
                if t1 and not (t3):
                    filtre[j] = False

                # Ep domine E
                if t2 and not (t3):
                    filtre[i] = False
                    break

    return np.array(tab)[filtre].tolist()


def Pareto_bis(tab):
    # la fonction modifie le dictionnaire tab.
    # cette fois ci tab est un dictionnaire.
    filtre = {}
    for k in tab:
        filtre[k] = []

    for k in tab:
        for i in range(len(tab[k])):
            E = tab[k][i]
            for kp in tab:
                for j in range(len(tab[kp])):
                    if k == kp and i == j:
                        buff = ""  # on ne fait rien
                    else:
                        Ep = tab[kp][j]
                        test = [E[r] - Ep[r] for r in range(len(E))]

                        t1 = all(l <= 0 for l in test)
                        t2 = all(l >= 0 for l in test)
                        t3 = all(l == 0 for l in test)

                        # E domine Ep
                        if t1 and not (t3):
                            if kp not in filtre:
                                filtre[kp] = [j]
                            else:
                                if j not in filtre[kp]:
                                    filtre[kp].append(j)

                        # Ep domine E
                        if t2 and not (t3):
                            if k not in filtre:
                                filtre[kp] = [i]
                            else:
                                if i not in filtre[k]:
                                    filtre[k].append(i)

    for k in filtre:
        buff = [tab[k][i] for i in range(len(tab[k])) if i not in filtre[k]]
        if len(buff) == 0:
            # dans ce cas la on peut venir supprimer la clé
            del tab[k]
        else:
            tab[k] = np.unique(buff, axis=0).tolist()


def Correc_Etiq(G, s, t, R, b):
    """
    R est le nombre de ressources que l'on a dans notre graphe.
    """

    ETIQ = {}  # dictionnaire d'etiquette
    for i in G.nodes:
        ETIQ[i] = []  # on initialise tout à l'ensemble vide.

    L = [s]
    ETIQ[s] = [[0] * (R + 1)]

    while len(L) != 0:
        # tant que la liste n'est pas vide

        # on prend le premier élément de la liste, c'est un choix arbitraire.
        i = L.pop(0)
        # print(len(L))
        # on regarde maintenant tous les successeurs de i
        for j in G.successors(i):
            # on boucle maintenant sur toutes les etiquettes de i
            for E in ETIQ[i]:
                test = [E[r] + G[i][j]["weight"][r] - b[r - 1] for r in range(1, R + 1)]
                bl = all(i <= 0 for i in test)

                if bl is True:
                    Ep = [E[r] + G[i][j]["weight"][r] for r in range(0, R + 1)]
                    pr = ETIQ[j]
                    if Ep not in pr:
                        pr.append(Ep)
                        ETIQ[j] = Pareto(pr)

                        if Ep in ETIQ[j]:
                            if j not in L:
                                L.append(j)

    # on choisit maintenant la meilleure etiquette.
    pcc = 0
    try:
        pcc = ETIQ[t][0]

        for k in range(len(ETIQ[t])):
            if ETIQ[t][k][0] < pcc[0]:
                pcc = ETIQ[t][k]

    except:
        print("trop de contraintes")

    return pcc


def trans(d):
    res = []
    for k in d:
        for E in d[k]:
            res.append(E)
    return res


def recherche1(d):
    pcc = []
    cp = 1
    pred = 0
    for k in d:
        for E in d[k]:
            if cp == 1:
                cp += 1
                pcc = E
                pred = k

            else:
                b = min(pcc[0], E[0])
                if b == E[0]:
                    pcc = E
                    pred = k
    return {"pred": pred, "pcc": pcc}


def recherche2(g, d, cont, R, curr, av):
    pcc = []
    pred = []
    cp = 1

    for k in d:
        for E in d[k]:
            test = [E[r] + g[curr][av]["weight"][r] - cont[r] for r in range(R + 1)]
            bl = all(i == 0 for i in test)
            if bl is True:
                if cp == 1:
                    pcc = E
                    pred = k
                    cp += 1
                else:
                    if E[0] < pcc[0]:
                        pcc = E
                        pred = k

    return {"pred": pred, "pcc": pcc}


class Impossible(Exception):
    pass


def Correc_Etiq_bis(G, s, t, R, b):
    tic = time.perf_counter()

    ETIQ = {}  # dictionnaire d'etiquette
    for i in G.nodes:
        ETIQ[i] = {}  # on initialise tout à l'ensemble vide.

    # la liste L nous indique quand on doit arréter l'algorithme.
    L = [s]  # on initialise avec simplement le sommet s au départ.

    ETIQ[s] = {s: [[0] * (R + 1)]}

    while len(L) != 0:
        i = L.pop(0)
        for j in G.successors(i):
            for E in trans(ETIQ[i]):
                # on regarde si on peut aller de i à j en respectant toutes les contraintes de ressources.
                test = [E[r] + G[i][j]["weight"][r] - b[r - 1] for r in range(1, R + 1)]
                bl = all(i <= 0 for i in test)

                # si il y a respect de toutes les contraintes on met à jour les Etiquettes.
                if bl is True:
                    # Ep est la nouvelle etiquette que on va rajouter à notre ensemble
                    Ep = [E[r] + G[i][j]["weight"][r] for r in range(0, R + 1)]

                    # on regarde si l'étiquette Ep est pas déjà présente dans le dictionnaire.
                    # si elle y est déjà on ne la rajoute pas.

                    if existe(Ep, ETIQ[j]) is False:
                        if i not in ETIQ[j]:
                            ETIQ[j][i] = [Ep]
                        else:
                            ETIQ[j][i].append(Ep)

                        # on supprime les Etiquettes dominées.
                        Pareto_bis(ETIQ[j])
                        if i in ETIQ[j]:
                            if Ep in ETIQ[j][i]:
                                if j not in L:
                                    L.append(j)

    # on choisit maintenant la meilleure etiquette
    # on va aussi chercher le plus court chemin.
    pcc = 0
    trace = []

    # variable pour se repérer.
    curr = t
    av = t

    # vecteur de vérification.
    cont = [0] * (R + 1)
    try:
        trace.append(t)
        d = recherche1(ETIQ[t])
        # mise à jour des sommets
        curr = d["pred"]
        av = t

        pcc = d["pcc"]
        cont = pcc
        if pcc == []:
            raise Impossible("probleme non faisable")

        # mise à jour de la trace.
        trace.append(curr)

        while curr != s:
            d = recherche2(G, ETIQ[curr], cont, R, curr, av)
            trace.append(d["pred"])
            tampon = curr
            curr = d["pred"]
            av = tampon
            cont = d["pcc"]
        if pcc == []:
            raise Impossible("probleme non faisable")

    except Impossible:
        toc = time.perf_counter()
        return {"etat": "Impossible", "temps": toc - tic}

    toc = time.perf_counter()
    return {"pcc": pcc, "trace": trace[::-1], "temps": toc - tic, "etat": "resolu"}


def MAJ_ORANT(gp, b, R, s, t, pccs, pcct, majorant):
    U = majorant
    for r in range(1, R + 1):
        bl = True
        for q in range(1, R + 1):
            if pccs[t][r][q] > b[q - 1]:
                bl = False
        if bl is True:
            if pccs[t][r][0] < U:
                U = pccs[t][r][0]

    for u, v in gp.edges():
        for P in pccs[u].values():
            for Pp in pcct[v].values():
                bl = True
                for r in range(1, R + 1):
                    if P[r] + gp[u][v]["weight"][r] + Pp[r] > b[r - 1]:
                        bl = False
                        break

                if bl is True and P[0] + gp[u][v]["weight"][0] + Pp[0] < U:
                    U = P[0] + gp[u][v]["weight"][0] + Pp[0]

    return U


def existe(Ep, d):
    res = False
    for k in d:
        if Ep in d[k]:
            res = True
            break
    return res


from pulp import *
import math
import networkx as nx
import time


def Bel_Ford(g, s, r, R):
    # L est un dictionnnaire qui représente la distance pour aller à chaque sommets
    L = {}
    L[s] = [0 for i in range(R + 1)]

    pred = {}
    pred[s] = s

    n = len(g.nodes())  # nombre de noeuds

    for i in g.nodes():
        if i != s:
            # initialisation à l'infini.
            L[i] = [0 for i in range(R + 1)]
            L[i][r] = math.inf
            # chaque sommet a lui même comme prédecesseur pour commencer.
            pred[i] = i

    for k in range(n - 2):
        # on regarde pour tous les noeuds.
        for v in g.nodes():
            # on propage la solution par les prédecesseurs

            for p in g.predecessors(v):
                lpv = g[p][v]["weight"][r]

                if L[p][r] + lpv < L[v][r]:
                    L[v][r] = L[p][r] + lpv
                    pred[v] = p
                    for k in range(R + 1):
                        L[v][k] = L[p][k] + g[p][v]["weight"][k]

    return {"L": L, "pred": pred}


def ELIM_SOMMETS(gp, b, R, s, t, U, pccs, pcct, chg):
    """
    Cette fonction va nous permettre de supprimer certains nœuds du graphe qui ne font pas parti de notre plus court chemin.
    la philosophie est assez simple. On considère un nœud $i$, et on regarde les plus courts chemins pour faire $s->i$ puis $i->t$
    selon toutes les ressources. Si selon la ressource 0, cette distance est strictement supérieure au majorant,
    alors on peut supprimer le sommet et si selon les autres ressources cette distance est supérieure strictement
    aux limites du problème, on peut aussi supprimer ce sommet.
    """
    res = chg
    nodes = [i for i in gp.nodes()]

    for i in nodes:
        # on ne supprime pas les noeuds s ou t du graphe.
        if i == s or i == t:
            pass
        else:
            for r in range(R + 1):
                if r == 0:
                    if pccs[i][0][0] + pcct[i][0][0] > U:
                        e = [(j, i) for j in gp.predecessors(i)] + [
                            (i, j) for j in gp.successors(i)
                        ]
                        gp.remove_edges_from(e)
                        gp.remove_node(i)
                        res = True
                        break
                else:
                    if pccs[i][r][r] + pcct[i][r][r] > b[r - 1]:
                        e = [(j, i) for j in gp.predecessors(i)] + [
                            (i, j) for j in gp.successors(i)
                        ]
                        gp.remove_edges_from(e)
                        gp.remove_node(i)
                        res = True
                        break
    return res


def PCC(gp, s, t, R):
    pccs = {}  # plus court chemin en partant depuis s
    pcct = {}  # plus court chemin pour aller jusqu'à t

    if s not in gp.nodes():
        print("pb 1")

    if t not in gp.nodes():
        print("pb 2")

    for u in gp.nodes():
        pccs[u] = {}
        pcct[u] = {}

        ## Init pour chaque ressource ##
        for r in range(R + 1):
            pccs[u][r] = {}
            pcct[u][r] = {}

    # on retourne le graphe.
    gp_bis = nx.DiGraph()
    gp_bis.add_nodes_from(gp.nodes())

    for i, j in gp.edges():
        gp_bis.add_weighted_edges_from([(j, i, gp[i][j]["weight"])])

    for r in range(R + 1):
        # on va chercher les plus courtes distances

        # sur les chemins de s à i pour tout i
        res = Bel_Ford(gp, s, r, R)["L"]

        # sur les chemins de i à t pour tout i
        res2 = Bel_Ford(gp_bis, t, r, R)["L"]

        for u in res:
            pccs[u][r] = res[u]

        for u in res2:
            pcct[u][r] = res2[u]

    return {"PCCS": pccs, "PCCT": pcct}


def ELIM_ARCS(gp, b, R, s, t, U, pccs, pcct, chg):
    """
    Cette fonction nous permet d'éliminer des arcs, et la philosophie est exactement
    la même que pour supprimer des sommets sauf que maintenant on considère un arc $(i,j)$, et on regarde $s->i$, $i->j$ et $j->t$
    et on compare ce chemin au majorant et aux limites de ressources.
    """
    res = chg
    edges = [(i, j) for i, j in gp.edges()]
    for i, j in edges:
        for r in range(R + 1):
            if r == 0:
                if pccs[i][0][0] + gp[i][j]["weight"][0] + pcct[j][0][0] > U:
                    gp.remove_edge(i, j)
                    res = True
                    break
            else:
                if pccs[i][r][r] + gp[i][j]["weight"][r] + pcct[j][r][r] > b[r - 1]:
                    gp.remove_edge(i, j)
                    res = True
                    break
    return res


def nettoyage(g, s, t, b, R):
    """
    Algorithme de prétraitement
    """

    tic = time.perf_counter()

    ### Données d'initialisation ###

    gp = nx.DiGraph()  # c'est le graphe réduit que on va retourner à la fin
    Cmax = 0
    V = len(g.edges())

    U = 0  # majorant du problème

    # copie de g dans le graphe gp.

    gp.add_nodes_from(g.nodes())

    for u, v in g.edges():
        gp.add_weighted_edges_from([(u, v, g[u][v]["weight"])])
        if Cmax < g[u][v]["weight"][0]:
            # on cherche le cout le plus important.
            Cmax = g[u][v]["weight"][0]

    U = Cmax * (V - 1)

    chg = True

    while chg is True:
        chg = False

        ### recherche des plus courts chemins ###
        P = PCC(gp, s, t, R)
        pccs = P["PCCS"]
        pcct = P["PCCT"]

        ### Recherche d'une solution Evidente ###
        ### + mise à jour du minorant ###
        sortie = False
        for r in range(1, R + 1):
            if pccs[t][r][r] > b[r - 1]:
                sortie = True
                break

        if sortie is True:
            break

        che = pccs[t][0]
        l = [che[k] - b[k - 1] for k in range(1, R + 1)]

        bl = all(i <= 0 for i in l)

        if bl is True:
            toc = time.perf_counter()
            # on construit un graphe avec seulement le chemin dedans.
            d = Bel_Ford(gp, s, 0, R)["pred"]
            trace = [t]
            curr = t

            while curr != s:
                trace.append(d[curr])
                curr = d[curr]

            gp = nx.DiGraph()
            gp.add_nodes_from(trace)
            trace = trace[::-1]

            for i in range(len(trace) - 1):
                u0 = trace[i]
                u1 = trace[i + 1]
                gp.add_weighted_edges_from([(u0, u1, g[u0][u1]["weight"])])

            break

        else:
            che[0]  # minorant du problème

        ### mise à jour du majorant ###
        U = MAJ_ORANT(gp, b, R, s, t, pccs, pcct, U)

        ### Elimination des sommets ###
        chg = ELIM_SOMMETS(gp, b, R, s, t, U, pccs, pcct, chg)

        ### Elimination des arcs ###
        chg = ELIM_ARCS(gp, b, R, s, t, U, pccs, pcct, chg)

    toc = time.perf_counter()

    return {"graphe": gp, "U": U, "Temps": toc - tic}
