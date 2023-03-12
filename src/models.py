from pulp import *


def prob_PCC(N, C, A, s, t, R, b):
    """
    Cette première fonction permet de créer le modèle Pulp correspondant au problème décrit précédemment.
    Cette fonction retrounera simplement le problème modéliser.
    """
    city = list(range(N))

    prob = LpProblem("PC_sans_contraintes", LpMinimize)

    # déclaration des variables.
    X = LpVariable.matrix("X", (city, city), 0, 1, LpInteger)  # matrice binaire

    # déclaration de l'objectif.
    prob += lpSum([lpSum([X[i][j] * C[i][j] for i in city]) for j in city])

    # déclaration des contraintes
    # avec la matrice d'adjacence on vérifie que le chemin existe.
    for i in range(N):
        for j in range(N):
            prob += X[i][j] <= A[i][j]  # contrainte liée à la matrice d'adjacence.

    prob += lpSum([X[i][t] for i in city]) == 1  # on arrive au puit que une seule fois
    prob += lpSum([X[s][i] for i in city]) == 1  # on part de la source

    prob += lpSum([X[t][j] for j in city]) == 0  # on part ne peut pas partir de t
    prob += lpSum([X[i][s] for i in city]) == 0  # on ne peut pas revenir à s.

    for j in range(0, N):
        if j != t and j != s:
            prob += lpSum([X[i][j] for i in city]) == lpSum([X[j][i] for i in city])

    # contraintes de ressources.

    for r in R:
        prob += (
            lpSum([lpSum([X[i][j] * R[r][i][j] for i in city]) for j in city]) <= b[r]
        )

    # on retourne le modèle Pulp réalisé

    return prob


def path_PCC(N, C, A, s, t, R, b):
    """
    La fonction suivante, va permettre de résoudre le problème avec le modèle Pulp précédent
    et de stocker le chemin emprunté tout en créant un affichage qui nous permet de voir la solution de notre problème.
    Cette fonction nous sera utile surtout dans la dernière partie.
    """

    L_chemin = []

    prob = prob_PCC(N, C, A, s, t, R, b)
    prob.solve(PULP_CBC_CMD())
    status = LpStatus[prob.status]
    print("Status:", status)

    if status == "Optimal":
        # stockage des variables du problème, dans un dictionnaires.
        vardict = {}
        for v in prob.variables():
            vardict[v.name] = v

        # on va représenter le chemin que on va prendre.
        bl = True
        k = s
        cons = {}

        while bl is True:
            for i in range(N):
                if vardict["X_" + str(k) + "_" + str(i)].value() == 1:
                    for r in R:
                        if r in cons:
                            cons[r] += R[r][k][i]
                        else:
                            cons[r] = R[r][k][i]
                    L_chemin.append(k)
                    k = i
                    if k == t:
                        bl = False
                        L_chemin.append(i)
        E = []
        E.append(value(prob.objective))

        for r in cons:
            E.append(cons[r])
        return {"statu": status, "trace": L_chemin, "consommations": E}
    else:
        return {"statu": status}
