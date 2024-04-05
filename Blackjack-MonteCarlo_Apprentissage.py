# -*- coding: utf-8 -*-
#############################################################################
#                           Méthode de Monte-Carlo
#
#        APPRENTISSAGE d'une stratégie (policy) au jeu du Black-Jack
#############################################################################

# Importation des modules
import random                     # génération de nombres aléatoires
import numpy as np                # manipulation de tableaux
import gymnasium as gym           # simulation du jeu du BlackJack

# Déclaration des constantes 
NB_PARTIES = 4000000   # nombre de parties à simuler
GAMMA = 0.95           # facteur d’actualisation pour l'estimation du gain futur

#-----------------------------------------------------------------------------
# APPRENTISSAGE d'une stratégie ==> estimation de Q(s,a) et P(s)
#-----------------------------------------------------------------------------
# Création de l'environnement pour simuler le jeu du BlackJack
env = gym.make('Blackjack-v1', render_mode='rgb_array')

# Initialisation des variables utilisées pour la méthode de Monte-Carlo
# (s = état du jeu = (score_joueur, score_croupier,avec_as_valeur11))
P = {}  #    P[s] = action définie par la stratégie pour l'état s
Q = {}  # Q[s][a] = score calculé pour l'action a dans l'état s
N = {}  # N[s][a] = nombre de sélections de l'action a dans l'état s
# Parcours des états possibles pour le jeu
for score_joueur in range(4,22):
    # Si score>=12, gestion des cas avec un as avec une valeur de 11
    for avec_as_valeur11 in range(0, 1+(score_joueur>=12)): 
        for score_croupier in range(1,11):
            state = (score_joueur, score_croupier, avec_as_valeur11)
            N[state] = [0,0]
            if score_joueur == 21 and not avec_as_valeur11:
                # si le score du joueur est 21, action=0 (se coucher)
                P[state] = 0
                Q[state] = [0,-1]
            else: 
                # sinon action initialisée au hasard
                P[state] = random.randint(0,1)
                Q[state] = [0,0]

# Simulation des parties successives
for no_partie in range(1,1+NB_PARTIES):
    
    # SIMULATION d'une partie
    partie = [] # mémorisation de la partie
    # choix aléatoire de l'état au début du jeu et de la 1ère action du joueur
    state, info = env.reset()
    action = random.randint(0,1) # action aléatoire ==> EXPLORATION
    # déroulement de la partie 
    terminated = False
    while not terminated:
        # exécution de l'action choisie
        next_state, reward, terminated, truncated, info = env.step(action)
        partie.append((state, action, reward))
        if not terminated: 
            state = next_state
            action = P[state] # action selon la stratégie ==> EXPLOITATION

    # ANALYSE de la partie
    G = 0.0 # gain cumulé des récompenses
    # partie rejouée "à l'envers"
    for state, action, reward in reversed(partie):
        # gain cumulé = récompense immédiate + gain futur déprécié
        G = reward + GAMMA * G
        # MàJ des estimations de Q[s][a]
        N[state][action] += 1
        Q[state][action] = Q[state][action] + (G-Q[state][action]) / N[state][action]
        # MàJ de la stratégie P[s]
        P_update = np.argmax(Q[state][:])
        if P_update != P[state]: P[state] = P_update

# Suppression de l'environnement du jeu BlackJack
env.close()

#-----------------------------------------------------------------------------
# Affichage des résultats avec les espérances de gains Q
#    state = (score_joueur, score_croupier, avec_as_valeur11)
#    Q[state,0] = gain espéré avec Stick / Q[state,1] = gain espéré avec Hit
#-----------------------------------------------------------------------------
print('=== Résultats sans as avec une valeur de 11 ===')
for state,action_value in sorted(Q.items()):
    if state[2] == 0:  # états sans as avec valeur de 11 (valeur de 1 possible)
        print(f'Joueur:{state[0]:2d} (as={state[2]})  Croupier:{state[1]:2d}',end="")
        print(f'  Q[stick={action_value[0]:+.2f},hit={action_value[1]:+.2f}]',end="")
        print("--> stick") if action_value[0]>action_value[1] else print("--> hit")
print('=== Résultats avec un as avec une valeur de 11 ===')
for state,action_value in sorted(Q.items()):
    if state[2] == 1:  # états avec as avec valeur de 11
        print(f'Joueur:{state[0]:2d} (as={state[2]})  Croupier:{state[1]:2d}',end="")
        print(f'  Q[stick={action_value[0]:+.2f},hit={action_value[1]:+.2f}]',end="")
        print("--> stick") if action_value[0]>action_value[1] else print("--> hit")