# -*- coding: utf-8 -*-
#############################################################################
#                           Méthode de Monte-Carlo
#
#        EVALUATION d'une stratégie (policy) au jeu du Black-Jack
#############################################################################

# Importation des modules
import random                     # génération de nombres aléatoires
import gymnasium as gym           # simulation du jeu du BlackJack
import matplotlib.pyplot as plt   # tracé de courbes/graphiques

# Déclaration des constantes 
NB_PARTIES = 100000   # nombre de parties à simuler

#-----------------------------------------------------------------------------
# Exemples de stratégies
#-----------------------------------------------------------------------------
# Choix aléatoire entre se coucher ou demander une carte
def strategie_aleatoire (state):
    return random.randint(0,1)

# Demande d'une carte si le score du joueur est inférieure à 17 
def strategie_croupier (state):
    return 0 if state[0] >= 17 else 1

#-----------------------------------------------------------------------------
# EVALUATION d'une stratégie
#-----------------------------------------------------------------------------
# Création de l'environnement pour simuler le jeu du BlackJack
env = gym.make('Blackjack-v1', render_mode='rgb_array')

# Mémorisation du gain_moyen et de ses évolutions (courbe)
gains_moyens = []
gain_moyen = 0

# Simulation des parties successives
for no_partie in range(1,1+NB_PARTIES):
    # initialisation aléatoire du jeu au début de la partie
    state, info = env.reset()
    # déroulement de la partie
    terminated = False
    while not terminated:     
        # choix et exécution d'une action : mettre la stratégie à évaluer
        action = strategie_aleatoire(state)
        new_state, reward, terminated, truncated, info = env.step(action)
        if terminated:
            # partie terminée --> MàJ du gain avec le résultat de la partie
            gain_moyen = gain_moyen + (reward-gain_moyen)/no_partie
            # mémorisation d'un point sur 100 pour afficher la courbe
            if no_partie % 100 == 0: 
                gains_moyens.append(gain_moyen)
        else:
            # partie non terminée --> itération pour une nouvelle action
            state = new_state

# Résultat 
print("Résultat moyen:",gain_moyen)

# Suppression de l'environnement du jeu BlackJack
env.close()

#-----------------------------------------------------------------------------
# Graphique avec l'évolution de l'estimation du gain moyen
#-----------------------------------------------------------------------------
plt.figure(figsize=(8,8),dpi=200)
plt.xlabel('Parties simulées (à multiplier par 100)',fontsize=18, labelpad=10)
plt.ylabel("Gain moyen",fontsize=18, labelpad=15)
plt.plot(gains_moyens,c='red',lw=5)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid(visible=True)
plt.show()