# Synthèse : qui risque de partir, pourquoi, et que faire

> **Données d'exemple.** Ce travail porte sur le jeu IBM Telco Customer Churn, un jeu **fictif** de 7 043 clients publié par IBM pour la démonstration. Les chiffres illustrent une méthode et ne décrivent aucun opérateur réel.

Sur les 7 043 clients, **1 869 sont partis, soit 26,5 %**. Le modèle de prévision attribue à chaque client un score de risque de départ.

## Trois profils de clients

| Profil | Part des clients | Taux de départ | Part de tous les départs | Ce qui les caractérise |
|---|---|---|---|---|
| **A. Récents, sans engagement** | 46 % (3 270) | **43,8 %** | **77 %** | 15 mois d'ancienneté en moyenne, 86 % sans engagement, 53 % en fibre, peu d'options (protection, streaming) |
| **B. Fidèles, gros forfaits** | 31 % (2 175) | 14,7 % | 17 % | 59 mois d'ancienneté, facture de 90 € par mois, 62 % en fibre, 76 % engagés 1 ou 2 ans |
| **C. Sans internet** | 23 % (1 598) | 7,3 % | 6 % | 95 % sans internet, facture de 22 € par mois, souvent payée par chèque |

**Trois départs sur quatre viennent du profil A.**

## Ce qui fait partir les clients

Les mêmes facteurs dominent dans les trois profils :

1. **Le type de contrat.** Un client sans engagement part beaucoup plus qu'un client engagé 1 ou 2 ans.
2. **L'ancienneté.** Le risque est le plus élevé dans la première année, puis il baisse régulièrement. Le surcroît de risque des premiers mois se concentre chez les clients sans engagement.
3. **Le montant de la facture et la fibre.** La fibre augmente le risque, nettement plus chez les clients sans engagement que chez les clients engagés.

Ces facteurs indiquent des **associations**, pas des causes prouvées.

## Recommandations par profil

- **Profil A (priorité).** Proposer, dans les 12 premiers mois, le passage à un engagement d'un an en échange d'un avantage. Pour les clients fibre sans engagement, proposer une option de protection ou d'assistance incluse.
- **Profil B.** Le risque est modéré mais ces clients pèsent lourd en chiffre d'affaires. Surveiller les fins d'engagement et les factures élevées, et proposer un geste de fidélité au renouvellement.
- **Profil C.** Le risque est faible. Pas de campagne de rétention dédiée. Le passage au prélèvement automatique peut se proposer à faible coût.

## Comment cibler une campagne

- Contacter les **10 % de clients au score le plus élevé** touche **28,6 % des futurs départs**, soit 2,9 fois mieux qu'un tirage au hasard.
- Contacter les **20 % les plus à risque** en touche **49,7 %**.

Nous avons simulé une campagne avec des hypothèses à remplacer par vos vrais chiffres : une action coûte 50 €, un client perdu coûte 500 €, une action réussit une fois sur deux. Le meilleur compromis consiste à contacter les clients dont le risque estimé dépasse **18 %**. Cela représente 52 % des clients et permet de toucher 89 % des futurs départs ; parmi les clients signalés, 45 % partiraient effectivement. C'est le réglage utilisé par l'outil de prévision.

Sur l'échantillon de test (1 409 clients), voici le coût total selon la stratégie :

| Stratégie | Coût total |
|---|---|
| Ne rien faire | 187 000 € |
| Contacter tout le monde | 163 950 € |
| Contacter les clients au risque estimé supérieur à 50 % | 150 100 € |
| **Contacter les clients au risque estimé supérieur à 18 %** | **140 300 €** |

Le seuil optimal change dès que les coûts changent. Il faut le recalculer avec les vrais chiffres.

## Fiabilité et limites

- **Qualité du classement.** Un client qui va partir reçoit un score plus élevé qu'un client qui va rester dans 84 % des cas (AUC 0,842, intervalle de confiance à 95 % : 0,819 à 0,862). Au réglage retenu (risque supérieur à 18 %), le modèle repère 89 % des départs. Le score de risque est calibré : un client à 30 % part environ trois fois sur dix.
- **Points faibles.** Le même seuil s'applique à tous les clients. Au réglage retenu, le modèle repère 63 % des départs du profil B et 64 % de ceux du profil C, mais avec beaucoup de fausses alertes (environ 3 alertes justes sur 10). Il ne repère aucun des 9 départs de clients engagés deux ans. Un seuil propre à chaque profil serait une piste, mais il faudrait plus de données : le test ne compte que 25 départs dans le profil C.
- **Données.** Il s'agit d'un jeu fictif, figé à une date, sans historique d'usage ni de réclamations. Un vrai projet demanderait des données réelles et datées, puis un test terrain avant de généraliser une action.
