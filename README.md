# Telco Customer Churn -- Prediction de desabonnement client

[![CI](https://github.com/artichautttt/telco-churn-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/artichautttt/telco-churn-ml/actions/workflows/ci.yml)

Projet de machine learning classique (from scratch, scikit-learn) : predire
si un client d'un operateur telecom va resilier son abonnement (churn), a
partir de ses donnees contractuelles et d'usage.

## Probleme

Classification binaire supervisee : `Churn` (Yes/No) a partir de 19
variables (donnees demographiques, services souscrits, type de contrat,
facturation).

**Dataset** : [IBM Telco Customer Churn](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)
-- 7 043 clients, 21 colonnes. C'est un **jeu d'exemple fictif** publie par
IBM pour la demonstration : les resultats illustrent une methode, pas un
operateur reel. Taux de churn observe : **26,5 %** (1 869 clients, classes
desequilibrees).

## Demarche

1. **EDA** (`notebooks/01_eda.ipynb`) -- structure des donnees, valeurs
   manquantes, distributions, correlations. Voir `reports/figures/`.
2. **Nettoyage** (`src/preprocessing.py`) -- `TotalCharges` (11 valeurs
   vides -> 0, correspondant a des clients avec `tenure == 0`), retrait de
   `customerID`, encodage de la cible, standardisation des variables
   numeriques + one-hot encoding des categorielles.
3. **Modelisation** (`src/train.py`) -- 3 modeles compares en validation
   croisee stratifiee 5-fold : regression logistique, random forest,
   XGBoost.
4. **Evaluation** (`src/evaluate.py`) -- metriques sur un holdout test set
   (20 %) jamais vu pendant la CV.
5. **Interpretabilite** (`src/interpret.py`) -- coefficients du modele
   retenu + valeurs SHAP.
6. **EDA SQL** (`sql/`, `src/bq_eda.py`) -- taux de churn par contrat,
   anciennete, service internet, paiement, contrat x anciennete, sur BigQuery.
7. **Typologies** (`src/segmentation.py`) -- k-prototypes vs K-means.
8. **Interactions** (`src/interactions.py`) -- valeurs d'interaction SHAP.
9. **Robustesse** (`src/robustness.py`) -- bootstrap, stabilite SHAP,
   calibration, performance par segment.
10. **Lecture metier** (`src/business.py`, `reports/synthese_metier.md`) --
    seuil par matrice de couts, gains et lift, synthese pour le marketing.

## Resultats (chiffres reels, reproductibles avec `random_state=42`)

### Comparaison des modeles a reequilibrage egal

Avant cette branche, le reequilibrage n'etait pas le meme pour les trois
modeles : `class_weight="balanced"` pour la regression logistique et la
random forest, **aucun** pour XGBoost (rappel churn CV de 0,516, artefact de
ce desequilibre de traitement). Les trois modeles sont maintenant a egalite :
`class_weight="balanced"` (regression logistique, random forest) et
`scale_pos_weight = n_negatifs / n_positifs = 2,7686`, calcule sur le train
set (XGBoost).

Metriques sur la **classe churn** (seuil 0,5) -- CV 5-fold stratifiee sur le
train set (moyenne +/- ecart-type) et test set (holdout 20 %, 374 churners
sur 1 409 clients) :

| Modele | AUC CV | Rappel CV | Precision CV | F1 CV | AUC test | Rappel test | Precision test | F1 test |
|---|---|---|---|---|---|---|---|---|
| **Regression logistique** | **0.8459** (+/-0.012) | **0.801** (+/-0.038) | 0.517 (+/-0.019) | **0.628** (+/-0.023) | **0.8417** | **0.783** | 0.505 | 0.614 |
| Random Forest | 0.8250 (+/-0.011) | 0.656 (+/-0.039) | 0.571 (+/-0.018) | 0.610 (+/-0.027) | 0.8195 | 0.634 | 0.549 | 0.588 |
| XGBoost (`scale_pos_weight`) | 0.8358 (+/-0.010) | 0.742 (+/-0.035) | 0.543 (+/-0.023) | 0.627 (+/-0.026) | 0.8312 | 0.749 | 0.525 | **0.617** |

Produit par `python -m src.train` (sauvegarde dans `models/cv_results.json`
et MLflow).

### Modele retenu : regression logistique

**Critere metier.** Pour la retention, rater un client qui va partir (faux
negatif) coute plus cher qu'une action de retention envoyee a tort (faux
positif) : le critere principal est le **rappel sur la classe churn**,
l'AUC servant a verifier que le classement global des clients reste bon.

**Regle de decision** (codee dans `src/train.select_model`, evaluee sur les
moyennes de CV, le test set restant reserve a l'evaluation) : on garde la
regression logistique -- plus simple, interpretable par ses coefficients,
deja servie par l'API -- si son ecart d'AUC avec le meilleur modele est
inferieur a 0,01 **et** si son rappel churn est au moins equivalent
(tolerance de 0,01) ; sinon on retient le meilleur modele.

**Resultat.** La regression logistique a a la fois la meilleure AUC CV
(0,8459 contre 0,8358 pour XGBoost) et le meilleur rappel CV (0,801 contre
0,742) : elle est conservee, et le test set confirme le meme ordre (AUC
0,842 vs 0,831 ; rappel 0,783 vs 0,749). XGBoost reequilibre a un F1 test
equivalent (0,617 vs 0,614) mais capte moins de churners. L'entrainement
(`models/best_model.pkl`), MLflow (tag `best_model` et
`selection_justification`), l'API et ce README designent donc le meme modele.
L'API sert sa version **recalibree** (meme classement, voir *Calibration*)
avec un seuil issu de la matrice de couts (voir *API de prediction*).

XGBoost reste entraine et sauvegarde (`models/xgboost_model.pkl`) : c'est le
modele a arbres utilise pour SHAP `TreeExplainer` et l'analyse des
interactions.

### Evaluation sur le test set (holdout, 20 %, jamais vu en CV)

| Metrique | Valeur |
|---|---|
| Accuracy | 0.739 |
| F1 | 0.614 |
| ROC-AUC | 0.842 |

```
              precision    recall  f1-score   support

    No Churn       0.90      0.72      0.80      1035
       Churn       0.51      0.78      0.61       374
```

Le modele privilegie le **rappel sur la classe churn (0.78)** au detriment
de sa precision (0.51) -- coherent avec `class_weight="balanced"` et
pertinent metier : dans un cas d'usage de retention client, rater un
client qui va partir (faux negatif) coute generalement plus cher qu'une
fausse alerte (faux positif) traitee par une action de retention inutile.

### Interpretabilite

Features augmentant le risque de churn (coefficients positifs) :
`InternetService_Fiber optic`, `Contract_Month-to-month`, `TotalCharges`,
`PaperlessBilling_Yes`.

Features reduisant le risque de churn (coefficients negatifs) :
`tenure`, `Contract_Two year`, `InternetService_DSL`, `MonthlyCharges`.

Coherent avec l'EDA : l'anciennete du client et l'engagement contractuel
long sont les facteurs protecteurs les plus forts ; un contrat sans
engagement et la fibre optique sont les facteurs de risque les plus forts.
Voir `reports/figures/feature_coefficients.png` et `shap_summary.png`.

## Analyse avancee (chiffres de l'execution `python scripts/run_all.py`, seeds fixes)

Toutes les valeurs ci-dessous sont lues dans les sorties du code
(`reports/*/summary.json`, `reports/*/*.csv`) et loguees dans MLflow (run
`advanced_analysis`).

### EDA SQL sur BigQuery (`sql/`, `src/bq_eda.py`)

Cinq requetes : taux de churn par type de contrat, tranche d'anciennete
(0-12, 13-24, 25-48, 49-72 mois), service internet, moyen de paiement, et
croisement contrat x anciennete. Le script charge le CSV avec un **schema
explicite** (`TotalCharges` NULL pour les 11 clients a anciennete nulle),
execute les requetes, exporte les resultats dans `reports/sql/*.csv` et
**recoupe chaque resultat avec le meme calcul en pandas**. Authentification
par ADC (`gcloud auth application-default login`), compatible Sandbox.

```bash
python -m src.bq_eda --project MON_PROJET                        # cree le dataset telco_churn
python -m src.bq_eda --project MON_PROJET --dataset MON_DATASET  # dataset existant, sans creation
```

**Execute le 28/10/2026** sur BigQuery (projet `mini-siem-taxi-pipeline`,
dataset existant `telco_churn`, region US) :

```bash
python -m src.bq_eda --project mini-siem-taxi-pipeline --dataset telco_churn --location US
```

Resultats exportes dans `reports/sql/*.csv`, identiques au recoupement
pandas pour les 5 requetes. Principaux taux de churn :

| Dimension | Taux de churn |
|---|---|
| Contrat | sans engagement 42,7 % ; un an 11,3 % ; deux ans 2,8 % |
| Anciennete | 0-12 mois 47,4 % ; 13-24 mois 28,7 % ; 25-48 mois 20,4 % ; 49-72 mois 9,5 % |
| Service internet | fibre 41,9 % ; DSL 19,0 % ; sans internet 7,4 % |
| Paiement | cheque electronique 45,3 % ; cheque papier 19,1 % ; virement auto. 16,7 % ; carte auto. 15,2 % |
| Contrat x anciennete | sans engagement 0-12 mois : 51,4 % ; deux ans 0-24 mois : 0 % (0 sur 158) |

### Typologies de clients (`src/segmentation.py`)

- **k-prototypes** (numeriques standardisees + categorielles brutes) compare a
  un **K-means** sur variables one-hot standardisees. La cible `Churn`
  n'entre pas dans la segmentation.
- **Choix de k** sur k = 2..8, sur un echantillon stratifie de 3 000 clients
  (memoire) : courbe de cout et silhouette (distance de Gower pour
  k-prototypes, sur 2 000 clients ; euclidienne pour K-means). Regle :
  silhouette maximale parmi k = 3..5. Retenu : **k = 3** pour k-prototypes
  (silhouette 0,316), **k = 4** pour K-means (0,221). Modeles finaux ajustes
  sur les 7 043 clients (k-prototypes `n_init=5`). Accord entre les deux
  segmentations : indice de Rand ajuste **0,650**.

| Segment (k-prototypes) | Clients | Taux de churn | Part des churners | Anciennete moy. | Facture moy. | Sans engagement | Fibre |
|---|---|---|---|---|---|---|---|
| 0 -- recents, sans engagement | 3 270 (46,4 %) | **43,8 %** | **76,7 %** | 15,5 mois | 69,1 | 86,3 % | 53,3 % |
| 1 -- fideles, gros forfaits | 2 175 (30,9 %) | 14,7 % | 17,1 % | 58,6 mois | 89,7 | 24,1 % | 62,3 % |
| 2 -- sans internet | 1 598 (22,7 %) | 7,3 % | 6,2 % | 31,2 mois | 21,8 | 33,2 % | 0 % |

K-means retrouve ces trois groupes et isole en plus les clients DSL sans
telephone (682 clients, 24,9 % de churn) ; detail dans
`reports/segmentation/profiles_kmeans.csv`.

**Facteurs SHAP par segment** (XGBoost) : le classement change peu. La
correlation de Spearman avec le classement global vaut 0,972 (segment 0),
0,984 (segment 1) et 0,968 (segment 2) ; `Contract` est premier partout.
Seuls changements notables : `InternetService` entre dans le top 3 du
segment 0 et `MonthlyCharges` passe 2e dans le segment 2.

### Interactions entre variables (`src/interactions.py`)

Valeurs d'interaction SHAP (`TreeExplainer`, XGBoost) calculees sur un
**echantillon stratifie de 1 000 clients du test set** (seed 42, pour
limiter la memoire), agregees par variable d'origine. Force d'une paire :
moyenne de |phi_ij + phi_ji| (log-odds).

| Rang | Paire | Force | Remarque |
|---|---|---|---|
| 1 | tenure x TotalCharges | 0,394 | **mecanique** : TotalCharges ~ tenure x MonthlyCharges |
| 2 | tenure x Contract | 0,294 | |
| 3 | MonthlyCharges x TotalCharges | 0,181 | **mecanique** |
| 4 | tenure x MonthlyCharges | 0,177 | |
| 5 | InternetService x Contract | 0,134 | |

Les paires avec `TotalCharges` refletent la construction de cette variable,
pas un comportement client. **Top sans TotalCharges** (interactions
informatives) : tenure x Contract (0,294), tenure x MonthlyCharges (0,177),
InternetService x Contract (0,134), tenure x PaymentMethod (0,122),
MonthlyCharges x Contract (0,121). MonthlyCharges x InternetService, demandee
explicitement, est faible (hors top 10). Figures :
`reports/figures/interactions/`.

Lecture :
- **tenure x Contract** : chez les clients sans engagement, l'interaction
  ajoute du risque pendant les ~12 premiers mois puis devient negative ;
  chez les clients en contrat deux ans, elle est negative pour les nouveaux
  clients et positive aux fortes anciennetes (les deux effets protecteurs ne
  s'additionnent pas).
- **InternetService x Contract** : le surcroit de risque de la fibre est
  plus fort chez les clients sans engagement (interaction positive) que chez
  les clients engages (negative).

**Ce que la regression logistique ne capte pas.** Sans terme d'interaction,
elle impose un effet de l'anciennete identique quel que soit le contrat
(-0,047 log-odds par mois) et des effets strictement additifs. Dans XGBoost,
la pente moyenne de l'effet de `tenure` varie selon le contrat
(-0,035 sans engagement, -0,023 un an, -0,033 deux ans), et la forme n'est
pas lineaire. Pourtant, ajouter explicitement les termes tenure x Contract
et MonthlyCharges x InternetService a la logistique ne change presque pas
son AUC CV : 0,8459 (+/-0,012) -> 0,8469 (+/-0,013). Ces interactions
existent mais pesent peu dans le classement global des clients.

### Robustesse (`src/robustness.py`)

**IC a 95 % par bootstrap** (1 000 tirages, test set, seuil 0,5) :

| Modele | AUC | Rappel churn |
|---|---|---|
| Regression logistique (servie) | 0,842 [0,819 ; 0,862] | 0,783 [0,740 ; 0,822] |
| XGBoost | 0,831 [0,806 ; 0,853] | 0,749 [0,705 ; 0,791] |

Les intervalles se recouvrent : l'avantage de la logistique sur ce test set
n'est pas significatif a lui seul ; il s'ajoute a sa simplicite.

**Stabilite du top 10 SHAP** (XGBoost ; Spearman deux a deux sur l'union des
top 10) :
- 5 folds de la CV : Spearman moyen **0,939** (min 0,888), recouvrement moyen
  des top 10 de 92 % (min 80 %) ;
- 5 seeds (nouveau decoupage + nouvelle seed) : Spearman moyen **0,910** (min
  0,855), recouvrement moyen 92 % (min 90 %).

Le trio de tete (Contract, tenure, MonthlyCharges) est le meme partout.

**Calibration** de la regression logistique : `class_weight="balanced"` la
rend **mal calibree** -- elle surestime le risque (probabilite moyenne 0,416
sur le test pour un taux reel de 0,265 ; Brier 0,169 ; ECE 0,150).
Recalibration par `CalibratedClassifierCV(cv=5, ensemble=False)` sur le
train : un seul modele ajuste sur tout le train, suivi d'une fonction
monotone apprise sur ses predictions hors-fold. Methode choisie sur le Brier
hors-fold du train : **Platt (sigmoid)**, Brier test 0,138 et ECE 0,023.
Platt est strictement croissante : le **classement est preserve a
l'identique** (AUC test 0,841703 avant et apres, correlation de Spearman 1,0
entre les deux scores ; verifie par `tests/test_api.py`). L'isotonique
(Brier 0,139, ECE 0,019) cree des ex aequo et baisse l'AUC a 0,840.
Figure : `reports/figures/calibration_curve.png`. C'est ce modele recalibre
(`models/calibrated_model.pkl`) que sert l'API.

**Performance par type de contrat et par segment** -- regression logistique
non recalibree au seuil 0,5 (reference), test set :

| Groupe | Clients | Churners | AUC | Rappel | Precision |
|---|---|---|---|---|---|
| Sans engagement | 773 | 329 | 0,742 | 0,866 | 0,516 |
| Un an | 300 | 36 | 0,743 | 0,222 | 0,286 |
| Deux ans | 336 | 9 | 0,760 | 0,000 | 0,000 |
| Segment 0 | 681 | 298 | 0,759 | 0,903 | 0,533 |
| Segment 1 | 399 | 51 | 0,768 | 0,471 | 0,329 |
| Segment 2 | 329 | 25 | 0,897 | 0,000 | 0,000 |

Au seuil 0,5, le modele ne repere presque aucun churner engage ou sans
internet : leurs probabilites restent sous le seuil. Les AUC intra-groupe
(0,74 a 0,90) montrent qu'il les classe pourtant mieux que le hasard. La
decision servie par l'API (probabilites recalibrees, seuil 0,18) en repere
davantage, sans seuil par segment (voir *Lecture metier* et *Limites*).

### Lecture metier (`src/business.py`)

**Seuil par matrice de couts.** Hypotheses par defaut, a remplacer par les
chiffres reels (options `--action-cost`, `--churn-loss`, `--success-rate`) :
action de retention 50 EUR, client perdu 500 EUR, succes d'une action sur un
vrai churner 50 %. Seuil choisi sur les probabilites recalibrees hors-fold
du train : **0,18** (seuil theorique pour des probabilites calibrees :
50 / (0,5 x 500) = 0,20). Sur le test set (1 409 clients) :

| Strategie | Clients contactes | Rappel | Precision | Cout total |
|---|---|---|---|---|
| Personne | 0 % | 0 | -- | 187 000 EUR |
| Tout le monde | 100 % | 1 | 0,265 | 163 950 EUR |
| Seuil 0,5 (probabilites recalibrees) | 22,5 % | 0,564 | 0,666 | 150 100 EUR |
| **Seuil 0,18 (servi par l'API)** | 52,2 % | **0,893** | **0,454** | **140 300 EUR** |

Au seuil servi, 736 clients du test sont signales, dont 334 vrais churners
sur 374. Le seuil et le modele sont ecrits dans `models/serving_config.json`,
lu par l'API.

**Decision servie par groupe** (probabilites recalibrees, seuil 0,18, test
set ; `reports/business/performance_by_group_at_threshold.csv`) :

| Groupe | Clients | Churners | Rappel | Precision |
|---|---|---|---|---|
| Sans engagement | 773 | 329 | 0,957 | 0,477 |
| Un an | 300 | 36 | 0,528 | 0,257 |
| Deux ans | 336 | 9 | 0,000 | 0,000 |
| Segment 0 | 681 | 298 | 0,960 | 0,503 |
| Segment 1 | 399 | 51 | 0,627 | 0,276 |
| Segment 2 | 329 | 25 | 0,640 | 0,314 |

**Gains et lift** (scores du modele servi, test set) : les 10 % de clients
au score le plus eleve contiennent **28,6 %** des churners (lift 2,86) ; les
20 % les plus a risque en contiennent **49,7 %** (lift 2,49). Figures :
`reports/figures/gains_lift.png`, `cost_vs_threshold.png`.

Synthese pour une equipe marketing : [`reports/synthese_metier.md`](reports/synthese_metier.md).

### Limites

- Jeu **fictif** (IBM), instantane sans dates : pas de validation temporelle
  ni d'historique d'usage ou de reclamations.
- Les facteurs SHAP et les interactions decrivent le modele, pas des causes :
  une action de retention doit etre validee par un test terrain.
- Les couts de la matrice de decision sont des hypotheses.
- Petits effectifs de churners dans certains groupes du test set (contrat
  deux ans : 9 ; segment 2 : 25) : les metriques par groupe sont fragiles.
- **Un seul seuil pour tous les clients.** Au seuil servi, le rappel reste
  nul chez les clients en contrat deux ans (0 sur 9) et la precision faible
  dans les segments 1 et 2. Un seuil par segment n'a **pas** ete mis en
  place : avec 9 et 25 churners dans le test, il serait surajuste au bruit.
  Piste d'amelioration : seuils (ou couts) par segment estimes sur davantage
  de donnees, par exemple sur les predictions hors-fold du train ou sur
  plusieurs periodes, et valides sur un jeu independant.
- Interactions SHAP estimees sur 1 000 clients du test, et choix de k sur
  3 000 clients, pour tenir en memoire.

## Test A/B de retention (effet SIMULE)

[`notebooks/02_ab_test_retention.ipynb`](notebooks/02_ab_test_retention.ipynb)
(fonctions : `src/ab_testing.py`, tests : `tests/test_ab_testing.py`) deroule un
test A/B d'une campagne de retention sur les clients du test set au-dessus du
seuil servi (0,18). **Aucune experience n'a eu lieu** : le jeu est public et
l'effet de la campagne est **simule** (le controle garde les vrais labels ;
dans le groupe traite, chaque churner est retenu avec une probabilite
parametree). Etapes : hypotheses H0/H1, dimensionnement (statsmodels),
randomisation avec controle du ratio et de l'equilibre, test A/A, test de deux
proportions, puissance simulee, garde-fou de cout sous les hypotheses du projet.
Chiffres de l'execution du 31/10/2026 (graines fixes, 2 000 tirages) : 736
clients cibles dont 334 churners (45,4 %) ; effet minimal detectable : baisse
relative de 22,3 % ; faux positifs du test A/A : 5,0 %. Un effet simule de
10 % n'est detecte que dans 21 % des tirages : un resultat non significatif
serait un manque de puissance, pas une absence d'effet. Les montants du
notebook sont des calculs sous hypotheses, pas des gains.

## Suivi de derive (derives INJECTEES)

`src/drift.py` mesure la derive des variables (PSI, KS ou chi-deux, correction
de Benjamini-Hochberg) et des predictions (PSI, part au-dessus du seuil 0,18),
en trois niveaux (aucun, surveillance, alerte ; convention PSI 0,1 / 0,25).
`python -m src.drift_report --current test|train|<fichier.csv>` ecrit un
rapport JSON et Markdown (exemple : `reports/drift/`) et le journalise dans MLflow local.
**Le jeu IBM Telco n'a aucune derive reelle ; la detection est validee sur des
derives injectees, pas sur un cas reel** ([notebook 03](notebooks/03_drift_monitoring.ipynb)).
Sur 1 000 redecoupages sans derive : 8,4 % de fausses alertes en surveillance,
0 % en alerte (seuils figes avant les scenarios). Anciennete -3 mois et prix
+5 % : alerte dans plus de 99 % des tirages. Derive conceptuelle simulee
(30 %) : invisible sur X, rappel au seuil 0,18 de 0,894 a 0,675.
Alternative outillee non utilisee ici : Evidently.
Le notebook 03 prend environ 17 min en local et n'est pas lance en CI (la CI couvre le module et ses 29 tests).

## Structure du projet

```
telco-churn-ml/
├── data/raw/telco_churn.csv       # dataset source (IBM, public)
├── notebooks/01_eda.ipynb         # exploration (notebook execute)
├── notebooks/02_ab_test_retention.ipynb  # test A/B de retention, effet simule
├── notebooks/03_drift_monitoring.ipynb   # suivi de derive, derives injectees
├── src/
│   ├── preprocessing.py           # chargement + nettoyage + pipeline sklearn
│   ├── train.py                   # entrainement + comparaison CV des 3 modeles
│   ├── evaluate.py                # evaluation sur holdout test set
│   ├── interpret.py               # coefficients + SHAP
│   ├── tracking.py                # tracking MLflow des entrainements
│   ├── schemas.py                 # validation Pydantic des inputs de l'API
│   ├── api.py                     # API FastAPI (POST /predict)
│   ├── bq_eda.py                  # chargement BigQuery + execution des requetes sql/
│   ├── segmentation.py            # k-prototypes vs K-means, profils, SHAP par segment
│   ├── interactions.py            # interactions SHAP, comparaison avec la logistique
│   ├── robustness.py              # bootstrap, stabilite SHAP, calibration, perf. par groupe
│   ├── business.py                # seuil par couts, gains et lift
│   ├── ab_testing.py              # test A/B : taille d'echantillon, randomisation, tests
│   ├── drift.py                   # derive : PSI, KS, chi-deux, Benjamini-Hochberg, niveaux d'alerte
│   ├── drift_report.py            # rapport de derive JSON + Markdown, journalisation MLflow
│   ├── shap_utils.py              # outils SHAP partages
│   └── log_analysis.py            # logging MLflow de l'analyse avancee
├── sql/                           # EDA en SQL BigQuery (5 requetes)
├── scripts/run_all.py             # reproduction en une commande (--steps)
├── tests/                         # pytest, un fichier par module
├── reports/
│   ├── synthese_metier.md         # synthese d'une page pour le marketing
│   ├── segmentation/ interactions/ robustness/ business/ drift/   # resultats (csv, json, md)
│   └── figures/                   # figures generees
├── models/                        # resultats .json versionnes ; modeles .pkl non versionnes
├── mlruns/                        # runs MLflow locaux (non versionne)
└── requirements.txt
```

## Reproduire les resultats

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

python scripts/run_all.py        # toute la chaine, une etape par processus (~5 min)
python scripts/run_all.py --list # etapes disponibles
python scripts/run_all.py --steps robustness business   # etapes choisies, dans l'ordre donne
pytest tests/ -q                 # tests (ceux sur les modeles necessitent run_all.py)
ruff check src tests scripts
```

**CI** (`.github/workflows/ci.yml`, actions `checkout@v7` et
`setup-python@v7`) : a chaque push sur `main` et a chaque PR, sur Python 3.12
(version minimale des dependances figees : numpy 2.5, xgboost 3.4 et shap
0.52 exigent Python >= 3.12) :
1. `ruff check` ;
2. `python -m src.build_serving` (~10 s) : produit seulement les artefacts
   de l'API -- regression logistique, recalibration, seuil de la matrice de
   couts -- avec les memes fonctions et seeds que la chaine complete, sans
   comparaison de modeles, segmentation ni SHAP ;
3. verification que le seuil recalcule est celui de
   `models/serving_config.json` (versionne) ;
4. `pytest` : tous les tests tournent, y compris ceux de l'API. Pas d'acces
   BigQuery en CI : les tests marques `bigquery` seraient ignores
   (`tests/conftest.py`) ; aucun test actuel n'en depend (ceux de
   `src.bq_eda` utilisent un faux client).

`run_all.py` enchaine train, evaluate, interpret, segmentation, interactions,
robustness, business et le logging MLflow ; il s'arrete a la premiere etape
en echec en indiquant laquelle. Tout tourne avec `n_jobs=1` pour limiter la
memoire. `make all` / `make test` / `make lint` font la meme chose sous
Linux/macOS. L'EDA BigQuery est a part (`python -m src.bq_eda`, voir plus
haut).

Les `.pkl` n'etant pas versionnes, les tests de `tests/test_api.py` qui
chargent les modeles sont ignores (skipped) tant que
`python scripts/run_all.py` ou `python -m src.build_serving` n'a pas ete
lance.

## Suivi des experiences (MLflow)

Chaque `python -m src.train` cree un run dans l'experience MLflow
`telco-churn`, en tracking local (`file:./mlruns`, aucun serveur dedie) :

- un run parent `model_comparison` : parametres globaux (split, CV, seed),
  tag `best_model`, metriques de CV du modele retenu, et le **pipeline final**
  (ColumnTransformer + modele) logue comme modele MLflow (artifact `model`) ;
- un run enfant par modele compare (`logistic_regression`, `random_forest`,
  `xgboost`) : hyperparametres (`get_params()`) et metriques de CV
  (moyenne et ecart-type sur 5 folds) -- accuracy, F1, ROC-AUC, et precision
  / rappel **par classe** (`precision_class0`, `recall_class1`, ...).

Consulter les runs :

```bash
# mlflow >= 3 exige cet opt-in pour le backend fichier local
set MLFLOW_ALLOW_FILE_STORE=true          # Windows (cmd) ; PowerShell : $env:MLFLOW_ALLOW_FILE_STORE="true"
mlflow ui --backend-store-uri file:./mlruns
# -> http://127.0.0.1:5000
```

Le tracking ne bloque jamais l'entrainement : en cas d'echec d'ecriture
MLflow, un avertissement est affiche et `models/best_model.pkl` reste valide.

## API de prediction (FastAPI)

L'API lit `models/serving_config.json`, ecrit par `python -m src.business` :

- **modele** : la regression logistique retenue par `train.py`, recalibree
  (Platt, `models/calibrated_model.pkl`). Elle enveloppe le pipeline complet
  (ColumnTransformer + modele) : le pretraitement est identique a
  l'entrainement, et la probabilite renvoyee est calibree ;
- **seuil de decision : 0,18**, optimise par la matrice de couts (etape 6)
  sur les probabilites calibrees hors-fold du train -- et non 0,5. Les couts
  sont des hypotheses (50 EUR par action, 500 EUR par client perdu, 50 % de
  succes) : relancer `python -m src.business --action-cost ... --churn-loss
  ... --success-rate ...` recalcule le seuil et la configuration.

Sur le test set, cette decision donne un **rappel churn de 0,893** et une
**precision de 0,454** (52,2 % des clients signales), pour une AUC
inchangee de 0,842.

```bash
python scripts/run_all.py                # si les modeles n'existent pas encore
uvicorn src.api:app --reload             # -> http://127.0.0.1:8000 (doc interactive : /docs)
```

Variables d'environnement : `SERVING_CONFIG` (autre fichier de
configuration), `MODEL_PATH` (remplace le modele de la configuration).

**Endpoint unique : `POST /predict`**, qui prend les 19 features d'un client
(memes noms de colonnes que le dataset, sans `customerID` ni `Churn`) :

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
       "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
       "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
       "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
       "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
       "PaymentMethod": "Electronic check", "MonthlyCharges": 95.7, "TotalCharges": 189.2}'
```

Reponse :

```json
{"churn_probability": 0.755437169675744, "churn_prediction": 1, "threshold": 0.18}
```

- `churn_probability` : probabilite de churn **calibree** ;
- `churn_prediction` : 1 si `churn_probability >= threshold` ;
- `threshold` : seuil de decision utilise (matrice de couts).

**Validation des inputs** (`src/schemas.py`) -- toute requete invalide est
rejetee en **422** avant d'atteindre le modele :

- types stricts, sans conversion implicite (`"95.7"` n'est pas accepte pour un
  nombre) et champs inconnus refuses ;
- categories limitees aux valeurs du dataset (`Literal`) : indispensable car
  le `OneHotEncoder(handle_unknown="ignore")` du pipeline encoderait
  silencieusement une categorie mal orthographiee (`"month-to-month"`) en
  zeros ;
- plages realistes : `tenure` 0-120 mois, `MonthlyCharges` 0-500,
  `TotalCharges` 0-50 000, `SeniorCitizen` 0/1 ;
- coherence entre services : `MultipleLines == "No phone service"` si et
  seulement si `PhoneService == "No"`, et les 6 options internet valent
  `"No internet service"` si et seulement si `InternetService == "No"`.
  Les 7 043 lignes du dataset respectent toutes ces regles.

## Contraintes du projet

- Machine learning classique uniquement (scikit-learn), pas de deep learning.
- Toutes les metriques ci-dessus sont issues d'executions reelles du code de
  ce depot, avec seed fixe (`random_state=42`) pour la reproductibilite.
