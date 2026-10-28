# Telco Customer Churn -- Prediction de desabonnement client

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
equivalent (0,617 vs 0,614) mais capte moins de churners. L'entrainement,
l'API (`models/best_model.pkl`), MLflow (tag `best_model` et
`selection_justification`) et ce README designent donc le meme modele.

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

**Non execute dans ce depot** : l'identite utilisee n'avait pas le droit
`bigquery.datasets.create` sur le projet disponible. Les CSV de
`reports/sql/` ne sont donc pas encore produits.

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

**Calibration** de la regression logistique servie :
`class_weight="balanced"` la rend **mal calibree** -- elle surestime le
risque (probabilite moyenne 0,416 sur le test pour un taux reel de 0,265 ;
Brier 0,169 ; ECE 0,150). Recalibration par `CalibratedClassifierCV(cv=5)`
sur le train, methode choisie sur le Brier hors-fold du train : **Platt
(sigmoid)**, Brier test 0,138 et ECE 0,023 (isotonique : 0,139 et 0,020),
AUC inchangee (0,842). Figure : `reports/figures/calibration_curve.png`.
Le modele recalibre (`models/calibrated_model.pkl`) est utilise pour la
lecture metier ; **l'API sert toujours le modele non recalibre** : son
`churn_probability` est un score de classement, pas une probabilite
calibree.

**Performance par type de contrat et par segment** (modele servi, test set,
seuil 0,5) :

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
(0,74 a 0,90) montrent qu'il les classe pourtant mieux que le hasard ; un
seuil par segment serait a etudier. Effectifs faibles (9 et 25 churners) :
ces chiffres sont fragiles.

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
| Seuil 0,5 (probabilites recalibrees) | 22,4 % | 0,564 | 0,668 | 150 050 EUR |
| **Seuil 0,18** | 52,4 % | **0,893** | 0,453 | **140 400 EUR** |

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
  deux ans : 9 ; segment 2 : 25).
- Interactions SHAP estimees sur 1 000 clients du test, et choix de k sur
  3 000 clients, pour tenir en memoire.
- L'EDA BigQuery n'a pas pu etre executee depuis cet environnement (droits
  GCP).

## Structure du projet

```
telco-churn-ml/
├── data/raw/telco_churn.csv       # dataset source (IBM, public)
├── notebooks/01_eda.ipynb         # exploration (notebook execute)
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
│   ├── shap_utils.py              # outils SHAP partages
│   └── log_analysis.py            # logging MLflow de l'analyse avancee
├── sql/                           # EDA en SQL BigQuery (5 requetes)
├── scripts/run_all.py             # reproduction en une commande (--steps)
├── tests/                         # pytest, un fichier par module
├── reports/
│   ├── synthese_metier.md         # synthese d'une page pour le marketing
│   ├── segmentation/ interactions/ robustness/ business/   # resultats (csv, json)
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
pytest tests/ -q                 # tests (les tests de l'API necessitent models/best_model.pkl)
ruff check src tests scripts
```

`run_all.py` enchaine train, evaluate, interpret, segmentation, interactions,
robustness, business et le logging MLflow ; il s'arrete a la premiere etape
en echec en indiquant laquelle. Tout tourne avec `n_jobs=1` pour limiter la
memoire. `make all` / `make test` / `make lint` font la meme chose sous
Linux/macOS. L'EDA BigQuery est a part (`python -m src.bq_eda`, voir plus
haut).

Les `.pkl` n'etant pas versionnes, les tests de l'API (`tests/test_api.py`)
sont ignores (skipped) tant que `python -m src.train` n'a pas ete lance.

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

L'API sert le pipeline complet sauvegarde par `train.py`
(`models/best_model.pkl` : ColumnTransformer + modele dans un seul objet
joblib), ce qui garantit un pretraitement identique a l'entrainement.
Le modele servi est celui retenu par la regle de selection de `train.py` (regression logistique, voir plus haut).

```bash
python -m src.train                      # si models/best_model.pkl n'existe pas encore
uvicorn src.api:app --reload             # -> http://127.0.0.1:8000 (doc interactive : /docs)
```

Le chemin du modele peut etre change via la variable d'environnement
`MODEL_PATH`.

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
{"churn_probability": 0.8950189298885548, "churn_prediction": 1}
```

`churn_prediction` vaut 1 si la probabilite de churn est >= 0,5.

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
