# Telco Customer Churn -- Prediction de desabonnement client

Projet de machine learning classique (from scratch, scikit-learn) : predire
si un client d'un operateur telecom va resilier son abonnement (churn), a
partir de ses donnees contractuelles et d'usage.

## Probleme

Classification binaire supervisee : `Churn` (Yes/No) a partir de 19
variables (donnees demographiques, services souscrits, type de contrat,
facturation).

**Dataset** : [IBM Telco Customer Churn](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)
-- 7 043 clients, 21 colonnes, dataset public reel. Taux de churn observe :
**26,5 %** (classes desequilibrees).

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

## Resultats (chiffres reels, reproductibles avec `random_state=42`)

### Validation croisee (5-fold stratifie, train set)

| Modele | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| **Regression logistique** | 0.749 (+/-0.015) | 0.628 (+/-0.023) | **0.846** (+/-0.012) |
| Random Forest | 0.778 (+/-0.012) | 0.610 (+/-0.027) | 0.825 (+/-0.011) |
| XGBoost | 0.797 (+/-0.010) | 0.574 (+/-0.020) | 0.839 (+/-0.009) |

**Modele retenu : regression logistique**, meilleur ROC-AUC en CV. Sur ce
dataset desequilibre, la regression logistique avec `class_weight="balanced"`
offre le meilleur compromis precision/rappel sur la classe minoritaire
(churn), ce que l'accuracy seule masquerait -- Random Forest et XGBoost ont
une accuracy plus elevee mais au prix d'un F1 plus faible (ils sous-predisent
la classe churn).

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
│   └── api.py                     # API FastAPI (POST /predict)
├── tests/
│   ├── test_preprocessing.py      # tests unitaires du pretraitement
│   └── test_api.py                # tests de l'API et du modele sauvegarde
├── reports/figures/               # figures generees (EDA, metriques, interpretabilite)
├── models/                        # resultats .json versionnes ; modeles .pkl non versionnes
├── mlruns/                        # runs MLflow locaux (non versionne)
└── requirements.txt
```

## Reproduire les resultats

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

python -m src.train              # entraine et compare les 3 modeles (CV) + tracking MLflow
python -m src.evaluate           # evalue le meilleur modele sur le holdout
python -m src.interpret          # coefficients + SHAP
pytest tests/ -q                 # tests (les tests de l'API necessitent models/best_model.pkl)
```

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
Le modele servi est celui retenu par `train.py` (meilleur ROC-AUC en CV).

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
