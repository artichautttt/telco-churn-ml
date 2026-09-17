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
│   └── interpret.py               # coefficients + SHAP
├── tests/test_preprocessing.py    # tests unitaires du pretraitement
├── reports/figures/               # figures generees (EDA, metriques, interpretabilite)
├── models/                        # modele entraine + resultats (non versionne)
└── requirements.txt
```

## Reproduire les resultats

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

pytest tests/ -q                 # tests unitaires du pretraitement
python -m src.train              # entraine et compare les 3 modeles (CV)
python -m src.evaluate           # evalue le meilleur modele sur le holdout
python -m src.interpret          # coefficients + SHAP
```

## Contraintes du projet

- Machine learning classique uniquement (scikit-learn), pas de deep learning.
- Toutes les metriques ci-dessus sont issues d'executions reelles du code de
  ce depot, avec seed fixe (`random_state=42`) pour la reproductibilite.
