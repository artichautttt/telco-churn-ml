# %% [markdown]
# # EDA -- Telco Customer Churn
#
# Objectif : comprendre la structure du dataset, detecter les valeurs
# manquantes/aberrantes, observer les distributions et les correlations
# avant de definir la strategie de nettoyage et de feature engineering.

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path.cwd().parent))
from src.preprocessing import clean_data, load_data

sns.set_theme(style="whitegrid")
FIG_DIR = Path("../reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# %%
df_raw = load_data("../data/raw/telco_churn.csv")
print(df_raw.shape)
df_raw.head()

# %% [markdown]
# ## Structure et types

# %%
df_raw.info()

# %% [markdown]
# `TotalCharges` est lu comme texte (`object`) alors qu'il devrait etre
# numerique -- signe de valeurs non numeriques dans la colonne.

# %%
non_numeric_mask = pd.to_numeric(df_raw["TotalCharges"], errors="coerce").isna()
print(f"Lignes TotalCharges non numeriques : {non_numeric_mask.sum()}")
df_raw.loc[non_numeric_mask, ["customerID", "tenure", "MonthlyCharges", "TotalCharges"]]

# %% [markdown]
# Les 11 lignes problematiques correspondent toutes a `tenure == 0` : des
# clients tout juste inscrits, rien n'a encore ete facture. Ce n'est pas une
# valeur manquante au hasard, c'est un total logiquement nul.

# %% [markdown]
# ## Valeurs manquantes (apres nettoyage)

# %%
df = clean_data(df_raw)
df.isna().sum().sum()

# %% [markdown]
# ## Distribution de la cible (Churn)

# %%
churn_rate = df["Churn"].mean()
print(f"Taux de churn : {churn_rate:.1%}")

fig, ax = plt.subplots(figsize=(5, 4))
df["Churn"].map({0: "No", 1: "Yes"}).value_counts().plot(kind="bar", ax=ax, color=["#4C72B0", "#C44E52"])
ax.set_title("Distribution de la cible Churn")
ax.set_xlabel("Churn")
ax.set_ylabel("Nombre de clients")
plt.tight_layout()
plt.savefig(FIG_DIR / "churn_distribution.png", dpi=120)
plt.show()

# %% [markdown]
# Desequilibre modere (~73% / ~27%) : accuracy seule sera trompeuse, il
# faudra regarder F1 / ROC-AUC / matrice de confusion.

# %% [markdown]
# ## Distributions des variables numeriques

# %%
numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, numeric_cols):
    sns.histplot(data=df, x=col, hue=df["Churn"].map({0: "No", 1: "Yes"}), kde=True, ax=ax, element="step")
    ax.set_title(col)
plt.tight_layout()
plt.savefig(FIG_DIR / "numeric_distributions.png", dpi=120)
plt.show()

# %% [markdown]
# - `tenure` : les clients qui partent (Churn=Yes) sont fortement concentres
#   sur les faibles anciennetes -- signal fort et attendu.
# - `MonthlyCharges` : les clients qui partent paient en moyenne plus cher.
# - `TotalCharges` correle fortement avec tenure (produit tenure x charges
#   mensuelles) -- a surveiller pour la colinearite.

# %% [markdown]
# ## Correlations (variables numeriques)

# %%
corr = df[numeric_cols + ["Churn"]].corr()
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
plt.tight_layout()
plt.savefig(FIG_DIR / "correlation_heatmap.png", dpi=120)
plt.show()
print(corr["Churn"].sort_values(ascending=False))

# %% [markdown]
# `tenure` et `TotalCharges` sont correlees a ~0.83 (attendu : TotalCharges
# ~ tenure * MonthlyCharges). Les deux restent informatives individuellement
# mais des modeles lineaires (regression logistique) seront sensibles a
# cette colinearite -- point a garder en tete au moment de comparer les
# coefficients.

# %% [markdown]
# ## Variables categorielles vs Churn

# %%
cat_cols = ["Contract", "InternetService", "PaymentMethod", "PaperlessBilling"]
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, col in zip(axes.ravel(), cat_cols):
    rate = df.groupby(col)["Churn"].mean().sort_values(ascending=False)
    rate.plot(kind="bar", ax=ax, color="#C44E52")
    ax.set_title(f"Taux de churn par {col}")
    ax.set_ylabel("Taux de churn")
    ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig(FIG_DIR / "categorical_churn_rates.png", dpi=120)
plt.show()

# %% [markdown]
# Signaux forts :
# - Contrat **Month-to-month** : taux de churn nettement plus eleve que
#   One year / Two year (engagement long = fidelisation).
# - Service internet **Fiber optic** : churn plus eleve que DSL ou "No"
#   (a investiguer -- prix plus eleve, ou qualite de service percue).
# - **Electronic check** comme methode de paiement associe a un churn
#   plus eleve que les paiements automatiques (carte/banque).
#
# ## Conclusions pour l'etape de nettoyage / feature engineering
#
# 1. `TotalCharges` : convertir en numerique, remplir les 11 valeurs vides
#    par 0 (clients tenure=0).
# 2. `customerID` : a retirer (pas de valeur predictive).
# 3. Cible `Churn` : encoder Yes/No -> 1/0.
# 4. Variables categorielles : one-hot encoding.
# 5. Variables numeriques : standardisation (utile pour la regression
#    logistique, neutre pour les modeles a base d'arbres).
# 6. Desequilibre de classes modere -> utiliser F1 / ROC-AUC en plus de
#    l'accuracy, envisager `class_weight="balanced"` pour la regression
#    logistique.
