# Rapport de derive

> Jeu public IBM Telco, sans derive reelle connue. La detection a ete validee sur des derives injectees (notebooks/03_drift_monitoring), pas sur un cas reel. Seuils PSI : convention d'usage, pas une regle.

- Genere le : 2026-10-31T20:43:51+00:00
- Modele : `models/calibrated_model.pkl`, seuil 0.18
- Reference : train (5634 lignes) ; courant : test (1409 lignes)
- Seuils : PSI surveillance 0.1, alerte 0.25, alpha 0.05 (p corrigee BH)
- **Niveau global : aucun** ; variables en surveillance : 0, en alerte : 0 sur 19

## Predictions

PSI 0.0139, KS 0.0208 (p = 0.7029) ; part au-dessus du seuil 0.18 : 51.77% (reference) contre 52.24% (courant) ; niveau : aucun.

## Variables

| Variable | Type | PSI | Test | Statistique | p | p corrigee | Niveau |
|---|---|---:|---|---:|---:|---:|---|
| MonthlyCharges | numerique | 0.0129 | ks | 0.0321 | 0.1912 | 0.7956 | aucun |
| TotalCharges | numerique | 0.0075 | ks | 0.0296 | 0.2710 | 0.8580 | aucun |
| tenure | numerique | 0.0057 | ks | 0.0212 | 0.6835 | 0.9291 | aucun |
| OnlineBackup | categorielle | 0.0044 | chi2 | 4.9367 | 0.0847 | 0.7956 | aucun |
| MultipleLines | categorielle | 0.0030 | chi2 | 3.3552 | 0.1868 | 0.7956 | aucun |
| StreamingTV | categorielle | 0.0028 | chi2 | 3.1273 | 0.2094 | 0.7956 | aucun |
| PhoneService | categorielle | 0.0017 | chi2 | 1.8321 | 0.1759 | 0.7956 | aucun |
| StreamingMovies | categorielle | 0.0010 | chi2 | 1.1532 | 0.5618 | 0.9291 | aucun |
| TechSupport | categorielle | 0.0008 | chi2 | 0.8699 | 0.6473 | 0.9291 | aucun |
| DeviceProtection | categorielle | 0.0004 | chi2 | 0.5006 | 0.7786 | 0.9291 | aucun |
| PaymentMethod | categorielle | 0.0004 | chi2 | 0.4530 | 0.9291 | 0.9291 | aucun |
| gender | categorielle | 0.0004 | chi2 | 0.4138 | 0.5200 | 0.9291 | aucun |
| OnlineSecurity | categorielle | 0.0003 | chi2 | 0.3810 | 0.8265 | 0.9291 | aucun |
| Dependents | categorielle | 0.0003 | chi2 | 0.3334 | 0.5637 | 0.9291 | aucun |
| SeniorCitizen | categorielle | 0.0002 | chi2 | 0.2729 | 0.6014 | 0.9291 | aucun |
| InternetService | categorielle | 0.0002 | chi2 | 0.2669 | 0.8751 | 0.9291 | aucun |
| Partner | categorielle | 0.0002 | chi2 | 0.2049 | 0.6508 | 0.9291 | aucun |
| Contract | categorielle | 0.0001 | chi2 | 0.1632 | 0.9216 | 0.9291 | aucun |
| PaperlessBilling | categorielle | 0.0001 | chi2 | 0.1137 | 0.7360 | 0.9291 | aucun |
