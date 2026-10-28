-- Taux de churn par moyen de paiement.
SELECT
  PaymentMethod AS payment_method,
  COUNT(*) AS n_clients,
  COUNTIF(Churn = 'Yes') AS n_churners,
  ROUND(SAFE_DIVIDE(COUNTIF(Churn = 'Yes'), COUNT(*)), 4) AS churn_rate
FROM `{table}`
GROUP BY payment_method
ORDER BY churn_rate DESC;
