-- Taux de churn par type de contrat.
SELECT
  Contract AS contract,
  COUNT(*) AS n_clients,
  COUNTIF(Churn = 'Yes') AS n_churners,
  ROUND(SAFE_DIVIDE(COUNTIF(Churn = 'Yes'), COUNT(*)), 4) AS churn_rate
FROM `{table}`
GROUP BY contract
ORDER BY churn_rate DESC;
