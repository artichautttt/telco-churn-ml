-- Taux de churn par service internet.
SELECT
  InternetService AS internet_service,
  COUNT(*) AS n_clients,
  COUNTIF(Churn = 'Yes') AS n_churners,
  ROUND(SAFE_DIVIDE(COUNTIF(Churn = 'Yes'), COUNT(*)), 4) AS churn_rate
FROM `{table}`
GROUP BY internet_service
ORDER BY churn_rate DESC;
