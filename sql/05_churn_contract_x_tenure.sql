-- Croisement type de contrat x tranche d'anciennete.
SELECT
  Contract AS contract,
  CASE
    WHEN tenure <= 12 THEN '00-12'
    WHEN tenure <= 24 THEN '13-24'
    WHEN tenure <= 48 THEN '25-48'
    ELSE '49-72'
  END AS tenure_band,
  COUNT(*) AS n_clients,
  COUNTIF(Churn = 'Yes') AS n_churners,
  ROUND(SAFE_DIVIDE(COUNTIF(Churn = 'Yes'), COUNT(*)), 4) AS churn_rate
FROM `{table}`
GROUP BY contract, tenure_band
ORDER BY contract, tenure_band;
