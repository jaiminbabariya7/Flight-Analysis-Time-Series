-- Cleans raw BTS On-Time Performance data
SELECT PARSE_DATE('%Y-%m-%d', CAST(fl_date AS STRING)) AS fl_date,
    UPPER(TRIM(carrier)) AS carrier, UPPER(TRIM(origin)) AS origin, UPPER(TRIM(dest)) AS dest,
    SAFE_CAST(dep_delay AS FLOAT64) AS dep_delay, SAFE_CAST(arr_delay AS FLOAT64) AS arr_delay,
    SAFE_CAST(distance AS FLOAT64) AS distance,
    CASE WHEN SAFE_CAST(arr_delay AS FLOAT64) <= 0 THEN 'on_time'
         WHEN SAFE_CAST(arr_delay AS FLOAT64) BETWEEN 1 AND 15 THEN 'minor_delay'
         WHEN SAFE_CAST(arr_delay AS FLOAT64) BETWEEN 16 AND 60 THEN 'moderate_delay'
         ELSE 'severe_delay' END AS delay_category,
    CURRENT_TIMESTAMP() AS _ingested_at
FROM flights.raw_flights
WHERE cancelled = 0 AND diverted = 0 AND origin IS NOT NULL