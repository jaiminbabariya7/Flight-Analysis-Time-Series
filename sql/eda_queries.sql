-- 1. Delay rate by carrier
SELECT carrier, COUNT(*) AS total, COUNTIF(arr_delay>15) AS delayed,
    ROUND(COUNTIF(arr_delay>15)/COUNT(*)*100,2) AS delay_pct, ROUND(AVG(arr_delay),2) AS avg_delay
FROM flights.cleaned_flights GROUP BY carrier ORDER BY delay_pct DESC;

-- 2. Delay by hour
SELECT CAST(crs_dep_time/100 AS INT64) AS dep_hour, COUNT(*) AS flights, ROUND(AVG(arr_delay),2) AS avg_delay
FROM flights.cleaned_flights GROUP BY dep_hour ORDER BY dep_hour;

-- 3. Top delayed routes
SELECT CONCAT(origin,' -> ',dest) AS route, COUNT(*) AS total, ROUND(AVG(arr_delay),2) AS avg_delay
FROM flights.cleaned_flights GROUP BY route HAVING COUNT(*)>500 ORDER BY avg_delay DESC LIMIT 20;