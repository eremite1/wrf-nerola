# Previsione WRF su Nerola

Corsa ICON2I **2026111412 UTC**, 48 ore (catena icon2i).

- `mappe/1km/<parametro>/AAAAMMGG_HH.png` e `mappe/3km/...`: una mappa per parametro (ora UTC nel nome del file).
- `index.html`: la pagina web (con `indice.json`, l'elenco di tutte le mappe).
- `sondaggi.html`: radiosondaggi virtuali (WRF + modelli di Mappe Lazio), dati in `dati/<dominio>/sondaggi.bin`.
- `meteogramma_nerola.png`: andamento nel punto di Nerola.
- `d01_tutto.npz`, `d02_tutto.npz`: i dati.
Confini: Natural Earth (pubblico dominio). Nomi dei paesi: GeoNames (CC BY 4.0).

Modello WRF 4.7.0, dati di partenza e al contorno: ECMWF IFS open data 0,1 gradi
(https://www.ecmwf.int/en/forecasts/datasets/open-data), licenza CC BY 4.0.
Dati modificati: ritagliati sull'Italia centrale, ricodificati e usati come condizioni iniziali e al contorno di WRF.
ECMWF non garantisce i dati e non risponde di errori o omissioni.
Previsione sperimentale: non usare per decisioni importanti o di sicurezza.
