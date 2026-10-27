#!/usr/bin/env python3
"""Confronto tra la previsione WRF e le osservazioni reali ("pagella").

Osservazioni: bollettini METAR degli aeroporti (NOAA Aviation Weather Center, dati pubblici), una o due al giorno
per ogni ora. Il programma:
  1. scarica le osservazioni delle ultime ore e le aggiunge allo storico (osservazioni.csv);
  2. prende i risultati dell'ultima corsa WRF (d01_tutto.npz, d02_tutto.npz), li legge nei punti delle stazioni
     e li aggiunge allo storico (previsioni/<corsa>.csv);
  3. confronta previsto e osservato e scrive pagella.json, che la pagina mostra.

Uso:  stazioni.py --storico CARTELLA --uscite CARTELLA_CON_NPZ --corsa AAAAMMGGHH
"""
import argparse
import csv
import json
import math
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

BBOX = (41.0, 11.3, 43.4, 14.3)     # lat min, lon min, lat max, lon max
URL = "https://aviationweather.gov/api/data/metar?bbox={},{},{},{}&format=json&hours={}"
KT = 1.852                           # km/h per nodo
GIORNI_FINESTRA = 7


def scarica(url, tentativi=4):
    ultimo = None
    for n in range(tentativi):
        try:
            rich = urllib.request.Request(url, headers={"User-Agent": "wrf-nerola"})
            with urllib.request.urlopen(rich, timeout=60) as r:
                return r.read().decode("utf-8")
        except Exception as e:  # noqa: BLE001
            ultimo = e
            time.sleep(5 * (n + 1))
    raise RuntimeError(f"download fallito: {ultimo}")


def ora_iso(t):
    return t.strftime("%Y-%m-%dT%H:00Z")


def leggi_metar(ore=72):
    testo = scarica(URL.format(*BBOX, ore)).strip()
    if not testo:
        return []
    dati = json.loads(testo)
    migliori = {}
    for m in dati:
        if m.get("temp") is None or m.get("obsTime") is None:
            continue
        t = datetime.fromtimestamp(int(m["obsTime"]), tz=timezone.utc)
        ora = (t + timedelta(minutes=30)).replace(minute=0, second=0, microsecond=0)   # ora piu' vicina
        dist = abs((t - ora).total_seconds())
        chiave = (m["icaoId"], ora)
        if chiave in migliori and migliori[chiave][0] <= dist:
            continue
        v = m.get("wspd")
        td = m.get("dewp")
        migliori[chiave] = (dist, {
            "id": m["icaoId"], "nome": (m.get("name") or m["icaoId"]).split(",")[0].strip(), "lat": m["lat"], "lon": m["lon"],
            "quota": m.get("elev"), "ora": ora_iso(ora), "T": m["temp"], "Td": "" if td is None else td,
            "V": "" if v is None else round(float(v) * KT, 1)})
    return [r for _, r in migliori.values()]


CAMPI_OSS = ["id", "nome", "lat", "lon", "quota", "ora", "T", "Td", "V"]


def unisci_osservazioni(percorso, nuove):
    vecchie = {}
    if percorso.exists():
        with open(percorso, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                vecchie[(r["id"], r["ora"])] = r
    for r in nuove:
        vecchie[(r["id"], r["ora"])] = {k: r[k] for k in CAMPI_OSS}
    limite = ora_iso(datetime.now(timezone.utc) - timedelta(days=60))
    righe = sorted((r for r in vecchie.values() if r["ora"] >= limite), key=lambda r: (r["ora"], r["id"]))
    with open(percorso, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, CAMPI_OSS)
        w.writeheader()
        w.writerows(righe)
    return righe


def dewpoint(q2, psfc):
    e = np.maximum(q2 * (psfc / 100.0) / (0.622 + q2), 1e-4)
    x = np.log(e / 6.112)
    return 243.5 * x / (17.67 - x)


def punto_vicino(lat, lon, la, lo):
    """Indice (i, j) della cella piu' vicina, o None se il punto e' fuori dal dominio."""
    c = math.cos(math.radians(lat))
    d = (la - lat) ** 2 + ((lo - lon) * c) ** 2
    i, j = np.unravel_index(np.argmin(d), d.shape)
    ny, nx = d.shape
    if i < 2 or j < 2 or i > ny - 3 or j > nx - 3:
        return None
    return int(i), int(j), float(np.sqrt(d[i, j]) * 111.0)


def estrai_previsione(cartella_uscite, corsa, stazioni):
    """Righe [corsa, valido, ore_dall_inizio, stazione, T_d01, Td_d01, V_d01, h_d01, T_d02, ...]."""
    campi = {}
    for dom in ("d01", "d02"):
        p = Path(cartella_uscite) / f"{dom}_tutto.npz"
        if not p.exists():
            continue
        z = np.load(p, allow_pickle=True)
        t2 = z["T2"] - 273.15
        td = dewpoint(z["Q2"], z["PSFC"])
        v = np.hypot(z["U10"], z["V10"]) * 3.6
        tempi = [datetime.strptime(str(t), "%Y-%m-%d_%H:%M:%S").replace(tzinfo=timezone.utc) for t in z["tempi"]]
        campi[dom] = (z["lat"], z["lon"], z["hgt"], t2, td, v, tempi)
    if not campi:
        return []
    inizio = datetime.strptime(corsa, "%Y%m%d%H").replace(tzinfo=timezone.utc)
    righe = []
    base = campi.get("d01") or campi["d02"]
    indici = {dom: {t: k for k, t in enumerate(c[6])} for dom, c in campi.items()}
    for s in stazioni.values():
        trovati = {}
        for dom, (lat, lon, hgt, t2, td, v, tempi) in campi.items():
            pv = punto_vicino(s["lat"], s["lon"], lat, lon)
            if pv:
                trovati[dom] = pv
        if not trovati:
            continue
        for t in base[6]:
            riga = [corsa, ora_iso(t), int(round((t - inizio).total_seconds() / 3600)), s["id"]]
            for dom in ("d01", "d02"):
                if dom in trovati and t in indici[dom]:
                    lat, lon, hgt, t2, td, v, tempi = campi[dom]
                    i, j, _ = trovati[dom]
                    kk = indici[dom][t]
                    riga += [round(float(t2[kk, i, j]), 2), round(float(td[kk, i, j]), 2), round(float(v[kk, i, j]), 1), round(float(hgt[i, j]))]
                else:
                    riga += ["", "", "", ""]
            righe.append(riga)
    return righe


CAMPI_PREV = ["corsa", "valido", "ore", "id", "T_d01", "Td_d01", "V_d01", "h_d01", "T_d02", "Td_d02", "V_d02", "h_d02"]


def salva_previsione(cartella, corsa, righe):
    p = Path(cartella) / "previsioni"
    p.mkdir(parents=True, exist_ok=True)
    with open(p / f"{corsa}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CAMPI_PREV)
        w.writerows(righe)


def leggi_previsioni(cartella, giorni):
    limite = (datetime.now(timezone.utc) - timedelta(days=giorni)).strftime("%Y%m%d%H")
    out = []
    for p in sorted((Path(cartella) / "previsioni").glob("*.csv")):
        if p.stem < limite:
            continue
        with open(p, newline="", encoding="utf-8") as f:
            out += list(csv.DictReader(f))
    return out


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


FASCE = [(0, 12, "0-12 ore"), (12, 24, "12-24 ore"), (24, 36, "24-36 ore"), (36, 49, "36-48 ore")]


def statistiche(coppie):
    """coppie: [(previsto, osservato)] -> errore medio assoluto, tendenza (previsto meno osservato), numero."""
    if not coppie:
        return None
    e = np.array([p - o for p, o in coppie])
    return {"mae": round(float(np.mean(np.abs(e))), 2), "bias": round(float(np.mean(e)), 2), "n": int(len(e))}


def pulisci(x):
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def calcola_pagella(oss_righe, prev_righe, giorni=GIORNI_FINESTRA):
    oss = {(r["id"], r["ora"]): r for r in oss_righe}
    stazioni = {}
    for r in oss_righe:
        stazioni[r["id"]] = {"id": r["id"], "nome": r["nome"], "lat": float(r["lat"]), "lon": float(r["lon"]), "quota": num(r["quota"])}
    limite = ora_iso(datetime.now(timezone.utc) - timedelta(days=giorni))
    riep = {d: {v: {f[2]: [] for f in FASCE} for v in ("T", "Td", "V")} for d in ("1km", "3km")}
    per_stazione = {}
    for r in prev_righe:
        o = oss.get((r["id"], r["valido"]))
        if not o or r["valido"] < limite:
            continue
        ore = int(r["ore"])
        fascia = next((f[2] for f in FASCE if f[0] <= ore < f[1]), None)
        if fascia is None:
            continue
        for dom, suff in (("1km", "d02"), ("3km", "d01")):
            for v in ("T", "Td", "V"):
                p, ob = num(r[f"{v}_{suff}"]), num(o[v])
                if p is None or ob is None:
                    continue
                riep[dom][v][fascia].append((p, ob))
                s = per_stazione.setdefault(r["id"], {}).setdefault(dom, {}).setdefault(v, [])
                if ore < 24:
                    s.append((p, ob))
    # serie recente: osservato e ultima previsione disponibile per ogni ora (previsione piu' recente tra quelle che coprono l'ora)
    ultime = {}
    for r in prev_righe:
        if r["valido"] < limite:
            continue
        k = (r["id"], r["valido"])
        if k not in ultime or r["corsa"] > ultime[k]["corsa"]:
            ultime[k] = r
    elenco = []
    for sid, st in sorted(stazioni.items()):
        voce = dict(st)
        voce["quota_modello"] = {}
        for dom, suff in (("1km", "d02"), ("3km", "d01")):
            qs = [num(r[f"h_{suff}"]) for r in prev_righe if r["id"] == sid and num(r[f"h_{suff}"]) is not None][:1]
            if qs:
                voce["quota_modello"][dom] = qs[0]
        voce["errori"] = {dom: {v: statistiche(c) for v, c in vv.items()} for dom, vv in per_stazione.get(sid, {}).items()}
        ore_serie = sorted({k[1] for k in ultime if k[0] == sid} | {k[1] for k in oss if k[0] == sid and k[1] >= limite})
        voce["serie"] = {
            "tempi": ore_serie,
            "oss": {v: [pulisci(num(oss[(sid, t)][v])) if (sid, t) in oss else None for t in ore_serie] for v in ("T", "Td", "V")},
            "wrf1": {v: [num(ultime[(sid, t)][f"{v}_d02"]) if (sid, t) in ultime else None for t in ore_serie] for v in ("T", "Td", "V")},
            "wrf3": {v: [num(ultime[(sid, t)][f"{v}_d01"]) if (sid, t) in ultime else None for t in ore_serie] for v in ("T", "Td", "V")},
        }
        if voce["errori"] or any(x is not None for x in voce["serie"]["oss"]["T"]):
            elenco.append(voce)
    return {
        "aggiornato": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "giorni": giorni,
        "fonte": "Osservazioni: bollettini METAR (NOAA Aviation Weather Center)",
        "riepilogo": {d: {v: {f: statistiche(c) for f, c in ff.items()} for v, ff in vv.items()} for d, vv in riep.items()},
        "stazioni": elenco,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--storico", required=True, help="cartella dello storico (osservazioni, previsioni, pagella.json)")
    ap.add_argument("--uscite", default="", help="cartella con d01_tutto.npz e d02_tutto.npz dell'ultima corsa")
    ap.add_argument("--corsa", default="", help="AAAAMMGGHH della corsa in --uscite")
    ap.add_argument("--ore", type=int, default=72, help="ore di osservazioni da scaricare")
    a = ap.parse_args()
    cart = Path(a.storico)
    cart.mkdir(parents=True, exist_ok=True)
    try:
        nuove = leggi_metar(a.ore)
        print("osservazioni scaricate:", len(nuove), "stazioni:", len({r['id'] for r in nuove}))
    except Exception as e:  # noqa: BLE001
        print("ATTENZIONE: osservazioni non scaricate:", e, file=sys.stderr)
        nuove = []
    oss = unisci_osservazioni(cart / "osservazioni.csv", nuove)
    stazioni = {r["id"]: {"id": r["id"], "lat": float(r["lat"]), "lon": float(r["lon"])} for r in oss}
    if a.uscite and a.corsa:
        righe = estrai_previsione(a.uscite, a.corsa, stazioni)
        print("previsioni estratte:", len(righe))
        if righe:
            salva_previsione(cart, a.corsa, righe)
    pag = calcola_pagella(oss, leggi_previsioni(cart, GIORNI_FINESTRA + 3))
    (cart / "pagella.json").write_text(json.dumps(pag, ensure_ascii=False, separators=(",", ":"), default=pulisci))
    print("pagella scritta:", len(pag["stazioni"]), "stazioni")


if __name__ == "__main__":
    main()
