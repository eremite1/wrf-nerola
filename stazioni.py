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
import gzip
import os
import json
import math
import sys
import time
import urllib.parse
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


NEROLA = (42.1606, 12.7856)   # centro del paese (prima era 3,5 km troppo a sud)
RETI_METEOHUB = ("dpcn-lazio", "dpcn-umbria", "mnw")     # rete regionale Lazio e Umbria (Protezione Civile), MeteoNetwork
BASE_MH = "https://meteohub.agenziaitaliameteo.it"
# variabile BUFR, livello e tipo di elaborazione degli strumenti standard
MH_VAR = {"B12101": ("103,2000,0,0", "T"), "B13003": ("103,2000,0,0", "RH"), "B11002": ("103,10000,0,0", "V")}
MH_PIOGGIA = ("B13011", "1,0,0,0")     # precipitazione accumulata (mm) su intervalli di 1, 5 o 60 minuti


def richiesta(url, dati=None, intestazioni=None, metodo=None):
    rich = urllib.request.Request(url, data=dati, headers={"User-Agent": "wrf-nerola", **(intestazioni or {})}, method=metodo)
    with urllib.request.urlopen(rich, timeout=180) as r:
        return r.read().decode("utf-8")


def leggi_meteohub(ore=72):
    """Osservazioni orarie delle reti della Protezione Civile (MeteoHub di ItaliaMeteo). Serve un account gratuito:
    nome utente e password stanno nei segreti METEOHUB_USERNAME e METEOHUB_PASSWORD."""
    utente, parola = os.environ.get("METEOHUB_USERNAME", ""), os.environ.get("METEOHUB_PASSWORD", "")
    if not utente or not parola:
        print("MeteoHub: nome utente e password non impostati, salto la rete della Protezione Civile")
        return []
    r = json.loads(richiesta(BASE_MH + "/auth/login", json.dumps({"username": utente, "password": parola}).encode(),
                             {"Content-Type": "application/json"}, "POST"))
    token = r if isinstance(r, str) else (r.get("token") or r.get("access_token") or r.get("jwt"))
    fine = datetime.now(timezone.utc)
    inizio = fine - timedelta(hours=ore)
    q = f"reftime: >={inizio:%Y-%m-%d %H:%M},<={fine:%Y-%m-%d %H:%M};license:CCBY_COMPLIANT"
    righe = []
    for rete in RETI_METEOHUB:
        par = urllib.parse.urlencode({"networks": rete, "q": q, "reliabilityCheck": "true", "lonmin": BBOX[1], "latmin": BBOX[0],
                                      "lonmax": BBOX[3], "latmax": BBOX[2]})
        try:
            risposta = json.loads(richiesta(BASE_MH + "/api/observations?" + par, intestazioni={"Authorization": "Bearer " + token,
                                                                                               "Accept": "application/json"}))
        except Exception as e:  # noqa: BLE001
            print(f"MeteoHub {rete}: errore {e}", file=sys.stderr)
            continue
        n0 = len(righe)
        for item in risposta.get("data", []):
            st = item["stat"]
            det = {d["var"]: d["val"] for d in st.get("details", [])}
            nome = str(det.get("B01019") or f"{st['lat']:.3f},{st['lon']:.3f}")
            sid = f"{rete}:{nome}"
            quota = det.get("B07030", det.get("B07031", ""))
            ore_v = {}
            pioggia = {}
            for prod in item.get("prod", []):
                if prod["var"] == MH_PIOGGIA[0] and prod.get("lev") == MH_PIOGGIA[1]:
                    try:
                        sec = int(prod.get("trange", "").split(",")[2])
                    except (IndexError, ValueError):
                        continue
                    if sec not in (60, 300, 3600):
                        continue
                    for v in prod.get("val", []):
                        if v.get("val") is None or v.get("rel") not in (1, None) or not 0 <= float(v["val"]) < 300:
                            continue
                        t = datetime.strptime(v["ref"][:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
                        h = ora_iso((t - timedelta(seconds=1)).replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))
                        acc = pioggia.setdefault(h, [0.0, 0])
                        acc[0] += float(v["val"])
                        acc[1] += sec
                    continue
                voce = MH_VAR.get(prod["var"])
                if not voce or prod.get("lev") != voce[0] or prod.get("trange") not in ("254,0,0", "0,0,0"):
                    continue
                for v in prod.get("val", []):
                    if v.get("val") is None or v.get("rel") not in (1, None):
                        continue
                    t = datetime.strptime(v["ref"][:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
                    h = (t + timedelta(minutes=30)).replace(minute=0, second=0, microsecond=0)
                    ore_v.setdefault(ora_iso(h), {}).setdefault(voce[1], []).append(float(v["val"]))
            for h in sorted(set(ore_v) | set(pioggia)):
                d = ore_v.get(h, {})
                t_c = None
                if "T" in d:
                    t_c = sum(d["T"]) / len(d["T"]) - 273.15
                    if not -35 < t_c < 50:
                        t_c = None
                p = ""
                if h in pioggia and pioggia[h][1] >= 0.9 * 3600:
                    p = round(pioggia[h][0], 2)
                if t_c is None and p == "":
                    continue
                td = ""
                if t_c is not None and "RH" in d:
                    rh = min(100.0, max(1.0, sum(d["RH"]) / len(d["RH"])))
                    g = math.log(rh / 100.0) + 17.62 * t_c / (243.12 + t_c)
                    td = round(243.12 * g / (17.62 - g), 1)
                vv = round(sum(d["V"]) / len(d["V"]) * 3.6, 1) if "V" in d and t_c is not None else ""
                righe.append({"id": sid, "nome": nome, "lat": st["lat"], "lon": st["lon"], "quota": quota, "ora": h,
                              "T": "" if t_c is None else round(t_c, 1), "Td": td, "V": vv, "P": p})
        print(f"MeteoHub {rete}: stazioni {len({r['id'] for r in righe[n0:]})}, righe {len(righe) - n0}")
    return righe


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
            "V": "" if v is None else round(float(v) * KT, 1), "P": ""})
    return [r for _, r in migliori.values()]


CAMPI_OSS = ["id", "nome", "lat", "lon", "quota", "ora", "T", "Td", "V", "P"]
VARS = ("T", "Td", "V", "P")


def unisci_osservazioni(percorso, nuove):
    vecchie = {}
    if percorso.exists():
        with open(percorso, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                vecchie[(r["id"], r["ora"])] = {c: r.get(c) or "" for c in CAMPI_OSS}
    for r in nuove:
        k = (r["id"], r["ora"])
        r = {c: r.get(c, "") for c in CAMPI_OSS}
        if k in vecchie and r["P"] == "" and vecchie[k].get("P"):
            r["P"] = vecchie[k]["P"]
        vecchie[k] = r
    limite = ora_iso(datetime.now(timezone.utc) - timedelta(days=60))
    righe = sorted((r for r in vecchie.values() if r["ora"] >= limite), key=lambda r: (r["ora"], r["id"]))
    with open(percorso, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, CAMPI_OSS)
        w.writeheader()
        w.writerows(righe)
    return righe


def distanza_nerola(lat, lon):
    return math.hypot(lat - NEROLA[0], (lon - NEROLA[1]) * math.cos(math.radians(NEROLA[0]))) * 111.0


RAGGIO_NEROLA = 4.5      # km: le stazioni entro questa distanza fanno la "osservazione di Nerola"


def leggi_stazione_nerola(cartella, ore=72):
    """Osservazioni orarie della stazione di Nerola (dati pubblicati sul sito, raccolti da stazione_nerola.py in nerola.csv)."""
    p = Path(cartella) / "nerola.csv"
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        letture = [(datetime.strptime(r["ts"], "%Y-%m-%dT%H:%M:00Z").replace(tzinfo=timezone.utc), r) for r in csv.DictReader(f)]
    limite = datetime.now(timezone.utc) - timedelta(hours=ore)
    letture = [(t, r) for t, r in letture if t >= limite]
    per_ora = {}
    for t, r in letture:
        per_ora.setdefault(ora_iso((t + timedelta(minutes=30)).replace(minute=0, second=0, microsecond=0)), []).append(r)
    pioggia = [(t, num(r["pioggia_oggi"])) for t, r in letture if num(r["pioggia_oggi"]) is not None]

    def cumulata(h):
        tb = datetime.strptime(h, "%Y-%m-%dT%H:00Z").replace(tzinfo=timezone.utc)
        vicini = [(abs((t - tb).total_seconds()), c) for t, c in pioggia if abs((t - tb).total_seconds()) <= 20 * 60]
        return min(vicini)[1] if vicini else None

    righe = []
    for h in sorted(per_ora):
        rr = per_ora[h]
        riga = {"id": "NEROLA", "nome": "Nerola (stazione MeteoNerola)", "lat": NEROLA[0], "lon": NEROLA[1], "quota": "", "ora": h}
        for v in ("T", "Td", "V"):
            xs = [num(r[v]) for r in rr if num(r[v]) is not None]
            riga[v] = round(sum(xs) / len(xs), 2) if xs else ""
        c1 = cumulata(h)
        c0 = cumulata(ora_iso(datetime.strptime(h, "%Y-%m-%dT%H:00Z").replace(tzinfo=timezone.utc) - timedelta(hours=1)))
        riga["P"] = "" if c1 is None or c0 is None else round(c1 - c0 if c1 >= c0 else c1, 2)
        if any(riga[v] != "" for v in ("T", "Td", "V", "P")):
            righe.append(riga)
    return righe


def osservazioni_nerola(oss_righe):
    """Nerola non ha una stazione ufficiale: usiamo la media delle stazioni entro pochi km (Moricone, Montelibretti, ...)."""
    vicine = {r["id"] for r in oss_righe if distanza_nerola(float(r["lat"]), float(r["lon"])) <= RAGGIO_NEROLA}
    per_ora = {}
    for r in oss_righe:
        if r["id"] in vicine:
            per_ora.setdefault(r["ora"], []).append(r)
    out = []
    for ora, rr in per_ora.items():
        riga = {"id": "NEROLA", "nome": "Nerola (media delle stazioni vicine)", "lat": NEROLA[0], "lon": NEROLA[1], "quota": "", "ora": ora}
        buono = False
        for v in VARS:
            xs = [float(x[v]) for x in rr if num(x.get(v)) is not None]
            riga[v] = round(sum(xs) / len(xs), 2) if len(xs) >= 2 else ""
            buono = buono or riga[v] != ""
        if buono:
            out.append(riga)
    return out, sorted(vicine)


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


def estrai_previsione(cartella_uscite, corsa, stazioni, file_dom=None):
    """Righe [corsa, valido, ore_dall_inizio, stazione, T, Td, V, h, P per ogni dominio]. P = pioggia dell'ora precedente (mm).
    file_dom: {suffisso: nome file npz}; di base d01 e d02 della catena principale (ECMWF)."""
    file_dom = file_dom or {"d01": "d01_tutto.npz", "d02": "d02_tutto.npz"}
    campi = {}
    for dom, nome in file_dom.items():
        p = Path(cartella_uscite) / nome
        if not p.exists():
            continue
        z = np.load(p, allow_pickle=True)
        t2 = z["T2"] - 273.15
        td = dewpoint(z["Q2"], z["PSFC"])
        v = np.hypot(z["U10"], z["V10"]) * 3.6
        tempi = [datetime.strptime(str(t), "%Y-%m-%d_%H:%M:%S").replace(tzinfo=timezone.utc) for t in z["tempi"]]
        if "RAINC" in z.files and "RAINNC" in z.files:
            cum = z["RAINC"] + z["RAINNC"]
            pr = np.full(cum.shape, np.nan)
            for k in range(1, len(tempi)):
                if tempi[k] - tempi[k - 1] == timedelta(hours=1):
                    pr[k] = np.maximum(cum[k] - cum[k - 1], 0)
        else:
            pr = np.full(t2.shape, np.nan)
        campi[dom] = (z["lat"], z["lon"], z["hgt"], t2, td, v, tempi, pr)
    if not campi:
        return []
    inizio = datetime.strptime(corsa, "%Y%m%d%H").replace(tzinfo=timezone.utc)
    righe = []
    base = campi.get("d01") or next(iter(campi.values()))
    indici = {dom: {t: k for k, t in enumerate(c[6])} for dom, c in campi.items()}
    for s in stazioni.values():
        trovati = {}
        for dom, c in campi.items():
            pv = punto_vicino(s["lat"], s["lon"], c[0], c[1])
            if pv:
                trovati[dom] = pv
        if not trovati:
            continue
        for t in base[6]:
            riga = [corsa, ora_iso(t), int(round((t - inizio).total_seconds() / 3600)), s["id"]]
            for dom in file_dom:
                if dom in trovati and t in indici[dom]:
                    lat, lon, hgt, t2, td, v, tempi, pr = campi[dom]
                    i, j, _ = trovati[dom]
                    kk = indici[dom][t]
                    p = float(pr[kk, i, j])
                    riga += [round(float(t2[kk, i, j]), 2), round(float(td[kk, i, j]), 2), round(float(v[kk, i, j]), 1), round(float(hgt[i, j])),
                             "" if math.isnan(p) else round(p, 2)]
                else:
                    riga += ["", "", "", "", ""]
            righe.append(riga)
    return righe


CAMPI_PREV = ["corsa", "valido", "ore", "id", "T_d01", "Td_d01", "V_d01", "h_d01", "P_d01", "T_d02", "Td_d02", "V_d02", "h_d02", "P_d02"]


def campi_prev(domini):
    return ["corsa", "valido", "ore", "id"] + [f"{v}_{d}" for d in domini for v in ("T", "Td", "V", "h", "P")]


def salva_previsione(cartella, corsa, righe, sotto="previsioni", domini=("d01", "d02")):
    p = Path(cartella) / sotto
    p.mkdir(parents=True, exist_ok=True)
    with gzip.open(p / f"{corsa}.csv.gz", "wt", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(campi_prev(domini))
        w.writerows(righe)
    vecchio = p / f"{corsa}.csv"
    if vecchio.exists():
        vecchio.unlink()


def _apri(p):
    return gzip.open(p, "rt", newline="", encoding="utf-8") if str(p).endswith(".gz") else open(p, newline="", encoding="utf-8")


def leggi_previsioni(cartella, giorni, sotto="previsioni"):
    limite = (datetime.now(timezone.utc) - timedelta(days=giorni)).strftime("%Y%m%d%H")
    out = []
    cart = Path(cartella) / sotto
    for p in sorted(cart.glob("*.csv*")):
        corsa = p.name.split(".")[0]
        if corsa < limite:
            if corsa < (datetime.now(timezone.utc) - timedelta(days=14)).strftime("%Y%m%d%H"):
                p.unlink()
            continue
        with _apri(p) as f:
            out += list(csv.DictReader(f))
    return out


# ---------------------------------------------------------------- altri modelli (stessi di "Mappe Lazio"), via Open-Meteo
MODELLI_OM = {"icon2i": "italia_meteo_arpae_icon_2i", "arome": "meteofrance_arome_france_hd", "ecmwf": "ecmwf_ifs",
              "ukmo": "ukmo_global_deterministic_10km", "iconeu": "dwd_icon_eu"}
URL_OM = "https://api.open-meteo.com/v1/forecast"
VAR_OM = (("T", "temperature_2m"), ("Td", "dew_point_2m"), ("V", "wind_speed_10m"), ("P", "precipitation"))
MAX_STAZIONI_OM = 70


def scegli_stazioni_om(stazioni):
    """Nerola, tutti gli aeroporti e le stazioni piu' vicine a Nerola (fino a MAX_STAZIONI_OM)."""
    ordine = sorted(stazioni.values(), key=lambda s: distanza_nerola(s["lat"], s["lon"]))
    sel = [s for s in ordine if s["id"] == "NEROLA"]
    sel += [s for s in ordine if len(s["id"]) == 4 and s["id"].isalpha() and s["id"] != "NEROLA" and distanza_nerola(s["lat"], s["lon"]) < 150]
    for s in ordine:
        if len(sel) >= MAX_STAZIONI_OM:
            break
        if s not in sel:
            sel.append(s)
    return sel


def scarica_altri_modelli(sel, adesso):
    """Righe {id, valido, modello, T, Td, V, P} per le prossime ore (Open-Meteo, ultima corsa di ciascun modello)."""
    righe = {}
    for k in range(0, len(sel), 25):
        gruppo = sel[k:k + 25]
        par = {"latitude": ",".join(f"{s['lat']:.4f}" for s in gruppo), "longitude": ",".join(f"{s['lon']:.4f}" for s in gruppo),
               "models": ",".join(MODELLI_OM.values()), "hourly": ",".join(v for _, v in VAR_OM), "timezone": "GMT",
               "forecast_days": 4, "past_days": 0, "wind_speed_unit": "kmh"}
        testo = scarica(URL_OM + "?" + urllib.parse.urlencode(par), tentativi=4)
        dati = json.loads(testo)
        if isinstance(dati, dict):
            dati = [dati]
        for s, d in zip(gruppo, dati):
            h = d.get("hourly", {})
            tempi = h.get("time", [])
            for mid, nome in MODELLI_OM.items():
                col = {v: h.get(f"{nom}_{nome}") for v, nom in VAR_OM}
                if not all(col.values()):
                    continue
                for n, t in enumerate(tempi):
                    valido = t[:13] + ":00Z"
                    if valido < ora_iso(adesso):
                        continue
                    vals = {v: col[v][n] for v in col}
                    if all(x is None for x in vals.values()):
                        continue
                    righe[(s["id"], valido, mid)] = vals
        time.sleep(1)
    return righe


CAMPI_SNAP = ["fetch", "valido", "ore", "id"] + [f"{v}_{m}" for m in MODELLI_OM for v in VARS]


def salva_snapshot(cartella, adesso, righe, max_ore=48):
    """Salva l'istantanea (solo le prossime 48 ore, 72 per Nerola) in modelli/<AAAAMMGGHH>.csv.gz."""
    fetch = adesso.replace(minute=0, second=0, microsecond=0)
    per = {}
    for (sid, valido, mid), vals in righe.items():
        ore = int(round((datetime.strptime(valido, "%Y-%m-%dT%H:00Z").replace(tzinfo=timezone.utc) - fetch).total_seconds() / 3600))
        if ore < 0 or ore > (72 if sid == "NEROLA" else max_ore):
            continue
        r = per.setdefault((sid, valido), {"fetch": ora_iso(fetch), "valido": valido, "ore": ore, "id": sid})
        for v in VARS:
            x = vals[v]
            r[f"{v}_{mid}"] = "" if x is None else x
    p = Path(cartella) / "modelli"
    p.mkdir(parents=True, exist_ok=True)
    with gzip.open(p / f"{fetch:%Y%m%d%H}.csv.gz", "wt", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, CAMPI_SNAP, restval="")
        w.writeheader()
        w.writerows(per.values())


def leggi_snapshot(cartella, giorni):
    cart = Path(cartella) / "modelli"
    limite = (datetime.now(timezone.utc) - timedelta(days=giorni)).strftime("%Y%m%d%H")
    out = []
    for p in sorted(cart.glob("*.csv.gz")):
        if p.name.split(".")[0] < limite:
            if p.name.split(".")[0] < (datetime.now(timezone.utc) - timedelta(days=14)).strftime("%Y%m%d%H"):
                p.unlink()
            continue
        with _apri(p) as f:
            out += list(csv.DictReader(f))
    return out


def ultimo_snapshot_vecchio(cartella, ore=5):
    cart = Path(cartella) / "modelli"
    nomi = sorted(p.name.split(".")[0] for p in cart.glob("*.csv.gz")) if cart.exists() else []
    if not nomi:
        return False
    t = datetime.strptime(nomi[-1], "%Y%m%d%H").replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - t < timedelta(hours=ore)


# ---------------------------------------------------------------- pagella
def num(x):
    try:
        if x is None or x == "":
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def pulisci(x):
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


FASCE = [(0, 12, "0-12 ore"), (12, 24, "12-24 ore"), (24, 36, "24-36 ore"), (36, 49, "36-48 ore")]
NOMI_FASCE = [f[2] for f in FASCE]
SOGLIA_PIOGGIA = 1.0     # mm in 6 ore: sopra questa soglia "ha piovuto"
NOMI_MODELLI = {
    "wrf1": ("WRF Nerola 1 km", "il nostro modello, risoluzione 1 km"),
    "wrf3": ("WRF Nerola 3 km", "il nostro modello, dominio esterno 3 km"),
    "wrfc": ("WRF corretto", "esperimento: il WRF che sbaglia meno, con correzione statistica degli errori passati, pioggia compresa (vedi sotto)"),
    "icon2i": ("ICON-2I", "ItaliaMeteo-ARPAE, 2 km"),
    "arome": ("AROME", "Meteo-France, 1,3 km"),
    "ecmwf": ("ECMWF IFS", "Centro europeo, 9 km"),
    "ukmo": ("UKMO", "Met Office, 10 km"),
    "iconeu": ("ICON-EU", "DWD, 7 km"),
}


# Tutti gli altri WRF (oltre a wrf1/wrf3 su ECMWF): id -> (cartella previsioni nello storico, dominio, nome)
FORZANTI = {"ecmwf": "ECMWF", "gfs": "GFS", "icon2i": "ICON-2I", "iconeu": "ICON-EU"}
RICETTE = {"f1": "WSM6 + MYNN + Noah", "f2": "Thompson + YSU + Noah", "f3": "Morrison + ACM2 + Noah-MP", "f4": "Thompson + MYJ + Noah"}
CATENE_EXTRA = {"gfs": ("prev_gfs", {"d01": "d01_tutto.npz", "d02": "d02_tutto.npz"}),
                "icon2i": ("prev_icon2i", {"d01": "d01_tutto.npz", "d02": "d02_tutto.npz"}),
                "iconeu": ("prev_iconeu", {"d01": "d01_tutto.npz", "d02": "d02_tutto.npz"}),
                "fine": ("prev_fine", {"d03": "d03_tutto.npz"})}


def registro_wrf():
    """id modello -> (cartella previsioni, suffisso colonna, nome, nota). Comprende tutti i WRF che girano."""
    r = {"wrf1": ("previsioni", "d02", "WRF 1 km · ECMWF", "catena principale"),
         "wrf3": ("previsioni", "d01", "WRF 3 km · ECMWF", "catena principale"),
         "wrf333": ("prev_fine", "d03", "WRF 333 m · ECMWF", "solo Nerola e dintorni (~20 km): poche stazioni")}
    for k in ("icon2i", "gfs", "iconeu"):
        r[f"wrf1_{k}"] = (f"prev_{k}", "d02", f"WRF 1 km · {FORZANTI[k]}", "catena su altri dati iniziali")
        r[f"wrf3_{k}"] = (f"prev_{k}", "d01", f"WRF 3 km · {FORZANTI[k]}", "catena su altri dati iniziali")
    for k in FORZANTI:
        for f, ric in RICETTE.items():
            r[f"wrf3_{k}_{f}"] = (f"prev_fis_{k}_{f}", "d01", f"WRF 3 km · {FORZANTI[k]} · {f.upper()}",
                                  f"variante di fisica: {ric}, una corsa al giorno")
    return r


def fascia_di(ore):
    return next((f[2] for f in FASCE if f[0] <= ore < f[1]), None)


def campioni_wrf(prev_righe, limite):
    """modello -> lista di (corsa, id, valido, ore, {T,Td,V,P})."""
    out = {"wrf1": [], "wrf3": []}
    for r in prev_righe:
        if r["valido"] < limite:
            continue
        ore = int(r["ore"])
        for m, suff in (("wrf1", "d02"), ("wrf3", "d01")):
            vals = {v: num(r.get(f"{v}_{suff}")) for v in VARS}
            if all(x is None for x in vals.values()):
                continue
            out[m].append((r["corsa"], r["id"], r["valido"], ore, vals))
    return out


def campioni_tutti_wrf(cart, giorni, limite):
    """Campioni di TUTTI i WRF registrati (esclusi wrf1/wrf3 gia' letti dalla catena principale)."""
    out, cache = {}, {}
    for m, (sotto, suff, _, _) in registro_wrf().items():
        if sotto == "previsioni":
            continue
        if sotto not in cache:
            cache[sotto] = leggi_previsioni(cart, giorni, sotto) if (Path(cart) / sotto).exists() else []
        lista = []
        for r in cache[sotto]:
            if r["valido"] < limite:
                continue
            vals = {v: num(r.get(f"{v}_{suff}")) for v in VARS}
            if all(x is None for x in vals.values()):
                continue
            lista.append((r["corsa"], r["id"], r["valido"], int(r["ore"]), vals))
        if lista:
            out[m] = lista
    return out


def campioni_om(snap_righe, limite):
    out = {m: [] for m in MODELLI_OM}
    for r in snap_righe:
        if r["valido"] < limite:
            continue
        ore = int(r["ore"])
        for m in MODELLI_OM:
            vals = {v: num(r.get(f"{v}_{m}")) for v in VARS}
            if all(x is None for x in vals.values()):
                continue
            out[m].append((r["fetch"], r["id"], r["valido"], ore, vals))
    return out


def correggi_wrf(campioni, oss, giorni=14, min_staz=4, min_tot=40, solo_corse=None):
    """WRF 1 km corretto: toglie al modello l'errore medio che aveva *prima* dell'inizio della corsa,
    per la stessa stazione e la stessa fascia di 3 ore del giorno (temperatura e punto di rugiada: differenza;
    vento: rapporto). Non usa mai dati successivi alla corsa: e' una vera previsione corretta."""
    storia = {}      # (id, fascia_oraria, var) -> [(valido, previsto, osservato)]
    for corsa, sid, valido, ore, vals in campioni:
        o = oss.get((sid, valido))
        if not o:
            continue
        b = int(valido[11:13]) // 3
        for v in ("T", "Td", "V"):
            p, ob = vals[v], num(o.get(v))
            if p is not None and ob is not None:
                storia.setdefault((sid, b, v), []).append((valido, p, ob))
                storia.setdefault((None, b, v), []).append((valido, p, ob))
        # pioggia: conto solo le ore in cui almeno uno dei due (modello o stazione) ha pioggia
        p, ob = vals.get("P"), num(o.get("P"))
        if p is not None and ob is not None and (p > 0.05 or ob > 0.05):
            storia.setdefault((sid, "P"), []).append((valido, p, ob))
            storia.setdefault((None, "P"), []).append((valido, p, ob))
    for lista in storia.values():
        lista.sort()
    out = []
    for corsa, sid, valido, ore, vals in campioni:
        if solo_corse is not None and corsa not in solo_corse:
            continue                     # la storia usa tutte le corse, ma correggo solo quelle richieste
        inizio = datetime.strptime(corsa, "%Y%m%d%H").replace(tzinfo=timezone.utc)
        fine = ora_iso(inizio)
        da = ora_iso(inizio - timedelta(days=giorni))
        b = int(valido[11:13]) // 3
        nuovi = dict(vals)
        for v in ("T", "Td", "V"):
            if vals[v] is None:
                continue
            for chiave, minimo in (((sid, b, v), min_staz), ((None, b, v), min_tot)):
                prec = [x for x in storia.get(chiave, []) if da <= x[0] < fine]
                if len(prec) >= minimo:
                    if v == "V":
                        sp = sum(x[1] for x in prec)
                        fattore = min(1.5, max(0.3, sum(x[2] for x in prec) / sp)) if sp > 0 else 1.0
                        nuovi[v] = vals[v] * fattore
                    else:
                        nuovi[v] = vals[v] - sum(x[1] - x[2] for x in prec) / len(prec)
                    break
        # pioggia: rapporto tra pioggia osservata e prevista nelle due settimane prima della corsa
        # (prima la stazione, se ha abbastanza ore piovose, altrimenti tutte le stazioni insieme)
        if vals.get("P") is not None:
            for chiave, minimo in (((sid, "P"), MIN_ORE_PIOGGIA_STAZ), ((None, "P"), MIN_ORE_PIOGGIA_TOT)):
                prec = [x for x in storia.get(chiave, []) if da <= x[0] < fine]
                sp = sum(x[1] for x in prec)
                if len(prec) >= minimo and sp >= 2.0:
                    fattore = min(2.5, max(0.3, sum(x[2] for x in prec) / sp))
                    nuovi["P"] = vals["P"] * fattore
                    break
        out.append((corsa, sid, valido, ore, nuovi))
    return out


MIN_ORE_PIOGGIA_STAZ = 8      # ore piovose (modello o stazione) per usare il fattore della singola stazione
MIN_ORE_PIOGGIA_TOT = 30      # ... per usare quello di tutte le stazioni insieme
SCALA_ERRORE = {"T": 1.5, "Td": 2.0, "V": 2.5, "P": 0.6}   # errore "tipico" di ogni grandezza: le rende confrontabili


def punteggi_wrf(campioni_per_modello, oss):
    """Per ogni WRF: lista ordinata di (valido, errore normalizzato) su tutte le grandezze, pioggia compresa."""
    out = {}
    for m, camp in campioni_per_modello.items():
        lista = []
        for corsa, sid, valido, ore, vals in camp:
            o = oss.get((sid, valido))
            if not o or sid == "NEROLA" or ore >= 49:
                continue
            e, n = 0.0, 0
            for v, sc in SCALA_ERRORE.items():
                p, ob = vals.get(v), num(o.get(v))
                if p is not None and ob is not None:
                    e += abs(p - ob) / sc
                    n += 1
            if n:
                lista.append((valido, e / n))
        lista.sort()
        out[m] = lista
    return out


def wrf_migliore_per_corsa(campioni_per_modello, oss, giorni=14, minimo=200):
    """Per ogni corsa sceglie il WRF con l'errore medio piu' basso nelle due settimane PRIMA della corsa
    (mai dati futuri). Ritorna {corsa: modello} e il punteggio attuale di ogni WRF."""
    import bisect
    pt = punteggi_wrf(campioni_per_modello, oss)
    chiavi = {m: [x[0] for x in l] for m, l in pt.items()}
    corse = sorted({c[0] for camp in campioni_per_modello.values() for c in camp})
    scelta = {}
    def media(m, da, a):
        i, j = bisect.bisect_left(chiavi[m], da), bisect.bisect_left(chiavi[m], a)
        sel = pt[m][i:j]
        return (sum(x[1] for x in sel) / len(sel), len(sel)) if sel else (None, 0)
    corse_di = {m: {c[0] for c in camp} for m, camp in campioni_per_modello.items()}
    for corsa in corse:
        inizio = datetime.strptime(corsa, "%Y%m%d%H").replace(tzinfo=timezone.utc)
        da, a = ora_iso(inizio - timedelta(days=giorni)), ora_iso(inizio)
        voti = {m: media(m, da, a) for m in pt if corsa in corse_di.get(m, ())}
        validi = {m: x for m, (x, n) in voti.items() if x is not None and n >= minimo}
        if validi:
            scelta[corsa] = min(validi, key=validi.get)
        elif corsa in corse_di.get("wrf1", ()):
            scelta[corsa] = "wrf1"
    adesso = ora_iso(datetime.now(timezone.utc) + timedelta(hours=1))
    punteggio = {}
    for m in pt:
        x, n = media(m, ora_iso(datetime.now(timezone.utc) - timedelta(days=giorni)), adesso)
        punteggio[m] = {"errore": None if x is None else round(x, 3), "n": n}
    return scelta, punteggio


def _coppie(campioni_per_modello, oss, solo=None):
    """var -> fascia -> modello -> {(id, valido): [(previsto, osservato)]}; per P: somme su 6 ore (finestre 00-06-12-18 UTC)."""
    cop = {v: {f: {} for f in NOMI_FASCE} for v in ("T", "Td", "V", "P6")}
    for m, camp in campioni_per_modello.items():
        serie = {}
        for snap, sid, valido, ore, vals in camp:
            if solo and sid != solo:
                continue
            o = oss.get((sid, valido))
            if not o:
                continue
            f = fascia_di(ore)
            for v in ("T", "Td", "V"):
                p, ob = vals[v], num(o.get(v))
                if f and p is not None and ob is not None:
                    cop[v][f].setdefault(m, {}).setdefault((sid, valido), []).append((p, ob))
            if vals["P"] is not None and num(o.get("P")) is not None:
                serie.setdefault((snap, sid), {})[valido] = (ore, vals["P"], num(o["P"]))
        for (snap, sid), s in serie.items():
            for valido, (ore, _, _) in s.items():
                if valido[11:13] not in ("00", "06", "12", "18"):
                    continue
                fine = datetime.strptime(valido, "%Y-%m-%dT%H:00Z").replace(tzinfo=timezone.utc)
                ore_finestra = [ora_iso(fine - timedelta(hours=k)) for k in range(6)]
                if not all(h in s for h in ore_finestra):
                    continue
                f = fascia_di(ore)
                if f:
                    cop["P6"][f].setdefault(m, {}).setdefault((sid, valido), []).append(
                        (sum(s[h][1] for h in ore_finestra), sum(s[h][2] for h in ore_finestra)))
    return cop


def _stat(coppie, pioggia=False):
    if not coppie:
        return None
    p = np.array([c[0] for c in coppie])
    o = np.array([c[1] for c in coppie])
    e = p - o
    r = {"mae": round(float(np.mean(np.abs(e))), 2), "bias": round(float(np.mean(e)), 2), "n": int(len(e))}
    if pioggia:
        a = int(np.sum((p >= SOGLIA_PIOGGIA) & (o >= SOGLIA_PIOGGIA)))
        b = int(np.sum((p >= SOGLIA_PIOGGIA) & (o < SOGLIA_PIOGGIA)))
        c = int(np.sum((p < SOGLIA_PIOGGIA) & (o >= SOGLIA_PIOGGIA)))
        r.update({"eventi": a + c, "pod": round(a / (a + c), 2) if a + c else None, "far": round(b / (a + b), 2) if a + b else None,
                  "csi": round(a / (a + b + c), 2) if a + b + c else None})
    return r


def riepilogo(campioni_per_modello, oss, solo=None):
    """var -> fascia -> modello -> statistiche, calcolate sulle stesse stazioni e ore per tutti i modelli (confronto equo)."""
    cop = _coppie(campioni_per_modello, oss, solo)
    out = {}
    for v, per_fascia in cop.items():
        out[v] = {}
        for f, per_modello in per_fascia.items():
            presenti = [m for m, d in per_modello.items() if d]
            comuni = None
            for m in presenti:
                k = set(per_modello[m])
                comuni = k if comuni is None else comuni & k
            out[v][f] = {}
            for m in presenti:
                lista = [c for k in (comuni or ()) for c in per_modello[m][k]]
                st = _stat(lista, v == "P6")
                if st:
                    out[v][f][m] = st
    return out


def confronto_wrf(camp, oss, adesso):
    """Classifica di TUTTI i WRF contro le stazioni (prime 48 ore) e ultime previsioni per Nerola."""
    reg = registro_wrf()
    wrf = {m: c for m, c in camp.items() if m in reg and c}
    classifica = {}
    for m, c in wrf.items():
        cop = _coppie({m: c}, {k: v for k, v in oss.items() if k[0] != "NEROLA" or m == "wrf333"})
        voce = {}
        for v in ("T", "V", "P6"):
            lista = [x for f in NOMI_FASCE for d in cop[v][f].get(m, {}).values() for x in d]
            st = _stat(lista, v == "P6")
            if st:
                voce[v] = st
        voce["corse"] = len({x[0] for x in c})
        voce["ultima"] = max(x[0] for x in c)
        classifica[m] = voce
    # Nerola: ultima previsione di ogni WRF, da 24 ore fa in avanti
    t0 = ora_iso(adesso - timedelta(hours=24))
    ultimo = {}
    for m, c in wrf.items():
        u = {}
        for corsa, sid, valido, ore, vals in c:
            if sid == "NEROLA" and valido >= t0 and (valido not in u or corsa > u[valido][0]):
                u[valido] = (corsa, vals)
        if u:
            ultimo[m] = u
    tempi = sorted({t for u in ultimo.values() for t in u} | {k[1] for k in oss if k[0] == "NEROLA" and k[1] >= t0})
    nerola = {"tempi": tempi,
              "oss": {v: [pulisci(num(oss[("NEROLA", t)][v])) if ("NEROLA", t) in oss else None for t in tempi] for v in ("T", "V", "P")},
              "prev": {m: {v: [pulisci(u[t][1][v]) if t in u else None for t in tempi] for v in ("T", "V", "P")} for m, u in ultimo.items()}}
    return {"modelli": {m: {"nome": reg[m][2], "nota": reg[m][3]} for m in reg},
            "classifica": classifica, "nerola": nerola}


def errori_orari(campioni, oss):
    """Errore medio del WRF per fascia di 3 ore (UTC), solo previsioni entro 48 ore."""
    acc = {v: [[] for _ in range(8)] for v in ("T", "Td", "V")}
    for corsa, sid, valido, ore, vals in campioni:
        o = oss.get((sid, valido))
        if not o or ore >= 49:
            continue
        for v in acc:
            p, ob = vals[v], num(o.get(v))
            if p is not None and ob is not None:
                acc[v][int(valido[11:13]) // 3].append((p, ob))
    return {v: [_stat(c) for c in lista] for v, lista in acc.items()}


def calcola_pagella(oss_righe, prev_righe, snap_righe, vicine_nerola, giorni=GIORNI_FINESTRA, extra_wrf=None):
    oss = {(r["id"], r["ora"]): r for r in oss_righe}
    stazioni = {}
    for r in oss_righe:
        stazioni[r["id"]] = {"id": r["id"], "nome": r["nome"], "lat": float(r["lat"]), "lon": float(r["lon"]), "quota": num(r["quota"])}
    adesso = datetime.now(timezone.utc)
    limite = ora_iso(adesso - timedelta(days=giorni))
    camp = campioni_wrf(prev_righe, limite)
    camp.update(campioni_om(snap_righe, limite))
    camp.update(extra_wrf or {})
    confronto = confronto_wrf(camp, oss, adesso)
    # WRF corretto: per ogni corsa parte dal WRF che nelle due settimane precedenti ha sbagliato meno
    # (escluso il 333 m: copre solo i dintorni di Nerola, il suo punteggio non e' confrontabile)
    candidati = {m: camp[m] for m in camp if m.startswith("wrf") and m != "wrf333" and camp[m]}
    scelta, punteggio_wrf = wrf_migliore_per_corsa(candidati, oss)
    corretti = []
    for m, lista in candidati.items():
        corretti += correggi_wrf(lista, oss, solo_corse={c for c, b in scelta.items() if b == m})
    camp["wrfc"] = corretti
    base_attuale = scelta[max(scelta)] if scelta else "wrf1"
    ordine = ["wrf1", "wrf3", "wrfc", "icon2i", "arome", "ecmwf", "ukmo", "iconeu"]
    camp = {m: camp[m] for m in ordine if camp.get(m)}
    riep = {"tutte": riepilogo(camp, {k: v for k, v in oss.items() if k[0] != "NEROLA"}),
            "nerola": riepilogo(camp, oss, solo="NEROLA")}
    # tabelle dell'errore del WRF per ora del giorno
    stima = {"1km": errori_orari(camp.get("wrf1", []), oss), "3km": errori_orari(camp.get("wrf3", []), oss)}
    # serie recenti: osservato e ultima previsione WRF disponibile per ogni ora
    ultime = {"d01": {}, "d02": {}}
    for r in prev_righe:
        if r["valido"] < limite:
            continue
        for dom in ultime:
            k = (r["id"], r["valido"])
            if k not in ultime[dom] or r["corsa"] > ultime[dom][k]["corsa"]:
                if num(r.get(f"T_{dom}")) is not None:
                    ultime[dom][k] = r
    # errori per stazione (prime 24 ore di previsione)
    per_stazione = {}
    for m, dom in (("wrf1", "1km"), ("wrf3", "3km")):
        for corsa, sid, valido, ore, vals in camp.get(m, []):
            o = oss.get((sid, valido))
            if not o or ore >= 24 or sid == "NEROLA":
                continue
            for v in ("T", "Td", "V"):
                p, ob = vals[v], num(o.get(v))
                if p is not None and ob is not None:
                    per_stazione.setdefault(sid, {}).setdefault(dom, {}).setdefault(v, []).append((p, ob))
    quota_modello = {}
    for r in prev_righe:
        q = quota_modello.setdefault(r["id"], {})
        for dom, suff in (("1km", "d02"), ("3km", "d01")):
            if dom not in q and num(r.get(f"h_{suff}")) is not None:
                q[dom] = num(r[f"h_{suff}"])
    elenco = []
    for sid, st in sorted(stazioni.items()):
        voce = dict(st)
        voce["dist_km"] = round(distanza_nerola(st["lat"], st["lon"]), 1)
        voce["quota_modello"] = quota_modello.get(sid, {})
        voce["errori"] = {dom: {v: _stat(c) for v, c in vv.items()} for dom, vv in per_stazione.get(sid, {}).items()}
        ore_serie = sorted({k[1] for k in ultime["d01"] if k[0] == sid} | {k[1] for k in ultime["d02"] if k[0] == sid}
                           | {k[1] for k in oss if k[0] == sid and k[1] >= limite})
        voce["serie"] = {
            "tempi": ore_serie,
            "oss": {v: [pulisci(num(oss[(sid, t)][v])) if (sid, t) in oss else None for t in ore_serie] for v in VARS},
            "wrf1": {v: [num(ultime["d02"][(sid, t)].get(f"{v}_d02")) if (sid, t) in ultime["d02"] else None for t in ore_serie] for v in VARS},
            "wrf3": {v: [num(ultime["d01"][(sid, t)].get(f"{v}_d01")) if (sid, t) in ultime["d01"] else None for t in ore_serie] for v in VARS},
        }
        if sid != "NEROLA" and (voce["errori"] or any(x is not None for x in voce["serie"]["oss"]["T"])):
            elenco.append(voce)
    elenco.sort(key=lambda v: v["dist_km"])
    for v in elenco[40:]:     # il grafico serve solo per le 40 stazioni piu' vicine a Nerola: il file resta leggero
        v["serie"] = {"tempi": [], "oss": {}, "wrf1": {}, "wrf3": {}}
    # Nerola: osservazione "virtuale" e ultime previsioni di tutti i modelli (da 24 ore fa a +72 ore)
    t0 = ora_iso(adesso - timedelta(hours=24))
    tempi = sorted({k[1] for k in oss if k[0] == "NEROLA" and k[1] >= t0} | {k[1] for k in ultime["d02"] if k[0] == "NEROLA" and k[1] >= t0}
                   | {c[2] for m in MODELLI_OM for c in camp.get(m, []) if c[1] == "NEROLA" and c[2] >= t0})
    ner = {"tempi": tempi, "vicine": vicine_nerola,
           "oss": {v: [pulisci(num(oss[("NEROLA", t)][v])) if ("NEROLA", t) in oss else None for t in tempi] for v in VARS}, "prev": {}}
    ner["prev"]["wrf1"] = {v: [num(ultime["d02"][("NEROLA", t)].get(f"{v}_d02")) if ("NEROLA", t) in ultime["d02"] else None for t in tempi] for v in VARS}
    ner["prev"]["wrf3"] = {v: [num(ultime["d01"][("NEROLA", t)].get(f"{v}_d01")) if ("NEROLA", t) in ultime["d01"] else None for t in tempi] for v in VARS}
    for m in list(MODELLI_OM) + ["wrfc"]:
        ultimo = {}
        for snap, sid, valido, ore, vals in camp.get(m, []):
            if sid == "NEROLA" and (valido not in ultimo or snap > ultimo[valido][0]):
                ultimo[valido] = (snap, vals)
        ner["prev"][m] = {v: [pulisci(ultimo[t][1][v]) if t in ultimo else None for t in tempi] for v in VARS}
    return {
        "aggiornato": adesso.strftime("%Y-%m-%dT%H:%MZ"),
        "giorni": giorni,
        "fonte": "Osservazioni: aeroporti (METAR, NOAA) e reti della Protezione Civile Lazio/Umbria e MeteoNetwork (MeteoHub, ItaliaMeteo)",
        "modelli": {m: {"nome": n, "nota": t} for m, (n, t) in NOMI_MODELLI.items()},
        "riepilogo": riep,
        "errori_orari": stima,
        "nerola": ner,
        "stazioni": elenco,
        "confronto_wrf": confronto,
        "wrf_corretto": {"base": base_attuale, "nome_base": registro_wrf().get(base_attuale, (0, 0, base_attuale))[2],
                         "punteggi": punteggio_wrf},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--storico", required=True, help="cartella dello storico (osservazioni, previsioni, pagella.json)")
    ap.add_argument("--fisica", default="", help="cartella con le varianti di fisica (ramo 'fisica')")
    ap.add_argument("--uscite", default="", help="cartella con d01_tutto.npz e d02_tutto.npz dell'ultima corsa")
    ap.add_argument("--corsa", default="", help="AAAAMMGGHH della corsa in --uscite")
    ap.add_argument("--ore", type=int, default=72, help="ore di osservazioni da scaricare")
    ap.add_argument("--stazione", default="", help="cartella con nerola.csv (dati della stazione di Nerola dal sito)")
    ap.add_argument("--altri-modelli", action="store_true", help="scarica anche le previsioni degli altri modelli (Open-Meteo)")
    a = ap.parse_args()
    cart = Path(a.storico)
    cart.mkdir(parents=True, exist_ok=True)
    try:
        nuove = leggi_metar(a.ore)
        print("METAR scaricati:", len(nuove), "stazioni:", len({r['id'] for r in nuove}))
    except Exception as e:  # noqa: BLE001
        print("ATTENZIONE: METAR non scaricati:", e, file=sys.stderr)
        nuove = []
    try:
        nuove += leggi_meteohub(a.ore)
    except Exception as e:  # noqa: BLE001
        print("ATTENZIONE: MeteoHub non scaricato:", e, file=sys.stderr)
    if a.stazione:
        try:
            ner = leggi_stazione_nerola(a.stazione, a.ore)
            print("stazione di Nerola (dal sito): ore", len(ner))
            nuove += ner
        except Exception as e:  # noqa: BLE001
            print("ATTENZIONE: stazione di Nerola non letta:", e, file=sys.stderr)
    oss = unisci_osservazioni(cart / "osservazioni.csv", nuove)
    stazioni = {r["id"]: {"id": r["id"], "lat": float(r["lat"]), "lon": float(r["lon"])} for r in oss}
    if any(r["id"] == "NEROLA" for r in oss):
        nerola, vicine = [], ["stazione"]       # c'e' la stazione vera: niente media delle vicine
    else:
        nerola, vicine = osservazioni_nerola(oss)
    print("Nerola: osservazione da", vicine, "ore con osservazione", len(nerola) or "(stazione)")
    oss_tutte = oss + nerola
    stazioni["NEROLA"] = {"id": "NEROLA", "lat": NEROLA[0], "lon": NEROLA[1]}
    if a.uscite and a.corsa:
        righe = estrai_previsione(a.uscite, a.corsa, stazioni)
        print("previsioni estratte:", len(righe))
        if righe:
            salva_previsione(cart, a.corsa, righe)
    # tutte le altre catene WRF (GFS, ICON-2I, ICON-EU, 333 m) e le varianti di fisica
    if a.uscite:
        for sub, (sotto, file_dom) in CATENE_EXTRA.items():
            try:
                c = json.loads((Path(a.uscite) / sub / "indice.json").read_text())["corsa"]
                r_ = estrai_previsione(Path(a.uscite) / sub, c, stazioni, file_dom)
                if r_:
                    salva_previsione(cart, c, r_, sotto, tuple(file_dom))
                print(f"WRF {sub}: corsa {c}, righe {len(r_)}")
            except Exception as e:  # noqa: BLE001
                print(f"ATTENZIONE: WRF {sub} non letto: {e}", file=sys.stderr)
    if a.fisica:
        try:
            ind = json.loads((Path(a.fisica) / "indice.json").read_text())
            for nome in ind.get("file", []):
                k, f = nome[:-4].split("_")
                c = ind["corse"].get(k)
                if not c:
                    continue
                r_ = estrai_previsione(a.fisica, c, stazioni, {"d01": nome})
                if r_:
                    salva_previsione(cart, c, r_, f"prev_fis_{k}_{f}", ("d01",))
                print(f"variante {k} {f}: corsa {c}, righe {len(r_)}")
        except Exception as e:  # noqa: BLE001
            print("ATTENZIONE: varianti di fisica non lette:", e, file=sys.stderr)
    if a.altri_modelli and not ultimo_snapshot_vecchio(cart):
        try:
            adesso = datetime.now(timezone.utc)
            righe = scarica_altri_modelli(scegli_stazioni_om(stazioni), adesso)
            print("altri modelli: valori scaricati", len(righe))
            if righe:
                salva_snapshot(cart, adesso, righe)
        except Exception as e:  # noqa: BLE001
            print("ATTENZIONE: altri modelli non scaricati:", e, file=sys.stderr)
    limite = ora_iso(datetime.now(timezone.utc) - timedelta(days=GIORNI_FINESTRA))
    extra = campioni_tutti_wrf(cart, GIORNI_FINESTRA + 3, limite)
    print("altri WRF in pagella:", {m: len(c) for m, c in extra.items()})
    pag = calcola_pagella(oss_tutte, leggi_previsioni(cart, GIORNI_FINESTRA + 3), leggi_snapshot(cart, GIORNI_FINESTRA + 3), vicine,
                          extra_wrf=extra)
    (cart / "pagella.json").write_text(json.dumps(pag, ensure_ascii=False, separators=(",", ":"), default=pulisci))
    print("pagella scritta:", len(pag["stazioni"]), "stazioni")


if __name__ == "__main__":
    main()
