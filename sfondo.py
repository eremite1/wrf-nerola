#!/usr/bin/env python3
"""Confini, costa, laghi, fiumi e nomi dei paesi da disegnare sotto le mappe.

  sfondo.py --scarica CARTELLA     scarica i dati (una volta) e ne fa un piccolo file sfondo.json
  (da mappe.py)  carica(CARTELLA)  legge sfondo.json; se manca, ritorna uno sfondo vuoto

Fonti (gratuite):
  - Natural Earth (pubblico dominio): province, costa, laghi, fiumi
  - GeoNames (CC BY 4.0): nomi e posizioni dei paesi
Se il download non riesce non e' un errore grave: le mappe escono lo stesso, senza confini.
"""
import argparse
import io
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

# riquadro tenuto nel file piccolo (piu' largo dei domini): lon_min, lon_max, lat_min, lat_max
RIQUADRO = (10.0, 15.5, 40.5, 43.8)
NE = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"
STRATI_NE = {
    "province": "ne_10m_admin_1_states_provinces.geojson",
    "costa": "ne_10m_coastline.geojson",
    "laghi": "ne_10m_lakes.geojson",
    "fiumi": "ne_10m_rivers_lake_centerlines.geojson",
}
GEONAMES = "https://download.geonames.org/export/dump/IT.zip"
# sedi di comune (PPLA3), capoluoghi di provincia (PPLA2) e di regione (PPLA); poi i paesi piu' piccoli (PPL) se hanno abitanti
CODICI_SEDE = {"PPLA", "PPLA2", "PPLA3", "PPLC"}


def scarica(url, tentativi=4):
    ultimo = None
    for n in range(tentativi):
        try:
            rich = urllib.request.Request(url, headers={"User-Agent": "wrf-nerola"})
            with urllib.request.urlopen(rich, timeout=180) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            ultimo = e
            time.sleep(5 * (n + 1))
    raise RuntimeError(f"download fallito: {url}: {ultimo}")


def _dentro(x, y):
    return RIQUADRO[0] <= x <= RIQUADRO[1] and RIQUADRO[2] <= y <= RIQUADRO[3]


def _linee(geom, solo_bordo=True):
    """Ritorna una lista di linee [[(lon, lat), ...], ...] da una geometria GeoJSON (linee o bordi di poligoni)."""
    t, c = geom.get("type"), geom.get("coordinates")
    if t == "LineString":
        return [c]
    if t == "MultiLineString":
        return list(c)
    if t == "Polygon":
        return list(c) if solo_bordo else [c[0]]
    if t == "MultiPolygon":
        return [anello for poligono in c for anello in (poligono if solo_bordo else poligono[:1])]
    return []


def _taglia(linea):
    """Tiene solo la parte di linea dentro il riquadro, arrotondando a 4 decimali (circa 10 m) per fare file piccoli."""
    out, corrente = [], []
    for x, y, *_ in linea:
        if RIQUADRO[0] - 0.3 <= x <= RIQUADRO[1] + 0.3 and RIQUADRO[2] - 0.3 <= y <= RIQUADRO[3] + 0.3:
            corrente.append((round(x, 4), round(y, 4)))
        elif len(corrente) > 1:
            out.append(corrente)
            corrente = []
        else:
            corrente = []
    if len(corrente) > 1:
        out.append(corrente)
    return out


def strato_ne(nome, file):
    dati = json.loads(scarica(NE + file))
    risultato = []
    for f in dati["features"]:
        g = f.get("geometry")
        p = f.get("properties") or {}
        if not g:
            continue
        if nome == "province" and (p.get("iso_a2") or p.get("ISO_A2")) != "IT":
            continue  # si tengono anche le province vicine fuori Italia? No: qui interessa l'Italia
        for linea in _linee(g, solo_bordo=True):
            risultato.extend(_taglia(linea))
    return risultato


def paesi():
    z = zipfile.ZipFile(io.BytesIO(scarica(GEONAMES)))
    out = []
    with z.open("IT.txt") as f:
        for riga in io.TextIOWrapper(f, encoding="utf-8"):
            c = riga.rstrip("\n").split("\t")
            if len(c) < 15 or c[6] != "P":
                continue
            lat, lon = float(c[4]), float(c[5])
            if not _dentro(lon, lat):
                continue
            pop = int(c[14] or 0)
            sede = c[7] in CODICI_SEDE
            if not sede and pop < 800:
                continue
            out.append({"nome": c[1], "lat": lat, "lon": lon, "pop": pop, "sede": sede})
    # i comuni (sedi) per primi e poi per numero di abitanti
    out.sort(key=lambda p: (not p["sede"], -p["pop"]))
    return out


def costruisci(cartella):
    cartella = Path(cartella)
    cartella.mkdir(parents=True, exist_ok=True)
    risultato, errori = {}, []
    for nome, file in STRATI_NE.items():
        try:
            risultato[nome] = strato_ne(nome, file)
            print(f"{nome}: {len(risultato[nome])} linee")
        except Exception as e:  # noqa: BLE001
            errori.append(f"{nome}: {e}")
            risultato[nome] = []
    try:
        risultato["paesi"] = paesi()
        print("paesi:", len(risultato["paesi"]))
    except Exception as e:  # noqa: BLE001
        errori.append(f"paesi: {e}")
        risultato["paesi"] = []
    risultato["errori"] = errori
    # se non e' arrivato nulla non lo salvo come "buono": la cache non va riempita con un file vuoto
    utile = any(risultato[k] for k in ("province", "costa", "laghi", "fiumi", "paesi"))
    if utile and not errori:
        (cartella / "sfondo.json").write_text(json.dumps(risultato, separators=(",", ":")))
    elif utile:
        (cartella / "sfondo_parziale.json").write_text(json.dumps(risultato, separators=(",", ":")))
    for e in errori:
        print("ATTENZIONE:", e, file=sys.stderr)
    return utile and not errori


def carica(cartella):
    """Sfondo pronto all'uso; se non c'e' (o e' solo parziale) usa quello che c'e', altrimenti uno vuoto."""
    vuoto = {"province": [], "costa": [], "laghi": [], "fiumi": [], "paesi": [], "errori": ["sfondo non disponibile"]}
    for nome in ("sfondo.json", "sfondo_parziale.json"):
        p = Path(cartella) / nome
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:  # noqa: BLE001
                pass
    return vuoto


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scarica", required=True, help="cartella dove mettere sfondo.json")
    a = ap.parse_args()
    ok = costruisci(a.scarica)
    print("sfondo completo" if ok else "sfondo incompleto o assente: le mappe usciranno con meno dettagli")


if __name__ == "__main__":
    main()
