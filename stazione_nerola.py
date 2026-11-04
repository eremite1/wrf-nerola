#!/usr/bin/env python3
"""Legge i dati della stazione meteo di Nerola pubblicati nella home di meteonerola.it e li aggiunge a uno storico.

La home contiene: i valori di adesso (temperatura, umidita', vento, pressione, pioggia di oggi, punto di rugiada)
e la temperatura delle ultime 24 ore ogni 5 minuti. Il programma li salva in nerola.csv, che la pagella usa come
"osservazione di Nerola".

Uso:  stazione_nerola.py --cartella CARTELLA   (legge/scrive CARTELLA/nerola.csv)
"""
import argparse
import csv
import json
import re
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

URL = "https://www.meteonerola.it/"
CAMPI = ["ts", "T", "Td", "V", "raffica", "RH", "pressione", "pioggia_oggi"]


def scarica(url, tentativi=4):
    ultimo = None
    for n in range(tentativi):
        try:
            rich = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (wrf-nerola)"})
            with urllib.request.urlopen(rich, timeout=60) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:  # noqa: BLE001
            ultimo = e
            time.sleep(5 * (n + 1))
    raise RuntimeError(f"download fallito: {ultimo}")


def numero(testo):
    m = re.search(r"-?\d+(?:[.,]\d+)?", testo or "")
    return float(m.group(0).replace(",", ".")) if m else None


def valore(html, etichetta):
    m = re.search(r"<dt>\s*" + etichetta + r"\s*</dt>\s*<dd>(.*?)</dd>", html, re.S)
    if not m:
        return None
    testo = re.sub(r"<small>.*?</small>", "", m.group(1), flags=re.S)
    return numero(re.sub(r"<[^>]*>", "", testo))


def leggi_home(html):
    """-> (riga con i valori di adesso o None, [(ts, temperatura)] delle ultime 24 ore)"""
    ts = None
    m = re.search(r'data-stampo="(\d+)-(\d+)"', html)
    if m:
        ts = int(m.group(1))
    serie = []
    m = re.search(r'id="dati-24h">(.*?)</script>', html, re.S)
    if m:
        serie = [(int(a) // 1000, float(b)) for a, b in json.loads(m.group(1)) if b is not None]
    riga = None
    mt = re.search(r'class="ora__temp"[^>]*>\s*(-?\d+(?:[.,]\d+)?)', html)
    if ts and mt:
        riga = {"ts": ts, "T": numero(mt.group(1)), "Td": valore(html, "Punto di rugiada"), "V": valore(html, "Vento"),
                "raffica": valore(html, "Raffica"), "RH": valore(html, "Umidit&agrave;|Umidità"), "pressione": valore(html, "Pressione"),
                "pioggia_oggi": valore(html, "Pioggia oggi")}
    return riga, serie


def iso(ts):
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:00Z")


def aggiorna(cartella, html):
    riga, serie = leggi_home(html)
    p = Path(cartella) / "nerola.csv"
    dati = {}
    if p.exists():
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                dati[r["ts"]] = r
    for ts, t in serie:
        r = dati.setdefault(iso(ts), {c: "" for c in CAMPI})
        r["ts"] = iso(ts)
        if r["T"] == "":
            r["T"] = t
    if riga:
        r = dati.setdefault(iso(riga["ts"]), {c: "" for c in CAMPI})
        for c in CAMPI:
            v = iso(riga["ts"]) if c == "ts" else riga[c]
            r[c] = "" if v is None else v
    limite = iso(int((datetime.now(timezone.utc) - timedelta(days=60)).timestamp()))
    righe = sorted((r for r in dati.values() if r["ts"] >= limite), key=lambda r: r["ts"])
    Path(cartella).mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, CAMPI)
        w.writeheader()
        w.writerows(righe)
    return len(serie), riga


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cartella", required=True)
    a = ap.parse_args()
    n, riga = aggiorna(a.cartella, scarica(URL))
    print("temperature ultime 24 ore:", n, "· valori di adesso:", riga)
    if not riga and not n:
        sys.exit("nessun dato letto dalla home")


if __name__ == "__main__":
    main()
