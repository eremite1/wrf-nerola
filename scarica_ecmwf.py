#!/usr/bin/env python3
"""Scarica i dati aperti ECMWF IFS a 0,1 gradi (licenza CC BY 4.0) che servono a WRF.

  scarica_ecmwf.py --trova --ore 48                 stampa l'ultima corsa completa (AAAAMMGGHH)
  scarica_ecmwf.py --corsa 2026102400 --ore 48 --out grezzi

Per ogni passo (0, 3, 6 ... ore) scrive grezzi/sfc_<ora>.grib2 e grezzi/pl_<ora>.grib2.
Dai file dell'ECMWF (circa 440 MB a passo, tutto il mondo) si prendono, con richieste a intervalli,
solo i campi che servono: la meta' o meno dei dati. Se l'indice non si capisce, si scarica il file intero.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

BASI = [
    "https://data.ecmwf.int/forecasts",
    "https://storage.googleapis.com/ecmwf-open-data",
    "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com",
]
# campi usati da WPS (vedi Vtable.ECMWF01)
CAMPI = {
    "pl": {"gh", "t", "u", "v", "r"},
    "sfc": {"2t", "2d", "10u", "10v", "msl", "sp", "skt", "lsm", "z", "sot", "vsw"},
}
MINIMO = {"pl": 30, "sfc": 8}      # sotto questo numero di voci l'indice non e' quello atteso


def nome(corsa, ora, tipo):
    g, h = corsa[:8], corsa[8:]
    return f"{g}/{h}z/ifs/0p1/oper/{g}{h}0000-{ora}h-oper-fc-{tipo}"


def esiste(corsa, ora, tipo):
    for base in BASI[:2]:
        try:
            if requests.head(f"{base}/{nome(corsa, ora, tipo)}.index", timeout=30).status_code == 200:
                return True
        except requests.RequestException:
            pass
    return False


def corsa_completa(corsa, ore):
    return esiste(corsa, ore, "sfc") and esiste(corsa, ore, "pl")


def trova(ore):
    """L'ultima corsa il cui ultimo passo e' gia' pubblicato (00 e 12 UTC; 06 e 18 solo fino a 90 ore)."""
    adesso = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    c = adesso.replace(hour=(adesso.hour // 6) * 6)
    for _ in range(16):                       # fino a 4 giorni indietro
        h = c.hour
        if h in (0, 12) or ore <= 90:
            t = c.strftime("%Y%m%d%H")
            if corsa_completa(t, ore):
                return t
        c -= timedelta(hours=6)
    sys.exit("Nessuna corsa ECMWF completa trovata negli ultimi giorni.")


def prendi(percorso, intervallo=None, tentativi=6):
    ultimo = None
    for i in range(tentativi):
        base = BASI[i % len(BASI)]
        try:
            h = {"Range": f"bytes={intervallo[0]}-{intervallo[1]}"} if intervallo else {}
            r = requests.get(f"{base}/{percorso}", headers=h, timeout=180)
            if r.status_code in (200, 206):
                atteso = intervallo[1] - intervallo[0] + 1 if intervallo else None
                if atteso is not None and len(r.content) != atteso:
                    raise IOError(f"{len(r.content)} byte invece di {atteso}")
                return r.content
            ultimo = f"{base}: HTTP {r.status_code}"
        except (requests.RequestException, IOError) as e:
            ultimo = f"{base}: {e}"
        print(f"{percorso.rsplit('/', 1)[-1]}: {ultimo}; riprovo ({i + 1}/{tentativi})", file=sys.stderr, flush=True)
        time.sleep(5 * (i + 1))
    sys.exit(f"Impossibile scaricare {percorso} ({ultimo})")


def intervalli(voci):
    """Unisce le voci contigue in una richiesta sola."""
    voci = sorted(voci, key=lambda r: r["_offset"])
    out = []
    for r in voci:
        a, b = r["_offset"], r["_offset"] + r["_length"] - 1
        if out and a == out[-1][1] + 1:
            out[-1][1] = b
        else:
            out.append([a, b])
    return out


def scarica(corsa, ora, tipo, destinazione, completo):
    base = nome(corsa, ora, tipo)
    dati = None
    if not completo:
        try:
            righe = [json.loads(r) for r in prendi(base + ".index").decode().splitlines() if r.strip()]
            voci = [r for r in righe if r.get("param") in CAMPI[tipo]]
            if len(voci) >= MINIMO[tipo] and all("_offset" in r and "_length" in r for r in voci):
                dati = b"".join(prendi(base + ".grib2", (a, b)) for a, b in intervalli(voci))
                stato = f"{len(voci)} campi"
            else:
                print(f"+{ora}h {tipo}: indice inatteso ({len(voci)} voci scelte): scarico il file intero", flush=True)
        except (ValueError, KeyError, TypeError) as e:
            print(f"+{ora}h {tipo}: indice illeggibile ({e}): scarico il file intero", flush=True)
    if dati is None:
        dati = prendi(base + ".grib2")
        stato = "file intero"
    if not dati.startswith(b"GRIB"):
        sys.exit(f"+{ora}h {tipo}: il file scaricato non e' un GRIB")
    destinazione.write_bytes(dati)
    print(f"ok +{ora:02d}h {tipo}: {stato}, {len(dati) / 1e6:.1f} MB", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corsa", default="", help="AAAAMMGGHH; vuoto = l'ultima completa")
    ap.add_argument("--ore", type=int, default=24)
    ap.add_argument("--out", default="grezzi")
    ap.add_argument("--trova", action="store_true", help="stampa solo la corsa scelta")
    ap.add_argument("--completi", action="store_true", help="scarica i file interi (piu' lento)")
    a = ap.parse_args()
    if a.ore % 3:
        sys.exit("--ore deve essere multiplo di 3")
    if a.corsa:
        if not corsa_completa(a.corsa, a.ore):
            sys.exit(f"La corsa {a.corsa} non e' (piu') disponibile fino a +{a.ore} h.")
        corsa = a.corsa
    else:
        corsa = trova(a.ore)
    if a.trova:
        print(corsa)
        return
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lavori = [(ora, tipo) for ora in range(0, a.ore + 1, 3) for tipo in ("sfc", "pl")]
    with ThreadPoolExecutor(4) as ex:
        for f in [ex.submit(scarica, corsa, ora, tipo, out / f"{tipo}_{ora}.grib2", a.completi) for ora, tipo in lavori]:
            f.result()
    print("scaricati", len(lavori), "file")


if __name__ == "__main__":
    main()
