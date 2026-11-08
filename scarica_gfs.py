#!/usr/bin/env python3
"""Scarica i dati GFS 0,25 gradi (NOAA, pubblico dominio) per il WRF, gia' ritagliati sull'Italia centrale.

  scarica_gfs.py --trova --ore 48                 stampa l'ultima corsa (AAAAMMGGHH) con tutte le ore disponibili
  scarica_gfs.py --corsa 2026110800 --ore 48 --out CARTELLA

Usa il servizio di ritaglio di NOMADS (filter_gfs_0p25.pl): per ogni scadenza (ogni 3 ore) un file GRIB2 con
tutti i livelli di pressione, il suolo e la superficie, solo nel riquadro che serve. Pochi MB per file.
"""
import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

BASE = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
ARCHIVIO = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/"
RIQUADRO = dict(leftlon=7, rightlon=19, toplat=47, bottomlat=37)
# variabili (Vtable.GFS) e livelli necessari a WPS
VARIABILI = ["HGT", "TMP", "RH", "UGRD", "VGRD", "PRES", "PRMSL", "MSLET", "SOILW", "TSOIL", "LAND", "ICEC",
             "WEASD", "SNOD", "SPFH"]
LIVELLI = ["all_lev"]
UA = {"User-Agent": "MeteoNerola-WRF/1.0"}


def url_file(corsa, ora):
    g, h = corsa[:8], corsa[8:10]
    q = {"dir": f"/gfs.{g}/{h}/atmos", "file": f"gfs.t{h}z.pgrb2.0p25.f{ora:03d}", **RIQUADRO, "subregion": ""}
    for v in VARIABILI:
        q[f"var_{v}"] = "on"
    for l in LIVELLI:
        q[l] = "on"
    return BASE, q


def esiste(corsa, ora):
    g, h = corsa[:8], corsa[8:10]
    u = f"{ARCHIVIO}gfs.{g}/{h}/atmos/gfs.t{h}z.pgrb2.0p25.f{ora:03d}.idx"
    try:
        r = requests.head(u, timeout=30, headers=UA)
        return r.status_code == 200
    except Exception:
        return False


def trova(ore, corsa=None):
    if corsa:
        return corsa if esiste(corsa, ore) else None
    t = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    t -= timedelta(hours=t.hour % 6)
    for _ in range(8):
        c = t.strftime("%Y%m%d%H")
        if esiste(c, ore) and esiste(c, 0):
            return c
        t -= timedelta(hours=6)
    return None


def scarica(corsa, ore, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for ora in range(0, ore + 1, 3):
        dest = out / f"gfs_{corsa}_f{ora:03d}.grib2"
        u, q = url_file(corsa, ora)
        for k in range(6):
            try:
                r = requests.get(u, params=q, timeout=180, headers=UA)
                if r.status_code == 200 and r.content[:4] == b"GRIB" and len(r.content) > 100000:
                    dest.write_bytes(r.content)
                    print(f"f{ora:03d}: {len(r.content) / 1e6:.1f} MB", flush=True)
                    break
                print(f"f{ora:03d}: risposta {r.status_code} ({len(r.content)} byte), riprovo", flush=True)
            except Exception as e:
                print(f"f{ora:03d}: {e}, riprovo", flush=True)
            time.sleep(10 * (k + 1))
        else:
            sys.exit(f"ERRORE: impossibile scaricare la scadenza f{ora:03d}")
        time.sleep(0.6)          # NOMADS accetta al massimo circa 120 richieste al minuto


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trova", action="store_true")
    ap.add_argument("--corsa")
    ap.add_argument("--ore", type=int, required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.trova:
        c = trova(a.ore, a.corsa)
        if not c:
            sys.exit("ERRORE: nessuna corsa GFS completa trovata")
        print(c)
        return
    scarica(a.corsa, a.ore, a.out)


if __name__ == "__main__":
    main()
