#!/usr/bin/env python3
"""Dai file grandi di WRF (wrfout_*) tiene solo i campi che servono e li scrive in piccoli file .npz.

  postprocess.py --run CARTELLA_DEL_RUN --blocco 1 --out CARTELLA_USCITA

Scrive d01_b1.npz e d02_b1.npz. Il vento e' ruotato verso nord/est (WRF lo da' lungo gli assi della griglia).
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

CAMPI = ["T2", "Q2", "PSFC", "RAINC", "RAINNC", "SNOWNC", "SWDOWN", "PBLH", "HFX", "LH", "TSK", "GLW"]
# campi a strati: (nome in WRF, nome nel file, quali strati tenere). Suolo: 4 strati (0-10, 10-40, 40-100, 100-200 cm);
# QCLOUD (acqua liquida delle nubi) solo il livello piu' basso del modello, serve per la nebbia.
STRATI = [("TSLB", "TSLB", slice(None)), ("SMOIS", "SMOIS", slice(None)), ("QCLOUD", "QCLOUD1", 0)]


def istante(nc):
    riga = np.ma.getdata(nc["Times"][0])
    return riga.tobytes().decode()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--blocco", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for dom in ("d01", "d02"):
        files = sorted(glob.glob(f"{a.run}/wrfout_{dom}_*"))
        if not files:
            sys.exit(f"nessun file wrfout_{dom}_* in {a.run}")
        tempi, dati, stat = [], {}, {}
        for f in files:
            nc = Dataset(f)
            tempi.append(istante(nc))
            if not stat:
                stat = {"lat": np.array(nc["XLAT"][0]), "lon": np.array(nc["XLONG"][0]), "hgt": np.array(nc["HGT"][0])}
                if "LANDMASK" in nc.variables:
                    stat["landmask"] = np.array(nc["LANDMASK"][0], dtype="float32")
            for k in CAMPI:
                if k in nc.variables:
                    dati.setdefault(k, []).append(np.array(nc[k][0], dtype="float32"))
            for nome, nome_file, quali in STRATI:
                if nome in nc.variables:
                    dati.setdefault(nome_file, []).append(np.array(nc[nome][0][quali], dtype="float32"))
            c, s = np.array(nc["COSALPHA"][0]), np.array(nc["SINALPHA"][0])
            u, v = np.array(nc["U10"][0]), np.array(nc["V10"][0])
            dati.setdefault("U10", []).append((u * c - v * s).astype("float32"))
            dati.setdefault("V10", []).append((v * c + u * s).astype("float32"))
            nc.close()
        np.savez_compressed(out / f"{dom}_b{a.blocco}.npz", tempi=np.array(tempi), **stat,
                            **{k: np.stack(v) for k, v in dati.items()})
        print(dom, "istanti:", len(tempi), tempi[0], "->", tempi[-1], "campi:", sorted(dati))


if __name__ == "__main__":
    main()
