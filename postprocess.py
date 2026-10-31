#!/usr/bin/env python3
"""Dai file grandi di WRF (wrfout_*) tiene solo i campi che servono e li scrive in piccoli file .npz.

  postprocess.py --run CARTELLA_DEL_RUN --blocco 1 --out CARTELLA_USCITA

Scrive d01_b1.npz e d02_b1.npz. Il vento e' ruotato verso nord/est (WRF lo da' lungo gli assi della griglia).
Dalla colonna d'aria (tutti i livelli) calcola anche il bulbo umido e gli indici per il tipo di precipitazione
(TW2, AFUS, ARIG, ZTW0: vedi colonna.py), usati dal nowcasting per distinguere neve, mista, pioggia e gelicidio.
Ogni 3 ore scrive anche d01_sond_b1.npz e d02_sond_b1.npz: il radiosondaggio virtuale di ogni punto della griglia
(quota, temperatura, punto di rugiada e vento sui livelli di pressione, vedi colonna.profili_pressione).
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import colonna  # noqa: E402

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
        tempi_sond, sond = [], {}          # radiosondaggi virtuali: solo ogni 3 ore (file a parte, vedi colonna.py)
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
            # colonna d'aria: bulbo umido a tutte le quote e indici per neve / pioggia / gelicidio (vedi colonna.py)
            if all(v in nc.variables for v in ("P", "PB", "T", "QVAPOR", "PH", "PHB")):
                try:
                    for k, v in colonna.da_wrfout(nc).items():
                        dati.setdefault(k, []).append(v)
                    if "QCLOUD" in nc.variables:
                        for k, v in colonna.nuvolosita_strati(nc).items():
                            dati.setdefault(k, []).append(v.astype("float32"))
                    if all(x in nc.variables for x in ("U", "V", "W", "COSALPHA", "SINALPHA")):
                        for k, v in colonna.parametri_nowcasting(nc).items():
                            dati.setdefault(k, []).append(v)
                except Exception as e:  # noqa: BLE001
                    print("ATTENZIONE: colonna d'aria non calcolata:", e, file=sys.stderr)
            if int(tempi[-1][11:13]) % 3 == 0 and all(x in nc.variables for x in
                                                      ("P", "PB", "T", "QVAPOR", "PH", "PHB", "U", "V", "COSALPHA", "SINALPHA")):
                try:
                    prof = colonna.profili_pressione(nc)
                    for k, v in prof.items():
                        sond.setdefault(k, []).append(v)
                    tempi_sond.append(tempi[-1])
                except Exception as e:  # noqa: BLE001
                    print("ATTENZIONE: radiosondaggio virtuale non calcolato:", e, file=sys.stderr)
            c, s = np.array(nc["COSALPHA"][0]), np.array(nc["SINALPHA"][0])
            u, v = np.array(nc["U10"][0]), np.array(nc["V10"][0])
            dati.setdefault("U10", []).append((u * c - v * s).astype("float32"))
            dati.setdefault("V10", []).append((v * c + u * s).astype("float32"))
            nc.close()
        np.savez_compressed(out / f"{dom}_b{a.blocco}.npz", tempi=np.array(tempi), **stat,
                            **{k: np.stack(v) for k, v in dati.items()})
        print(dom, "istanti:", len(tempi), tempi[0], "->", tempi[-1], "campi:", sorted(dati))
        if tempi_sond and all(len(v) == len(tempi_sond) for v in sond.values()):
            np.savez_compressed(out / f"{dom}_sond_b{a.blocco}.npz", tempi=np.array(tempi_sond),
                                livelli=np.array(colonna.LIVELLI_SONDAGGIO, dtype="int16"),
                                **{k: np.stack(v) for k, v in sond.items()})
            print(dom, "radiosondaggi virtuali:", len(tempi_sond), "istanti")


if __name__ == "__main__":
    main()
