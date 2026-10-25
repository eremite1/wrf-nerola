"""Legge e unisce i file delle uscite (uno per blocco di previsione) scritti da postprocess.py."""
import glob

import numpy as np


def carica(cartella, dominio):
    """Ritorna un dizionario con tempi (ordinati, senza doppioni), lat, lon, hgt e i campi [tempo, y, x].
    Se un istante compare in due blocchi (la ripresa lo riscrive) vale l'ultimo."""
    file = sorted(glob.glob(f"{cartella}/{dominio}_b*.npz"))
    if not file:
        return None
    per_tempo = {}
    stat = None
    for f in file:
        z = np.load(f, allow_pickle=False)
        if stat is None:
            stat = {k: z[k] for k in ("lat", "lon", "hgt", "landmask") if k in z.files}
        campi = [k for k in z.files if k not in ("tempi", "lat", "lon", "hgt", "landmask")]
        for i, t in enumerate(z["tempi"]):
            per_tempo[str(t)] = {k: z[k][i] for k in campi}
    tempi = sorted(per_tempo)
    out = dict(stat)
    out["tempi"] = tempi
    for k in per_tempo[tempi[0]]:
        out[k] = np.stack([per_tempo[t][k] for t in tempi])
    return out


def pioggia(d):
    """Pioggia accumulata dall'inizio (mm), [tempo, y, x]."""
    return d["RAINC"] + d["RAINNC"]
