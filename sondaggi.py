"""Radiosondaggi virtuali per la pagina web (pagina/sondaggi.html).

Legge i file {dominio}_sond_b*.npz scritti da postprocess.py (profili sui livelli di pressione, ogni 3 ore) e
aggiunge il livello del suolo dai campi di superficie (pressione al suolo, temperatura e umidita' a 2 m, vento a 10 m).
Scrive un solo file binario per dominio, ordinato per punto della griglia, cosi' la pagina scarica solo i pochi
kilobyte del punto cliccato (richiesta "Range", come per il meteogramma):

  dati/<dominio>/sondaggi.bin   interi a 16 bit (little endian), [punto, istante, valore]
  per ogni istante: 6 valori del suolo   pressione (decimi di hPa), quota (m), temperatura, rugiada, vento est,
                                         vento nord (decimi di grado e di m/s)
                    poi, per ogni livello di pressione, 5 valori: quota (m), temperatura, rugiada, vento est,
                                         vento nord (decimi). -32768 = nessun dato (livello sotto il suolo)
"""
import glob
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

MANCANTE = -32768
N_SUOLO = 6
N_LIVELLO = 5


def _carica(cartella, dom):
    file = sorted(glob.glob(f"{cartella}/{dom}_sond_b*.npz"))
    if not file:
        return None
    per_tempo, livelli = {}, None
    for f in file:
        z = np.load(f, allow_pickle=False)
        if livelli is None:
            livelli = [int(x) for x in z["livelli"]]
        elif [int(x) for x in z["livelli"]] != livelli:
            raise ValueError("livelli diversi tra i blocchi")
        for i, t in enumerate(z["tempi"]):
            per_tempo[str(t)] = np.stack([z[k][i] for k in ("SZ", "ST", "STD", "SU", "SV")], axis=1)  # [livello, 5, y, x]
    return livelli, per_tempo


def _q(campo, scala):
    v = np.round(np.asarray(campo, dtype="float64") * scala)
    return np.where(np.isfinite(v), np.clip(v, -32767, 32767), MANCANTE).astype("int16")


def _suolo(d, i):
    """Valori al suolo all'istante i del dizionario delle uscite: [6, y, x]."""
    p = d["PSFC"][i] / 100.0
    q = np.maximum(d["Q2"][i], 1e-6)
    e = np.maximum(q * p / (0.622 + q), 1e-3)
    td = 243.5 * np.log(e / 6.112) / (17.67 - np.log(e / 6.112))
    t = d["T2"][i] - 273.15
    td = np.minimum(td, t)
    return np.stack([_q(p, 10), _q(d["hgt"] + 2.0, 1), _q(t, 10), _q(td, 10), _q(d["U10"][i], 10), _q(d["V10"][i], 10)])


def scrivi(cartella_uscite, dom, d, out, cartella_dom, precedente=""):
    """Scrive dati/<cartella_dom>/sondaggi.bin e ritorna la voce per indice.json (None se non ci sono dati).
    d: le uscite gia' caricate (uscite.carica). Se mancano i profili (per esempio rigenerando solo le mappe) e
    l'indice precedente e' della stessa corsa, ricopia il file gia' pubblicato."""
    dest = Path(out) / "dati" / cartella_dom
    dest.mkdir(parents=True, exist_ok=True)
    letti = _carica(cartella_uscite, dom)
    if letti is None:
        return _ricopia(precedente, cartella_dom, dest)
    livelli, per_tempo = letti
    indice_t = {str(t): i for i, t in enumerate(d["tempi"])}
    tempi = [t for t in sorted(per_tempo) if t in indice_t]
    if not tempi:
        return None
    blocchi = []
    for t in tempi:
        suolo = _suolo(d, indice_t[t])                                   # [6, y, x]
        prof = per_tempo[t].reshape(len(livelli) * N_LIVELLO, *suolo.shape[1:])
        blocchi.append(np.concatenate([suolo, prof]).reshape(suolo.shape[0] + prof.shape[0], -1))  # [valori, punti]
    tutto = np.stack(blocchi)                                            # [istante, valore, punto]
    np.ascontiguousarray(tutto.transpose(2, 0, 1)).astype("<i2").tofile(dest / "sondaggi.bin")
    print(dom, "radiosondaggi virtuali:", len(tempi), "istanti,", len(livelli), "livelli,",
          round((dest / "sondaggi.bin").stat().st_size / 1e6, 1), "MB")
    return {"file": f"dati/{cartella_dom}/sondaggi.bin", "livelli": livelli, "n_suolo": N_SUOLO,
            "n_livello": N_LIVELLO,
            "tempi": [datetime.strptime(t[:16], "%Y-%m-%d_%H:%M").strftime("%Y-%m-%dT%H:%MZ") for t in tempi]}


def _ricopia(precedente, cartella_dom, dest):
    if not precedente or not Path(precedente).exists():
        return None
    try:
        vecchio = json.loads(Path(precedente).read_text())
        voce = vecchio["domini"][cartella_dom]["sondaggi"]
        sorgente = Path(precedente).parent / voce["file"]
        if sorgente.exists():
            shutil.copyfile(sorgente, dest / "sondaggi.bin")
            print(cartella_dom, "radiosondaggi virtuali: ricopiati quelli gia' pubblicati")
            return voce
    except Exception:  # noqa: BLE001
        pass
    return None
