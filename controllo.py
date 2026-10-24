#!/usr/bin/env python3
"""Controlla che i risultati siano plausibili prima di pubblicarli. Esce con 1 se qualcosa non va.

  controllo.py --uscite CARTELLA --ore 48 [--rapporto controllo.txt]
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import uscite  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uscite", required=True)
    ap.add_argument("--ore", type=int, required=True)
    ap.add_argument("--rapporto", default="")
    a = ap.parse_args()
    righe, errori = [], []

    def ok(msg):
        righe.append("ok     " + msg)

    def ko(msg):
        righe.append("ERRORE " + msg)
        errori.append(msg)

    for dom in ("d01", "d02"):
        d = uscite.carica(a.uscite, dom)
        if d is None:
            ko(f"{dom}: nessun file di risultati")
            continue
        n = len(d["tempi"])
        (ok if n == a.ore + 1 else ko)(f"{dom}: istanti {n} (attesi {a.ore + 1}), da {d['tempi'][0]} a {d['tempi'][-1]}")
        for k in ("T2", "PSFC", "U10", "V10", "RAINC", "RAINNC", "SWDOWN"):
            if k in d and not np.isfinite(d[k]).all():
                ko(f"{dom}: {k} contiene valori non numerici")
        t2 = d["T2"] - 273.15
        (ok if -35 <= t2.min() and t2.max() <= 48 else ko)(f"{dom}: temperatura a 2 m tra {t2.min():.1f} e {t2.max():.1f} C (accettati -35..48)")
        vento = np.hypot(d["U10"], d["V10"])
        (ok if vento.max() < 70 else ko)(f"{dom}: vento a 10 m massimo {vento.max():.1f} m/s (accettato < 70)")
        p = d["PSFC"] / 100.0
        (ok if 500 <= p.min() and p.max() <= 1100 else ko)(f"{dom}: pressione al suolo tra {p.min():.0f} e {p.max():.0f} hPa (accettati 500..1100)")
        pioggia = uscite.pioggia(d)
        orario = np.diff(pioggia, axis=0) if len(pioggia) > 1 else np.zeros(1)
        (ok if orario.min() > -0.01 and orario.max() < 150 and pioggia.max() < 800 else ko)(
            f"{dom}: pioggia oraria massima {orario.max():.1f} mm, totale massimo {pioggia.max():.0f} mm (accettati < 150 e < 800)")
        if "SWDOWN" in d:
            (ok if -1 <= d["SWDOWN"].min() and d["SWDOWN"].max() <= 1200 else ko)(
                f"{dom}: radiazione solare tra {d['SWDOWN'].min():.0f} e {d['SWDOWN'].max():.0f} W/m2 (accettati 0..1200)")
        salto = np.abs(np.diff(t2, axis=0)).max() if len(t2) > 1 else 0.0
        (ok if salto < 15 else ko)(f"{dom}: massimo salto di temperatura in un'ora {salto:.1f} C (accettato < 15)")
    righe.append("RISULTATO: " + ("NON PLAUSIBILE, non si pubblica" if errori else "plausibile"))
    testo = "\n".join(righe)
    print(testo)
    if a.rapporto:
        open(a.rapporto, "w").write(testo + "\n")
    sys.exit(1 if errori else 0)


if __name__ == "__main__":
    main()
