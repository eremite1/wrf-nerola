#!/usr/bin/env python3
"""Disegna le mappe (ogni 3 ore) e il meteogramma di Nerola dai risultati di WRF.

  mappe.py --uscite CARTELLA --corsa 2026102400 --out CARTELLA_MAPPE
"""
import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import uscite  # noqa: E402

ROMA = ZoneInfo("Europe/Rome")
CITTA = {"Nerola": (42.13, 12.77), "Roma": (41.90, 12.50), "Tivoli": (41.96, 12.80),
         "Monterotondo": (42.05, 12.62), "Rieti": (42.40, 12.86)}
NEROLA = CITTA["Nerola"]
FONTE = "WRF su dati ECMWF IFS 0,1 gradi (CC BY 4.0, dati modificati)"


def quando(t):
    u = datetime.strptime(t, "%Y-%m-%d_%H:%M:%S").replace(tzinfo=timezone.utc)
    return u, u.astimezone(ROMA)


def mappa(d, i, etichetta, corsa, cartella, lim):
    lon, lat, hgt = d["lon"], d["lat"], d["hgt"]
    t2 = d["T2"][i] - 273.15
    u, v = d["U10"][i], d["V10"][i]
    vento = np.hypot(u, v) * 3.6
    pioggia = uscite.pioggia(d)
    ultime = pioggia[i] - pioggia[max(i - 3, 0)]
    fig, ax = plt.subplots(1, 3, figsize=(19, 6.4))
    m0 = ax[0].pcolormesh(lon, lat, t2, cmap="coolwarm", vmin=lim["t"][0], vmax=lim["t"][1], shading="auto")
    ax[0].contour(lon, lat, hgt, levels=np.arange(200, 3000, 200), colors="k", linewidths=0.4, alpha=0.5)
    fig.colorbar(m0, ax=ax[0], label="Temperatura a 2 m (C)", shrink=0.8)
    ax[0].set_title("Temperatura a 2 m")
    m1 = ax[1].pcolormesh(lon, lat, vento, cmap="viridis", vmin=0, vmax=lim["w"], shading="auto")
    s = 4 if etichetta == "3 km" else 3
    ax[1].quiver(lon[::s, ::s], lat[::s, ::s], u[::s, ::s], v[::s, ::s], color="w", scale=120, width=0.003)
    fig.colorbar(m1, ax=ax[1], label="Vento a 10 m (km/h)", shrink=0.8)
    ax[1].set_title("Vento a 10 m")
    m2 = ax[2].pcolormesh(lon, lat, np.where(ultime < 0.1, np.nan, ultime), cmap="Blues", vmin=0, vmax=lim["r"], shading="auto")
    ax[2].contour(lon, lat, hgt, levels=np.arange(200, 3000, 400), colors="0.5", linewidths=0.4)
    fig.colorbar(m2, ax=ax[2], label="Pioggia nelle ultime 3 ore (mm)", shrink=0.8)
    ax[2].set_title("Pioggia (ultime 3 ore)")
    for a in ax:
        a.set_aspect(1.0 / np.cos(np.radians(42.1)))
        a.set_xlabel("Longitudine")
        a.set_ylabel("Latitudine")
        for nome, (la, lo) in CITTA.items():
            if lat.min() < la < lat.max() and lon.min() < lo < lon.max():
                a.plot([lo], [la], "k*" if nome == "Nerola" else "ko", markersize=11 if nome == "Nerola" else 4)
                a.annotate(nome, (lo, la), textcoords="offset points", xytext=(4, 4), fontsize=8)
    utc, loc = quando(d["tempi"][i])
    c = datetime.strptime(corsa, "%Y%m%d%H")
    fig.suptitle(f"Dominio {etichetta}  |  corsa ECMWF {c:%d/%m/%Y %H} UTC  |  valido {utc:%d/%m %H:%M} UTC ({loc:%H:%M} ora locale)",
                 fontsize=12)
    fig.text(0.5, 0.01, FONTE, ha="center", fontsize=8, color="0.4")
    nome_file = f"{cartella}/{utc:%Y%m%d_%H}.png"
    fig.savefig(nome_file, dpi=90, bbox_inches="tight")
    plt.close(fig)


def meteogramma(d, corsa, destinazione):
    lat, lon = d["lat"], d["lon"]
    j = np.unravel_index(np.argmin((lat - NEROLA[0]) ** 2 + ((lon - NEROLA[1]) * np.cos(np.radians(42.1))) ** 2), lat.shape)
    t = [quando(x)[1] for x in d["tempi"]]
    t2 = d["T2"][:, j[0], j[1]] - 273.15
    vento = np.hypot(d["U10"][:, j[0], j[1]], d["V10"][:, j[0], j[1]]) * 3.6
    pr = uscite.pioggia(d)[:, j[0], j[1]]
    orario = np.concatenate([[0.0], np.diff(pr)])
    p = d["PSFC"][:, j[0], j[1]] / 100.0
    fig, ax = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    ax[0].plot(t, t2, color="tab:red")
    ax[0].set_ylabel("Temperatura 2 m (C)")
    ax[1].plot(t, vento, color="tab:green")
    ax[1].set_ylabel("Vento 10 m (km/h)")
    ax[2].bar(t, orario, width=0.035, color="tab:blue")
    ax[2].set_ylabel("Pioggia oraria (mm)")
    ax[3].plot(t, p, color="tab:purple")
    ax[3].set_ylabel("Pressione al suolo (hPa)")
    for x in ax:
        x.grid(alpha=0.3)
    ax[3].xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %H", tz=ROMA))
    ax[3].xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 6), tz=ROMA))
    fig.autofmt_xdate()
    c = datetime.strptime(corsa, "%Y%m%d%H")
    fig.suptitle(f"Nerola (punto di griglia a {lat[j]:.3f} N, {lon[j]:.3f} E, quota modello {d['hgt'][j]:.0f} m)\n"
                 f"corsa ECMWF {c:%d/%m/%Y %H} UTC, orari in ora locale", fontsize=11)
    fig.text(0.5, 0.005, FONTE, ha="center", fontsize=8, color="0.4")
    fig.savefig(destinazione, dpi=100, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uscite", required=True)
    ap.add_argument("--corsa", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    for dom, etichetta in (("d02", "1 km"), ("d01", "3 km")):
        d = uscite.carica(a.uscite, dom)
        if d is None:
            sys.exit(f"mancano i risultati di {dom}")
        cart = out / ("mappe_1km" if dom == "d02" else "mappe_3km")
        cart.mkdir(parents=True, exist_ok=True)
        pr = uscite.pioggia(d)
        tre = np.stack([pr[i] - pr[max(i - 3, 0)] for i in range(len(pr))])
        lim = {"t": (float(np.floor((d["T2"] - 273.15).min())), float(np.ceil((d["T2"] - 273.15).max()))),
               "w": max(20.0, float(np.ceil(np.hypot(d["U10"], d["V10"]).max() * 3.6))),
               "r": max(2.0, float(np.ceil(tre.max())))}
        for i in range(0, len(d["tempi"]), 3):
            mappa(d, i, etichetta, a.corsa, str(cart), lim)
        print(dom, "mappe:", len(range(0, len(d["tempi"]), 3)))
        if dom == "d02":
            meteogramma(d, a.corsa, str(out / "meteogramma_nerola.png"))
            np.savez_compressed(out / "d02_tutto.npz", **{k: v for k, v in d.items() if k != "tempi"}, tempi=np.array(d["tempi"]))
        else:
            np.savez_compressed(out / "d01_tutto.npz", **{k: v for k, v in d.items() if k != "tempi"}, tempi=np.array(d["tempi"]))


if __name__ == "__main__":
    main()
