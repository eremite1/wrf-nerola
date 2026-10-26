#!/usr/bin/env python3
"""Disegna le mappe (un parametro per immagine) e il meteogramma di Nerola dai risultati di WRF.

  mappe.py --uscite CARTELLA --corsa 2026102400 --out CARTELLA_PUBBLICA [--sfondo CARTELLA_SFONDO]

Scrive:
  mappe/<dominio>/<parametro>/AAAAMMGG_HH.png   (ora UTC nel nome del file)
  indice.json                                   (elenco di tutte le mappe, per le pagine web)
  meteogramma_nerola.png, d01_tutto.npz, d02_tutto.npz
"""
import argparse
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.patheffects  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sfondo as sfondo_mod  # noqa: E402
import uscite  # noqa: E402

ROMA = ZoneInfo("Europe/Rome")
NEROLA = (42.13, 12.77)
FONTE = "WRF su dati ECMWF IFS 0,1 gradi (CC BY 4.0, dati modificati)"
DOMINI = (("d02", "1km", "1 km"), ("d01", "3km", "3 km"))
# paesi sempre scritti se dentro la mappa, anche se sul sito di GeoNames mancassero
PAESI_FISSI = {"Nerola": NEROLA}


# ---------------------------------------------------------------- calcoli sui campi

def rh2(d):
    t = d["T2"] - 273.15
    p = d["PSFC"] / 100.0
    q = d["Q2"]
    e = q * p / (0.622 + q)
    es = 6.112 * np.exp(17.67 * t / (t + 243.5))
    return np.clip(100.0 * e / es, 0, 100)


def pressione_mare(d):
    h = d["hgt"][None]
    return d["PSFC"] / 100.0 * np.exp(9.80665 * h / (287.05 * (d["T2"] + 0.0065 * h / 2)))


def visibilita(d):
    """Visibilita' stimata (metri) dall'acqua liquida del primo livello del modello (formula di Kunkel)."""
    rho = d["PSFC"] / (287.05 * d["T2"])
    lwc = np.maximum(d["QCLOUD1"] * rho * 1000.0, 0)  # g/m3
    with np.errstate(divide="ignore"):
        vis = np.where(lwc > 1e-4, 1000.0 * 0.027 / np.maximum(lwc, 1e-4) ** 0.88, 99999.0)
    return np.minimum(vis, 99999.0)


def finestra(a, i, n, f):
    return f(a[i - n:i + 1], axis=0)


# ---------------------------------------------------------------- definizione dei parametri
# Parametri a classi (colori a gradini): "bordi" sono le soglie, i colori e le etichette sono uno per classe
# (len(bordi)+1); le classi in "nascondi" non si colorano (sono "niente da segnalare").
BLU = ["#bfe3f7", "#7fc4ea", "#3f9be0", "#2a6cc7", "#2f9e44", "#f2c230", "#f08a24", "#d63a2f", "#b0207a", "#6a1b9a"]


def classi_pioggia(soglie):
    etichette = [f"< {soglie[0]:g}"] + [f"{a:g} - {b:g}" for a, b in zip(soglie[:-1], soglie[1:])] + [f"> {soglie[-1]:g}"]
    return {"bordi": soglie, "colori": ["#ffffff"] + BLU[:len(soglie)], "etichette": etichette, "nascondi": {0}}


PIOGGIA = {
    1: [0.1, 0.5, 1, 2, 4, 8, 15, 30],
    3: [0.1, 0.5, 1, 2, 5, 10, 20, 40],
    6: [0.2, 1, 2, 5, 10, 20, 40, 60],
    12: [0.2, 1, 5, 10, 20, 40, 60, 100],
    24: [0.2, 1, 5, 10, 20, 40, 80, 120],
    0: [0.2, 1, 5, 10, 20, 40, 80, 120],
}


def p_pioggia(n):
    def f(d, i):
        pr = uscite.pioggia(d)
        return pr[i] - (pr[i - n] if n else pr[0])
    titolo = "Pioggia totale dall'inizio" if n == 0 else f"Pioggia nelle ultime {n} {'ora' if n == 1 else 'ore'}"
    return {"id": "pioggia_totale" if n == 0 else f"pioggia_{n}h", "titolo": titolo, "unita": "mm", "calcola": f,
            "classi": classi_pioggia(PIOGGIA[n]), "passo": 1 if n == 1 else 3, "inizio": max(n, 1) if n else 3,
            "nota": "WRF, pioggia convettiva e a grande scala"}


PARAMETRI = [
    {"id": "temperatura", "titolo": "Temperatura a 2 m", "unita": "°C", "calcola": lambda d, i: d["T2"][i] - 273.15,
     "cmap": "RdYlBu_r", "limiti": "auto", "passo": 3, "inizio": 0, "quote": True},
    {"id": "vento", "titolo": "Vento a 10 m", "unita": "km/h", "calcola": lambda d, i: np.hypot(d["U10"][i], d["V10"][i]) * 3.6,
     "cmap": "viridis", "limiti": (0, 60), "passo": 3, "inizio": 0, "frecce": True},
    {"id": "umidita", "titolo": "Umidità relativa a 2 m", "unita": "%", "calcola": lambda d, i: rh2(d)[i],
     "cmap": "YlGnBu", "limiti": (20, 100), "passo": 3, "inizio": 0},
    {"id": "pressione", "titolo": "Pressione ridotta al livello del mare", "unita": "hPa",
     "calcola": lambda d, i: pressione_mare(d)[i], "cmap": "Spectral_r", "limiti": "auto", "passo": 3, "inizio": 0},
    p_pioggia(1), p_pioggia(3), p_pioggia(6), p_pioggia(12), p_pioggia(24), p_pioggia(0),
    {"id": "tmin24", "titolo": "Temperatura minima nelle ultime 24 ore", "unita": "°C",
     "calcola": lambda d, i: finestra(d["T2"], i, 24, np.min) - 273.15, "cmap": "RdYlBu_r", "limiti": "auto",
     "passo": 3, "inizio": 24},
    {"id": "tmax24", "titolo": "Temperatura massima nelle ultime 24 ore", "unita": "°C",
     "calcola": lambda d, i: finestra(d["T2"], i, 24, np.max) - 273.15, "cmap": "RdYlBu_r", "limiti": "auto",
     "passo": 3, "inizio": 24},
    {"id": "gelata", "titolo": "Rischio di gelata (minima delle ultime 24 ore)", "unita": "",
     "calcola": lambda d, i: finestra(d["T2"], i, 24, np.min) - 273.15, "passo": 3, "inizio": 24,
     "classi": {"bordi": [-2, 0, 2], "colori": ["#5b2a86", "#2a6cc7", "#8fd0f0", "#ffffff"],
                "etichette": ["gelata forte (< -2 °C)", "gelata (-2..0 °C)", "rischio brina (0..2 °C)", "nessun rischio"],
                "nascondi": {3}},
     "nota": "temperatura a 2 m; sul prato di notte fa 2-3 °C meno"},
    {"id": "nebbia", "titolo": "Nebbia: visibilità stimata", "unita": "",
     "calcola": lambda d, i: visibilita(d)[i], "passo": 3, "inizio": 3, "serve": "QCLOUD1",
     "classi": {"bordi": [200, 500, 1000, 5000], "colori": ["#4b4b4b", "#7d7d7d", "#b0b0b0", "#dcdcdc", "#ffffff"],
                "etichette": ["< 200 m", "200-500 m", "500-1000 m", "1-5 km", "> 5 km"], "nascondi": {4}},
     "nota": "stimata dall'acqua delle nubi nel primo livello del modello; nelle prime ore il modello parte senza nubi"},
    {"id": "suolo_temp", "titolo": "Temperatura del suolo (0-10 cm)", "unita": "°C",
     "calcola": lambda d, i: d["TSLB"][i, 0] - 273.15, "cmap": "RdYlBu_r", "limiti": "auto", "passo": 3, "inizio": 0,
     "serve": "TSLB", "terra": True},
    {"id": "suolo_umidita_10", "titolo": "Umidità del suolo (0-10 cm)", "unita": "% volume",
     "calcola": lambda d, i: d["SMOIS"][i, 0] * 100, "cmap": "YlGnBu", "limiti": (5, 50), "passo": 3, "inizio": 0,
     "serve": "SMOIS", "terra": True},
    {"id": "suolo_umidita_40", "titolo": "Umidità del suolo (10-40 cm)", "unita": "% volume",
     "calcola": lambda d, i: d["SMOIS"][i, 1] * 100, "cmap": "YlGnBu", "limiti": (5, 50), "passo": 3, "inizio": 0,
     "serve": "SMOIS", "terra": True},
]


# ---------------------------------------------------------------- disegno

def quando(t):
    u = datetime.strptime(t, "%Y-%m-%d_%H:%M:%S").replace(tzinfo=timezone.utc)
    return u, u.astimezone(ROMA)


def km(la1, lo1, la2, lo2):
    return np.hypot((la1 - la2) * 111.0, (lo1 - lo2) * 111.0 * np.cos(np.radians(la1)))


def scegli_paesi(sfondo, lat, lon, minimo_km, massimo, vicini=False):
    """Paesi dentro la mappa, i piu' grandi per primi, senza accavallamenti. Nerola c'e' sempre."""
    dentro = [(n, la, lo) for n, (la, lo) in PAESI_FISSI.items() if lat.min() < la < lat.max() and lon.min() < lo < lon.max()]
    scelti = list(dentro)
    elenco = sfondo.get("paesi", [])
    if vicini:  # dominio piccolo: prima i comuni piu' vicini a Nerola, cosi' compaiono anche i paesi piccoli attorno
        elenco = sorted(elenco, key=lambda p: (not p.get("sede"), km(p["lat"], p["lon"], *NEROLA)))
    for p in elenco:
        if p["nome"] == "Rome":
            p = dict(p, nome="Roma")
        if len(scelti) >= massimo:
            break
        if not (lat.min() + 0.02 < p["lat"] < lat.max() - 0.02 and lon.min() + 0.02 < p["lon"] < lon.max() - 0.03):
            continue
        if any(km(p["lat"], p["lon"], la, lo) < minimo_km for _, la, lo in scelti):
            continue
        if p["nome"] in {n for n, _, _ in scelti}:
            continue
        scelti.append((p["nome"], p["lat"], p["lon"]))
    return scelti


def disegna_sfondo(ax, sfondo):
    def linee(chiave, **stile):
        ls = sfondo.get(chiave) or []
        if ls:
            ax.add_collection(LineCollection(ls, **stile))
    linee("laghi", colors="#2a6cc7", linewidths=0.7)
    linee("fiumi", colors="#2a6cc7", linewidths=0.6, alpha=0.8)
    linee("province", colors="k", linewidths=0.7, linestyles="--", alpha=0.7)
    linee("costa", colors="k", linewidths=1.0)


FIG = (8.6, 6.4)   # pollici
DPI = 85
AREA_MAPPA = [0.085, 0.14, 0.60, 0.72]   # sinistra, basso, larghezza, altezza (frazioni della figura)
AREA_LEGENDA = [0.715, 0.22, 0.022, 0.56]

# aggiunte ai parametri: linee di livello, e come mostrare il valore quando si passa il puntatore sulla pagina web
EXTRA = {
    "temperatura": {"isolinee": True},
    "tmin24": {"isolinee": True},
    "tmax24": {"isolinee": True},
    "suolo_temp": {"isolinee": True},
    "pressione": {"isolinee": True},
    "gelata": {"unita_dato": "°C", "etichetta_dato": "Temperatura minima 24 h"},
    "nebbia": {"unita_dato": "m", "etichetta_dato": "Visibilità stimata", "scala": 1, "clip": 30000},
}
for _p in PARAMETRI:
    _p.pop("quote", None)  # niente curve di livello del terreno: sporcano la mappa
    _p.update(EXTRA.get(_p["id"], {}))


def passo_isolinee(vmin, vmax):
    for passo in (0.5, 1, 2, 4, 5, 10, 20):
        if (vmax - vmin) / passo <= 8:
            return passo
    return 20


def leviga(z, volte=2):
    """Media mobile 3x3: le linee di livello sono piu' pulite (a 1 km il campo e' molto irregolare)."""
    z = np.array(z, dtype="float64")
    for _ in range(volte):
        p = np.pad(z, 1, mode="edge")
        z = sum(p[a:a + z.shape[0], b:b + z.shape[1]] for a in range(3) for b in range(3)) / 9.0
    return z


def disegna(d, p, i, dom, etichetta, corsa, sfondo, paesi, cartella, lim):
    lon, lat, hgt = d["lon"], d["lat"], d["hgt"]
    z = np.array(p["calcola"](d, i), dtype="float64")
    if p.get("terra") and "landmask" in d:
        z = np.where(d["landmask"] > 0.5, z, np.nan)
    fig = plt.figure(figsize=FIG)
    ax = fig.add_axes(AREA_MAPPA)
    cax = fig.add_axes(AREA_LEGENDA)
    cl = p.get("classi")
    if cl:
        mostrate = [k for k in range(len(cl["colori"])) if k not in cl["nascondi"]]
        nuovo = {k: n for n, k in enumerate(mostrate)}  # le classi nascoste non compaiono nemmeno nella legenda
        classe = np.digitize(z, cl["bordi"]).astype("float64")
        classe = np.vectorize(lambda x: nuovo.get(int(x), np.nan), otypes=[float])(classe)
        classe[np.isnan(z)] = np.nan
        cmap = ListedColormap([cl["colori"][k] for k in mostrate])
        mappa_col = ax.pcolormesh(lon, lat, classe, cmap=cmap, vmin=-0.5, vmax=len(mostrate) - 0.5, shading="auto")
        cb = fig.colorbar(mappa_col, cax=cax)
        cb.set_ticks(range(len(mostrate)))
        cb.set_ticklabels([cl["etichette"][k] for k in mostrate])
        if p["unita"]:
            cb.set_label(p["unita"])
    else:
        vmin, vmax = lim[p["id"]] if p.get("limiti") == "auto" else p.get("limiti")
        mappa_col = ax.pcolormesh(lon, lat, z, cmap=p["cmap"], vmin=vmin, vmax=vmax, shading="auto")
        cb = fig.colorbar(mappa_col, cax=cax, extend="both" if p.get("limiti") != "auto" else "neither")
        cb.set_label(p["unita"])
        if p.get("isolinee") and p.get("limiti") == "auto":
            passo = passo_isolinee(vmin, vmax)
            livelli = np.arange(np.ceil(vmin / passo) * passo, vmax + 1e-9, passo)
            zs = leviga(np.nan_to_num(z, nan=np.nanmean(z)))
            if len(livelli) and np.nanmin(zs) < livelli.max() and np.nanmax(zs) > livelli.min():
                cs = ax.contour(lon, lat, zs, levels=livelli, colors="#1a1a1a", linewidths=0.7, alpha=0.75)
                testi = ax.clabel(cs, fmt=lambda v: f"{v:g}", fontsize=8, inline=True, inline_spacing=3)
                for t in testi:
                    t.set_path_effects([matplotlib.patheffects.withStroke(linewidth=1.8, foreground="white")])
    if p.get("frecce"):
        s = 4 if etichetta == "3 km" else 3
        ax.quiver(lon[::s, ::s], lat[::s, ::s], d["U10"][i][::s, ::s], d["V10"][i][::s, ::s], color="w", scale=120, width=0.003)
    disegna_sfondo(ax, sfondo)
    ax.set_xlim(lon.min(), lon.max())
    ax.set_ylim(lat.min(), lat.max())
    ax.set_aspect(1.0 / np.cos(np.radians(42.1)))
    ax.set_xlabel("Longitudine")
    ax.set_ylabel("Latitudine")
    for nome, la, lo in paesi:
        grande = nome == "Nerola"
        ax.plot([lo], [la], "k*" if grande else "ko", markersize=11 if grande else 3.5, markeredgecolor="w", markeredgewidth=0.6)
        t = ax.annotate(nome, (lo, la), textcoords="offset points", xytext=(4, 4), fontsize=8 if not grande else 9,
                        fontweight="bold" if grande else "normal")
        t.set_path_effects([matplotlib.patheffects.withStroke(linewidth=2, foreground="white")])
    utc, loc = quando(d["tempi"][i])
    c = datetime.strptime(corsa, "%Y%m%d%H")
    fig.suptitle(f"{p['titolo']}  ({etichetta})", fontsize=13, fontweight="bold", y=0.975)
    ax.set_title(f"valido {utc:%d/%m/%Y %H:%M} UTC ({loc:%H:%M} ora locale)  |  corsa ECMWF {c:%d/%m %H} UTC", fontsize=9)
    nota = p.get("nota", "")
    fig.text(0.5, 0.012, (nota + "\n" if nota else "") + FONTE, ha="center", va="bottom", fontsize=7.5, color="0.35")
    ax.apply_aspect()
    pos = ax.get_position()
    layout = {"asse": [round(pos.x0, 5), round(pos.y0, 5), round(pos.width, 5), round(pos.height, 5)],
              "pixel": [int(round(FIG[0] * DPI)), int(round(FIG[1] * DPI))]}
    nome_file = f"{cartella}/{utc:%Y%m%d_%H}.png"
    buf = io.BytesIO()
    fig.savefig(buf, dpi=DPI, format="png")
    plt.close(fig)
    buf.seek(0)
    # palette a 256 colori (ottimizzata): le immagini pesano un terzo e la legenda resta fedele
    Image.open(buf).convert("RGB").quantize(256, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE).save(nome_file, optimize=True)
    return f"{utc:%Y%m%d_%H}.png", utc, z, layout


def quantizza(z, scala=10, clip=None):
    """Numeri interi a 16 bit per la pagina web (-32768 = nessun dato)."""
    z = np.asarray(z, dtype="float64")
    if clip is not None:
        z = np.minimum(z, clip)
    q = np.round(z * scala)
    q = np.where(np.isfinite(q), np.clip(q, -32767, 32767), -32768)
    return q.astype("<i2")


def campi_serie(d):
    """Campi orari per il meteogramma di un punto qualsiasi: (id, titolo, unita, scala, array [t, y, x])."""
    pr = uscite.pioggia(d)
    elenco = [("T2", "Temperatura a 2 m", "°C", 10, d["T2"] - 273.15),
              ("UMID", "Umidità relativa a 2 m", "%", 10, rh2(d)),
              ("VENTO", "Vento a 10 m", "km/h", 10, np.hypot(d["U10"], d["V10"]) * 3.6),
              ("PIOGGIA", "Pioggia dall'inizio", "mm", 10, pr - pr[0]),
              ("PRESS", "Pressione al livello del mare", "hPa", 10, pressione_mare(d))]
    if "TSLB" in d:
        elenco.append(("TSUOLO", "Temperatura del suolo 0-10 cm", "°C", 10, d["TSLB"][:, 0] - 273.15))
    if "SMOIS" in d:
        elenco.append(("UMSUOLO", "Umidità del suolo 0-10 cm", "% vol", 10, d["SMOIS"][:, 0] * 100))
    return elenco


def limiti_automatici(d, parametri, frames):
    lim = {}
    for p in parametri:
        if p.get("limiti") != "auto" or p.get("classi"):
            continue
        vals = []
        for i in frames[p["id"]]:
            z = np.asarray(p["calcola"](d, i), dtype="float64")
            if p.get("terra") and "landmask" in d:
                z = np.where(d["landmask"] > 0.5, z, np.nan)
            vals.append((np.nanmin(z), np.nanmax(z)))
        if vals:
            lo = np.floor(min(v[0] for v in vals))
            hi = np.ceil(max(v[1] for v in vals))
            lim[p["id"]] = (float(lo), float(hi if hi > lo else lo + 1))
    return lim


def meteogramma(d, corsa, destinazione):
    lat, lon = d["lat"], d["lon"]
    j = np.unravel_index(np.argmin((lat - NEROLA[0]) ** 2 + ((lon - NEROLA[1]) * np.cos(np.radians(42.1))) ** 2), lat.shape)
    t = [quando(x)[1] for x in d["tempi"]]
    t2 = d["T2"][:, j[0], j[1]] - 273.15
    umid = rh2(d)[:, j[0], j[1]]
    vento = np.hypot(d["U10"][:, j[0], j[1]], d["V10"][:, j[0], j[1]]) * 3.6
    pr = uscite.pioggia(d)[:, j[0], j[1]]
    orario = np.concatenate([[0.0], np.diff(pr)])
    p = pressione_mare(d)[:, j[0], j[1]]
    righe = 6
    fig, ax = plt.subplots(righe, 1, figsize=(11, 2.0 * righe + 1), sharex=True)
    k = 0
    ax[k].plot(t, t2, color="tab:red", label="aria 2 m")
    if "TSLB" in d:
        ax[k].plot(t, d["TSLB"][:, 0, j[0], j[1]] - 273.15, color="saddlebrown", linestyle="--", label="suolo 0-10 cm")
    ax[k].axhline(0, color="0.6", linewidth=0.8)
    ax[k].legend(loc="upper right", fontsize=8)
    ax[k].set_ylabel("Temperatura (°C)")
    k += 1
    ax[k].plot(t, umid, color="tab:cyan")
    ax[k].set_ylabel("Umidità 2 m (%)")
    k += 1
    ax[k].plot(t, vento, color="tab:green")
    ax[k].set_ylabel("Vento 10 m (km/h)")
    k += 1
    ax[k].bar(t, orario, width=0.035, color="tab:blue")
    ax2 = ax[k].twinx()
    ax2.plot(t, pr - pr[0], color="navy", linewidth=1)
    ax2.set_ylabel("totale (mm)", fontsize=8)
    ax[k].set_ylabel("Pioggia oraria (mm)")
    k += 1
    ax[k].plot(t, p, color="tab:purple")
    ax[k].set_ylabel("Pressione (hPa)")
    k += 1
    if "SMOIS" in d:
        ax[k].plot(t, d["SMOIS"][:, 0, j[0], j[1]] * 100, color="tab:brown", label="0-10 cm")
        ax[k].plot(t, d["SMOIS"][:, 1, j[0], j[1]] * 100, color="tab:olive", label="10-40 cm")
        ax[k].legend(loc="upper right", fontsize=8)
        ax[k].set_ylabel("Umidità suolo (%)")
    else:
        ax[k].plot(t, d["SWDOWN"][:, j[0], j[1]], color="orange")
        ax[k].set_ylabel("Radiazione (W/m2)")
    for x in ax:
        x.grid(alpha=0.3)
        x.ticklabel_format(axis="y", useOffset=False)
    ax[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %H", tz=ROMA))
    ax[-1].xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 6), tz=ROMA))
    fig.autofmt_xdate()
    c = datetime.strptime(corsa, "%Y%m%d%H")
    fig.suptitle(f"Nerola (punto di griglia a {lat[j]:.3f} N, {lon[j]:.3f} E, quota modello {d['hgt'][j]:.0f} m)\n"
                 f"corsa ECMWF {c:%d/%m/%Y %H} UTC, orari in ora locale", fontsize=11)
    fig.tight_layout(rect=[0, 0.025, 1, 0.94])
    fig.text(0.5, 0.005, FONTE, ha="center", fontsize=8, color="0.4")
    fig.savefig(destinazione, dpi=100, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uscite", required=True)
    ap.add_argument("--corsa", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sfondo", default="")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sf = sfondo_mod.carica(a.sfondo) if a.sfondo else sfondo_mod.carica("/nonesiste")
    if sf.get("errori"):
        print("ATTENZIONE sfondo:", "; ".join(sf["errori"]))
    indice = {"corsa": a.corsa, "creato": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), "fonte": FONTE, "domini": {}}
    for dom, cartella_dom, etichetta in DOMINI:
        d = uscite.carica(a.uscite, dom)
        if d is None:
            sys.exit(f"mancano i risultati di {dom}")
        n = len(d["tempi"])
        parametri = [p for p in PARAMETRI if p.get("serve") is None or p["serve"] in d]
        frames = {p["id"]: [i for i in range(p["inizio"], n, p["passo"])] for p in parametri}
        parametri = [p for p in parametri if frames[p["id"]]]
        lim = limiti_automatici(d, parametri, frames)
        paesi = scegli_paesi(sf, d["lat"], d["lon"], 5.0 if etichetta == "1 km" else 16.0, 30 if etichetta == "1 km" else 26,
                              vicini=etichetta == "1 km")
        voce = {"etichetta": etichetta, "parametri": {}}
        ny, nx = d["lat"].shape
        cart_dati = out / "dati" / cartella_dom
        cart_dati.mkdir(parents=True, exist_ok=True)
        layout = None
        for p in parametri:
            cart = out / "mappe" / cartella_dom / p["id"]
            cart.mkdir(parents=True, exist_ok=True)
            files, valori = [], []
            for i in frames[p["id"]]:
                nome, utc, z, lay = disegna(d, p, i, dom, etichetta, a.corsa, sf, paesi, str(cart), lim)
                layout = layout or lay
                files.append({"file": f"mappe/{cartella_dom}/{p['id']}/{nome}", "valido_utc": utc.strftime("%Y-%m-%dT%H:%MZ")})
                valori.append(quantizza(z, p.get("scala", 10), p.get("clip")).reshape(-1))
            (cart_dati / f"{p['id']}.bin").write_bytes(np.stack(valori).tobytes())
            voce["parametri"][p["id"]] = {
                "titolo": p["titolo"], "unita": p["unita"], "immagini": files,
                "dati": f"dati/{cartella_dom}/{p['id']}.bin", "scala": p.get("scala", 10),
                "unita_dato": p.get("unita_dato", p["unita"]), "etichetta_dato": p.get("etichetta_dato", p["titolo"])}
        # griglia (per trovare il punto sotto il puntatore) e serie orarie (per il meteogramma di un punto)
        np.concatenate([d["lat"].reshape(-1), d["lon"].reshape(-1), d["hgt"].reshape(-1)]).astype("<f4").tofile(cart_dati / "griglia.bin")
        campi = campi_serie(d)
        serie = np.stack([quantizza(arr, sc).reshape(arr.shape[0], -1) for _, _, _, sc, arr in campi])  # [campo, t, punto]
        np.ascontiguousarray(serie.transpose(2, 0, 1)).tofile(cart_dati / "serie.bin")                    # [punto, campo, t]
        luoghi = [{"nome": nome, "lat": round(la, 4), "lon": round(lo, 4)}
                  for nome, la, lo in scegli_paesi(sf, d["lat"], d["lon"], 2.5, 120, vicini=True)]
        voce.update({
            "layout": {**layout, "estensione": [float(d["lon"].min()), float(d["lon"].max()), float(d["lat"].min()), float(d["lat"].max())],
                       "ny": int(ny), "nx": int(nx)},
            "griglia": f"dati/{cartella_dom}/griglia.bin",
            "serie": {"file": f"dati/{cartella_dom}/serie.bin", "n_t": n, "tempi": [quando(t)[0].strftime("%Y-%m-%dT%H:%MZ") for t in d["tempi"]],
                      "campi": [{"id": c[0], "titolo": c[1], "unita": c[2], "scala": c[3]} for c in campi]},
            "luoghi": luoghi})
        indice["domini"][cartella_dom] = voce
        print(dom, "parametri:", len(parametri), "immagini:", sum(len(v["immagini"]) for v in voce["parametri"].values()))
        salva = {k: v for k, v in d.items() if k != "tempi"}
        np.savez_compressed(out / f"{dom}_tutto.npz", **salva, tempi=np.array(d["tempi"]))
        if dom == "d02":
            meteogramma(d, a.corsa, str(out / "meteogramma_nerola.png"))
    (out / "indice.json").write_text(json.dumps(indice, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
