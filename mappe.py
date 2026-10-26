#!/usr/bin/env python3
"""Disegna le mappe (un parametro per immagine) e il meteogramma di Nerola dai risultati di WRF.

  mappe.py --uscite CARTELLA --corsa 2026102400 --out CARTELLA_PUBBLICA [--sfondo CARTELLA_SFONDO]

Scrive:
  mappe/<dominio>/<parametro>/AAAAMMGG_HH.png   (ora UTC nel nome del file)
  dati/<dominio>/...                            (numeri per la pagina web: valori, griglia, serie orarie, quote)
  indice.json                                   (elenco di tutte le mappe, per la pagina web)
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
import matplotlib.cm as mcm  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import matplotlib.transforms  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sfondo as sfondo_mod  # noqa: E402
import uscite  # noqa: E402

ROMA = ZoneInfo("Europe/Rome")
NEROLA = (42.13, 12.77)
FONTE = "WRF su dati ECMWF IFS 0,1° (CC BY 4.0, dati modificati)"
CREDITI = "Rilievo: Terrain Tiles (SRTM/GMTED) · Confini: ISTAT, Natural Earth · Nomi: GeoNames"
DOMINI = (("d02", "1km", "1 km"), ("d01", "3km", "3 km"))

FIG = (7.6, 6.6)   # pollici
DPI = 85
AREA_MAPPA = [0.03, 0.09, 0.80, 0.76]   # sinistra, basso, larghezza, altezza (frazioni della figura)
AREA_LEGENDA = [0.725, 0.22, 0.022, 0.56]
GRADO_PER_KM_LAT = 1 / 111.0

# ---- costanti della stima di raffreddamento notturno sul prato (da tarare con le misure vere)
SIGMA = 5.670374e-8


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


def bulbo_umido(t, rh):
    """Temperatura di bulbo umido (C) dalla temperatura (C) e dall'umidita' relativa (%), formula di Stull (2011)."""
    rh = np.clip(rh, 5, 99)
    return (t * np.arctan(0.151977 * np.sqrt(rh + 8.313659)) + np.arctan(t + rh) - np.arctan(rh - 1.676331)
            + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh) - 4.686035)


def campi_gelo(d):
    """Campi orari per il gelo, tutti calcolati dal modello (niente stime inventate):
    temperatura a 2 m, bulbo umido, temperatura della superficie del suolo (TSK) e classe oraria di gelo.
    La nuvolosita' (0 sereno, 1 coperto) viene dalla radiazione a onda lunga verso il basso (GLW) confrontata con
    quella del cielo sereno (formula di Brutsaert)."""
    t2 = d["T2"] - 273.15
    rh = rh2(d)
    tw = bulbo_umido(t2, rh)
    vento = np.hypot(d["U10"], d["V10"])
    ts = np.minimum(d["TSK"] - 273.15, t2) if "TSK" in d else t2
    if "GLW" in d:
        tk = d["T2"]
        e = d["Q2"] * (d["PSFC"] / 100.0) / (0.622 + d["Q2"])
        eps = np.clip(1.24 * (np.maximum(e, 0.1) / tk) ** (1 / 7), 0.55, 0.95)
        cielo_sereno = eps * SIGMA * tk ** 4
        coperto = 0.97 * SIGMA * tk ** 4
        nuv = np.clip((d["GLW"] - cielo_sereno) / (coperto - cielo_sereno), 0, 1)
        nuv = np.where(d["GLW"] < 1.0, 0.5, nuv)   # primo istante: la radiazione non e' ancora calcolata
    else:
        nuv = np.full_like(t2, 0.5)
    # classe oraria: 0 nessuna, 1 brina, 2 gelata, 3 gelata forte
    forte = (t2 <= -3) | ((tw <= -3) & (t2 <= -1))
    gelata = (t2 <= 0) | ((tw <= 0) & (ts <= 0))
    brina = (ts <= 0) & (tw <= 2) & (vento <= 4) & (nuv < 0.6)
    classe = np.select([forte, gelata, brina], [3, 2, 1], 0).astype("float64")
    return t2, tw, ts, classe


_CACHE_GELO = {}


def gelo(d):
    k = id(d)
    if k not in _CACHE_GELO:
        _CACHE_GELO.clear()
        _CACHE_GELO[k] = campi_gelo(d)
    return _CACHE_GELO[k]


def finestra(a, i, n, f):
    return f(a[i - n:i + 1], axis=0)


CLASSI_GELO = ["nessuna", "brina", "gelata", "gelata forte"]


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
            "nota": "pioggia convettiva e a grande scala"}


PARAMETRI = [
    {"id": "temperatura", "titolo": "Temperatura a 2 m", "unita": "°C", "calcola": lambda d, i: d["T2"][i] - 273.15,
     "cmap": "RdYlBu_r", "limiti": "auto", "passo": 3, "inizio": 0, "isolinee": True},
    {"id": "vento", "titolo": "Vento a 10 m", "unita": "km/h", "calcola": lambda d, i: np.hypot(d["U10"][i], d["V10"][i]) * 3.6,
     "cmap": "viridis", "limiti": (0, 60), "passo": 3, "inizio": 0, "frecce": True},
    {"id": "umidita", "titolo": "Umidità relativa a 2 m", "unita": "%", "calcola": lambda d, i: rh2(d)[i],
     "cmap": "YlGnBu", "limiti": (20, 100), "passo": 3, "inizio": 0},
    {"id": "pressione", "titolo": "Pressione ridotta al livello del mare", "unita": "hPa",
     "calcola": lambda d, i: pressione_mare(d)[i], "cmap": "Spectral_r", "limiti": "auto", "passo": 3, "inizio": 0, "isolinee": True},
    p_pioggia(1), p_pioggia(3), p_pioggia(6), p_pioggia(12), p_pioggia(24), p_pioggia(0),
    {"id": "tmin24", "titolo": "Temperatura minima a 2 m nelle ultime 24 ore", "unita": "°C",
     "calcola": lambda d, i: finestra(gelo(d)[0], i, 24, np.min), "cmap": "RdYlBu_r", "limiti": "auto",
     "passo": 3, "inizio": 24, "isolinee": True},
    {"id": "tmax24", "titolo": "Temperatura massima a 2 m nelle ultime 24 ore", "unita": "°C",
     "calcola": lambda d, i: finestra(gelo(d)[0], i, 24, np.max), "cmap": "RdYlBu_r", "limiti": "auto",
     "passo": 3, "inizio": 24, "isolinee": True},
    {"id": "twmin24", "titolo": "Temperatura minima di bulbo umido (ultime 24 ore)", "unita": "°C",
     "calcola": lambda d, i: finestra(gelo(d)[1], i, 24, np.min), "cmap": "RdYlBu_r", "limiti": "auto",
     "passo": 3, "inizio": 24, "isolinee": True,
     "nota": "bulbo umido: temperatura a cui scende l'aria evaporando acqua; vicino a 0 °C con aria secca c'è rischio di gelo"},
    {"id": "tprato24", "titolo": "Temperatura minima della superficie del suolo (ultime 24 ore)", "unita": "°C",
     "calcola": lambda d, i: finestra(gelo(d)[2], i, 24, np.min), "cmap": "RdYlBu_r", "limiti": "auto",
     "passo": 3, "inizio": 24, "isolinee": True,
     "nota": "temperatura della superficie del suolo calcolata dal modello (WRF, campo TSK)"},
    {"id": "gelata", "titolo": "Brina e gelate (ultime 24 ore)", "unita": "",
     "calcola": lambda d, i: finestra(gelo(d)[3], i, 24, np.max), "passo": 3, "inizio": 24,
     "classi": {"bordi": [0.5, 1.5, 2.5], "colori": ["#ffffff", "#8fd8f2", "#2f6fd6", "#6a2c91"],
                "etichette": CLASSI_GELO, "nascondi": {0}},
     "scala": 1, "etichetta_dato": "Brina / gelata", "nota": "classe peggiore nelle 24 ore, da temperatura a 2 m, bulbo umido, suolo, cielo e vento"},
    {"id": "nebbia", "titolo": "Nebbia: visibilità stimata", "unita": "",
     "calcola": lambda d, i: visibilita(d)[i], "passo": 3, "inizio": 3, "serve": "QCLOUD1",
     "classi": {"bordi": [200, 500, 1000, 5000], "colori": ["#4b4b4b", "#7d7d7d", "#b0b0b0", "#dcdcdc", "#ffffff"],
                "etichette": ["< 200 m", "200-500 m", "500-1000 m", "1-5 km", "> 5 km"], "nascondi": {4}},
     "unita_dato": "m", "etichetta_dato": "Visibilità stimata", "scala": 1, "clip": 30000,
     "nota": "stimata dall'acqua delle nubi nel primo livello del modello; nelle prime ore il modello parte senza nubi"},
    {"id": "suolo_temp", "titolo": "Temperatura del suolo (0-10 cm)", "unita": "°C",
     "calcola": lambda d, i: d["TSLB"][i, 0] - 273.15, "cmap": "RdYlBu_r", "limiti": "auto", "passo": 3, "inizio": 0,
     "serve": "TSLB", "terra": True, "isolinee": True},
    {"id": "suolo_umidita_10", "titolo": "Umidità del suolo (0-10 cm)", "unita": "% volume",
     "calcola": lambda d, i: d["SMOIS"][i, 0] * 100, "cmap": "YlGnBu", "limiti": (5, 50), "passo": 3, "inizio": 0,
     "serve": "SMOIS", "terra": True},
    {"id": "suolo_umidita_40", "titolo": "Umidità del suolo (10-40 cm)", "unita": "% volume",
     "calcola": lambda d, i: d["SMOIS"][i, 1] * 100, "cmap": "YlGnBu", "limiti": (5, 50), "passo": 3, "inizio": 0,
     "serve": "SMOIS", "terra": True},
]
for _p in PARAMETRI:
    _p.setdefault("unita_dato", _p["unita"])
    _p.setdefault("etichetta_dato", _p["titolo"])
    _p.setdefault("scala", 10)


# ---------------------------------------------------------------- utilita' di calcolo e disegno

def quando(t):
    u = datetime.strptime(t, "%Y-%m-%d_%H:%M:%S").replace(tzinfo=timezone.utc)
    return u, u.astimezone(ROMA)


def km(la1, lo1, la2, lo2):
    return np.hypot((la1 - la2) * 111.0, (lo1 - lo2) * 111.0 * np.cos(np.radians(la1)))


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


def sovra(a, f):
    """Ingrandisce una griglia di f volte con interpolazione lineare (le mappe a classi vengono morbide, non a quadretti)."""
    a = np.asarray(a, dtype="float64")
    ny, nx = a.shape
    y = np.linspace(0, ny - 1, (ny - 1) * f + 1)
    x = np.linspace(0, nx - 1, (nx - 1) * f + 1)
    iy = np.clip(np.floor(y).astype(int), 0, ny - 2)
    ix = np.clip(np.floor(x).astype(int), 0, nx - 2)
    wy = (y - iy)[:, None]
    wx = (x - ix)[None, :]
    return (a[iy][:, ix] * (1 - wx) + a[iy][:, ix + 1] * wx) * (1 - wy) + (a[iy + 1][:, ix] * (1 - wx) + a[iy + 1][:, ix + 1] * wx) * wy


def ritaglia_dem(dem, lon0, lon1, lat0, lat1, margine=0.02):
    """Pezzo di rilievo che copre l'estensione; ritorna (quote, [lon_min, lon_max, lat_min, lat_max], passo, lon_inizio, lat_inizio)."""
    if not dem:
        return None
    for chiave in ("fine", "largo"):
        if chiave not in dem:
            continue
        z, l0, a0, passo = dem[chiave]
        ny, nx = z.shape
        if l0 <= lon0 - margine and l0 + (nx - 1) * passo >= lon1 + margine and a0 <= lat0 - margine and a0 + (ny - 1) * passo >= lat1 + margine:
            j0 = int(np.floor((lon0 - margine - l0) / passo)); j1 = int(np.ceil((lon1 + margine - l0) / passo)) + 1
            i0 = int(np.floor((lat0 - margine - a0) / passo)); i1 = int(np.ceil((lat1 + margine - a0) / passo)) + 1
            sub = z[i0:i1, j0:j1]
            return sub, passo, l0 + j0 * passo, a0 + i0 * passo
    return None


def rilievo_rgb(sub, passo, lon_ini, lat_ini):
    """Rilievo come due immagini: la base (tinta leggera per la quota, azzurrina sul mare) e le ombre/luci da sovrapporre
    ai colori (luce da nord-ovest)."""
    lat = lat_ini + np.arange(sub.shape[0]) * passo
    dy = passo * 111000.0
    dx = passo * 111000.0 * np.cos(np.radians(lat))[:, None]
    z = sub.astype("float64")
    gy = np.gradient(z, dy, axis=0)
    gx = np.gradient(z, axis=1) / dx
    ve = 2.0   # esagerazione verticale: i rilievi si leggono meglio
    pend = np.arctan(ve * np.hypot(gx, gy))
    esp = np.arctan2(gy, -gx)
    zen, az = np.radians(45.0), np.radians(315.0)
    hs = np.clip(np.cos(zen) * np.cos(pend) + np.sin(zen) * np.sin(pend) * np.cos(az - esp), 0, 1)
    piano = np.cos(zen)
    quota = np.clip(z / 1600.0, 0, 1)
    basso = np.array([0.93, 0.95, 0.90])
    alto = np.array([0.84, 0.78, 0.68])
    base = basso[None, None, :] * (1 - quota[..., None]) + alto[None, None, :] * quota[..., None]
    base = np.where((z <= 0)[..., None], np.array([0.80, 0.88, 0.96])[None, None, :], base)
    ombra = np.clip((piano - hs) / piano, 0, 1) * 0.75     # versante in ombra: nero trasparente
    luce = np.clip((hs - piano) / (1 - piano), 0, 1) * 0.45  # versante illuminato: bianco trasparente
    ov = np.zeros(z.shape + (4,))
    ov[..., :3] = np.where((hs < piano)[..., None], 0.0, 1.0)
    ov[..., 3] = np.where(hs < piano, ombra, luce)
    return np.clip(base, 0, 1), ov


def candidati(sf, lat, lon, vicini):
    """Paesi dentro la mappa, in ordine di importanza. Nerola e' sempre il primo."""
    elenco = [{"nome": "Nerola", "lat": NEROLA[0], "lon": NEROLA[1], "pop": 0, "sede": True, "quota": None}]
    dentro = []
    for p in sf.get("paesi", []):
        if lat.min() + 0.015 < p["lat"] < lat.max() - 0.015 and lon.min() + 0.02 < p["lon"] < lon.max() - 0.02 and p["nome"] != "Nerola":
            dentro.append(p)
    if vicini:
        grandi = sorted([p for p in dentro if p["pop"] >= 20000], key=lambda p: -p["pop"])
        altri = sorted([p for p in dentro if p["pop"] < 20000], key=lambda p: (not p["sede"], km(p["lat"], p["lon"], *NEROLA)))
        elenco += grandi + altri
    else:
        elenco += sorted([p for p in dentro if p["sede"] or p["pop"] >= 3000], key=lambda p: -p["pop"])
    visti, out = set(), []
    for p in elenco:
        if p["nome"] not in visti:
            visti.add(p["nome"])
            out.append(p)
    return out


def posiziona_etichette(cand, lon, lat, massimo, estensione=None):
    """Sceglie quali nomi scrivere e dove, in modo che non si sovrappongano. Ritorna [(nome, lat, lon, dx, dy, ha, va)]."""
    fig = plt.figure(figsize=FIG, dpi=DPI)
    ax = fig.add_axes(AREA_MAPPA)
    ax.set_xlim(lon.min(), lon.max())
    ax.set_ylim(lat.min(), lat.max())
    ax.set_aspect(1.0 / np.cos(np.radians(42.1)))
    ax.apply_aspect()
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    occupati, scelte = [], []
    ass = ax.get_window_extent()

    def sovrappone(bb):
        return any(bb.overlaps(o) for o in occupati) or not (ass.x0 <= bb.x0 and bb.x1 <= ass.x1 and ass.y0 <= bb.y0 and bb.y1 <= ass.y1)

    posizioni = [(5, 4, "left", "bottom"), (5, -4, "left", "top"), (-5, 4, "right", "bottom"), (-5, -4, "right", "top"),
                 (0, 7, "center", "bottom"), (0, -7, "center", "top")]
    for p in cand:
        if len(scelte) >= massimo:
            break
        px, py = ax.transData.transform((p["lon"], p["lat"]))
        grande = p["nome"] == "Nerola"
        marcatore = matplotlib.transforms.Bbox.from_bounds(px - 4, py - 4, 8, 8)
        if any(marcatore.overlaps(o) for o in occupati):
            continue
        for dx, dy, ha, va in posizioni:
            t = ax.annotate(p["nome"], (p["lon"], p["lat"]), textcoords="offset points", xytext=(dx, dy), ha=ha, va=va,
                            fontsize=9.5 if grande else 8.5, fontweight="bold")
            bb = t.get_window_extent(rend)
            bb = matplotlib.transforms.Bbox.from_extents(bb.x0 - 4, bb.y0 - 3, bb.x1 + 4, bb.y1 + 3)
            t.remove()
            if not sovrappone(bb):
                occupati += [bb, marcatore]
                scelte.append((p["nome"], p["lat"], p["lon"], dx, dy, ha, va))
                break
    plt.close(fig)
    return scelte


def disegna_sfondo(ax, sfondo, sul_rilievo):
    def linee(chiave, **stile):
        ls = sfondo.get(chiave) or []
        if ls:
            ax.add_collection(LineCollection(ls, **stile))
    linee("comuni", colors="#3d3d3d", linewidths=0.35, alpha=0.55, zorder=3)
    linee("province", colors="#1a1a1a", linewidths=0.9, linestyles=(0, (5, 3)), alpha=0.85, zorder=3)
    linee("regioni", colors="#1a1a1a", linewidths=1.4, alpha=0.9, zorder=3)
    linee("laghi", colors="#2a6cc7", linewidths=0.8, zorder=3)
    linee("fiumi", colors="#2a6cc7", linewidths=0.7, alpha=0.85, zorder=3)
    linee("costa", colors="#1a1a1a", linewidths=1.0, zorder=3)


def disegna(d, p, i, dom, etichetta, corsa, ctx, cartella, lim):
    lon, lat = d["lon"], d["lat"]
    z = np.array(p["calcola"](d, i), dtype="float64")
    if p.get("terra") and "landmask" in d:
        z = np.where(d["landmask"] > 0.5, z, np.nan)
    fig = plt.figure(figsize=FIG, facecolor="white")
    ax = fig.add_axes(AREA_MAPPA)
    cax = fig.add_axes(AREA_LEGENDA)
    if ctx["rilievo"] is not None:
        (base, ombre), est = ctx["rilievo"]
        ax.imshow(base, extent=est, origin="lower", aspect="auto", interpolation="bilinear", zorder=0)
        ax.imshow(ombre, extent=est, origin="lower", aspect="auto", interpolation="bilinear", zorder=1.5)
    f = ctx["sovra"]
    cl = p.get("classi")
    if cl:
        codice = np.digitize(sovra(z, f), cl["bordi"]).astype("float64")
        mostrate = [k for k in range(len(cl["colori"])) if k not in cl["nascondi"]]
        nuovo = {k: n for n, k in enumerate(mostrate)}
        mappa = np.full(codice.shape, np.nan)
        for k, n in nuovo.items():
            mappa[codice == k] = n
        if p.get("terra") and "landmask" in d:
            mappa[sovra(d["landmask"], f) < 0.5] = np.nan
        cmap = mcolors.ListedColormap([cl["colori"][k] for k in mostrate])
        ax.pcolormesh(sovra(lon, f), sovra(lat, f), mappa, cmap=cmap, vmin=-0.5, vmax=len(mostrate) - 0.5, shading="auto", alpha=0.85, zorder=1)
        sm = mcm.ScalarMappable(norm=mcolors.Normalize(-0.5, len(mostrate) - 0.5), cmap=cmap)
        cb = fig.colorbar(sm, cax=cax)
        cb.set_ticks(range(len(mostrate)))
        cb.set_ticklabels([cl["etichette"][k] for k in mostrate])
        cb.ax.tick_params(labelsize=9, length=0)
        if p["unita"]:
            cb.set_label(p["unita"])
    else:
        vmin, vmax = lim[p["id"]] if p.get("limiti") == "auto" else p.get("limiti")
        ax.pcolormesh(lon, lat, z, cmap=p["cmap"], vmin=vmin, vmax=vmax, shading="auto" if np.isnan(z).any() else "gouraud", zorder=1)
        sm = mcm.ScalarMappable(norm=mcolors.Normalize(vmin, vmax), cmap=p["cmap"])
        cb = fig.colorbar(sm, cax=cax, extend="both" if p.get("limiti") != "auto" else "neither")
        cb.set_label(p["unita"])
        cb.ax.tick_params(labelsize=8.5)
        if p.get("isolinee") and p.get("limiti") == "auto":
            passo = passo_isolinee(vmin, vmax)
            livelli = np.arange(np.ceil(vmin / passo) * passo, vmax + 1e-9, passo)
            zs = leviga(np.nan_to_num(z, nan=np.nanmean(z)))
            if len(livelli) and np.nanmin(zs) < livelli.max() and np.nanmax(zs) > livelli.min():
                cs = ax.contour(lon, lat, zs, levels=livelli, colors="#1a1a1a", linewidths=0.7, alpha=0.7, zorder=2.5)
                testi = ax.clabel(cs, fmt=lambda v: f"{v:g}", fontsize=8, inline=True, inline_spacing=3)
                for t in testi:
                    t.set_path_effects([pe.withStroke(linewidth=2, foreground="white")])
                    t.set_zorder(4)
    if p.get("frecce"):
        s = 4 if etichetta == "3 km" else 3
        q = ax.quiver(lon[::s, ::s], lat[::s, ::s], d["U10"][i][::s, ::s], d["V10"][i][::s, ::s], color="white", scale=120,
                      width=0.0035, edgecolor="#222222", linewidth=0.4, zorder=2.5)
    disegna_sfondo(ax, ctx["sfondo"], ctx["rilievo"] is not None)
    ax.set_xlim(lon.min(), lon.max())
    ax.set_ylim(lat.min(), lat.max())
    ax.set_aspect(1.0 / np.cos(np.radians(42.1)))
    ax.set_xticks([])
    ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_color("#111111")
        s_.set_linewidth(1.3)
    for nome, la, lo, dx, dy, ha, va in ctx["etichette"]:
        grande = nome == "Nerola"
        ax.plot([lo], [la], marker="*" if grande else "o", color="#d6341f" if grande else "#111111", markersize=12 if grande else 4.2,
                markeredgecolor="white", markeredgewidth=0.7, zorder=5)
        ax.annotate(nome, (lo, la), textcoords="offset points", xytext=(dx, dy), ha=ha, va=va, fontsize=9.5 if grande else 8.5,
                    fontweight="bold", color="#111111", zorder=6,
                    bbox=dict(boxstyle="square,pad=0.12", facecolor="white", edgecolor="none", alpha=0.85))
    ax.text(0.985, 0.02, "www.meteonerola.it", transform=ax.transAxes, ha="right", va="bottom", fontsize=9.5, fontweight="bold",
            color="#111111", zorder=7, bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#333333", linewidth=0.8, alpha=0.92))
    utc, loc = quando(d["tempi"][i])
    utc0 = quando(d["tempi"][0])[0]
    ore = int(round((utc - utc0).total_seconds() / 3600))
    c = datetime.strptime(corsa, "%Y%m%d%H")
    # disposizione: mappa in alto a sinistra sotto il titolo, barra dei colori a destra; il gruppo e' centrato nella figura
    ax.apply_aspect()
    bx = ax.get_position()
    larg_etichette = 0.15 if cl else 0.085
    totale = bx.width + 0.022 + 0.024 + larg_etichette
    x0 = max(0.015, (1 - totale) / 2)
    y0 = AREA_MAPPA[1] + AREA_MAPPA[3] - bx.height
    ax.set_position([x0, y0, bx.width, bx.height])
    cax.set_position([x0 + bx.width + 0.022, y0 + 0.01, 0.024, bx.height - 0.02])
    xr = x0 + bx.width + 0.022 + 0.024 + larg_etichette * 0.55
    gg = ["Lun", "Mar", "Mer", "Gio", "Ven", "Sab", "Dom"][loc.weekday()]
    mesi = ["gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre"]
    tit = p["titolo"]
    dim = 20 if len(tit) <= 24 else 16 if len(tit) <= 38 else 13
    fig.text(x0, 0.945, tit, fontsize=dim, fontweight="bold", ha="left", va="center", color="#111111")
    fig.text(x0, 0.885, f"{gg} {loc.day} {mesi[loc.month - 1]}, ore {loc:%H:%M}", fontsize=13, ha="left", va="center", color="#222222")
    fig.text(xr, 0.945, f"WRF {etichetta}", fontsize=16, fontweight="bold", ha="right", va="center", color="#1d4e89")
    fig.text(xr, 0.885, f"+{ore}h", fontsize=13, fontweight="bold", ha="right", va="center", color="#555555")
    nota = p.get("nota", "")
    fig.text(x0, 0.012, (nota + "\n" if nota else "") + f"{FONTE} · corsa {c:%d/%m %H} UTC · ora italiana ({utc:%H} UTC)\n" + CREDITI,
             ha="left", va="bottom", fontsize=6.8, color="#6b7480", style="italic")
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
              ("PRESS", "Pressione al livello del mare", "hPa", 10, pressione_mare(d)),
              ("TW", "Temperatura di bulbo umido", "°C", 10, gelo(d)[1]),
              ("TPRATO", "Temperatura della superficie del suolo", "°C", 10, gelo(d)[2])]
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
    ax[k].plot(t, gelo(d)[2][:, j[0], j[1]], color="tab:blue", linewidth=1, linestyle=":", label="prato (stima)")
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
    cartella_sfondo = a.sfondo or "/nonesiste"
    sf = sfondo_mod.carica(cartella_sfondo)
    dem = sfondo_mod.carica_dem(cartella_sfondo)
    for avviso in sf.get("errori", []) + sf.get("avvisi", []):
        print("ATTENZIONE sfondo:", avviso)
    if not dem:
        print("ATTENZIONE: rilievo non disponibile, mappe senza ombreggiatura e senza quota reale")
    indice = {"corsa": a.corsa, "creato": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), "fonte": FONTE, "crediti": CREDITI,
              "domini": {}}
    for dom, cartella_dom, etichetta in DOMINI:
        d = uscite.carica(a.uscite, dom)
        if d is None:
            sys.exit(f"mancano i risultati di {dom}")
        n = len(d["tempi"])
        parametri = [p for p in PARAMETRI if p.get("serve") is None or p["serve"] in d]
        frames = {p["id"]: [i for i in range(p["inizio"], n, p["passo"])] for p in parametri}
        parametri = [p for p in parametri if frames[p["id"]]]
        lim = limiti_automatici(d, parametri, frames)
        piccolo = etichetta == "1 km"
        cand = candidati(sf, d["lat"], d["lon"], vicini=piccolo)
        etichette = posiziona_etichette(cand, d["lon"], d["lat"], 16 if piccolo else 14)
        print(dom, "etichette:", len(etichette), "su", len(cand), "candidati")
        lon0, lon1, lat0, lat1 = float(d["lon"].min()), float(d["lon"].max()), float(d["lat"].min()), float(d["lat"].max())
        pezzo = ritaglia_dem(dem, lon0, lon1, lat0, lat1)
        rilievo = None
        if pezzo is not None:
            sub, passo, li, ai = pezzo
            rilievo = (rilievo_rgb(sub, passo, li, ai), [li, li + (sub.shape[1] - 1) * passo, ai, ai + (sub.shape[0] - 1) * passo])
        ctx = {"sfondo": sf, "rilievo": rilievo, "etichette": etichette, "sovra": 3 if piccolo else 2}
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
                nome, utc, z, lay = disegna(d, p, i, dom, etichetta, a.corsa, ctx, str(cart), lim)
                layout = layout or lay
                files.append({"file": f"mappe/{cartella_dom}/{p['id']}/{nome}", "valido_utc": utc.strftime("%Y-%m-%dT%H:%MZ")})
                valori.append(quantizza(z, p["scala"], p.get("clip")).reshape(-1))
            (cart_dati / f"{p['id']}.bin").write_bytes(np.stack(valori).tobytes())
            meta = {"titolo": p["titolo"], "unita": p["unita"], "immagini": files, "dati": f"dati/{cartella_dom}/{p['id']}.bin",
                    "scala": p["scala"], "unita_dato": p["unita_dato"], "etichetta_dato": p["etichetta_dato"]}
            if p.get("classi") and p["scala"] == 1 and p["unita_dato"] == "":
                meta["classi_etichette"] = [e.replace("\n", ", ") for e in p["classi"]["etichette"]]
            voce["parametri"][p["id"]] = meta
        # griglia (per trovare il punto sotto il puntatore) e serie orarie (per il meteogramma di un punto)
        np.concatenate([d["lat"].reshape(-1), d["lon"].reshape(-1), d["hgt"].reshape(-1)]).astype("<f4").tofile(cart_dati / "griglia.bin")
        campi = campi_serie(d)
        serie = np.stack([quantizza(arr, sc).reshape(arr.shape[0], -1) for _, _, _, sc, arr in campi])  # [campo, t, punto]
        np.ascontiguousarray(serie.transpose(2, 0, 1)).tofile(cart_dati / "serie.bin")                    # [punto, campo, t]
        luoghi = [{"nome": p_["nome"], "lat": round(p_["lat"], 4), "lon": round(p_["lon"], 4), "quota": p_.get("quota")}
                  for p_ in cand[:150]]
        voce.update({
            "layout": {**layout, "estensione": [lon0, lon1, lat0, lat1], "ny": int(ny), "nx": int(nx)},
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
