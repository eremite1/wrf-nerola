#!/usr/bin/env python3
"""Prepara i dati iniziali e al contorno del WRF dai modelli ICON (DWD ICON-EU, ItaliaMeteo ICON-2I)
scrivendoli direttamente nel "formato intermedio" di WPS (quello che produce ungrib), cosi' metgrid li legge
senza bisogno di file GRIB e tabelle Vtable.

I dati vengono dall'archivio pubblico di Open-Meteo (CC BY 4.0), letto con motore/fonti.py di meteo-mappe.

  ICON-EU (7 km):  17 livelli di pressione da 1000 a 50 hPa, superficie e suolo: tutto da ICON-EU.
  ICON-2I (2,2 km): superficie e livelli 1000, 925, 850, 700, 500, 250 hPa da ICON-2I; sopra 250 hPa
                    (200-50 hPa) e strati del suolo da ICON-EU (ICON-2I non li pubblica), portati sulla
                    griglia di ICON-2I.

La pressione al suolo (che non e' pubblicata) si ricava dalla pressione al livello del mare, dalla temperatura
a 2 m e dalla quota del terreno (rilievo Terrain Tiles mediato sulle celle del modello), con la formula
ipsometrica; la stessa quota viene data a WPS come SOILHGT, cosi' le due cose sono coerenti.

  intermedio_icon.py --modello iconeu --trova --ore 48              stampa la corsa AAAAMMGGHH
  intermedio_icon.py --modello iconeu --corsa 2026110900 --ore 48 --out CARTELLA
"""
import argparse
import os
import struct
import sys
from datetime import datetime, timedelta

import numpy as np

sys.path.insert(0, os.environ.get("MOTORE_MAPPE", "/home/runner/meteo-mappe"))
from motore import fonti  # noqa: E402

RIQUADRO = (9.0, 17.0, 39.0, 45.5)            # lon0, lon1, lat0, lat1 (il dominio a 3 km e' circa 11-14,7 E, 41-43,3 N)
EU = "dwd_icon_eu"
I2I = "italia_meteo_arpae_icon_2i"
LIV_EU = [1000, 950, 925, 900, 850, 800, 700, 600, 500, 400, 300, 250, 200, 150, 100, 70, 50]
LIV_2I = [1000, 925, 850, 700, 500, 250]
SOPRA_2I = [200, 150, 100, 70, 50]
G, RD = 9.80665, 287.05


# ------------------------------------------------------------------ formato intermedio di WPS (versione 5)
def riempi(v):
    """Buchi (NaN, es. livelli sotto il terreno) riempiti con il valore valido piu' vicino."""
    v = np.asarray(v, "float64")
    buchi = ~np.isfinite(v)
    if not buchi.any():
        return v
    if buchi.all():
        raise ValueError("campo tutto vuoto")
    from scipy.ndimage import distance_transform_edt
    _, (jj, ii) = distance_transform_edt(buchi, return_indices=True)
    return v[jj, ii]


def _rec(f, dati):
    f.write(struct.pack(">i", len(dati)))
    f.write(dati)
    f.write(struct.pack(">i", len(dati)))


def scrivi_campo(f, hdate, campo, unita, descr, livello, lat, lon, valori):
    """Un campo su griglia regolare lat/lon (lat e lon 1D crescenti), valori [ny, nx]."""
    ny, nx = valori.shape
    _rec(f, struct.pack(">i", 5))
    _rec(f, struct.pack(">24sf32s9s25s46sfiii", hdate.ljust(24).encode(), 0.0, b"Open-Meteo ICON".ljust(32),
                        campo.ljust(9).encode(), unita.ljust(25).encode(), descr[:46].ljust(46).encode(),
                        float(livello), nx, ny, 0))
    _rec(f, struct.pack(">8sfffff", b"SWCORNER", float(lat[0]), float(lon[0]),
                        float(lat[1] - lat[0]), float(lon[1] - lon[0]), 6367.47))
    _rec(f, struct.pack(">i", 0))                  # venti riferiti alla terra (nord-est)
    _rec(f, np.ascontiguousarray(riempi(valori), dtype=">f4").tobytes())


# ------------------------------------------------------------------ lettura
class Lettore:
    def __init__(self, dataset, info, valido):
        self.f = fonti.FileOM(dataset, info, valido)
        self.lat = self.lon = None

    def __call__(self, nome):
        la, lo, v = self.f.leggi(nome, RIQUADRO)
        v = np.asarray(v, dtype="float64")
        if la[0] > la[-1]:
            la, v = la[::-1], v[::-1]
        if self.lat is None:
            self.lat, self.lon = np.asarray(la, "float64"), np.asarray(lo, "float64")
        return v

    def chiudi(self):
        self.f.chiudi()


def uv_da_velocita(vel, dire):
    r = np.radians(dire)
    return -vel * np.sin(r), -vel * np.cos(r)


def su_griglia(lat_da, lon_da, v, lat_a, lon_a):
    from scipy.interpolate import RegularGridInterpolator
    f = RegularGridInterpolator((lat_da, lon_da), v, bounds_error=False, fill_value=None)
    LA, LO = np.meshgrid(lat_a, lon_a, indexing="ij")
    return f(np.stack([LA, LO], -1))


_QUOTA = {}


def quota_e_mare(lat, lon):
    """Quota media del terreno e frazione di terra di ogni cella (rilievo Terrain Tiles, circa 30-150 m)."""
    k = (lat.size, lon.size, round(lat[0], 4), round(lon[0], 4))
    if k in _QUOTA:
        return _QUOTA[k]
    import sfondo
    dlat, dlon = abs(lat[1] - lat[0]), abs(lon[1] - lon[0])
    passo = min(dlat, dlon) / 4
    z, (lon0, lat0, p) = sfondo.dem((lon[0] - dlon, lon[-1] + dlon, lat[0] - dlat, lat[-1] + dlat), 9, passo)
    z = np.asarray(z, "float64")
    quota = np.zeros((lat.size, lon.size))
    terra = np.zeros_like(quota)
    for j, la in enumerate(lat):
        j0 = max(int(round((la - dlat / 2 - lat0) / p)), 0); j1 = int(round((la + dlat / 2 - lat0) / p)) + 1
        for i, lo in enumerate(lon):
            i0 = max(int(round((lo - dlon / 2 - lon0) / p)), 0); i1 = int(round((lo + dlon / 2 - lon0) / p)) + 1
            blocco = z[j0:j1, i0:i1]
            terra[j, i] = np.mean(blocco > 0.5) if blocco.size else 0.0
            quota[j, i] = np.mean(np.maximum(blocco, 0.0)) if blocco.size else 0.0
    _QUOTA[k] = (quota, (terra >= 0.5).astype("float64"))
    return _QUOTA[k]


def pressione_suolo(pmsl_pa, t2_k, rh2, quota):
    """Formula ipsometrica con temperatura virtuale media dello strato (gradiente standard per lo strato fittizio)."""
    es = 611.2 * np.exp(17.67 * (t2_k - 273.15) / (t2_k - 29.65))
    q = 0.622 * (rh2 / 100 * es) / pmsl_pa
    tv = t2_k * (1 + 0.61 * q) + 0.0065 * quota / 2
    return pmsl_pa * np.exp(-G * quota / (RD * tv))


def suolo_noah(L):
    """Strati del suolo ICON-EU riportati ai 4 strati del modello Noah (0-10, 10-40, 40-100, 100-200 cm)."""
    t0, t6, t18, t54 = (L(f"soil_temperature_{c}cm") + 273.15 for c in (0, 6, 18, 54))
    sm = {k: L(f"soil_moisture_{k}cm") for k in ("0_to_1", "1_to_3", "3_to_9", "9_to_27", "27_to_81")}
    st = {"ST000010": (t0 + 2 * t6) / 3,
          "ST010040": t18 + (25 - 18) / (54 - 18) * (t54 - t18),
          "ST040100": t54, "ST100200": t54}
    sm_ = {"SM000010": (1 * sm["0_to_1"] + 2 * sm["1_to_3"] + 6 * sm["3_to_9"] + 1 * sm["9_to_27"]) / 10,
           "SM010040": (17 * sm["9_to_27"] + 13 * sm["27_to_81"]) / 30,
           "SM040100": sm["27_to_81"], "SM100200": sm["27_to_81"]}
    return st, sm_, t0


def scrivi_istante(modello, info, valido, cartella):
    t = datetime.strptime(valido[:16], "%Y-%m-%dT%H:%M")
    hdate = t.strftime("%Y-%m-%d_%H:00:00")
    eu = Lettore(EU, info[EU], valido)
    principale = Lettore(I2I, info[I2I], valido) if modello == "icon2i" else eu
    try:
        P = principale
        t2 = P("temperature_2m") + 273.15
        lat, lon = P.lat, P.lon
        porta = (lambda v: v) if P is eu else (lambda v: su_griglia(eu.lat, eu.lon, v, lat, lon))
        if P is not eu:
            eu("temperature_2m")                                  # carica la griglia di ICON-EU
        rh2 = np.clip(P("relative_humidity_2m"), 1, 100)
        pmsl = P("pressure_msl") * 100.0
        if P is eu:
            u10, v10 = P("wind_u_component_10m"), P("wind_v_component_10m")
        else:
            u10, v10 = uv_da_velocita(P("wind_speed_10m"), P("wind_direction_10m"))
        quota, terra = quota_e_mare(lat, lon)
        psfc = pressione_suolo(pmsl, t2, rh2, quota)
        st, sm, t0_eu = suolo_noah(eu)
        skin = (P("soil_temperature_0cm") + 273.15) if P is not eu else t0_eu
        neve = P("snow_depth")
        nome = os.path.join(cartella, f"ICON_{hdate[:13]}")      # diventa FILE:... in WPS (i ":" non sono ammessi negli artefatti)
        with open(nome, "wb") as f:
            w = lambda c, u, d, lv, v: scrivi_campo(f, hdate, c, u, d, lv, lat, lon, v)  # noqa: E731
            # livelli di pressione
            if P is eu:
                livelli = [(p, eu) for p in LIV_EU]
            else:
                livelli = [(p, P) for p in LIV_2I] + [(p, eu) for p in SOPRA_2I]
            for p, S in livelli:
                conv = (lambda v: v) if S is P else porta
                tt = conv(S(f"temperature_{p}hPa")) + 273.15
                rh = np.clip(conv(S(f"relative_humidity_{p}hPa")), 0, 100)
                gh = conv(S(f"geopotential_height_{p}hPa"))
                if S is eu:
                    uu, vv = conv(S(f"wind_u_component_{p}hPa")), conv(S(f"wind_v_component_{p}hPa"))
                else:
                    uu, vv = uv_da_velocita(S(f"wind_speed_{p}hPa"), S(f"wind_direction_{p}hPa"))
                for c, u, d, v in (("TT", "K", "Temperature", tt), ("RH", "%", "Relative Humidity", rh),
                                   ("UU", "m s-1", "U", uu), ("VV", "m s-1", "V", vv), ("GHT", "m", "Height", gh)):
                    w(c, u, d, p * 100.0, v)
            # superficie
            for c, u, d, v in (("TT", "K", "Temperature 2 m", t2), ("RH", "%", "Relative Humidity 2 m", rh2),
                               ("UU", "m s-1", "U 10 m", u10), ("VV", "m s-1", "V 10 m", v10),
                               ("PSFC", "Pa", "Surface Pressure", psfc), ("SOILHGT", "m", "Terrain field of source analysis", quota),
                               ("LANDSEA", "proprtn", "Land/Sea flag", terra), ("SKINTEMP", "K", "Skin temperature", skin),
                               ("SNOW", "kg m-2", "Water equivalent of accum snow depth", neve * 250.0),
                               ("SNOWH", "m", "Physical Snow Depth", neve), ("SEAICE", "proprtn", "Ice flag", np.zeros_like(t2))):
                w(c, u, d, 200100.0, v)
            w("PMSL", "Pa", "Sea-level Pressure", 201300.0, pmsl)
            for c, v in st.items():
                w(c, "K", f"T of {c[2:5]}-{c[5:8]} cm ground layer", 200100.0, porta(v))
            for c, v in sm.items():
                w(c, "m3 m-3", f"Soil Moist {c[2:5]}-{c[5:8]} cm layer", 200100.0, porta(v))
        print(f"{hdate}: {os.path.getsize(nome) / 1e6:.1f} MB, griglia {lat.size}x{lon.size}, "
              f"T2 Nerola circa {t2[np.abs(lat - 42.16).argmin(), np.abs(lon - 12.79).argmin()] - 273.15:.1f} °C", flush=True)
    finally:
        eu.chiudi()
        if principale is not eu:
            principale.chiudi()


def corse(modello, ore, corsa=None):
    """Informazioni (latest.json o corsa richiesta) per i dataset necessari, con tutte le ore disponibili."""
    ds = [EU] + ([I2I] if modello == "icon2i" else [])
    base = fonti.ultimo(ds[-1])
    if base is None:
        return None, None
    rif = corsa and f"{corsa[:4]}-{corsa[4:6]}-{corsa[6:8]}T{corsa[8:10]}:00"
    candidati = [rif] if rif else []
    if not rif:
        t = datetime.strptime(base["reference_time"][:16], "%Y-%m-%dT%H:%M")
        # ICON-EU esce ogni 3 ore, ma le corse 03/09/15/21 arrivano solo a +30 h: provo a ritroso ogni 3 ore
        # (prima il passo era 6 ore e, partendo da una corsa "corta", trovava solo corse corte)
        passo, quante = (12, 4) if modello == "icon2i" else (3, 9)
        candidati = [(t - timedelta(hours=passo * k)).strftime("%Y-%m-%dT%H:%M") for k in range(quante)]
    for c in candidati:
        info = {d: fonti.info_corsa(d, c) for d in ds}
        if any(v is None for v in info.values()):
            continue
        t0 = datetime.strptime(c, "%Y-%m-%dT%H:%M")
        servono = [(t0 + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%MZ") for h in range(0, ore + 1, 3)]
        if all(all(s in info[d]["valid_times"] for s in servono) for d in ds):
            return t0.strftime("%Y%m%d%H"), info
    return None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modello", choices=("iconeu", "icon2i"), required=True)
    ap.add_argument("--ore", type=int, required=True)
    ap.add_argument("--trova", action="store_true")
    ap.add_argument("--corsa")
    ap.add_argument("--out")
    a = ap.parse_args()
    corsa, info = corse(a.modello, a.ore, a.corsa)
    if not corsa:
        sys.exit("ERRORE: nessuna corsa completa trovata")
    if a.trova:
        print(corsa)
        return
    os.makedirs(a.out, exist_ok=True)
    t0 = datetime.strptime(corsa, "%Y%m%d%H")
    for h in range(0, a.ore + 1, 3):
        scrivi_istante(a.modello, info, (t0 + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%MZ"), a.out)


if __name__ == "__main__":
    main()
