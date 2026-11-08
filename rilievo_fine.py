#!/usr/bin/env python3
"""Mette nel dominio a 333 m (geo_em.d03.nc) il rilievo vero, a circa 30 m, mediato sulle celle del modello.

I dati geografici standard di WRF hanno il rilievo a circa 1 km (GMTED2010 30"): a 333 m le colline di Nerola e
i Lucretili sarebbero smussati. Qui si prende il rilievo delle Terrain Tiles (SRTM, circa 30 m, vedi sfondo.py),
si fa la media sull'area di ogni cella (333 x 333 m) e la si scrive in HGT_M. Vicino ai bordi si sfuma verso il
rilievo del dominio a 1 km, per non creare scalini tra un dominio e l'altro.

Uso: rilievo_fine.py geo_em.d03.nc
"""
import sys

import numpy as np
from netCDF4 import Dataset

import sfondo

BORDO = 6          # celle di sfumatura verso il rilievo standard


def main(percorso):
    nc = Dataset(percorso, "r+")
    lat = np.array(nc["XLAT_M"][0], dtype="float64")
    lon = np.array(nc["XLONG_M"][0], dtype="float64")
    vecchio = np.array(nc["HGT_M"][0], dtype="float64")
    m = 0.01
    z, (lon0, lat0, passo) = sfondo.dem((lon.min() - m, lon.max() + m, lat.min() - m, lat.max() + m), 13, 0.0003)
    z = z.astype("float64")
    # media su ogni cella: tutti i punti fini entro mezza cella (circa 167 m) dal centro
    dlat = 333.3 / 111320.0 / 2
    dlon = dlat / np.cos(np.radians(lat.mean()))
    nuovo = np.empty_like(vecchio)
    for j in range(lat.shape[0]):
        for i in range(lat.shape[1]):
            i0 = int(round((lon[j, i] - dlon - lon0) / passo)); i1 = int(round((lon[j, i] + dlon - lon0) / passo))
            j0 = int(round((lat[j, i] - dlat - lat0) / passo)); j1 = int(round((lat[j, i] + dlat - lat0) / passo))
            nuovo[j, i] = z[max(j0, 0):j1 + 1, max(i0, 0):i1 + 1].mean()
    ny, nx = vecchio.shape
    dist = np.minimum.reduce([np.arange(ny)[:, None] + 0 * np.arange(nx)[None], (ny - 1 - np.arange(ny))[:, None] + 0 * np.arange(nx)[None],
                              np.arange(nx)[None] + 0 * np.arange(ny)[:, None], (nx - 1 - np.arange(nx))[None] + 0 * np.arange(ny)[:, None]])
    w = np.clip(dist / BORDO, 0, 1)
    finale = w * nuovo + (1 - w) * vecchio
    print(f"rilievo d03: standard {vecchio.min():.0f}-{vecchio.max():.0f} m, nuovo {nuovo.min():.0f}-{nuovo.max():.0f} m, "
          f"differenza media {np.abs(finale - vecchio).mean():.0f} m, massima {np.abs(finale - vecchio).max():.0f} m")
    nc["HGT_M"][0] = finale.astype("float32")
    nc.close()


if __name__ == "__main__":
    main(sys.argv[1])
