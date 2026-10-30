"""Colonna d'aria: temperatura di bulbo umido a tutte le quote e indici per il tipo di precipitazione.

Criterio (dell'autore del progetto, coerente con il metodo di Bourgouin 2000 sulle "aree" calde e fredde):
  - bulbo umido sotto 0 gradi: l'acqua diventa ghiaccio
  - tra 0 e 2 gradi: la neve tende a mantenersi
  - sopra 2 gradi: la neve tende a fondere
  e non conta solo il suolo ma tutta la colonna: se in quota c'e' uno strato sopra 2 gradi abbastanza spesso, la neve
  fonde e arriva liquida; se poi vicino al suolo c'e' uno strato sotto 0, la goccia ri-gela (granuli di ghiaccio) o
  gela al contatto con il suolo (gelicidio), ma NON torna neve.

Per ogni punto e ogni ora si calcolano (tra il suolo e 500 hPa, circa 5,5 km, dove nasce la precipitazione):
  TW2   bulbo umido del livello piu' basso del modello (gradi C)
  AFUS  "area di fusione": somma, strato per strato, di (bulbo umido - 2) x spessore dove il bulbo umido supera
        2 gradi (gradi C x metri). Dice quanta "occasione di fondere" trova la neve scendendo.
  ARIG  "area di rigelo": somma di (0 - bulbo umido) x spessore negli strati sotto 0 che stanno SOTTO lo strato di
        fusione piu' basso (gradi C x metri). Dice quanta occasione ha la goccia di ri-gelare prima del suolo.
  ZTW0  quota (m sul mare) sopra la quale il bulbo umido resta sempre sotto 0 fino a 500 hPa ("quota neve" teorica).
Il bulbo umido e' calcolato risolvendo l'equazione psicrometrica alla pressione di ogni livello (non con formule
valide solo al livello del mare).
"""
import numpy as np

G = 9.81


def pressione_vapore_saturo(tc):
    """hPa, sull'acqua (Bolton 1980)."""
    return 6.112 * np.exp(17.67 * tc / (tc + 243.5))


def bulbo_umido(tc, e, p):
    """Bulbo umido (gradi C) da temperatura (C), pressione del vapore (hPa) e pressione (hPa): equazione psicrometrica
    es(Tw) - e = gamma * p * (T - Tw), risolta con il metodo di Newton."""
    tc = np.asarray(tc, dtype="float64")
    e = np.minimum(np.asarray(e, dtype="float64"), pressione_vapore_saturo(tc))
    p = np.asarray(p, dtype="float64")
    td = 243.5 * np.log(np.maximum(e, 1e-3) / 6.112) / (17.67 - np.log(np.maximum(e, 1e-3) / 6.112))
    tw = tc - (tc - td) / 3.0
    for _ in range(8):
        gamma = 0.00066 * (1 + 0.00115 * tw)
        es = pressione_vapore_saturo(tw)
        f = es - e - gamma * p * (tc - tw)
        des = es * 17.67 * 243.5 / (tw + 243.5) ** 2
        df = des + gamma * p - 0.00066 * 0.00115 * p * (tc - tw)
        tw = tw - f / df
    return np.minimum(tw, tc)


def indici(T_k, p_pa, qv, z_w, hgt, p_cima=500.0):
    """T_k, p_pa, qv: [livello, y, x] sui livelli di massa; z_w: quota (m) delle interfacce [livello+1, y, x];
    hgt: quota del suolo [y, x]. Ritorna un dizionario di campi [y, x]."""
    tc = T_k - 273.15
    p = p_pa / 100.0
    e = qv * p / (0.622 + qv)
    tw = bulbo_umido(tc, e, p)
    dz = np.diff(z_w, axis=0)                       # spessore di ogni strato
    zc = 0.5 * (z_w[1:] + z_w[:-1])                  # quota del centro dello strato
    dentro = p >= p_cima
    caldo = (tw > 2.0) & dentro
    afus = np.sum(np.where(caldo, (tw - 2.0) * dz, 0.0), axis=0)
    nz = tw.shape[0]
    k = np.arange(nz)[:, None, None]
    # livello dello strato di fusione piu' basso (nz se non c'e')
    k_caldo = np.where(caldo, k, nz).min(axis=0)
    sotto = k < k_caldo[None]
    freddo = (tw < 0.0) & sotto & (k_caldo[None] < nz)
    arig = np.sum(np.where(freddo, -tw * dz, 0.0), axis=0)
    # quota sopra la quale il bulbo umido resta sotto 0 fino a 500 hPa
    sopra_zero = (tw > 0.0) & dentro
    k_alto = np.where(sopra_zero, k, -1).max(axis=0)          # ultimo livello (dal basso) ancora sopra 0
    ztw0 = np.where(k_alto < 0, hgt, np.take_along_axis(z_w, np.clip(k_alto + 1, 0, nz)[None], axis=0)[0])
    return {"TW2": tw[0].astype("float32"), "AFUS": afus.astype("float32"), "ARIG": arig.astype("float32"),
            "ZTW0": ztw0.astype("float32")}


def da_wrfout(nc):
    """Calcola gli indici da un file wrfout gia' aperto (netCDF4.Dataset)."""
    P = np.array(nc["P"][0]) + np.array(nc["PB"][0])
    theta = np.array(nc["T"][0]) + 300.0
    T = theta * (P / 1e5) ** 0.2854
    qv = np.maximum(np.array(nc["QVAPOR"][0]), 0.0)
    z_w = (np.array(nc["PH"][0]) + np.array(nc["PHB"][0])) / G
    hgt = np.array(nc["HGT"][0])
    return indici(T, P, qv, z_w, hgt)


# ------------------------------------------------------------------------------------------------- classificazione

PIOGGIA, MISTA, NEVE, GELICIDIO, GRANULI = 0, 1, 2, 3, 4
# soglie di Bourgouin (2000) in J/kg, convertite in gradi C x metri: area[J/kg] = g/273,15 x area[C x m]
A_NEVE = 5.6 * 273.15 / G        # circa 156: sotto, la neve non fonde abbastanza -> neve
A_PIOGGIA = 13.2 * 273.15 / G    # circa 368: sopra, la neve fonde del tutto -> liquida


def classifica(afus, arig, tw_suolo):
    """0 pioggia, 1 mista, 2 neve, 3 gelicidio (pioggia che gela al suolo), 4 granuli di ghiaccio."""
    afus = np.asarray(afus, dtype="float64")
    arig = np.asarray(arig, dtype="float64")
    tws = np.asarray(tw_suolo, dtype="float64")
    out = np.full(afus.shape, PIOGGIA, dtype=np.uint8)
    out[(afus >= A_NEVE) & (afus < A_PIOGGIA)] = MISTA
    out[(afus < A_NEVE) & (tws <= 2.0)] = NEVE
    liquida = afus >= A_PIOGGIA
    # Bourgouin: strato freddo profondo -> la goccia ri-gela in volo (granuli), sottile -> gela al suolo (gelicidio)
    pa = afus * G / 273.15
    na = arig * G / 273.15
    out[liquida & (na > 66 + 0.66 * pa)] = GRANULI
    out[liquida & (na <= 66 + 0.66 * pa) & (tws < 0.0) & (arig > 0)] = GELICIDIO
    return out


if __name__ == "__main__":
    # prove con profili inventati (dal basso in alto, ogni 250 m)
    z = np.arange(0, 6001, 250.0)
    def prova(nome, tw_fn):
        zc = 0.5 * (z[1:] + z[:-1])
        tc = tw_fn(zc)
        p = 1013.25 * np.exp(-zc / 8000) * 100
        qv = 0.622 * pressione_vapore_saturo(tc) / (p / 100) * 0.999   # aria quasi satura: bulbo umido ~ temperatura
        r = indici((tc + 273.15)[:, None, None], p[:, None, None], qv[:, None, None], z[:, None, None], np.zeros((1, 1)))
        c = classifica(r["AFUS"], r["ARIG"], r["TW2"])
        nomi = ["pioggia", "mista", "neve", "gelicidio", "granuli di ghiaccio"]
        print(f"{nome:55s} TW2 {r['TW2'][0,0]:5.1f}  AFUS {r['AFUS'][0,0]:6.0f}  ARIG {r['ARIG'][0,0]:6.0f}  ZTW0 {r['ZTW0'][0,0]:5.0f} -> {nomi[c[0,0]]}")
    prova("freddo ovunque (-2 al suolo)", lambda zc: -2 - 0.0065 * zc)
    prova("1 grado al suolo, freddo sopra (neve si mantiene)", lambda zc: 1 - 0.0065 * zc)
    prova("4 gradi al suolo, zero a 600 m", lambda zc: 4 - 0.0065 * zc)
    prova("10 gradi al suolo (pioggia)", lambda zc: 10 - 0.0065 * zc)
    prova("strato caldo +5 tra 1000 e 2000 m, -3 al suolo (sottile)",
          lambda zc: np.where(zc < 300, -3, np.where(zc < 1000, -3 + (zc - 300) / 700 * 8, np.where(zc < 2000, 5, 5 - (zc - 2000) * 0.0065))))
    prova("strato caldo +3 tra 1500 e 2000 m, -8 sotto 1200 m (spesso)",
          lambda zc: np.where(zc < 1200, -8, np.where(zc < 1500, -8 + (zc - 1200) / 300 * 11, np.where(zc < 2000, 3, 3 - (zc - 2000) * 0.0065))))


# ------------------------------------------------------------------------------------------------- parametri per il nowcasting

def _a_livello(campo, p, p_liv):
    """Valore di un campo 3D [livello, y, x] alla pressione p_liv (hPa), interpolato in log(p). NaN sotto il suolo."""
    lp = np.log(p)
    target = np.log(p_liv)
    nz = p.shape[0]
    # primo livello (dal basso) con pressione sotto il bersaglio
    sopra = (p < p_liv)
    k = np.argmax(sopra, axis=0)
    k = np.clip(k, 1, nz - 1)
    lp1 = np.take_along_axis(lp, (k - 1)[None], 0)[0]
    lp2 = np.take_along_axis(lp, k[None], 0)[0]
    c1 = np.take_along_axis(campo, (k - 1)[None], 0)[0]
    c2 = np.take_along_axis(campo, k[None], 0)[0]
    w = np.clip((target - lp1) / np.where(lp2 != lp1, lp2 - lp1, 1e-9), 0, 1)
    out = c1 + w * (c2 - c1)
    return np.where(p[0] < p_liv, np.nan, out)


def parametri_nowcasting(nc):
    """Dal wrfout: vento che spinge i temporali (media 850-500 hPa) e a 700 hPa, moto verticale e umidita' relativa a
    700 hPa, convergenza al suolo, energia disponibile per i temporali (MUCAPE semplificata, metodo della particella).
    Il vento e' ruotato verso nord/est."""
    P = (np.array(nc["P"][0]) + np.array(nc["PB"][0])) / 100.0
    theta = np.array(nc["T"][0]) + 300.0
    T = theta * (P / 1000.0) ** 0.2854
    qv = np.maximum(np.array(nc["QVAPOR"][0]), 0.0)
    U = np.array(nc["U"][0]); V = np.array(nc["V"][0]); W = np.array(nc["W"][0])
    u = 0.5 * (U[:, :, 1:] + U[:, :, :-1]); v = 0.5 * (V[:, 1:, :] + V[:, :-1, :]); w = 0.5 * (W[1:] + W[:-1])
    c = np.array(nc["COSALPHA"][0]); s = np.array(nc["SINALPHA"][0])
    ue = u * c - v * s
    ve = v * c + u * s
    e = qv * P / (0.622 + qv)
    ur = np.clip(100 * e / pressione_vapore_saturo(T - 273.15), 0, 100)
    out = {}
    # vento medio 850-500 hPa (pesato in pressione): e' quello che sposta le celle temporalesche
    pesi = np.where((P <= 850) & (P >= 500), 1.0, 0.0)
    dp = np.abs(np.gradient(P, axis=0)) * pesi
    den = np.maximum(dp.sum(0), 1e-6)
    out["USTEER"] = (ue * dp).sum(0) / den
    out["VSTEER"] = (ve * dp).sum(0) / den
    for nome, campo in (("U700", ue), ("V700", ve), ("W700", w), ("RH700", ur)):
        out[nome] = _a_livello(campo, P, 700.0)
    # energia per i temporali: particella del livello piu' instabile nei primi 3 km, sollevata (adiabatica secca fino
    # alla condensazione, poi umida); CAPE = somma di g (Tparticella - Tambiente)/Tambiente dz dove positivo
    z = (np.array(nc["PH"][0]) + np.array(nc["PHB"][0])) / G
    zc = 0.5 * (z[1:] + z[:-1])
    out["CAPE"] = _cape(T, P, qv, zc, z)
    return {k: np.asarray(v, dtype="float32") for k, v in out.items()}


def _cape(T, P, qv, zc, zw):
    Rd, cp, Lv, eps = 287.04, 1004.0, 2.5e6, 0.622
    nz = T.shape[0]
    hgt = zw[0]
    # temperatura potenziale equivalente (Bolton) per scegliere la particella piu' instabile nei primi 3 km
    e = qv * P / (eps + qv)
    td = 243.5 * np.log(np.maximum(e, 1e-3) / 6.112) / (17.67 - np.log(np.maximum(e, 1e-3) / 6.112)) + 273.15
    tl = 1 / (1 / (td - 56) + np.log(T / td) / 800) + 56
    the = T * (1000 / P) ** (0.2854 * (1 - 0.28 * qv)) * np.exp((3.376 / tl - 0.00254) * qv * 1000 * (1 + 0.81 * qv))
    bassi = (zc - hgt[None]) < 3000
    k0 = np.argmax(np.where(bassi, the, -1), axis=0)
    Tp = np.take_along_axis(T, k0[None], 0)[0]
    qp = np.take_along_axis(qv, k0[None], 0)[0]
    Pp = np.take_along_axis(P, k0[None], 0)[0]
    th_e_p = np.take_along_axis(the, k0[None], 0)[0]
    tlp = np.take_along_axis(tl, k0[None], 0)[0]
    plcl = Pp * (tlp / Tp) ** (1 / 0.2854)
    cape = np.zeros(Tp.shape)
    k = np.arange(nz)[:, None, None]
    for j in range(nz):
        sopra = j >= k0
        Pj, Tj = P[j], T[j]
        secco = Pj >= plcl
        Tsec = Tp * (Pj / Pp) ** 0.2854
        # sopra la condensazione: temperatura con la stessa theta-e della particella (Newton)
        Tu = np.where(secco, Tsec, Tj)
        for _ in range(6):
            es = 6.112 * np.exp(17.67 * (Tu - 273.15) / (Tu - 29.65))
            qs = eps * es / np.maximum(Pj - es, 1e-3)
            th = Tu * (1000 / Pj) ** (0.2854 * (1 - 0.28 * qs)) * np.exp((3.376 / np.maximum(Tu - 0.5, 150) - 0.00254) * qs * 1000 * (1 + 0.81 * qs))
            f = th - th_e_p
            Tu = Tu - f / np.maximum(th / Tu * (1 + 2.5e6 * qs / (461.5 * Tu)), 1e-3)
        Tpar = np.where(secco, Tsec, Tu)
        dz = zw[j + 1] - zw[j]
        b = 9.81 * (Tpar - Tj) / Tj
        cape += np.where(sopra & (b > 0) & (Pj > 150), b * dz, 0.0)
    return np.minimum(cape, 6000)
