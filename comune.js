// Codice comune alle pagine sondaggi.html e colonna.html (WRF Nerola + modelli di Mappe Lazio).
// Ogni pagina definisce la funzione disegnaTutto(), chiamata quando arrivano i dati o cambia l'ora.
"use strict";
// ------------------------------------------------------------------------------------------------ dentro il sito
// Quando la pagina e' incorporata in meteonerola.it (scheda "Modelli meteo"), il menu tra le due pagine lo fa il sito:
// i collegamenti portano alla scheda giusta della pagina del sito e il punto scelto finisce nell'indirizzo del sito.
const INCORPORATA = (() => { try { return window.parent !== window; } catch (e) { return true; } })();
if (INCORPORATA) document.documentElement.classList.add("incorporata");
function linkPagina(pagina, lat, lon) {                 // pagina: "sondaggi" o "colonna"
  const punto = lat.toFixed(4) + "," + lon.toFixed(4);
  if (INCORPORATA) { try { return { href: window.top.location.pathname + "#" + pagina + "/" + punto, target: "_top" }; } catch (e) { /* altro sito */ } }
  return { href: pagina + ".html#" + punto, target: "" };
}
function segnaPuntoNelSito(pagina, lat, lon) {
  if (!INCORPORATA) return;
  try { window.top.history.replaceState(null, "", "#" + pagina + "/" + lat.toFixed(4) + "," + lon.toFixed(4)); } catch (e) { /* niente */ }
}

// ------------------------------------------------------------------------------------------------ impostazioni
const NEROLA = { lat: 42.1606, lon: 12.7856 };
const LAZIO = [11.3, 14.1, 41.0, 42.8];                // area di Mappe Lazio: lon0, lon1, lat0, lat1
const RAW_WRF = "https://raw.githubusercontent.com/eremite1/wrf-nerola/risultati/";
const LIVELLI_OM = [1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 200, 150, 100];
// AROME: bordo orientale del territorio coperto (indicativo; dove non ci sono dati il modello risulta "fuori area")
const AROME_BORDO = [[43.3, 13.29], [42.8, 13.14], [41.0, 12.81], [40.6, 12.74]];

const MODELLI = [
  { id: "wrf1",   nome: "WRF 1 km",  ris: "1 km",   col: "--m1", tipo: "wrf", dom: "1km", passo_km: 1 },
  { id: "wrf3",   nome: "WRF 3 km",  ris: "3 km",   col: "--m2", tipo: "wrf", dom: "3km", passo_km: 3 },
  { id: "icon2i", nome: "ICON-2I",   ris: "2,2 km", col: "--m3", tipo: "om", api: ["italia_meteo_arpae_icon_2i"] },
  { id: "arome",  nome: "AROME",     ris: "2,5 km", col: "--m4", tipo: "om", api: ["meteofrance_arome_france"] },
  { id: "ecmwf",  nome: "ECMWF IFS", ris: "9 km",   col: "--m5", tipo: "om", api: ["ecmwf_ifs", "ecmwf_ifs025"] },
  { id: "ukmo",   nome: "UKMO",      ris: "10 km",  col: "--m6", tipo: "om", api: ["ukmo_global_deterministic_10km"] },
  { id: "iconeu", nome: "ICON-EU",   ris: "7 km",   col: "--m7", tipo: "om", api: ["dwd_icon_eu"] },
];

const $ = (id) => document.getElementById(id);
const nf0 = new Intl.NumberFormat("it-IT", { maximumFractionDigits: 0 });
const nf1 = new Intl.NumberFormat("it-IT", { minimumFractionDigits: 1, maximumFractionDigits: 1 });
const colore = (m) => getComputedStyle(document.documentElement).getPropertyValue(m.col).trim();
const fmt = (v, f = nf0, u = "") => (v == null || !isFinite(v)) ? "–" : f.format(v) + u;

// ------------------------------------------------------------------------------------------------ fisica
// Costanti e formule standard; nessuna correzione empirica.
const RD = 287.04, CP = 1005.7, LV = 2.501e6, EPS = 0.622, G = 9.80665, KAP = RD / CP;
const es = (tc) => 6.112 * Math.exp(17.67 * tc / (tc + 243.5));                     // hPa (Bolton 1980)
const mescolanza = (p, tc) => EPS * es(tc) / Math.max(p - es(tc), 1e-3);              // kg/kg, saturazione a tc
const rugiadaDaE = (e) => { const l = Math.log(Math.max(e, 1e-4) / 6.112); return 243.5 * l / (17.67 - l); };
const rugiadaDaUR = (tc, ur) => rugiadaDaE(es(tc) * Math.max(ur, 0.5) / 100);
const theta = (tc, p) => (tc + 273.15) * Math.pow(1000 / p, KAP);
function thetaE(tc, tdc, p) {                                                         // Bolton (1980), eq. 43
  const T = tc + 273.15, Td = tdc + 273.15, r = mescolanza(p, tdc);
  const TL = 1 / (1 / (Td - 56) + Math.log(T / Td) / 800) + 56;
  return T * Math.pow(1000 / p, 0.2854 * (1 - 0.28 * r)) * Math.exp((3.376 / TL - 0.00254) * r * 1000 * (1 + 0.81 * r));
}
function bulboUmido(tc, tdc, p) {                       // equazione psicrometrica alla pressione p (come colonna.py)
  const e = Math.min(es(tdc), es(tc));
  let tw = tc - (tc - tdc) / 3;
  for (let i = 0; i < 10; i++) {
    const gam = 0.00066 * (1 + 0.00115 * tw), est = es(tw);
    const f = est - e - gam * p * (tc - tw);
    const df = est * 17.67 * 243.5 / Math.pow(tw + 243.5, 2) + gam * p - 0.00066 * 0.00115 * p * (tc - tw);
    tw -= f / df;
  }
  return Math.min(tw, tc);
}
function lcl(tc, tdc, p) {                              // Bolton (1980), eq. 15: temperatura e pressione di condensazione
  const T = tc + 273.15, Td = Math.min(tdc, tc) + 273.15;
  const TL = 1 / (1 / (Td - 56) + Math.log(T / Td) / 800) + 56;
  return { t: TL - 273.15, p: p * Math.pow(TL / T, 1 / KAP) };
}
function dTdlnp(T, p) {                                 // adiabatica satura (pseudo-adiabatica), T in K, p in hPa
  const rs = mescolanza(p, T - 273.15);
  return (RD * T + LV * rs) / (CP + LV * LV * rs * EPS / (RD * T * T));
}
function satura(T0c, p0, p1) {                          // integra l'adiabatica satura da p0 a p1 (Runge-Kutta 4)
  let T = T0c + 273.15, lp = Math.log(p0);
  const lp1 = Math.log(p1), n = Math.max(2, Math.ceil(Math.abs(lp1 - lp) / 0.01)), h = (lp1 - lp) / n;
  for (let i = 0; i < n; i++) {
    const p = Math.exp(lp);
    const k1 = dTdlnp(T, p), k2 = dTdlnp(T + h * k1 / 2, Math.exp(lp + h / 2));
    const k3 = dTdlnp(T + h * k2 / 2, Math.exp(lp + h / 2)), k4 = dTdlnp(T + h * k3, Math.exp(lp + h));
    T += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6; lp += h;
  }
  return T - 273.15;
}
function particella(tc, tdc, p0, pressioni) {           // temperatura della particella sollevata alle pressioni date
  const c = lcl(tc, tdc, p0);
  let pPrec = c.p, tPrec = c.t;
  return pressioni.map((p) => {
    if (p >= c.p) return (tc + 273.15) * Math.pow(p / p0, KAP) - 273.15;
    const t = satura(tPrec, pPrec, p); pPrec = p; tPrec = t; return t;
  });
}

// ------------------------------------------------------------------------------------------------ profili
// Un profilo e' una lista di livelli {p, z, t, td, u, v} dal suolo verso l'alto (p decrescente); il primo e' il suolo.
function interp(prof, p, k) {
  if (!prof || prof.length < 2 || p > prof[0].p + 0.01) return null;
  const lp = Math.log(p);
  for (let i = 0; i < prof.length - 1; i++) {
    const a = prof[i], b = prof[i + 1];
    if (p <= a.p && p >= b.p) {
      if (a[k] == null || b[k] == null) return null;
      const w = (Math.log(a.p) - lp) / (Math.log(a.p) - Math.log(b.p) || 1);
      return a[k] + w * (b[k] - a[k]);
    }
  }
  return null;
}
function interpZ(prof, z, k) {
  for (let i = 0; i < prof.length - 1; i++) {
    const a = prof[i], b = prof[i + 1];
    if (z >= a.z && z <= b.z && a[k] != null && b[k] != null) return a[k] + (z - a.z) / ((b.z - a.z) || 1) * (b[k] - a[k]);
  }
  return null;
}
function quotaDiCambio(prof, valore) {                  // quota piu' alta in cui il valore passa da >0 a <=0
  const v = prof.map(valore);
  if (v.every((x) => x <= 0)) return { z: prof[0].z, alSuolo: true };
  for (let i = prof.length - 2; i >= 0; i--) {
    if (v[i] > 0 && v[i + 1] <= 0) return { z: prof[i].z + v[i] / (v[i] - v[i + 1]) * (prof[i + 1].z - prof[i].z) };
    if (v[i + 1] > 0) return null;
  }
  return null;
}
function areeParticella(prof, t0, td0, p0) {            // CAPE e CIN (J/kg) di una particella che parte da p0
  const passi = [];
  for (let p = p0; p >= Math.max(prof[prof.length - 1].p, 100); p -= 5) passi.push(p);
  const tp = particella(t0, td0, p0, passi);
  let cape = 0, cin = 0, sopraLfc = false, negativa = 0;
  for (let i = 0; i < passi.length - 1; i++) {
    const te1 = interp(prof, passi[i], "t"), te2 = interp(prof, passi[i + 1], "t");
    if (te1 == null || te2 == null) continue;
    const b = 0.5 * ((tp[i] - te1) / (te1 + 273.15) + (tp[i + 1] - te2) / (te2 + 273.15));
    const area = RD * 0.5 * ((te1 + 273.15) + (te2 + 273.15)) * b * Math.log(passi[i] / passi[i + 1]);
    if (area > 0) { cape += area; if (!sopraLfc) { sopraLfc = true; cin = negativa; } }
    else if (!sopraLfc) negativa += area;
  }
  return { cape, cin: sopraLfc ? cin : null };
}
function indici(prof) {
  const s = prof[0], T = (p) => interp(prof, p, "t"), Td = (p) => interp(prof, p, "td"), Z = (p) => interp(prof, p, "z");
  const out = {};
  const c = lcl(s.t, s.td, s.p);
  out.lclP = c.p; out.lclZ = (() => { const z = Z(c.p); return z == null ? null : z - s.z; })();
  const sb = areeParticella(prof, s.t, s.td, s.p);
  out.cape = sb.cape; out.cin = sb.cin;
  let mu = 0;                                             // particella piu' instabile nei 300 hPa piu' bassi
  for (const l of prof) if (l.p >= s.p - 300 && l.td != null) mu = Math.max(mu, areeParticella(prof, l.t, l.td, l.p).cape);
  out.mucape = mu;
  const t500 = T(500), t850 = T(850), td850 = Td(850), t700 = T(700), td700 = Td(700);
  out.li = t500 == null ? null : t500 - particella(s.t, s.td, s.p, [500])[0];
  out.si = (t500 == null || t850 == null || td850 == null) ? null : t500 - particella(t850, td850, 850, [500])[0];
  out.k = [t850, t500, td850, t700, td700].some((x) => x == null) ? null : (t850 - t500) + td850 - (t700 - td700);
  out.tt = [t850, td850, t500].some((x) => x == null) ? null : t850 + td850 - 2 * t500;
  out.gradiente = (t850 == null || t500 == null || Z(500) == null || Z(850) == null) ? null : (t850 - t500) / (Z(500) - Z(850)) * 1000;
  let pw = 0;                                             // acqua precipitabile fino a 300 hPa (mm)
  for (let i = 0; i < prof.length - 1; i++) {
    const a = prof[i], b = prof[i + 1];
    if (b.p < 300 || a.td == null || b.td == null) break;
    pw += 0.5 * (mescolanza(a.p, a.td) + mescolanza(b.p, b.td)) * (a.p - b.p) * 100 / G;
  }
  out.pw = pw;
  out.zero = quotaDiCambio(prof, (l) => l.t);
  out.zeroTw = quotaDiCambio(prof, (l) => l.td == null ? l.t : bulboUmido(l.t, l.td, l.p));
  const u6 = interpZ(prof, s.z + 6000, "u"), v6 = interpZ(prof, s.z + 6000, "v");
  out.shear = (u6 == null || s.u == null) ? null : Math.hypot(u6 - s.u, v6 - s.v);
  return out;
}

// ------------------------------------------------------------------------------------------------ dati WRF
let baseWrf = null, indiceWrf = null;
const griglie = {};
async function caricaIndiceWrf() {
  for (const base of ["", RAW_WRF]) {
    try {
      const r = await fetch(base + "indice.json?t=" + Date.now());
      if (!r.ok) continue;
      indiceWrf = await r.json(); baseWrf = base; return;
    } catch (e) { /* provo l'altro indirizzo */ }
  }
}
async function griglia(dom) {
  if (griglie[dom]) return griglie[dom];
  const D = indiceWrf.domini[dom];
  const r = await fetch(baseWrf + D.griglia + "?v=" + encodeURIComponent(indiceWrf.creato));
  const a = new Float32Array(await r.arrayBuffer());
  const n = D.layout.ny * D.layout.nx;
  griglie[dom] = { n, ny: D.layout.ny, nx: D.layout.nx, lat: a.subarray(0, n), lon: a.subarray(n, 2 * n), hgt: a.subarray(2 * n, 3 * n) };
  return griglie[dom];
}
function vicino(g, lat, lon) {
  const c = Math.cos(lat * Math.PI / 180);
  let m = Infinity, k = -1;
  for (let i = 0; i < g.n; i++) {
    const dy = g.lat[i] - lat, dx = (g.lon[i] - lon) * c, d = dy * dy + dx * dx;
    if (d < m) { m = d; k = i; }
  }
  return { k, km: 111.2 * Math.sqrt(m) };
}
async function datiWrf(m, lat, lon) {
  if (!indiceWrf) return { stato: "WRF non disponibile" };
  const D = indiceWrf.domini[m.dom];
  if (!D) return { stato: "dominio non pubblicato" };
  const g = await griglia(m.dom);
  const v = vicino(g, lat, lon);
  if (v.km > m.passo_km * 0.75) return { stato: "fuori area" };
  const S = D.sondaggi;
  if (!S) return { stato: "dalla prossima corsa" };
  const nv = S.n_suolo + S.livelli.length * S.n_livello, nt = S.tempi.length;
  const inizio = v.k * nt * nv * 2, fine = inizio + nt * nv * 2 - 1;
  const r = await fetch(baseWrf + S.file + "?v=" + encodeURIComponent(indiceWrf.creato), { headers: { Range: `bytes=${inizio}-${fine}` } });
  if (!r.ok) throw new Error("HTTP " + r.status);
  let buf = await r.arrayBuffer();
  if (r.status !== 206) buf = buf.slice(inizio, fine + 1);
  const a = new Int16Array(buf), M = -32768;
  const profili = {};
  S.tempi.forEach((t, it) => {
    const o = it * nv;
    if (a[o] === M) return;
    const prof = [{ p: a[o] / 10, z: a[o + 1], t: a[o + 2] / 10, td: a[o + 3] / 10, u: a[o + 4] / 10, v: a[o + 5] / 10 }];
    S.livelli.forEach((p, j) => {
      const b = o + S.n_suolo + j * S.n_livello;
      if (a[b] === M || a[b + 1] === M || p >= prof[0].p) return;
      prof.push({ p, z: a[b], t: a[b + 1] / 10, td: a[b + 2] === M ? null : a[b + 2] / 10,
                  u: a[b + 3] === M ? null : a[b + 3] / 10, v: a[b + 4] === M ? null : a[b + 4] / 10 });
    });
    if (prof.length > 3) profili[t] = prof;
  });
  const c = indiceWrf.corsa;
  return { profili, quotaModello: g.hgt[v.k], distanza: v.km,
           corsa: `${c.slice(6, 8)}/${c.slice(4, 6)} ${c.slice(8, 10)} UTC` };
}

// ------------------------------------------------------------------------------------------------ dati Open-Meteo
function urlOm(api, lat, lon) {
  const v = ["temperature_2m", "dew_point_2m", "surface_pressure", "wind_speed_10m", "wind_direction_10m"];
  for (const p of LIVELLI_OM) for (const k of ["temperature", "relative_humidity", "geopotential_height", "wind_speed", "wind_direction"]) v.push(`${k}_${p}hPa`);
  return "https://api.open-meteo.com/v1/forecast?latitude=" + lat.toFixed(4) + "&longitude=" + lon.toFixed(4) +
         "&hourly=" + v.join(",") + "&models=" + api + "&wind_speed_unit=ms&timezone=GMT&forecast_days=4" +
         "&cell_selection=nearest&elevation=nan";
}
function venti(vel, dir) {
  if (vel == null || dir == null) return [null, null];
  const r = dir * Math.PI / 180;
  return [-vel * Math.sin(r), -vel * Math.cos(r)];
}
async function datiOm(m, lat, lon) {
  let j = null, ultimoErr = null;
  for (const api of m.api) {
    try {
      const r = await fetch(urlOm(api, lat, lon));
      if (!r.ok) { ultimoErr = new Error("HTTP " + r.status); continue; }
      j = await r.json(); break;
    } catch (e) { ultimoErr = e; }
  }
  if (!j) throw ultimoErr || new Error("nessuna risposta");
  const H = j.hourly, profili = {};
  H.time.forEach((t, i) => {
    const ps = H.surface_pressure[i], t2 = H.temperature_2m[i];
    if (ps == null || t2 == null) return;
    const td2 = H.dew_point_2m[i];
    const [u, v] = venti(H.wind_speed_10m[i], H.wind_direction_10m[i]);
    const liv = [];
    for (const p of LIVELLI_OM) {
      const tt = H[`temperature_${p}hPa`]?.[i], z = H[`geopotential_height_${p}hPa`]?.[i];
      if (tt == null || z == null || p >= ps - 0.5) continue;
      const ur = H[`relative_humidity_${p}hPa`]?.[i];
      const [lu, lv] = venti(H[`wind_speed_${p}hPa`]?.[i], H[`wind_direction_${p}hPa`]?.[i]);
      liv.push({ p, z, t: tt, td: ur == null ? null : Math.min(rugiadaDaUR(tt, ur), tt), u: lu, v: lv });
    }
    if (liv.length < 3) return;
    // quota del suolo del modello: equazione ipsometrica dal primo livello sopra il suolo
    const a = liv[0], tm = 0.5 * (t2 + a.t) + 273.15;
    const zs = a.z - RD * tm / G * Math.log(ps / a.p);
    profili[t + "Z"] = [{ p: ps, z: zs, t: t2, td: td2 == null ? null : Math.min(td2, t2), u, v }, ...liv];
  });
  if (!Object.keys(profili).length) return { stato: "fuori area" };
  const primo = Object.values(profili)[0][0];
  return { profili, quotaModello: primo.z };
}

// ------------------------------------------------------------------------------------------------ stato della pagina
const stato = { punto: null, risultati: {}, tempi: [], i: 0, visibili: new Set(MODELLI.map((m) => m.id)), vista: "grafico", gen: 0 };

function etichettaTempo(iso) {
  const d = new Date(iso);
  const s = d.toLocaleString("it-IT", { weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit", timeZone: "Europe/Rome" });
  return s.charAt(0).toUpperCase() + s.slice(1);
}
function aggiornaTempi() {
  const adesso = Date.now() - 3 * 3600e3, tutti = new Set();
  for (const r of Object.values(stato.risultati)) if (r && r.profili) for (const t of Object.keys(r.profili)) if (Date.parse(t) >= adesso) tutti.add(t);
  const vecchio = stato.tempi[stato.i];
  stato.tempi = [...tutti].sort();
  let i = vecchio ? stato.tempi.indexOf(vecchio) : -1;
  if (i < 0) {                       // l'ora piu' vicina a adesso tra quelle ogni 3 ore (ci sono anche i WRF)
    let m = Infinity;
    stato.tempi.forEach((t, j) => { const d = Math.abs(Date.parse(t) - Date.now()) + (new Date(t).getUTCHours() % 3 ? 1e9 : 0); if (d < m) { m = d; i = j; } });
  }
  stato.i = Math.max(0, i);
  $("cursore").max = Math.max(0, stato.tempi.length - 1);
  $("cursore").value = stato.i;
}
function profiloDi(id) {
  const r = stato.risultati[id];
  return r && r.profili ? r.profili[stato.tempi[stato.i]] || null : null;
}
function statoModello(m) {
  const r = stato.risultati[m.id];
  if (r === undefined) return "caricamento…";
  if (r.errore) return "errore";
  if (r.stato) return r.stato;
  if (!profiloDi(m.id)) return m.tipo === "wrf" ? "solo ogni 3 ore" : "nessun dato a quest'ora";
  return "";
}
const disponibili = () => MODELLI.filter((m) => profiloDi(m.id));
const mostrati = () => disponibili().filter((m) => stato.visibili.has(m.id));

// ------------------------------------------------------------------------------------------------ mappa
let mappa, marcatore;
function riquadro(lat0, lat1, lon0, lon1) { return [[lat0, lon0], [lat0, lon1], [lat1, lon1], [lat1, lon0]]; }
async function preparaMappa() {
  mappa = L.map("mappa", { zoomSnap: 0.25 }).fitBounds([[LAZIO[2], LAZIO[0]], [LAZIO[3], LAZIO[1]]]);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 13, attribution: "© OpenStreetMap" }).addTo(mappa);
  const voci = [];
  const aggiungi = (nome, nota, col, forma, tratt) => {
    const strato = L.polygon(forma, { color: col, weight: 2, fillOpacity: 0.04, dashArray: tratt, interactive: false }).addTo(mappa);
    voci.push({ nome, nota, col, strato, tratt });
  };
  const cv = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  aggiungi("ICON-2I · ECMWF · UKMO · ICON-EU", "tutto il Lazio (e oltre)", cv("--m5"), riquadro(LAZIO[2], LAZIO[3], LAZIO[0], LAZIO[1]), "6 4");
  aggiungi("AROME", "solo a ovest della linea (indicativo)", cv("--m4"),
           [[43.3, 10.9], ...AROME_BORDO, [40.6, 10.9]], "2 4");
  await caricaIndiceWrf();
  if (indiceWrf) {
    for (const [dom, nome, col] of [["3km", "WRF 3 km", cv("--m2")], ["1km", "WRF 1 km", cv("--m1")]]) {
      if (!indiceWrf.domini[dom]) continue;
      const g = await griglia(dom), b = [];
      const P = (y, x) => [g.lat[y * g.nx + x], g.lon[y * g.nx + x]];
      for (let x = 0; x < g.nx; x++) b.push(P(0, x));
      for (let y = 0; y < g.ny; y++) b.push(P(y, g.nx - 1));
      for (let x = g.nx - 1; x >= 0; x--) b.push(P(g.ny - 1, x));
      for (let y = g.ny - 1; y >= 0; y--) b.push(P(y, 0));
      aggiungi(nome, dom === "1km" ? "Nerola, alta risoluzione" : "Lazio centrale", col, b, null);
    }
  }
  const leg = $("legenda");
  voci.reverse().forEach((v) => {
    const l = document.createElement("label");
    l.innerHTML = `<input type="checkbox" checked><span class="riq" style="border-color:${v.col};border-style:${v.tratt ? "dashed" : "solid"}"></span><span>${v.nome} <small>${v.nota}</small></span>`;
    l.querySelector("input").addEventListener("change", (e) => e.target.checked ? v.strato.addTo(mappa) : v.strato.remove());
    leg.appendChild(l);
  });
  mappa.on("click", (e) => scegli(e.latlng.lat, e.latlng.lng));
}

async function scegli(lat, lon) {
  const gen = ++stato.gen;
  stato.punto = { lat, lon };
  stato.risultati = {};
  history.replaceState(null, "", "#" + lat.toFixed(4) + "," + lon.toFixed(4));
  for (const id of ["link-sondaggi", "link-colonna"]) { const l = document.getElementById(id); if (l) l.hash = lat.toFixed(4) + "," + lon.toFixed(4); }
  segnaPuntoNelSito(PAGINA, lat, lon);
  if (!marcatore) marcatore = L.circleMarker([lat, lon], { radius: 7, color: "#fff", weight: 2, fillColor: "#d6341f", fillOpacity: 1 }).addTo(mappa);
  else marcatore.setLatLng([lat, lon]);
  $("intestazione").innerHTML = `<h2>${lat.toFixed(3)}° N, ${lon.toFixed(3)}° E</h2><div class="sub">Carico i dati dei modelli…</div>`;
  disegnaTutto();
  await Promise.all(MODELLI.map(async (m) => {
    let r;
    try { r = m.tipo === "wrf" ? await datiWrf(m, lat, lon) : await datiOm(m, lat, lon); }
    catch (e) { r = { errore: String(e.message || e) }; }
    if (gen !== stato.gen) return;
    stato.risultati[m.id] = r;
    aggiornaTempi(); disegnaTutto();
  }));
}

// ------------------------------------------------------------------------------------------------ disegno SVG
const NS = "http://www.w3.org/2000/svg";
function el(nome, attr, figlio) {
  const e = document.createElementNS(NS, nome);
  for (const k in attr) e.setAttribute(k, attr[k]);
  if (figlio) e.textContent = figlio;
  return e;
}
function percorso(punti) {
  let d = "", su = false;
  for (const [x, y] of punti) {
    if (x == null || !isFinite(x)) { su = false; continue; }
    d += (su ? "L" : "M") + x.toFixed(1) + " " + y.toFixed(1); su = true;
  }
  return d;
}
