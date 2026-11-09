#!/usr/bin/env python3
"""Scrive il namelist.input di WRF per i due domini su Nerola (3 km e 1 km).

Uso:
  namelist.py --corsa 2026102400 --da 0 --a 24 --nlev 15 --nsoil 4 > namelist.input

--corsa  e' l'inizio della previsione (AAAAMMGGHH, UTC)
--da/--a sono ore dall'inizio della corsa. Con --da maggiore di zero il calcolo
         riparte dal file di ripresa (restart) scritto dal blocco precedente.
Per real.exe si usa --da 0 --a (durata totale della previsione).
"""
import argparse
import re
import sys
from datetime import datetime, timedelta

MODELLO = """&time_control
 run_days = 0,
 run_hours = @ORE@,
 run_minutes = 0,
 run_seconds = 0,
 start_year = @SY@, @SY@,
 start_month = @SMO@, @SMO@,
 start_day = @SD@, @SD@,
 start_hour = @SH@, @SH@,
 start_minute = 00, 00,
 start_second = 00, 00,
 end_year = @EY@, @EY@,
 end_month = @EMO@, @EMO@,
 end_day = @ED@, @ED@,
 end_hour = @EH@, @EH@,
 end_minute = 00, 00,
 end_second = 00, 00,
 interval_seconds = 10800,
 input_from_file = .true., .true.,
 history_interval = 60, 60,
 frames_per_outfile = 1, 1,
 restart = @RESTART@,
 restart_interval = @RINT@,
 io_form_history = 2,
 io_form_restart = 2,
 io_form_input = 2,
 io_form_boundary = 2,
 debug_level = 0,
/
&domains
 time_step = 18,
 time_step_fract_num = 0,
 time_step_fract_den = 1,
@ADATTIVO@ max_dom = 2,
 e_we = 81, 73,
 e_sn = 81, 73,
 e_vert = 45, 45,
 p_top_requested = 5000,
 num_metgrid_levels = @NLEV@,
 num_metgrid_soil_levels = @NSOIL@,
 dx = 3000, 1000,
 dy = 3000, 1000,
 grid_id = 1, 2,
 parent_id = 0, 1,
 i_parent_start = 1, 29,
 j_parent_start = 1, 29,
 parent_grid_ratio = 1, 3,
 parent_time_step_ratio = 1, 3,
 feedback = 1,
 smooth_option = 0,
/
&physics
 mp_physics = 6, 6,
 ra_lw_physics = 4, 4,
 ra_sw_physics = 4, 4,
 radt = 10, 6,
 sf_sfclay_physics = 5, 5,
 sf_surface_physics = 2, 2,
 bl_pbl_physics = 5, 5,
 bldt = 0, 0,
 cu_physics = 0, 0,
 cudt = 0, 0,
 num_soil_layers = 4,
 num_land_cat = 21,
 surface_input_source = 1,
 sf_urban_physics = 0, 0,
 icloud = 1,
 isfflx = 1,
 ifsnow = 1,
 slope_rad = 1, 1,
 topo_shading = 1, 1,
/
&dynamics
 hybrid_opt = 2,
 w_damping = 1,
 diff_opt = 2, 2,
 km_opt = 4, 4,
 diff_6th_opt = 0, 0,
 diff_6th_factor = 0.12, 0.12,
 base_temp = 290.,
 damp_opt = 3,
 zdamp = 5000., 5000.,
 dampcoef = 0.2, 0.2,
 khdif = 0, 0,
 kvdif = 0, 0,
 non_hydrostatic = .true., .true.,
 moist_adv_opt = 1, 1,
 scalar_adv_opt = 1, 1,
 gwd_opt = 0, 0,
/
&bdy_control
 spec_bdy_width = 5,
 specified = .true., .false.,
 nested = .false., .true.,
/
&namelist_quilt
 nio_tasks_per_group = 0,
 nio_groups = 1,
/
"""


# terzo dominio (333 m su Nerola, catena "fine"): valore per ogni chiave; le altre ripetono quello del dominio 2
TERZO = {"e_we": "61", "e_sn": "61", "dx": "333.3333", "dy": "333.3333", "grid_id": "3", "parent_id": "2",
         "i_parent_start": "27", "j_parent_start": "27", "parent_grid_ratio": "3", "parent_time_step_ratio": "3",
         "radt": "2", "starting_time_step": "2", "max_time_step": "4", "min_time_step": "1",
         "specified": ".false.", "nested": ".true."}


def tre_domini(t):
    """Aggiunge il terzo dominio a ogni riga con un valore per dominio, e lo rende un annidamento a senso unico."""
    out = []
    for riga in t.splitlines():
        m = re.match(r"^( *)(\w+)( *= *)(.+?),\s*$", riga)
        if m and m.group(4).count(",") == 1:
            chiave = m.group(2)
            v2 = m.group(4).split(",")[1].strip()
            riga = f"{riga.rstrip().rstrip(',')}, {TERZO.get(chiave, v2)},"
        out.append(riga)
    t = "\n".join(out) + "\n"
    t = t.replace(" max_dom = 2,", " max_dom = 3,").replace(" feedback = 1,", " feedback = 0,")
    t = t.replace(" adaptation_domain = 2,", " adaptation_domain = 3,")
    return t


# metodi di calcolo ("fisica") per il confronto delle varianti: F1 e' quella della catena principale
FISICHE = {
    "F1": {},                                                                         # WSM6, MYNN, Noah
    "F2": {"mp_physics": "8", "bl_pbl_physics": "1", "sf_sfclay_physics": "1"},       # Thompson, YSU
    "F3": {"mp_physics": "10", "bl_pbl_physics": "7", "sf_sfclay_physics": "1",
           "sf_surface_physics": "4"},                                                # Morrison, ACM2, Noah-MP
    "F4": {"mp_physics": "8", "bl_pbl_physics": "2", "sf_sfclay_physics": "2"},       # Thompson, MYJ
}


def un_dominio(t):
    """Solo il dominio a 3 km: ogni riga con un valore per dominio tiene il primo."""
    out = []
    for riga in t.splitlines():
        m = re.match(r"^( *)(\w+)( *= *)(.+?),\s*$", riga)
        if m and m.group(4).count(",") == 1:
            riga = f"{m.group(1)}{m.group(2)}{m.group(3)}{m.group(4).split(',')[0].strip()},"
        out.append(riga)
    t = "\n".join(out) + "\n"
    return t.replace(" max_dom = 2,", " max_dom = 1,").replace(" adaptation_domain = 2,", " adaptation_domain = 1,")


def con_fisica(t, codice):
    for chiave, valore in FISICHE[codice].items():
        t = re.sub(rf"^( *{chiave} *= *)([^\n]*)$", lambda m: m.group(1) + ", ".join([valore] * m.group(2).count(",")) + ",",
                   t, flags=re.M)
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corsa", required=True)
    ap.add_argument("--da", type=int, default=0)
    ap.add_argument("--a", type=int, required=True)
    ap.add_argument("--nlev", type=int, default=15)
    ap.add_argument("--nsoil", type=int, default=4)
    ap.add_argument("--fisso", action="store_true",
                    help="passo di tempo fisso (18 s e 6 s) invece di quello adattivo: si usa se l'adattivo fallisce")
    ap.add_argument("--domini", type=int, default=2, choices=(1, 2, 3), help="1 = solo 3 km; 3 = aggiunge il dominio a 333 m su Nerola")
    ap.add_argument("--fisica", default="F1", choices=sorted(FISICHE), help="metodo di calcolo (varianti di confronto)")
    a = ap.parse_args()
    if a.a <= a.da:
        sys.exit("--a deve essere maggiore di --da")
    c = datetime.strptime(a.corsa, "%Y%m%d%H")
    inizio = c + timedelta(hours=a.da)
    fine = c + timedelta(hours=a.a)
    ore = a.a - a.da
    sost = {
        "ORE": ore,
        "SY": inizio.strftime("%Y"), "SMO": inizio.strftime("%m"), "SD": inizio.strftime("%d"), "SH": inizio.strftime("%H"),
        "EY": fine.strftime("%Y"), "EMO": fine.strftime("%m"), "ED": fine.strftime("%d"), "EH": fine.strftime("%H"),
        "RESTART": ".true." if a.da > 0 else ".false.",
        "RINT": ore * 60,          # un file di ripresa alla fine del blocco
        "NLEV": a.nlev, "NSOIL": a.nsoil,
        # passo di tempo ADATTIVO: invece di avanzare sempre di 18 s (3 km) e 6 s (1 km), WRF allunga il passo quando
        # i venti sono deboli e lo accorcia quando sono forti, restando sempre stabile (numero di Courant 1,2).
        # Con tempo normale il calcolo diventa molto piu' veloce; le uscite restano alle ore esatte.
        "ADATTIVO": "" if a.fisso else (
            " use_adaptive_time_step = .true.,\n"
            " step_to_output_time = .true.,\n"
            " target_cfl = 1.2, 1.2,\n"
            " target_hcfl = 0.84, 0.84,\n"
            " max_step_increase_pct = 5, 51,\n"
            " starting_time_step = 18, 6,\n"
            " max_time_step = 36, 12,\n"
            " min_time_step = 9, 3,\n"
            " adaptation_domain = 2,\n"),
    }
    t = MODELLO
    for k, v in sost.items():
        t = t.replace("@" + k + "@", str(v))
    if a.domini == 3:
        t = tre_domini(t)
    elif a.domini == 1:
        t = un_dominio(t)
    t = con_fisica(t, a.fisica)
    if "@" in t:
        sys.exit("segnaposto non sostituito: " + t[t.index("@"):t.index("@") + 20])
    sys.stdout.write(t)


if __name__ == "__main__":
    main()
