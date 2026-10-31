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
 max_dom = 2,
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corsa", required=True)
    ap.add_argument("--da", type=int, default=0)
    ap.add_argument("--a", type=int, required=True)
    ap.add_argument("--nlev", type=int, default=15)
    ap.add_argument("--nsoil", type=int, default=4)
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
    }
    t = MODELLO
    for k, v in sost.items():
        t = t.replace("@" + k + "@", str(v))
    if "@" in t:
        sys.exit("segnaposto non sostituito: " + t[t.index("@"):t.index("@") + 20])
    sys.stdout.write(t)


if __name__ == "__main__":
    main()
