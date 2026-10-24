#!/usr/bin/env bash
# Ritaglia i dati ECMWF sull'Italia centrale (latitudini da sud a nord, come vuole WPS) e rimette le etichette
# giuste agli strati di suolo (WPS le riconosce per profondita' in cm).
#   prepara_grib.sh CARTELLA_GREZZI CARTELLA_FINALE ORE
# Legge sfc_<ora>.grib2 e pl_<ora>.grib2, scrive gli stessi nomi nella cartella finale.
set -euo pipefail
RAW=$(cd "$1" && pwd); mkdir -p "$2"; FIN=$(cd "$2" && pwd); ORE=$3

passo() {
  local s=$1 T k p D1 D2
  T=$(mktemp -d)
  for t in sfc pl; do
    cdo -s -f grb2 invertlat -sellonlatbox,9.5,16,39.5,44.5 "$RAW/${t}_$s.grib2" "$T/${t}.grib2" 2>&1 | tail -3
  done
  cp "$T/pl.grib2" "$FIN/pl_$s.grib2"
  grib_copy -w shortName!=sot,shortName!=vsw "$T/sfc.grib2" "$T/nosoil.grib2"
  : > "$T/soil.grib2"
  for k in 1 2 3 4; do
    case $k in
      1) D1=0;   D2=7;;
      2) D1=7;   D2=28;;
      3) D1=28;  D2=100;;
      4) D1=100; D2=289;;
    esac
    for p in sot vsw; do
      grib_copy -w shortName=$p,level=$k "$T/sfc.grib2" "$T/tmp.grib2"
      grib_set -s typeOfFirstFixedSurface=106,scaleFactorOfFirstFixedSurface=2,scaledValueOfFirstFixedSurface=$D1,typeOfSecondFixedSurface=106,scaleFactorOfSecondFixedSurface=2,scaledValueOfSecondFixedSurface=$D2 "$T/tmp.grib2" "$T/tmp2.grib2"
      cat "$T/tmp2.grib2" >> "$T/soil.grib2"
    done
  done
  cat "$T/nosoil.grib2" "$T/soil.grib2" > "$FIN/sfc_$s.grib2"
  rm -rf "$T"
  echo "ritagliato +${s} h"
}
export -f passo
export RAW FIN
seq 0 3 "$ORE" | xargs -P 4 -I{} bash -c 'set -euo pipefail; passo {}'
n=$(ls "$FIN"/*.grib2 | wc -l)
[ "$n" -eq $(( (ORE / 3 + 1) * 2 )) ] || { echo "ERRORE: file finali $n invece di $(( (ORE / 3 + 1) * 2 ))"; exit 1; }
ls -lh "$FIN" | head -n 8
