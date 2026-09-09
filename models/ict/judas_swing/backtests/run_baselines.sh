#!/bin/sh
PY="C:/personal/ICT_v5/.venv/Scripts/python.exe"
echo "################ A. LEGACY ZERO-COST BASELINE ################"
"$PY" -u portfolio.py --config configs/judas_fx.yaml --split 2025-11-01 \
      --save results/baseline/fx_zerocost.csv
echo
echo "################ B. HISTORICAL-SPREAD BASELINE ################"
"$PY" -u portfolio.py --config configs/judas_fx.yaml --split 2025-11-01 \
      --set 'costs={spread_mode: historical, gap_fill: open}' \
      --save results/baseline/fx_spread.csv
