#!/bin/sh
# Chained doctrine sweeps. Each grid is its own process, so the six 1m data
# loads happen once per grid rather than once per config.
PY="C:/personal/ICT_v5/.venv/Scripts/python.exe"
for g in bias confirm manage; do
  echo "######## GRID: $g ########"
  "$PY" -u portfolio.py --config configs/judas_fx.yaml --grid "$g" --split 2025-11-01
  echo
done
