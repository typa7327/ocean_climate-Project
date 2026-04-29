"""Run the Ocean-Climate ensemble experiment without installing the package.

Usage (from repo root):
    python run.py

Note: this works because run_lorenz_ensemble.py runs as a side-effect of
being imported (the experiment code is at module level, not in a function).
After you complete the lab and wrap everything in main(), this file will
no longer do anything — switch to `pip install -e .` and run `run-lorenz`.
"""
from ocn_cli_analysis.ocean_climate_ensemble import main
main()