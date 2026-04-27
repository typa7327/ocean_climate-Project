"""
ocean_climate_ensemble.py
=========================
CESM2 Large Ensemble — Ocean Temperature × ENSO Analysis
---------------------------------------------------------
Pipeline
  1. Load TEMP over the ENSO box (CLI-specified region), member 0 only
  2. Compute Niño-3.4 index and ONI
  3. Classify ENSO phase (El Niño / La Niña / Neutral)
  4. Time series: ONI and TEMP anomaly
  5. Lag-correlation: TEMP vs ONI at lags −12 … +12 months
  6. Correlation map: pointwise TEMP × ONI with significance stippling
  7. Composite maps: El Niño / La Niña / difference

Usage
-----
python ocean_climate_ensemble.py \\
    --component ocn \\
    --scenario historical \\
    --forcing cmip6 \\
    --time-slice 1990-01 2014-12 \\
    --lat-box 5 -5 \\
    --lon-box -155 -120
"""

import argparse

import numpy as np
import xarray as xr

from loader import _get_dataset, compute_anomaly, detrend, standardize
from enso import compute_nino34, compute_oni, classify_enso
from lag import lag_correlation
from correlation import correlation_map, significance_mask
from composite import composite_by_phase, composite_difference
import timeseries


def load_enso_box(members, *, component, scenario, forcing,
                  time_slice, lat_box, lon_box):
    """Load TEMP over the specified box for given member(s)."""
    print(f"Loading TEMP — members={members} …")
    da = _get_dataset(
        "TEMP",
        component=component,
        scenario=scenario,
        forcing=forcing,
        time_slice=time_slice,
        lat=slice(*lat_box),
        lon=slice(*lon_box),
        members=members,
    )
    da = da.load()
    print(f"  → loaded shape: {da.dims} = {dict(da.sizes)}")
    return da


def spatial_mean_ts(da: xr.DataArray) -> xr.DataArray:
    """Average over all spatial dims → pure time (× member_id) series."""
    spatial = [d for d in da.dims if d in ("nlat", "nlon", "lat", "lon")]
    return da.mean(dim=spatial)


def main():
    parser = argparse.ArgumentParser(
        description="CESM2-LE Ocean Temperature × ENSO analysis"
    )
    parser.add_argument("--component",  type=str, required=True,
                        choices=["atm", "ocn"])
    parser.add_argument("--scenario",   type=str, required=True,
                        choices=["historical", "ssp370"])
    parser.add_argument("--forcing",    type=str, required=True,
                        choices=["cmip6", "smbb"])
    parser.add_argument("--time-slice", type=str, nargs=2, required=True,
                        metavar=("START", "END"),
                        help="e.g. 1990-01 2014-12")
    parser.add_argument("--lat-box",    type=float, nargs=2, required=True,
                        metavar=("NORTH", "SOUTH"),
                        help="e.g. 5 -5  (north first)")
    parser.add_argument("--lon-box",    type=float, nargs=2, required=True,
                        metavar=("WEST", "EAST"),
                        help="e.g. -155 -120  (negative °W accepted)")

    args = parser.parse_args()

    # Unpack CLI args into local config
    COMPONENT  = args.component
    SCENARIO   = args.scenario          # already .lower()'d by argparse type=
    FORCING    = args.forcing
    TIME_SLICE = tuple(args.time_slice)
    LAT_BOX    = tuple(args.lat_box)
    LON_BOX    = tuple(args.lon_box)
    LAGS       = list(range(-12, 13))   # months; positive = ONI leads TEMP
    ALPHA      = 0.05                   # significance level for correlation maps

    # -----------------------------------------------------------------------
    # 1. Load data — member 0 only
    # -----------------------------------------------------------------------
    da_m0 = load_enso_box(
        members=0,
        component=COMPONENT,
        scenario=SCENARIO,
        forcing=FORCING,
        time_slice=TIME_SLICE,
        lat_box=LAT_BOX,
        lon_box=LON_BOX,
    )  # (time, nlat, nlon)

    # -----------------------------------------------------------------------
    # 2. Niño-3.4 index and ONI
    # -----------------------------------------------------------------------
    print("\nComputing Niño-3.4 index …")

    nino34_m0 = compute_nino34(da_m0)   # (time,)
    oni_m0    = compute_oni(nino34_m0)  # (time,) — has NaN at edges from rolling

    # Drop the NaN-padded edge time steps from the ONI rolling mean so that
    # phase labels and field arrays share the exact same positional indices.
    # Squeeze first so oni_m0 is strictly 1-D (time,) before extracting values.
    oni_1d     = oni_m0.squeeze()
    valid_time = np.where(~np.isnan(oni_1d.values))[0]
    oni_m0    = oni_1d.isel(time=valid_time)
    da_valid  = da_m0.isel(time=valid_time)   # field trimmed to same window

    phase_m0  = classify_enso(oni_m0)

    # -----------------------------------------------------------------------
    # 3. Temperature anomaly fields for correlation / composite
    # -----------------------------------------------------------------------
    print("Computing temperature anomalies …")
    # Compute anomaly on the full da_m0 then trim to the valid window
    anom_full = da_m0.groupby("time.month") - da_m0.groupby("time.month").mean("time")
    anom_full = anom_full.assign_coords(time=da_m0.time)
    anom_m0   = anom_full.isel(time=valid_time)   # (time, z_t, nlat, nlon)

    # Spatially averaged anomaly time series (for lag plot)
    ts_m0 = spatial_mean_ts(anom_m0)

    # -----------------------------------------------------------------------
    # 4. Time series: ONI and TEMP anomaly
    # -----------------------------------------------------------------------
    print("\nPlotting time series …")
    timeseries.plot_timeseries(
        oni_m0,
        title="ONI — Member 0",
        xlabel="Time",
        ylabel="ONI (°C)",
    )
    timeseries.plot_timeseries(
        ts_m0,
        title="ENSO-box TEMP Anomaly — Member 0",
        xlabel="Time",
        ylabel="TEMP Anomaly (°C)",
    )

    # -----------------------------------------------------------------------
    # 5. Lag correlation: spatially averaged TEMP vs ONI
    # -----------------------------------------------------------------------
    print("\nComputing lag correlations …")
    lagcorr_m0 = lag_correlation(ts_m0, oni_m0, LAGS)

    timeseries.plot_lag_correlation(
        lagcorr_m0,
        title="Lag Correlation: ENSO-box TEMP vs ONI",
        member_label="Member 0",
    )

    # -----------------------------------------------------------------------
    # 6. Correlation maps (pointwise TEMP anomaly vs ONI)
    # -----------------------------------------------------------------------
    print("\nBuilding correlation maps …")

    corr_map_m0 = correlation_map(anom_m0, oni_m0)
    sig_map_m0  = significance_mask(anom_m0, oni_m0, alpha=ALPHA)

    timeseries.plot_correlation_map(
        corr_map_m0,
        sig_mask=sig_map_m0,
        title="TEMP–ONI Correlation Map — Member 0",
    )

    # -----------------------------------------------------------------------
    # 7. Composite maps
    # -----------------------------------------------------------------------
    print("\nBuilding composite maps …")

    comp_m0 = composite_by_phase(anom_m0, phase_m0)
    diff_m0 = composite_difference(anom_m0, phase_m0)
    en_m0   = comp_m0.get("El Nino")
    ln_m0   = comp_m0.get("La Nina")

    if en_m0 is not None and ln_m0 is not None and diff_m0 is not None:
        timeseries.plot_composite_map(
            en_m0, ln_m0, diff_m0,
            title_prefix="TEMP Composite (Member 0)",
        )
    else:
        print("  Skipping composite map — one or more phases absent from time slice")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n=== Analysis complete ===")
    print(f"Time period : {TIME_SLICE[0]} – {TIME_SLICE[1]}")
    print(f"ENSO box    : {LAT_BOX[0]}°N – {abs(LAT_BOX[1])}°S, "
          f"{abs(LON_BOX[0])}°W – {abs(LON_BOX[1])}°W")
    print(f"Lag range   : {LAGS[0]} … {LAGS[-1]} months")

    max_lag = int(lagcorr_m0["lag"].isel(lag=int(lagcorr_m0.argmax("lag"))).values)
    print(f"Peak lag    : {max_lag} months")

    return {
        "nino34_m0": nino34_m0,
        "oni_m0": oni_m0,
        "phase_m0": phase_m0,
        "corr_map_m0": corr_map_m0,
        "sig_map_m0": sig_map_m0,
        "lagcorr_m0": lagcorr_m0,
        "comp_m0": comp_m0,
        "diff_m0": diff_m0,
    }


if __name__ == "__main__":
    results = main()