# from mld import compute_mld_density
# from stratification import compute_stratification_index
# from enso import compute_nino34, classify_enso
# from lag import lag_correlation
# from composites import composite_by_phase
import timeseries
import matplotlib.pyplot as plt
from loader import _get_dataset, list_variables, compute_anomaly, detrend, standardize, rolling_mean
# import correlation
# import composite
# import lag
# import enso
import argparse


def main():
    parser = argparse.ArgumentParser(description="Compute Mixed Layer Depth")

    parser.add_argument("--component", type=str, required=True, choices=["atm", "ocn"])
    parser.add_argument("--scenario", type=str, required=True, choices=["historical", "SSP370"])
    parser.add_argument("--forcing", type=str, required=True, choices=["cmip6", "smbb"])

    parser.add_argument("--time-slice", type=str, nargs=2, required=True)
    parser.add_argument("--lat-box", type=float, nargs=2, required=True)
    parser.add_argument("--lon-box", type=float, nargs=2, required=True)

    args = parser.parse_args()

    # Convert CLI → variables
    TIME_SLICE = tuple(args.time_slice)
    LAT_BOX    = tuple(args.lat_box)
    LON_BOX    = tuple(args.lon_box)

    COMPONENT  = args.component
    SCENARIO   = args.scenario
    FORCING    = args.forcing

    LAGS  = list(range(-12, 13))
    ALPHA = 0.05

    # ✅ NOW this works because variables exist in scope
    da = _get_dataset(
        "TEMP",
        component=COMPONENT,
        scenario=SCENARIO,
        forcing=FORCING,
        time_slice=TIME_SLICE,
        lat=slice(*LAT_BOX),
        lon=slice(*LON_BOX),
        members=0,
    )


    # Data download from URL
    da = da.load()

    # print("OCN - HISTORICAL - CMIP6")
    # print(list_variables("ocn", "historical", "cmip6"))
    # print("OCN - SSP370 - CMIP6")
    # print(list_variables("ocn", "ssp370", "cmip6"))
    # print("OCN - SSP370 - SMBB")
    # print(list_variables("ocn", "ssp370", "smbb"))
    # print("OCN - HISTORICAL - SMBB")
    # print(list_variables("ocn", "historical", "smbb"))

    # Spatial mean → time series
    ts = da.mean(["nlat", "nlon"])

    ts_anom = compute_anomaly(ts.to_dataset(name="TEMP"), "TEMP")
    ts_detrended = detrend(ts.to_dataset(name="TEMP"), "TEMP")
    ts_std = standardize(ts.to_dataset(name="TEMP"), "TEMP")
    ts_smooth = rolling_mean(ts.to_dataset(name="TEMP"), "TEMP", window=3)

    # --------------------------------------------
    # PLOT
    # --------------------------------------------
    timeseries.plot_timeseries(ts, 
                            title="CESM2-LE TIMESERIES FUNC TEMP — ENSO box (member 0, historical)", 
                            xlabel="Depth (cm)", 
                            ylabel="Time")

    timeseries.plot_timeseries(ts,
                            title="Raw SST Timeseries (ENSO box)",
                            xlabel="Time",
                            ylabel="Temperature")

    timeseries.plot_timeseries(ts_anom,
                            title="SST Anomaly (ENSO box)",
                            xlabel="Time",
                            ylabel="Anomaly")

    timeseries.plot_timeseries(ts_smooth,
                            title="Smoothed SST (3-month mean)", 
                            xlabel="Time",
                            ylabel="Temperature")

if __name__ == "__main__":
    main()