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


def main():
    """
    1. Compute ocean structure
    2. Compute climate indices
    3. Perform lag correlation analysis
    4. Compute composites
    5. Display

    Returns:
      dict of results
    """

# Open lazily — no data downloaded yet
da = _get_dataset(
    "TEMP",
    component="ocn",
    scenario="historical",
    forcing="cmip6",
    time_slice=("1999-01", "2000-12"),
    lat=slice(5, -5),      # Pacific ENSO
    lon=slice(-155.0, -120.0),  # negative °W values; converted to 0–360 internally
    members=0,                  # first ensemble member only
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