# from .profiles.mld import compute_mld_density
# from .profiles.stratification import compute_stratification_index
# from .indices.enso import compute_nino34, classify_enso
# from .analysis.lag import lag_correlation
# from .analysis.composites import composite_by_phase
import matplotlib.pyplot as plt
from loader import _get_dataset


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
    lat=slice(5, 5),      # Pacific ENSO
    lon=slice(-170.0, -120.0),  # negative °W values; converted to 0–360 internally
    members=0,                  # first ensemble member only
)

da = da.load()

# Spatial mean → time series, convert K → °C
ts = da.mean(["nlat", "nlon"]) - 273.15


# --------------------------------------------
# REPLACE WITH MAPS.PY & TIMESERIES.PY
# --------------------------------------------

fig, ax = plt.subplots(figsize=(10, 4))
ts.plot(ax=ax)
ax.set_title("CESM2-LE TEMP — Colorado box (member 0, historical)")
ax.set_ylabel("Temperature (°C)")
ax.set_xlabel("Time")
plt.tight_layout()
plt.savefig("trefht_colorado.png", dpi=150)
print("Saved trefht_colorado.png")
plt.show()



if __name__ == "__main__":
    main()