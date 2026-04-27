import numpy as np
import xarray as xr


# ---------------------------------------------------------------------------
# Niño-3.4 index
# ---------------------------------------------------------------------------

def compute_nino34(da: xr.DataArray) -> xr.DataArray:
    """
    Spatial mean of SST anomaly over the Niño-3.4 box.

    Works with both regular (atm) and curvilinear POP (ocn) grids.
    ``da`` should already be subsetted to the ENSO box region.
    Anomaly is computed internally from the monthly climatology.
    """
    da = da.squeeze()
    # Monthly anomaly
    clim = da.groupby("time.month").mean("time")
    anom = da.groupby("time.month") - clim

    reduce_dims = [d for d in anom.dims
                   if d in ("nlat", "nlon", "lat", "lon", "z_t", "z_w", "depth", "lev")]
    nino34 = anom.mean(dim=reduce_dims)

    nino34 = nino34.assign_coords(time=da["time"])

    return nino34.rename("nino34")


def compute_oni(nino34: xr.DataArray) -> xr.DataArray:
    """3-month centred running mean of the Niño-3.4 index (ONI)."""
    return nino34.rolling(time=3, center=True).mean().rename("oni")


def classify_enso(oni: xr.DataArray, threshold: float = 0.5) -> xr.DataArray:
    """
    Classify each time step as 'El Nino', 'La Nina', or 'Neutral'.
    """
    phase = xr.where(oni > threshold, "El Nino",
            xr.where(oni < -threshold, "La Nina", "Neutral"))
    return phase.rename("enso_phase")