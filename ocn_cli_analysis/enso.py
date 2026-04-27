import numpy as np
import xarray as xr


# ---------------------------------------------------------------------------
# Niño-3.4 index
# ---------------------------------------------------------------------------

def compute_nino34(da: xr.DataArray) -> xr.DataArray:
    """Spatial mean of SST anomaly over the Niño-3.4 box.

    Works with both regular (atm) and curvilinear POP (ocn) grids.
    ``da`` should already be subsetted to the ENSO box region.
    Anomaly is computed internally from the monthly climatology.

    Parameters
    ----------
    da : xr.DataArray
        Temperature DataArray, already spatially subsetted to the ENSO box.
        Dimensions must include ``time`` plus spatial dims
        (``nlat``/``nlon`` for ocean, ``lat``/``lon`` for atm).

    Returns
    -------
    xr.DataArray
        1-D (or member_id × time) Niño-3.4 index in °C anomaly.
    """
    # Collapse all non-time, non-spatial dims (e.g. member_id) before computing
    da = da.squeeze()
    # Monthly climatological anomaly
    clim = da.groupby("time.month").mean("time")
    anom = da.groupby("time.month") - clim

    # Collapse all spatial + depth dims → pure (time,) series
    reduce_dims = [d for d in anom.dims
                   if d in ("nlat", "nlon", "lat", "lon", "z_t", "z_w", "depth", "lev")]
    nino34 = anom.mean(dim=reduce_dims)

    # Restore time coordinate if groupby subtraction dropped the labels
    nino34 = nino34.assign_coords(time=da["time"])

    return nino34.rename("nino34")


def compute_oni(nino34: xr.DataArray) -> xr.DataArray:
    """3-month centred running mean of the Niño-3.4 index (ONI)."""
    return nino34.rolling(time=3, center=True).mean().rename("oni")


def classify_enso(oni: xr.DataArray, threshold: float = 0.5) -> xr.DataArray:
    """Classify each time step as 'El Nino', 'La Nina', or 'Neutral'.

    Parameters
    ----------
    oni : xr.DataArray
        ONI time series (time dimension required; optional member_id dim).
    threshold : float
        ±threshold in °C.

    Returns
    -------
    xr.DataArray of str labels.
    """
    phase = xr.where(oni > threshold, "El Nino",
            xr.where(oni < -threshold, "La Nina", "Neutral"))
    return phase.rename("enso_phase")