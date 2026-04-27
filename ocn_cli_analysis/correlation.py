import numpy as np
import xarray as xr
from scipy import stats


_SPATIAL_DIMS = ("nlat", "nlon", "lat", "lon")
_DEPTH_DIMS   = ("z_t", "z_w", "depth", "lev")


def _prep_field(field: xr.DataArray) -> xr.DataArray:
    """Reduce to (time, nlat, nlon) or (time, lat, lon).

    Averages depth dims (z_t, z_w, depth, lev) and squeezes any remaining
    size-1 dims (e.g. member_id) so xr.corr always produces a clean 2-D map.
    """
    depth_dims = [d for d in field.dims if d in _DEPTH_DIMS]
    if depth_dims:
        field = field.mean(depth_dims)
    # Drop any leftover size-1 dimensions (e.g. member_id)
    field = field.squeeze()
    return field


def _prep_index(index: xr.DataArray) -> xr.DataArray:
    """Ensure index is strictly 1-D over time."""
    extra = [d for d in index.dims if d != "time"]
    if extra:
        index = index.mean(extra)
    return index


def correlation_map(field: xr.DataArray, index: xr.DataArray) -> xr.DataArray:
    """Pointwise Pearson correlation between a spatial field and a 1-D index.

    Depth dimensions (z_t, z_w, depth, lev) are averaged out automatically
    so the output is always 2-D (nlat × nlon or lat × lon).

    Parameters
    ----------
    field : xr.DataArray
        Shape (time, [z_t,] nlat, nlon). Anomalies recommended.
    index : xr.DataArray
        1-D (time,) ENSO index.

    Returns
    -------
    xr.DataArray
        2-D correlation map.
    """
    return xr.corr(_prep_field(field), _prep_index(index), dim="time").rename("correlation")


def regression_map(field: xr.DataArray, index: xr.DataArray) -> xr.DataArray:
    """Pointwise linear regression slope."""
    field = _prep_field(field)
    index = _prep_index(index)
    cov = xr.cov(field, index, dim="time")
    var = index.var("time")
    return (cov / var).rename("regression")


def significance_mask(
    field: xr.DataArray,
    index: xr.DataArray,
    alpha: float = 0.05,
) -> xr.DataArray:
    """Boolean mask: True where the correlation is statistically significant.

    Uses a two-tailed t-test with n-2 degrees of freedom.

    Returns
    -------
    xr.DataArray of bool, same 2-D spatial shape as the correlation map.
    """
    field = _prep_field(field)
    index = _prep_index(index)
    corr = xr.corr(field, index, dim="time").rename("correlation")
    n = field.sizes["time"]
    t = corr * np.sqrt((n - 2) / (1 - corr ** 2 + 1e-12))
    p = xr.apply_ufunc(
        lambda t_vals: 2 * stats.t.sf(np.abs(t_vals), df=n - 2),
        t,
        dask="parallelized",
        output_dtypes=[float],
    )
    return (p < alpha).rename("significant")