import logging
import os
import scipy.signal
import numpy as np
import xarray as xr

BUCKET = "ncar-cesm2-lens"
_STORAGE_OPTIONS = {"anon": True}
_OCN_GRID_URL = f"s3://{BUCKET}/ocn/static/grid.zarr"

SCENARIOS = ("historical", "ssp370")
FORCINGS = ("cmip6", "smbb")
COMPONENTS = ("atm", "ocn")


def _zarr_url(component: str, scenario: str, forcing: str, variable: str) -> str:
    return f"s3://{BUCKET}/{component}/monthly/cesm2LE-{scenario}-{forcing}-{variable}.zarr"

# ds = _get_dataset()
def _get_dataset(variable: str,
    component: str = "atm",
    scenario: str = "historical",
    forcing: str = "cmip6",
    time_slice: tuple | None = None,
    lat: float | tuple | None = None,
    lon: float | tuple | None = None,
    members: int | list[int] | None = None,
) -> xr.DataArray:
    """Open a CESM2 Large Ensemble variable from the NCAR AWS S3 archive.

    Data are loaded lazily — nothing is downloaded until you call .compute(),
    .load(), or pass the result to a plotting or analysis function.

    Parameters
    ----------
    variable : str
        Variable name, e.g. ``"TREFHT"`` (atm) or ``"TEMP"`` (ocn).
        Use ``list_variables()`` to see what is available.
    component : {"atm", "ocn"}
        Model component.
    scenario : {"historical", "ssp370"}
        Forcing scenario.
    forcing : {"cmip6", "smbb"}
        Ensemble forcing type.
    time_slice : (start, stop) or None
        ISO date strings, e.g. ``("1990-01", "2000-12")``.
    lat : float, (south, north), or slice, or None
        Latitude. Scalar → nearest grid point. Tuple or slice → range.
        For ``component="ocn"`` (curvilinear POP grid) only scalar and
        tuple are supported; slices are treated as tuples.
    lon : float, (west, east), or slice, or None
        Longitude. Negative values (°W) are accepted and converted to 0–360
        to match the model grid.
    members : int or list[int] or None
        Ensemble member index/indices (0-based). ``None`` returns all members.

    Returns
    -------
    xr.DataArray
        Lazy DataArray. Call ``.load()`` or ``.compute()`` to download.

    Examples
    --------
    Near-surface temperature at a single point, first member, 1990–2000::

        da = open_cesm2le(
            "TREFHT",
            time_slice=("1990", "2000"),
            lat=40.0,
            lon=-105.0,
            members=0,
        )
        da.load().plot()

    Temperature over a spatial box::

        da = open_cesm2le(
            "TREFHT",
            time_slice=("1990-01", "2000-12"),
            lat=slice(40.0, 43.0),
            lon=slice(-108.0, -105.0),
            members=0,
        )
        da.load().mean(["lat", "lon"]).plot()

    Ocean temperature nearest to a point::

        da = open_cesm2le("TEMP", component="ocn", lat=30.0, lon=-140.0)
    """
    if component not in COMPONENTS:
        raise ValueError(f"component must be one of {COMPONENTS}")
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {SCENARIOS}")
    if forcing not in FORCINGS:
        raise ValueError(f"forcing must be one of {FORCINGS}")

    url = _zarr_url(component, scenario, forcing, variable)
    ds = xr.open_zarr(url, storage_options=_STORAGE_OPTIONS, consolidated=True)
    da = ds[variable]

    # Ensemble member selection
    if members is not None:
        if isinstance(members, int):
            members = [members]
        da = da.isel(member_id=members)

    # Time selection
    if time_slice is not None:
        da = da.sel(time=slice(*time_slice))

    # Spatial selection
    if component == "atm":
        da = _sel_atm(da, lat, lon)
    elif component == "ocn":
        grid = xr.open_zarr(_OCN_GRID_URL, storage_options=_STORAGE_OPTIONS, consolidated=True)
        da = _sel_ocn(da, grid, lat, lon)

    return da

def list_variables(
    component: str = "atm",
    scenario: str = "historical",
    forcing: str = "cmip6",
) -> list[str]:
    """Return variable names available for a given component/scenario/forcing.

    Parameters
    ----------
    component : {"atm", "ocn"}
    scenario : {"historical", "ssp370"}
    forcing : {"cmip6", "smbb"}

    Returns
    -------
    list[str]
        Sorted list of variable name strings.

    Examples
    --------
    >>> list_variables("ocn", "ssp370", "cmip6")
    ['DIC', 'DOC', 'O2', 'PD', 'SALT', 'TEMP', ...]
    """
    import s3fs  # deferred — only needed for listing

    fs = s3fs.S3FileSystem(anon=True)
    prefix = f"{BUCKET}/{component}/monthly"
    tag = f"cesm2LE-{scenario}-{forcing}-"
    entries = fs.ls(prefix)
    return sorted(
        e.split("/")[-1][len(tag) : -len(".zarr")]
        for e in entries
        if e.split("/")[-1].startswith(tag) and e.endswith(".zarr")
    )

def _to_360(v: float) -> float:
    """Normalize a longitude to 0–360 range.

    Negative values (°W) are shifted by +360.  Positive values — including
    360 itself — are left unchanged so that ``lon=360`` stays as 360
    rather than collapsing to 0 via ``360 % 360``.
    """
    return v + 360 if v < 0 else v


def _sel_lon(da: xr.DataArray, lo_raw: float, hi_raw: float, dim: str) -> xr.DataArray:
    """Select a longitude range on a 0–360 grid, handling wraparound correctly.

    Cases
    -----
    - span >= 360°  : global — return da unchanged
    - lo <= hi      : normal slice, e.g. -135→-60 becomes 225→300
    - lo > hi       : range crosses the prime meridian, e.g. -179→179 becomes
                      181→179; concat [181,360) + [0,179]
    """
    if abs(hi_raw - lo_raw) >= 360.0:
        return da

    lo = _to_360(lo_raw)
    hi = _to_360(hi_raw)

    if lo <= hi:
        return da.sel({dim: slice(lo, hi)})

    # Wraparound: e.g. lo=181, hi=179 → [181,360) ∪ [0,179]
    part1 = da.sel({dim: slice(lo, None)})
    part2 = da.sel({dim: slice(None, hi)})
    return xr.concat([part1, part2], dim=dim)


def _sel_atm(da: xr.DataArray, lat, lon) -> xr.DataArray:
    """Subset an atmosphere DataArray on its regular lat/lon grid.

    ``lat`` / ``lon`` may be:
    - a scalar  → nearest-neighbour lookup
    - a 2-tuple → bounding range, e.g. ``(40.0, 43.0)``
    - a slice   → passed directly, e.g. ``slice(40.0, 43.0)``
    """
    if lat is not None:
        if np.isscalar(lat):
            da = da.sel(lat=float(lat), method="nearest")
        elif isinstance(lat, slice):
            da = da.sel(lat=lat)
        else:
            da = da.sel(lat=slice(min(lat), max(lat)))
    if lon is not None:
        if np.isscalar(lon):
            da = da.sel(lon=_to_360(float(lon)), method="nearest")
        elif isinstance(lon, slice):
            lo_raw = float(lon.start) if lon.start is not None else -180.0
            hi_raw = float(lon.stop) if lon.stop is not None else 180.0
            da = _sel_lon(da, lo_raw, hi_raw, dim="lon")
        else:
            da = _sel_lon(da, float(lon[0]), float(lon[1]), dim="lon")
    return da


def _sel_ocn(da: xr.DataArray, ds: xr.Dataset, lat, lon) -> xr.DataArray:
    """Subset an ocean DataArray on the curvilinear POP grid.

    The POP grid has 2-D coordinate arrays TLAT/TLONG (nlat × nlon), so
    standard label-based indexing does not apply.  For a scalar lat/lon we
    find the nearest grid cell; for ranges we take a conservative bounding-box
    slice (cells outside the mask but inside the rectangle may be included).
    """
    if lat is None and lon is None:
        return da

    tlat = ds["TLAT"]    # (nlat, nlon)
    tlong = ds["TLONG"]  # (nlat, nlon), values 0–360

    # Normalise slice → tuple so the scalar/range branches below cover both
    if isinstance(lat, slice):
        lat = (lat.start, lat.stop)
    if isinstance(lon, slice):
        lon = (lon.start, lon.stop)

    if np.isscalar(lat) and np.isscalar(lon):
        lon_norm = _to_360(float(lon))
        dist = (tlat - float(lat)) ** 2 + (tlong - lon_norm) ** 2
        flat_idx = int(dist.values.argmin())
        j, i = np.unravel_index(flat_idx, tlat.shape)
        return da.isel(nlat=int(j), nlon=int(i))

    if lat is not None and lon is not None:
        lat_min, lat_max = sorted(float(v) for v in lat)
        lo_raw, hi_raw = float(lon[0]), float(lon[1])
        if abs(hi_raw - lo_raw) >= 360.0:
            lon_mask = xr.ones_like(tlong, dtype=bool)  # global — keep all
        else:
            lo = _to_360(lo_raw)
            hi = _to_360(hi_raw)
            if lo <= hi:
                lon_mask = (tlong >= lo) & (tlong <= hi)
            else:
                # Wraparound: [lo,360) ∪ [0,hi]
                lon_mask = (tlong >= lo) | (tlong <= hi)
        mask = (tlat >= lat_min) & (tlat <= lat_max) & lon_mask
        rows = np.where(mask.any(dim="nlon").values)[0]
        cols = np.where(mask.any(dim="nlat").values)[0]
        if rows.size and cols.size:
            da = da.isel(
                nlat=slice(int(rows[0]), int(rows[-1]) + 1),
                nlon=slice(int(cols[0]), int(cols[-1]) + 1),
            )

    return da

# def subset_region(ds, lat_bounds, lon_bounds):
#     return ds.sel(lat=slice(*lat_bounds), lon=slice(*lon_bounds))

def subset_time(ds, start, end):
    return ds.sel(time=slice(start, end))

def resample_monthly(ds):
    return ds.resample(time='1M').mean()

def resample_seasonal(ds):
    return ds.groupby('time.season').mean('time')

def compute_anomaly(ds, var):
    clim = ds[var].groupby('time.month').mean('time')
    return ds[var].groupby('time.month') - clim

def detrend(ds, var):
    return xr.apply_ufunc(
        scipy.signal.detrend,
        ds[var],
        kwargs={'axis': 0}
    )

def standardize(ds, var):
    return (ds[var] - ds[var].mean('time')) / ds[var].std('time')

def rolling_mean(ds, var, window=3):
    return ds[var].rolling(time=window, center=True).mean()

