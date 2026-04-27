import xarray as xr


def _to_time_series(da: xr.DataArray) -> xr.DataArray:
    """Collapse all non-time dimensions so the result is strictly 1-D (time,)."""
    extra = [d for d in da.dims if d != "time"]
    if extra:
        da = da.mean(extra)
    return da


def lag_correlation(
    field: xr.DataArray,
    index: xr.DataArray,
    lags: list[int],
) -> xr.DataArray:
    """Time series lag-correlation between a spatially averaged field and an index.

    Parameters
    ----------
    field : xr.DataArray
        Time series, any non-time dimensions are averaged out automatically.
    index : xr.DataArray
        1-D ENSO index time series.
    lags : list of int
        Lags in months. Positive → index leads field.

    Returns
    -------
    xr.DataArray
        1-D array of correlation coefficients, indexed by ``lag``.
    """
    field = _to_time_series(field)
    index = _to_time_series(index)

    results = []
    for lag in lags:
        shifted = index.shift(time=lag)
        combined = xr.Dataset({"field": field, "shifted": shifted}).dropna("time")
        corr = float(xr.corr(combined["field"], combined["shifted"], dim="time"))
        results.append(corr)

    return xr.DataArray(
        results,
        coords={"lag": xr.DataArray(lags, dims="lag")},
        dims="lag",
    ).rename("lag_correlation")


def lead_lag_regression(
    field: xr.DataArray,
    index: xr.DataArray,
    lags: list[int],
) -> xr.DataArray:
    """Regression slope at each lag (field regressed onto index).

    Returns
    -------
    xr.DataArray
        Regression coefficients indexed by ``lag``.
    """
    field = _to_time_series(field)
    index = _to_time_series(index)

    results = []
    for lag in lags:
        shifted = index.shift(time=lag)
        combined = xr.Dataset({"field": field, "shifted": shifted}).dropna("time")
        f, s = combined["field"], combined["shifted"]
        results.append(float(xr.cov(f, s, dim="time") / s.var("time")))

    return xr.DataArray(
        results,
        coords={"lag": xr.DataArray(lags, dims="lag")},
        dims="lag",
    ).rename("lag_regression")


def max_correlation_lag(lag_corr: xr.DataArray) -> int:
    """Return the lag at which correlation is maximum."""
    return int(lag_corr["lag"].isel(lag=int(lag_corr.argmax("lag"))).values)