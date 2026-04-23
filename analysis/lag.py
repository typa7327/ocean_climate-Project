import xarray as xr

def lag_correlation(ds, var, index, lags):
    results = []
    for lag in lags:
        shifted = ds[index].shift(time=lag)
        results.append(xr.corr(ds[var], shifted, dim='time'))
    return xr.concat(results, dim='lag')

def lead_lag_regression(ds, var, index, lags):
    results = []
    for lag in lags:
        shifted = ds[index].shift(time=lag)
        slope = xr.cov(ds[var], shifted, dim='time') / shifted.var('time')
        results.append(slope)
    return xr.concat(results, dim='lag')

def max_correlation_lag(lag_corr, lags):
    return lags[lag_corr.argmax('lag')]