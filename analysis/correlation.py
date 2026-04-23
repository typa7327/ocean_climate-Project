import xarray as xr
import numpy as np

def compute_correlation(ds, var1, var2):
    return xr.corr(ds[var1], ds[var2], dim='time')

def compute_regression(ds, x, y):
    return xr.cov(ds[x], ds[y], dim='time') / ds[x].var('time')

def correlation_map(field, index):
    return xr.corr(field, index, dim='time')

def significance_test(corr, n):
    t = corr * np.sqrt((n - 2) / (1 - corr**2))
    return t