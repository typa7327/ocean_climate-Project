import xarray as xr
import scipy.signal

def load_netcdf(filepath):
    return xr.open_dataset(filepath)

def subset_region(ds, lat_bounds, lon_bounds):
    return ds.sel(lat=slice(*lat_bounds), lon=slice(*lon_bounds))

def subset_time(ds, start, end):
    return ds.sel(time=slice(start, end))

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

