import xarray as xr
import numpy as np
from loader import _get_dataset, compute_anomaly, rolling_mean


# def compute_depth_mean_temperature(ds, temp_var='temperature', depth_range=(0, 2000)):
#     """
#     Mean temperature over a depth range.
#     """
#     subset = ds.sel(depth=slice(*depth_range))
#     return subset[temp_var].mean('depth').rename('depth_mean_temp')


# def compute_temperature_gradient(ds, temp_var='temperature'):
#     """
#     Vertical temperature gradient (dT/dz).
#     """
#     return ds[temp_var].differentiate('depth').rename('dTdz')


# def compute_thermocline_strength(ds, temp_var='temperature'):
#     """
#     Maximum vertical temperature gradient (strength of thermocline).
#     """
#     grad = compute_temperature_gradient(ds, temp_var)
#     return grad.max('depth').rename('thermocline_strength')


# def compute_thermocline_depth(ds, temp_var='temperature'):
#     """
#     Depth of maximum temperature gradient (thermocline depth).
#     """
#     grad = compute_temperature_gradient(ds, temp_var)
#     return ds['depth'].isel(depth=grad.argmax('depth')).rename('thermocline_depth')


def compute_temperature_anomaly(ds, temp_var='TEMP'):
    """
    Temperature anomaly using shared utility function.
    """
    return compute_anomaly(ds, temp_var).rename('temp_anomaly')


def smooth_temperature(da, window=3):
    """
    Smooth temperature time series using rolling mean.
    Expects DataArray converted to Dataset if needed.
    """
    ds = da.to_dataset(name='TEMP')
    return rolling_mean(ds, 'TEMP', window=window).rename('temp_smoothed')


# def compute_mixed_layer_temperature(ds, temp_var='TEMP', mld=None):
#     """
#     Compute mean temperature within the mixed layer.

#     Parameters:
#     - ds: dataset with temperature and depth
#     - mld: mixed layer depth (DataArray)

#     Returns:
#     - mixed layer temperature
#     """
#     if mld is None:
#         raise ValueError("MLD must be provided")

#     temp = ds[temp_var]

#     # mask values deeper than MLD
#     masked = temp.where(ds['depth'] <= mld)

#     return masked.mean('depth').rename('ml_temp')


def compute_temperature_variance(ds, temp_var='TEMP'):
    """
    Temporal variance of temperature.
    """
    return ds[temp_var].var('time').rename('temp_variance')