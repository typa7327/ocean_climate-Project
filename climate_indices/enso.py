import xarray as xr
from data.loader import subset_region #(ds, lat_bounds, lon_bounds)

def compute_nino(sst):
    region = sst.sel(lat=slice(5, -5), lon=slice(190, 240))
    return region.mean(dim=['lat', 'lon'])

def compute_nino_subset(sst, lat_bounds, lon_bounds):
    region = subset_region(sst, lat_bounds, lon_bounds)
    return region.mean(dim=['lat', 'lon'])

def compute_oni(nino):
    return nino.rolling(time=3, center=True).mean()

def classify_enso(nino, threshold=0.5):
    return xr.where(
        compute_oni(nino) > threshold, 'El Nino',
        xr.where(compute_oni(nino) < -threshold, 'La Nina'),
        xr.where(compute_oni(nino) < threshold and compute_oni(nino) > -threshold, 'Neutral')
    )