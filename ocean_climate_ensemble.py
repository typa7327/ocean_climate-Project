# from data.loader import load_netcdf
from netCDF4 import Dataset
# import xarray as xr
# import matplotlib as plt


def main():
    """
    1. Compute ocean structure
    2. Compute climate indices
    3. Perform lag correlation analysis
    4. Compute composites
    5. Display

    Returns:
      dict of results
    """

    # return {
    #     'dataset': ds,
    #     'lag_corr_mld': lag_corr_mld,
    #     'lag_corr_heat': lag_corr_heat,
    #     'composites_mld': composites_mld,
    #     'composites_heat': composites_heat
    # }

    file = "/Users/tpachuda/Desktop/ATOC4815/ocean_climate-Project/test_data/OceanSODA_ETHZ-v2023.OCADS.01_1982-2022 (1).nc"
    # Code to check variables within netCDF4 files

    ds = Dataset(file)
    print(ds.variables)


if __name__ == "__main__":
    main()