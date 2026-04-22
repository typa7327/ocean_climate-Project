import matplotlib.pyplot as plt
import cartopy.crs as ccrs

def plot_map(da, title=None):
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
    da.plot(ax=ax, transform=ccrs.PlateCarree())
    ax.coastlines()
    if title:
        plt.title(title)
    plt.show()

def plot_correlation_map(corr, title=None):
    plot_map(corr, title)

def plot_composite_map(comp, title=None):
    plot_map(comp, title)