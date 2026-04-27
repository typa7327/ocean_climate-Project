import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr


def plot_correlation_map(corr: xr.DataArray,sig_mask: xr.DataArray = None,title: str = "Correlation Map",cmap: str = "RdBu_r",save: bool = True,) -> None:
    """Pcolormesh correlation map on the POP curvilinear grid.
 
    Uses TLAT/TLONG if present, otherwise falls back to nlat/nlon indices.
    Stippling marks statistically significant grid cells.
 
    Parameters
    ----------
    corr : xr.DataArray
        2-D correlation map (nlat × nlon).
    sig_mask : xr.DataArray, optional
        Boolean mask (True = significant). Stippled on the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
 
    # Try to use geographic coordinates if attached
    if "TLAT" in corr.coords and "TLONG" in corr.coords:
        lats = corr["TLAT"].values
        lons = corr["TLONG"].values
        vals = corr.values
        pcm = ax.pcolormesh(lons, lats, vals, cmap=cmap, vmin=-1, vmax=1,
                            shading="auto")
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        if sig_mask is not None:
            sig = sig_mask.values
            # Stipple significant points
            ax.contourf(lons, lats, sig.astype(float), levels=[0.5, 1.5],
                        hatches=["..."], colors="none")
    else:
        pcm = corr.plot(ax=ax, cmap=cmap, vmin=-1, vmax=1, add_colorbar=False)
        if sig_mask is not None:
            sig_mask.plot.contourf(ax=ax, levels=[0.5, 1.5],
                                   hatches=["..."], colors="none", add_colorbar=False)
 
    plt.colorbar(pcm, ax=ax, label="Pearson r", shrink=0.8)
    ax.set_title(title)
    plt.tight_layout()
    if save:
        fname = title.replace(" ", "_").replace("/", "-") + ".png"
        plt.savefig(fname, dpi=150)
        print(f"Saved {fname}")
    plt.show()
    plt.close()

def plot_composite_map(comp_en: xr.DataArray,comp_ln: xr.DataArray,diff: xr.DataArray,title_prefix: str = "TEMP Composite",cmap: str = "RdBu_r",save: bool = True,) -> None:
    """Three-panel composite map: El Niño | La Niña | Difference.
 
    Parameters
    ----------
    comp_en : xr.DataArray
        El Niño composite (2-D spatial).
    comp_ln : xr.DataArray
        La Niña composite.
    diff : xr.DataArray
        El Niño − La Niña.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    panels = [
        (comp_en, f"{title_prefix} — El Niño"),
        (comp_ln, f"{title_prefix} — La Niña"),
        (diff,    f"{title_prefix} — El Niño minus La Niña"),
    ]
    vmax = float(max(abs(comp_en).max(), abs(comp_ln).max(), abs(diff).max()))
    for ax, (da, ptitle) in zip(axes, panels):
        if "TLAT" in da.coords and "TLONG" in da.coords:
            pcm = ax.pcolormesh(da["TLONG"].values, da["TLAT"].values,
                                da.values, cmap=cmap, vmin=-vmax, vmax=vmax,
                                shading="auto")
            ax.set_xlabel("Lon")
            ax.set_ylabel("Lat")
        else:
            pcm = da.plot(ax=ax, cmap=cmap, vmin=-vmax, vmax=vmax,
                          add_colorbar=False)
        plt.colorbar(pcm, ax=ax, shrink=0.8, label="°C")
        ax.set_title(ptitle)
    plt.tight_layout()
    if save:
        fname = (title_prefix + "_composites").replace(" ", "_") + ".png"
        plt.savefig(fname, dpi=150)
        print(f"Saved {fname}")
    plt.show()
    plt.close()