import matplotlib.pyplot as plt
import xarray as xr
import numpy as np


def plot_timeseries(
    ts,
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    save: bool = True,
) -> None:
    """Plot a single time series (DataArray or Dataset)."""
    fig, ax = plt.subplots(figsize=(10, 4))
    if isinstance(ts, xr.Dataset):
        for var in ts.data_vars:
            ts[var].plot(ax=ax, label=var)
        ax.legend()
    else:
        ts.plot(ax=ax)
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    plt.tight_layout()
    if save and title:
        fname = title.replace(" ", "_").replace("/", "-") + ".png"
        plt.savefig(fname, dpi=150)
        print(f"Saved {fname}")
    plt.show()
    plt.close()


def plot_member_vs_ensemble(
    member_ts: xr.DataArray,
    ensemble_ts: xr.DataArray,
    title: str = "Member 0 vs Ensemble Mean",
    ylabel: str = "Temperature (°C)",
    save: bool = True,
) -> None:
    """Overlay a single ensemble member against the ensemble mean.

    Parameters
    ----------
    member_ts : xr.DataArray
        Time series for one member (1-D over ``time``).
    ensemble_ts : xr.DataArray
        Ensemble mean time series (1-D over ``time``).
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    member_ts.plot(ax=ax, color="steelblue", alpha=0.8, label="Member 0")
    ensemble_ts.plot(ax=ax, color="firebrick", linewidth=2, label="Ensemble mean")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time")
    ax.legend()
    plt.tight_layout()
    if save:
        fname = title.replace(" ", "_").replace("/", "-") + ".png"
        plt.savefig(fname, dpi=150)
        print(f"Saved {fname}")
    plt.show()
    plt.close()


def plot_lag_correlation(
    lag_corr: xr.DataArray,
    title: str = "Lag Correlation: TEMP vs ONI",
    member_label: str = "Member 0",
    ensemble_lag_corr: xr.DataArray = None,
    save: bool = True,
) -> None:
    """Bar/line plot of lag-correlation coefficients.

    Parameters
    ----------
    lag_corr : xr.DataArray
        Lag-correlation for member 0, indexed by ``lag``.
    ensemble_lag_corr : xr.DataArray, optional
        Same but for the ensemble mean.
    """
    lags = lag_corr["lag"].values.astype(float)
    vals = lag_corr.squeeze().values.astype(float)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(lags, vals, width=0.6, color="steelblue",
           alpha=0.8, label=member_label)
    if ensemble_lag_corr is not None:
        ax.bar(lags + 0.2, ensemble_lag_corr.values, width=0.35,
               color="firebrick", alpha=0.8, label="Ensemble mean")
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xlabel("Lag (months, positive = ONI leads TEMP)")
    ax.set_ylabel("Pearson r")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if save:
        fname = title.replace(" ", "_").replace("/", "-") + ".png"
        plt.savefig(fname, dpi=150)
        print(f"Saved {fname}")
    plt.show()
    plt.close()


def plot_correlation_map(
    corr: xr.DataArray,
    sig_mask: xr.DataArray = None,
    title: str = "Correlation Map",
    cmap: str = "RdBu_r",
    save: bool = True,
) -> None:
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
        # No geographic coords — build explicit integer index grids for axes
        import numpy as np
        vals = corr.squeeze().values          # ensure 2-D (nlat, nlon)
        nrows, ncols = vals.shape
        X, Y = np.meshgrid(np.arange(ncols), np.arange(nrows))
        pcm = ax.pcolormesh(X, Y, vals, cmap=cmap, vmin=-1, vmax=1, shading="auto")
        ax.set_xlabel("nlon index")
        ax.set_ylabel("nlat index")
        if sig_mask is not None:
            sig = sig_mask.squeeze().values.astype(float)
            ax.contourf(X, Y, sig, levels=[0.5, 1.5], hatches=["..."], colors="none")

    plt.colorbar(pcm, ax=ax, label="Pearson r", shrink=0.8)
    ax.set_title(title)
    plt.tight_layout()
    if save:
        fname = title.replace(" ", "_").replace("/", "-") + ".png"
        plt.savefig(fname, dpi=150)
        print(f"Saved {fname}")
    plt.show()
    plt.close()


def plot_composite_map(
    comp_en: xr.DataArray,
    comp_ln: xr.DataArray,
    diff: xr.DataArray,
    title_prefix: str = "TEMP Composite",
    cmap: str = "RdBu_r",
    save: bool = True,
) -> None:
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
    import numpy as np

    def _to_2d(da):
        """Squeeze member_id + average z_t → 2-D (nlat, nlon)."""
        da = da.squeeze()
        depth_dims = [d for d in da.dims if d in ("z_t", "z_w", "depth", "lev")]
        if depth_dims:
            da = da.mean(depth_dims)
        return da

    comp_en = _to_2d(comp_en)
    comp_ln = _to_2d(comp_ln)
    diff    = _to_2d(diff)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    panels = [
        (comp_en, f"{title_prefix} — El Niño"),
        (comp_ln, f"{title_prefix} — La Niña"),
        (diff,    f"{title_prefix} — El Niño minus La Niña"),
    ]
    vmax = float(max(float(abs(comp_en).max()), float(abs(comp_ln).max()), float(abs(diff).max())))
    if vmax == 0:
        vmax = 1.0
    for ax, (da, ptitle) in zip(axes, panels):
        vals = da.values
        nrows, ncols = vals.shape
        if "TLAT" in da.coords and "TLONG" in da.coords:
            X, Y = da["TLONG"].values, da["TLAT"].values
            ax.set_xlabel("Longitude (°E)")
            ax.set_ylabel("Latitude (°N)")
        else:
            X, Y = np.meshgrid(np.arange(ncols), np.arange(nrows))
            ax.set_xlabel("nlon index")
            ax.set_ylabel("nlat index")
        pcm = ax.pcolormesh(X, Y, vals, cmap=cmap, vmin=-vmax, vmax=vmax,
                            shading="auto")
        plt.colorbar(pcm, ax=ax, shrink=0.8, label="°C")
        ax.set_title(ptitle)
    plt.tight_layout()
    if save:
        fname = (title_prefix + "_composites").replace(" ", "_") + ".png"
        plt.savefig(fname, dpi=150)
        print(f"Saved {fname}")
    plt.show()
    plt.close()


def plot_multiple_timeseries(ds, vars):
    """Quick multi-variable overlay plot from a Dataset."""
    fig, ax = plt.subplots(figsize=(10, 4))
    for v in vars:
        ds[v].plot(ax=ax, label=v)
    ax.legend()
    plt.tight_layout()
    plt.show()
    plt.close()