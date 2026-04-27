import numpy as np
import xarray as xr


def composite_by_phase(
    field: xr.DataArray,
    phase: xr.DataArray,
) -> dict[str, xr.DataArray]:
    """Mean of ``field`` for each ENSO phase label.

    Parameters
    ----------
    field : xr.DataArray
        Field with a ``time`` dimension. Must have the same number of time
        steps as ``phase`` — positional alignment is used.
    phase : xr.DataArray
        1-D string labels (``"El Nino"``, ``"La Nina"``, ``"Neutral"``).

    Returns
    -------
    dict mapping phase label → mean DataArray (or None if phase absent).
    """
    field = field.squeeze()
    phase = phase.squeeze()

    # Use raw numpy values — avoids all xarray coordinate alignment issues
    phase_vals = np.asarray(phase).ravel()
    n_phase    = len(phase_vals)
    n_field    = field.sizes["time"]

    if n_phase != n_field:
        # Trim both to the overlapping interior (drop NaN-padded edges from ONI)
        n = min(n_phase, n_field)
        # Find the first/last non-NaN-labelled positions
        # phase_vals contains strings so there are no NaNs; just trim symmetrically
        trim = (max(n_phase, n_field) - n) // 2
        if n_phase > n_field:
            phase_vals = phase_vals[trim: trim + n_field]
        else:
            field = field.isel(time=slice(trim, trim + n_phase))
            n = n_phase
        print(f"  [composite] aligned to {n} time steps "
              f"(original phase={n_phase}, field={n_field})")

    composites = {}
    for label in ("El Nino", "La Nina", "Neutral"):
        idx = np.where(phase_vals == label)[0]
        count = len(idx)
        if count == 0:
            print(f"  [composite] no '{label}' months — skipping")
            composites[label] = None
        else:
            composites[label] = field.isel(time=idx).mean("time")
            print(f"  [composite] '{label}': {count} months")
    return composites


def composite_difference(
    field: xr.DataArray,
    phase: xr.DataArray,
) -> xr.DataArray | None:
    """El Niño minus La Niña composite difference.

    Returns None if either phase is absent from the time slice.
    """
    comp = composite_by_phase(field, phase)
    if comp["El Nino"] is None or comp["La Nina"] is None:
        print("  [composite_difference] one phase missing — cannot compute difference")
        return None
    return (comp["El Nino"] - comp["La Nina"]).rename("ElNino_minus_LaNina")


def event_composite(
    field: xr.DataArray,
    index: xr.DataArray,
    threshold: float = 1.0,
) -> xr.DataArray:
    """Mean of ``field`` during strong ENSO events (|index| > threshold)."""
    index_vals = np.asarray(index).ravel()
    idx = np.where(np.abs(index_vals) > threshold)[0]
    return field.isel(time=idx).mean("time").rename("event_composite")