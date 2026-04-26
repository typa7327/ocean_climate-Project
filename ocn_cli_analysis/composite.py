def composite_by_phase(ds, var, phase_var='enso_phase'):
    return ds[var].groupby(ds[phase_var]).mean('time')

def composite_difference(ds, var, phase_var='enso_phase'):
    comp = composite_by_phase(ds, var, phase_var)
    return comp.sel({phase_var: 'El Nino'}) - comp.sel({phase_var: 'La Nina'})

def event_composite(ds, var, index, threshold=1.0):
    events = ds.where(abs(ds[index]) > threshold, drop=True)
    return events[var].mean('time')