# Ocean_Climate-Project
Final project for ATOC 4815. 

A package designed to implement comparison of how long term climate patterns affect ocean profiles. Connects to public NCAR CESM2-LE models. The goal of this project was to allow for prior done and accessible data conversion to access different ocean variables and connect them to climate indices with available statistical modeling.

Run options for component, scenario, forcing, variable, index, time, lat, lon

Run with:   python ocean_climate_ensemble.py --component ocn --scenario historical --forcing cmip6 --variable TEMP --index enso --time-slice 1998-01 2000-12 --lat-box 5 -5 --lon-box -155 -120

Currently focused on the comparison between ENSO/ONI vs SST
Currently working outputs: Lag Correlation, Correlation Projection, Timeseries

Future Additions: Indices: AMO, NAO, PDO, etc.
                  Ocean profiles: Salinity, MLD, Heat, Thermocline, Chemical Concentrations, etc.
                  Graphs: Vertical Profiles for all variables (meters, not cm), Composite Maps (phase differences)

https://pypi.org/project/ocean-climate/1.0.0/
pip install ocean-climate