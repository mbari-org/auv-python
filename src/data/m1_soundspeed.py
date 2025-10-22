#! /usr/bin/env python
"""
Read most recent profile of temperature and practical salinity from the MBARI M1
mooring in Monterey Bay and return a profile of sound speed as a function of
depth.

This uses the opendap URL produced on an hourly basis as part of MBARI's SSDS
realtime data system.

Using Ferret to access the data:
================================
The most recent profile is retrieved using the SET REGION/L=2156:2156 statement
where the number 2156 is seen as the last index for the L axis (TIME) seen in
the output of the SHOW DATA/VAR statement. Below is a terminal session showing
how to access the data:

[ssdsadmin@elvis ~]$ ferret
        NOAA/PMEL TMAP
        FERRET v7.43 (optimized)
        Linux 3.10.0-862.11.6.el7.x86_64 64-bit - 09/14/18
        22-Oct-25 09:20

yes? USE "http://dods.mbari.org/opendap/data/ssdsdata/deployments/m1/202507/OS_MBARI-M1_20250724_R_TS.nc"
yes? SHOW DATA/VAR
     currently SET data sets:
    1> http://dods.mbari.org/opendap/data/ssdsdata/deployments/m1/202507/OS_MBARI-M1_20250724_R_TS.nc  (default)
     Hourly Gridded MBARI Mooring M1 Sea Water Temperature and Salinity Observations
 name     title                             I         J         K         L
 PSAL     Hourly sea_water_salinity        1:1       1:1       1:11      1:2156
             1 on grid GEN1 with -1.E+34 for missing data
             X=122.5W(-122.5):121.5W(-121.5)  Y=36.3N:37.3N  Z=0:325
 PSAL_QC  quality flag                     1:1       1:1       1:11      1:2156
               on grid GEN1 with -1.E+34 for missing data
             X=122.5W(-122.5):121.5W(-121.5)  Y=36.3N:37.3N  Z=0:325
 TEMP     Hourly sea_water_temperature     1:1       1:1       1:11      1:2156
             celsius on grid GEN1 with -1.E+34 for missing data
             X=122.5W(-122.5):121.5W(-121.5)  Y=36.3N:37.3N  Z=0:325
 TEMP_QC  quality flag                     1:1       1:1       1:11      1:2156
               on grid GEN1 with -1.E+34 for missing data
             X=122.5W(-122.5):121.5W(-121.5)  Y=36.3N:37.3N  Z=0:325
 TIME_QC  Quality flag for time axis, 1:   ...       ...       ...       1:2156
             flag on grid GEN2 with -1.E+34 for missing data

 POSITION_QC
          Quality flag for Latitude and L  1:1       ...       ...       ...
               on grid GEN3 with -1.E+34 for missing data
             X=122.5W(-122.5):121.5W(-121.5)
 DEPTH_QC Quality flag for depth axis, 1:  ...       ...       1:11      ...
               on grid GEN4 with -1.E+34 for missing data
             Z=0:325

  time range: 24-JUL-2025 18:30 to 22-OCT-2025 13:30

yes? SET REGION/L=2156:2156
yes? LIST TEMP, PSAL
             DATA SET: http://dods.mbari.org/opendap/data/ssdsdata/deployments/m1/202507/OS_MBARI-M1_20250724_R_TS.nc
             Hourly Gridded MBARI Mooring M1 Sea Water Temperature and Salinity Observations
             DEPTH (m): 0 to 325
             LONGITUDE: 122W(-122)
             LATITUDE: 36.8N
             TIME: 22-OCT-2025 13:30
 Column  1: TEMP is Hourly sea_water_temperature (celsius)
 Column  2: PSAL is Hourly sea_water_salinity (1)
              TEMP   PSAL
1      /  1:  16.38  33.32
10     /  2:  16.39  33.32
20     /  3:  15.43  33.28
40     /  4:  13.30  33.42
60     /  5:  11.95  33.51
80     /  6:  11.33  33.61
100    /  7:  11.01  33.63
150    /  8:  10.09  33.81
200    /  9:   9.67  33.93
250    / 10:   9.12  34.08
300    / 11:   7.92  34.09
yes? quit


Using Python to access the data:
================================
The Xarray library and variety of Python packages provides similar ease-of-use
capability in more modern computational environments. This module provides that
implementation. There are two dependencies that need to be installed via pip or
some other package manager:
    gsw
    xarray

e.g. pip install gsw xarray

__author__ = "Mike McCann"
__copyright__ = "Copyright 2025, Monterey Bay Aquarium Research Institute"
"""  # noqa: E501

import gsw
import xarray as xr

# Source for realtime M1 mooring data
url = (
    "http://dods.mbari.org/opendap/data/ssdsdata/deployments/m1/202507/OS_MBARI-M1_20250724_R_TS.nc"
)
ds = xr.open_dataset(url)

# Select the most recent profile by indexing the TIME dimension
latest = ds.isel(TIME=-1)
temp = latest["TEMP"].to_numpy.flatten()
psal = latest["PSAL"].to_numpy.flatten()
depth = latest["DEPTH"].to_numpy.flatten()

# Convert practical salinity to absolute salinity using lat and lon of M1
# mooring from the index data in the dataset
lon = ds["LONGITUDE"].to_numpy.item()
lat = ds["LATITUDE"].to_numpy.item()
abs_sal = gsw.SA_from_SP(psal, depth, lon, lat)

# Print out a header showing time, lat, lon and data source similar to Ferret output
time_str = str(latest["TIME"].to_numpy)
time_str = time_str.split(".")[0] + " UTC"  # Remove fractional seconds
print("Most recent sound speed profile from M1 mooring")  # noqa: T201
print("===============================================")  # noqa: T201
print(f"Data source: {url}")  # noqa: T201
print(f"Title:       {ds.title}")  # noqa: T201
print(f"Latitude:    {lat:.2f}")  # noqa: T201
print(f"Longitude:   {lon:.2f}")  # noqa: T201
print(f"Time:        {time_str}")  # noqa: T201
print()  # noqa: T201

# Calculate sound speed using the Gibbs Seawater (GSW) Oceanographic Toolbox
# Print out the profile of sound speed as a table
soundspeed = gsw.sound_speed(abs_sal, temp, depth)
print(f"{'Depth (m)':>10} {'Sound Speed (m/s)':>20}")  # noqa: T201
for d, c in zip(depth, soundspeed, strict=True):
    print(f"{d:10.2f} {c:20.2f}")  # noqa: T201
