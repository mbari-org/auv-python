#!/usr/bin/env python
"""
Align calibrated data producing a netCDF file with coordinates on each variable

Read calibrated data from netCDF files created by calibrate.py, use the
best available (e.g. filtered, nudged) coordinate variables to assign to
each measured (record) variable. The original instrument sampling interval
is preserved with the coordinate varaibles interpolated onto that original
time axis.
"""

__author__ = "Mike McCann"
__copyright__ = "Copyright 2021, Monterey Bay Aquarium Research Institute"

import cf_xarray  # Needed for the .cf accessor
import logging
import os
import sys
import time
from collections import OrderedDict, namedtuple
from datetime import datetime
from socket import gethostname

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import xarray as xr
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from matplotlib import patches
from scipy.interpolate import interp1d
from seawater import eos80
from shapely.geometry import LineString, Point


class Align_NetCDF:
    def process_command_line(self):
        pass

    def process_cal(self):
        pass

    def write_netcdf(self):
        pass


if __name__ == "__main__":

    align_netcdf = Align_NetCDF()
    align_netcdf.process_command_line()
    p_start = time.time()
    netcdf_dir = align_netcdf.process_logs()
    align_netcdf.write_netcdf(netcdf_dir)
    align_netcdf.logger.info(f"Time to process: {(time.time() - p_start):.2f} seconds")
