#!/usr/bin/env python
'''
Base classes for reading and writing data for MBARI's Dorado class AUVs.

--
Mike McCann
MBARI 30 March 2020
'''

import os
import sys
import csv
import time
import coards
import glob
import numpy as np
import numpy.ma as ma
from collections import namedtuple
from datetime import datetime
from pupynere import netcdf_file
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from seawater import eos80
from subprocess import Popen, PIPE


class AUV(object):
    '''
    Container for common methods to be reused by BEDs processing software.
    Initially used by bed2netcdf.py and bed2x3d.py to read the data.
    '''

    def add_global_metadata(self):
        '''Use instance variables to write metadata specific for the data that are written
        '''

        iso_now = datetime.utcnow().isoformat() + 'Z'

        self.nc_file.netcdf_version = '4'
        self.nc_file.Conventions = 'CF-1.6'
        self.nc_file.date_created = iso_now
        self.nc_file.date_update = iso_now
        self.nc_file.date_modified = iso_now
        self.nc_file.featureType = 'trajectory'

        self.nc_file.comment = ('Autonomous Underwater Vehicle data....'
                               'https://bitbucket.org/mbari/auv-python')

        self.nc_file.time_coverage_start = coards.from_udunits(self.time[0], self.time.units).isoformat() + 'Z'
        self.nc_file.time_coverage_end = coards.from_udunits(self.time[-1], self.time.units).isoformat() + 'Z'

        self.nc_file.distribution_statement = 'Any use requires prior approval from MBARI'
        self.nc_file.license = self.nc_file.distribution_statement
        self.nc_file.useconst = 'Not intended for legal use. Data may contain inaccuracies.'
        self.nc_file.history = 'Created by "%s" on %s' % (' '.join(sys.argv), iso_now,)


