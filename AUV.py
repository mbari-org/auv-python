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

    def __init__(self):
        '''
        Initialize arrays
        '''
        self.secList = []
        self.pSecList = []

    def add_global_metadata(self):
        '''Use instance variables to write metadata specific for the data that are written
        '''

        iso_now = datetime.utcnow().isoformat() + 'Z'

        summary = 'Benthic Event Detector data from an in situ instrument designed to capture turbidity currents. '
        if self.args.seconds_offset:
            summary += 'Time adjusted with --seconds_offset {}'.format(self.args.seconds_offset)
        if self.args.seconds_slope:
            summary += ' --seconds_slope {}. '.format(self.args.seconds_slope)
        if self.args.bar_offset:
            summary += 'Pressure adjusted with --bar_offset {}'.format(self.args.bar_offset)
        if self.args.bar_slope:
            summary += '--bar_slope {}. '.format(self.args.bar_slope)
        if self.args.yaw_offset:
            summary += 'Yaw angle adjusted with --yaw_offset {}.'.format(self.args.yaw_offset)
        if self.args.trajectory and self.args.beg_depth and self.args.end_depth:
            summary += ' Positions extracted from thalweg file {} between depths {} and {}.'.format(
                        self.args.trajectory, self.args.beg_depth, self.args.end_depth)
        if self.args.trajectory:
            summary += ' Positions extracted from thalweg file {} between depths {} and {}.'.format(
                        self.args.trajectory, self.bed_depth[0], self.bed_depth[-1])

        summary += ' Data read from input file(s) {}.'.format(self.args.input)

        for v in ('seconds_offset', 'seconds_slope', 'bar_offset', 'bar_slope', 'yaw_offset'):
            setattr(self.ncFile, v, getattr(self.args, v))

        self.ncFile.summary = summary
        self.ncFile.netcdf_version = '3.6'
        self.ncFile.Conventions = 'CF-1.6'
        self.ncFile.date_created = iso_now
        self.ncFile.date_update = iso_now
        self.ncFile.date_modified = iso_now
        self.ncFile.featureType = self.featureType
        if self.featureType == 'trajectory':
            self.ncFile.geospatial_lat_min = np.min(self.latitude[:])
            self.ncFile.geospatial_lat_max = np.max(self.latitude[:])
            self.ncFile.geospatial_lon_min = np.min(self.longitude[:])
            self.ncFile.geospatial_lon_max = np.max(self.longitude[:])
            self.ncFile.geospatial_lat_units = 'degree_north'
            self.ncFile.geospatial_lon_units = 'degree_east'

            self.ncFile.geospatial_vertical_min= np.min(self.depth[:])
            self.ncFile.geospatial_vertical_max= np.max(self.depth[:])
            self.ncFile.geospatial_vertical_units = 'm'
            self.ncFile.geospatial_vertical_positive = 'down'

            self.ncFile.comment = ('BED devices measure 3 axes of acceleration and rotation about those'
                                   ' 3 axes at 50 Hz. They also measure pressure at 1 Hz during an event.'
                                   ' Those data are represented in this file as variables XA, YA, ZA, XR,'
                                   ' YR, ZR, and P. Additional variables are computed by the bed2netcdf.py'
                                   ' program; see the long_name and comment attributes for explanations.'
                                   ' Source code for the calucations are in the bed2netcdf.py, BEDS.py,'
                                   ' and util.py files on the subversion source code control server at MBARI:'
                                   ' http://kahuna.shore.mbari.org/viewvc/svn/BEDs/trunk/BEDs/Visualization/py/.')

        self.ncFile.time_coverage_start = coards.from_udunits(self.time[0], self.time.units).isoformat() + 'Z'
        self.ncFile.time_coverage_end = coards.from_udunits(self.time[-1], self.time.units).isoformat() + 'Z'

        self.ncFile.distribution_statement = 'Any use requires prior approval from the MBARI BEDS PI: Dr. Charles Paull'
        self.ncFile.license = self.ncFile.distribution_statement
        self.ncFile.useconst = 'Not intended for legal use. Data may contain inaccuracies.'
        self.ncFile.history = 'Created by "%s" on %s' % (' '.join(sys.argv), iso_now,)


