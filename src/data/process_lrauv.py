#!/usr/bin/env python
"""
Process LRAUV data from NetCDF4 log files to resampled .nc files.
(This replaces the legacy lrauvNc4ToNetcdf.py script in STOQS.)

Find LRAUV log files in smb://atlas.shore.mbari.org/LRAUV<vehicle>/missionlogs
and run the data through standard science data processing to calibrated,
aligned, and resampled netCDF files.  Use a standard set of processing options;
more flexibility is available via the inndividual processing modules.

Limit processing to specific steps by providing arguments:
    --extract
    --combine
    --resample
    --archive
    --cleanup
If none provided then perform all steps.

Uses command line arguments from nc42netcdfs.py and combine.py.
"""

__author__ = "Mike McCann"
__copyright__ = "Copyright 2025, Monterey Bay Aquarium Research Institute"

from process import Processor


class LRAUVProcessor(Processor):
    pass


if __name__ == "__main__":
    VEHICLE = "tethys"
    LRAUV_DIR = "/Volumes/LRAUV"
    # It's possible that we might need calibration files for some sensors
    # in the future, so point to a potential directory where they can be found.
    CALIBRATION_DIR = "/Volumes/DMO/MDUC_CORE_CTD_200103/Calibration Files"
    MOUNT_DIR = "smb://atlas.shore.mbari.org/LRAUV"
    START_YEAR = 2012

    proc = LRAUVProcessor(VEHICLE, LRAUV_DIR, MOUNT_DIR, CALIBRATION_DIR)
    proc.process_command_line()
    proc.process_log_files()
