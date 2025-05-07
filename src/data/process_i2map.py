#!/usr/bin/env python
"""
Scan master/i2MAP directory and process all missions found there

Find all the i2MAP missions in smb://thalassa.shore.mbari.org/M3/master/i2MAP
and run the data through standard science data processing to calibrated,
aligned, and resampled netCDF files.  Use a standard set of processing options;
more flexibility is available via the inndividual processing modules.

Limit processing to specific steps by providing arguments:
    --download_process
    --calibrate
    --resample
    --archive
    --cleanup
If none provided then perform all steps.

Uses command line arguments from logs2netcdf.py and calibrate.py.
"""

__author__ = "Mike McCann"
__copyright__ = "Copyright 2021, Monterey Bay Aquarium Research Institute"

from process import Processor


class I2mapProcessor(Processor):
    pass


if __name__ == "__main__":
    VEHICLE = "i2map"
    VEHICLE_DIR = "/Volumes/M3/master/i2MAP"
    CALIBRATION_DIR = "/Volumes/DMO/MDUC_CORE_CTD_200103/Calibration Files"
    MOUNT_DIR = "smb://thalassa.shore.mbari.org/M3"
    START_YEAR = 2017

    proc = I2mapProcessor(VEHICLE, VEHICLE_DIR, MOUNT_DIR, CALIBRATION_DIR)
    proc.process_command_line()
    proc.process_missions(START_YEAR)
