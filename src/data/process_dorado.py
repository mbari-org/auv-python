#!/usr/bin/env python
"""
Process Dorado/Gulper data from vehicle .log files to resampled .nc files.
(This replaces the Legacy SSDS Portal/Matlab processing steps.)

Find Dorado/Gulper missions in cifs://atlas.shore.mbari.org/AUVCTD/missionlogs
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
__copyright__ = "Copyright 2022, Monterey Bay Aquarium Research Institute"

from process import Processor


class DoradoProcessor(Processor):
    pass


if __name__ == "__main__":
    VEHICLE = "dorado"
    VEHICLE_DIR = "/Volumes/AUVCTD/missionlogs"
    MOUNT_DIR = "cifs://atlas.shore.mbari.org/AUVCTD"
    START_YEAR = 2003

    proc = DoradoProcessor(VEHICLE, VEHICLE_DIR, MOUNT_DIR)
    proc.process_command_line()
    proc.process_missions(START_YEAR)
