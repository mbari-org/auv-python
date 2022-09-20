#!/usr/bin/env python
"""
Scan master/i2MAP directory and process all missions found there

Find all the i2MAP missions in cifs://titan.shore.mbari.org/M3/master/i2MAP
and run the data through standard science data processing to calibrated,
aligned, and resampled netCDF files.  Use a standard set of processing options;
more flexibility is available via the inndividual processing modules.

Limit processing to specific steps by providing arugments:
    --download_process
    --calibrate
    --resample
    --archive
If none provided then perform all steps.

Uses command line arguments from logs2netcdf.py and calibrate.py.
"""

__author__ = "Mike McCann"
__copyright__ = "Copyright 2021, Monterey Bay Aquarium Research Institute"

from process import Processor


TEST_LIST = [
    # "2020.008.00",
    "2020.041.02",
    "2020.041.02",
    "2020.055.01",
    "2020.181.02",
    "2020.210.03",
    "2020.224.04",
    "2020.258.01",
    "2020.266.01",
    "2021.264.03",
]


class I2mapProcessor(Processor):
    pass


if __name__ == "__main__":
    VEHICLE = "i2map"
    VEHICLE_DIR = "/Volumes/M3/master/i2MAP"
    MOUNT_DIR = "smb://titan.shore.mbari.org/M3"
    START_YEAR = 2017

    proc = I2mapProcessor(VEHICLE, VEHICLE_DIR, MOUNT_DIR)
    proc.process_command_line()
    proc.args.use_m3 = True
    if not proc.args.start_year:
        proc.args.start_year = START_YEAR

    if proc.args.mission:
        # mission is string like: 2021.062.01
        year = int(proc.args.mission.split(".")[0])
        missions = proc.mission_list(start_year=year, end_year=year)
        if proc.args.mission in missions:
            proc.process_mission(
                proc.args.mission,
                src_dir=missions[proc.args.mission],
            )
        else:
            proc.logger.error(
                "Mission %s not found in missions: %s",
                proc.args.mission,
                missions,
            )
    elif proc.args.start_year and proc.args.end_year:
        missions = proc.mission_list(
            start_year=proc.args.start_year,
            end_year=proc.args.end_year,
        )
        for mission in missions:
            if (
                int(mission.split(".")[1]) < proc.args.start_yd
                or int(mission.split(".")[1]) > proc.args.end_yd
            ):
                continue
            proc.process_mission(mission, src_dir=missions[mission])
