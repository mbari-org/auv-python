#!/usr/bin/env python
"""
Process Dorado/Gulper data from vehicle .log files to resampled .nc files.
(This replaces the Legacy SSDS Portal/Matlab processing steps.)

Find Dorado/Gulper missions in cifs://atlas.shore.mbari.org/AUVCTD/missionlogs
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
__copyright__ = "Copyright 2022, Monterey Bay Aquarium Research Institute"

from process import Processor


class DoradoProcessor(Processor):
    pass


if __name__ == "__main__":
    VEHICLE = "dorado"
    VEHICLE_DIR = "/Volumes/AUVCTD/missionlogs"
    MOUNT_DIR = "cifs://atlas.shore.mbari.org/AUVCTD"
    START_YEAR = 2003

    proc = DoradoProcessor(VEHICLE, VEHICLE_DIR + "/2022", MOUNT_DIR)
    proc.process_command_line()
    proc.args.use_m3 = False
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
        # TODO: Parallelize this with asyncio
        for mission in missions:
            if (
                int(mission.split(".")[1]) < proc.args.start_yd
                or int(mission.split(".")[1]) > proc.args.end_yd
            ):
                continue
            proc.process_mission(mission, src_dir=missions[mission])
