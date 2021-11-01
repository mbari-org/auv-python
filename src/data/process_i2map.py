#!/usr/bin/env python
"""
Scan master/i2MAP directory and process all missions found there

Find all the i2MAP missions in cifs://titan.shore.mbari.org/M3/master/i2MAP
and run the data through standard science data processing to calibrated and
aligned netCDF files.

Uses command line arguments from logs2netcdf.py and calibrate_align.py.
"""

__author__ = "Mike McCann"
__copyright__ = "Copyright 2021, Monterey Bay Aquarium Research Institute"

import logging
import subprocess

from logs2netcdfs import AUV_NetCDF, BASE_PATH
from calibrate_align import CalAligned_NetCDF


class Processor:

    logger = logging.getLogger(__name__)
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        "%(levelname)s %(asctime)s %(filename)s "
        "%(funcName)s():%(lineno)d %(message)s"
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.DEBUG)

    # Assume that we are running on a Mac and need the find(1) syntax for it
    I2MAP_DIR = "/Volumes/M3/master/i2MAP"
    REGEX = r".*[0-9][0-9][0-9][0-9]\.[0-9][0-9][0-9]\.[0-9][0-9]"
    FIND_CMD = f'find -E {I2MAP_DIR} -regex "{REGEX}" | sort'

    def mission_list(self, start_year: int = 2006, end_year: int = 2030) -> list:
        missions = []
        self.logger.debug("Executing %s", self.FIND_CMD)
        for line in subprocess.getoutput(self.FIND_CMD).split("\n"):
            self.logger.debug(line)
            if "No such file or directory" in line:
                raise FileNotFoundError(line)
            mission = line.split("/")[-1]
            try:
                year = int(mission.split(".")[0])
            except ValueError:
                self.logger.warning("Cannot parse year from %s", mission)
            if start_year <= year and year <= end_year:
                missions.append(mission)
        return missions

    def process_mission(self, mission: str) -> None:
        auv_netcdf = AUV_NetCDF()
        auv_netcdf.process_command_line()
        auv_netcdf.download_process_logs(vehicle="i2map", name=mission)

        cal_netcdf = CalAligned_NetCDF()
        cal_netcdf.process_command_line()
        try:
            netcdf_dir = cal_netcdf.process_logs(vehicle="i2map", name=mission)
            cal_netcdf.write_netcdf(netcdf_dir)
        except FileNotFoundError as e:
            cal_netcdf.logger.error("%s", e)


if __name__ == "__main__":
    proc = Processor()
    for mission in proc.mission_list():
        proc.process_mission(mission)
