#!/usr/bin/env python
"""
Scan master/i2MAP directory and process all missions found there

Find all the i2MAP missions in cifs://titan.shore.mbari.org/M3/master/i2MAP
and run the data through standard science data processing to calibrated and
aligned netCDF files.

Limit processing to specific steps by providing arugments:
    --download_process
    --calibrate
    --resample
If none provided then perform all steps.

Uses command line arguments from logs2netcdf.py and calibrate_align.py.
"""

__author__ = "Mike McCann"
__copyright__ = "Copyright 2021, Monterey Bay Aquarium Research Institute"

import argparse
import logging
import platform
import subprocess
import sys

from calibrate_align import CalAligned_NetCDF
from logs2netcdfs import BASE_PATH, AUV_NetCDF

TEST_LIST = [
    "2020.008.00",
    "2020.041.02",
    "2020.041.02",
    "2020.055.01",
    "2020.181.02",
    "2020.210.03",
    "2020.224.04",
    "2020.258.01",
    "2020.266.01",
]


class Processor:
    VEHICLE = "i2map"
    logger = logging.getLogger(__name__)
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        "%(levelname)s %(asctime)s %(filename)s "
        "%(funcName)s():%(lineno)d %(message)s"
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    _log_levels = (logging.WARN, logging.INFO, logging.DEBUG)

    I2MAP_DIR = "/Volumes/M3/master/i2MAP"
    REGEX = r".*\/[0-9][0-9][0-9][0-9]\.[0-9][0-9][0-9]\.[0-9][0-9]"
    if platform.system() == "Darwin":
        FIND_CMD = f'find -E {I2MAP_DIR} -regex "{REGEX}" | sort'
    else:
        FIND_CMD = f'find {I2MAP_DIR} -regex "{REGEX}" | sort'

    def mission_list(self, start_year: int, end_year: int) -> list:
        missions = []
        self.logger.debug("Executing %s", self.FIND_CMD)
        for line in subprocess.getoutput(self.FIND_CMD).split("\n"):
            self.logger.debug(line)
            if "No such file or directory" in line:
                self.logger.error("%s", line)
                self.logger.info("Is smb://titan.shore.mbari.org/M3 mounted?")
                return missions
            mission = line.split("/")[-1]
            try:
                year = int(mission.split(".")[0])
            except ValueError:
                self.logger.warning("Cannot parse year from %s", mission)
            if start_year <= year and year <= end_year:
                missions.append(mission)
        return missions

    def download_process(self, mission: str) -> None:
        auv_netcdf = AUV_NetCDF()
        auv_netcdf.args = argparse.Namespace()
        auv_netcdf.args.base_path = self.args.base_path
        auv_netcdf.args.local = self.args.local
        auv_netcdf.args.noinput = self.args.noinput
        auv_netcdf.args.clobber = self.args.clobber
        auv_netcdf.args.noreprocess = self.args.noreprocess
        auv_netcdf.args.auv_name = self.VEHICLE
        auv_netcdf.args.mission = mission
        auv_netcdf.set_portal()
        auv_netcdf.args.verbose = self.args.verbose
        auv_netcdf.logger.setLevel(self._log_levels[self.args.verbose])
        auv_netcdf.commandline = self.commandline
        auv_netcdf.download_process_logs()

    def calibrate(self, mission: str) -> None:
        cal_netcdf = CalAligned_NetCDF()
        cal_netcdf.args = argparse.Namespace()
        cal_netcdf.args.base_path = self.args.base_path
        cal_netcdf.args.local = self.args.local
        cal_netcdf.args.noinput = self.args.noinput
        cal_netcdf.args.clobber = self.args.clobber
        cal_netcdf.args.noreprocess = self.args.noreprocess
        cal_netcdf.args.auv_name = self.VEHICLE
        cal_netcdf.args.mission = mission
        cal_netcdf.args.plot = None
        cal_netcdf.args.verbose = self.args.verbose
        cal_netcdf.logger.setLevel(self._log_levels[self.args.verbose])
        cal_netcdf.commandline = self.commandline
        try:
            netcdf_dir = cal_netcdf.process_logs()
            cal_netcdf.write_netcdf(netcdf_dir)
        except FileNotFoundError as e:
            cal_netcdf.logger.error("%s %s", mission, e)

    def process_mission(self, mission: str) -> None:
        if (not self.args.calibrate and not self.args.download_process) or (
            self.args.calibrate and self.args.download_process
        ):
            self.download_process(mission)
            self.calibrate(mission)
        elif self.args.download_process:
            self.download_process(mission)
        elif self.args.calibrate:
            self.calibrate(mission)

    def process_command_line(self):
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter,
            description=__doc__,
        )
        parser.add_argument(
            "--base_path",
            action="store",
            default=BASE_PATH,
            help="Base directory for missionlogs and"
            " missionnetcdfs, default: auv_data",
        )
        parser.add_argument(
            "--local",
            action="store_true",
            help="Specify if files are local in the MISSION directory",
        )
        parser.add_argument(
            "--clobber",
            action="store_true",
            help="Use with --noinput to overwrite existing" " downloaded log files",
        )
        parser.add_argument(
            "--noinput",
            action="store_true",
            help="Execute without asking for a response, e.g. "
            " to not ask to re-download file",
        )
        parser.add_argument(
            "--noreprocess",
            action="store_true",
            help="Use with --noinput to not re-process existing"
            " downloaded log files",
        )
        parser.add_argument(
            "--start_year",
            action="store",
            type=int,
            default=2000,
            help="Begin processing at this year",
        )
        parser.add_argument(
            "--end_year",
            action="store",
            type=int,
            default=2100,
            help="End processing at this year",
        )
        parser.add_argument(
            "--download_process",
            action="store_true",
            help="Download and process instrument logs to netCDF files",
        )
        parser.add_argument(
            "--calibrate",
            action="store_true",
            help="Calibrate and adjust original instrument netCDF data",
        )
        parser.add_argument(
            "--resample",
            action="store_true",
            help="Resample all instrument data to common times",
        )
        parser.add_argument(
            "-v",
            "--verbose",
            type=int,
            choices=range(3),
            action="store",
            default=0,
            const=1,
            nargs="?",
            help="verbosity level: "
            + ", ".join(
                [f"{i}: {v}" for i, v, in enumerate(("WARN", "INFO", "DEBUG"))]
            ),
        )

        self.args = parser.parse_args()
        self.logger.setLevel(self._log_levels[self.args.verbose])
        self.commandline = " ".join(sys.argv)


if __name__ == "__main__":
    proc = Processor()
    proc.process_command_line()
    ##for mission in proc.mission_list(
    ##    start_year=proc.args.start_year, end_year=proc.args.end_year
    ##):
    for mission in TEST_LIST:
        proc.process_mission(mission)
