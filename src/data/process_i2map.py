#!/usr/bin/env python
"""
Find all the i2MAP missions in cifs://titan.shore.mbari.org/M3/master/i2MAP
and run the data through standard science data processing to calibrated and
aligned netCDF files.
"""

__author__ = "Mike McCann"
__copyright__ = "Copyright 2021, Monterey Bay Aquarium Research Institute"

import argparse
import logging
import subprocess
import sys

from logs2netcdfs import AUV_NetCDF

BASE_PATH = "auv_data"


class Processor:

    logger = logging.getLogger(__name__)
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        "%(levelname)s %(asctime)s %(filename)s "
        "%(funcName)s():%(lineno)d %(message)s"
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    _log_levels = (logging.WARN, logging.INFO, logging.DEBUG)

    # Assume that we are running on a Mac
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
        auv_netcdf.args = argparse.Namespace()
        auv_netcdf.args.base_path = self.args.base_path
        auv_netcdf.args.local = self.args.local
        auv_netcdf.args.auv_name = "i2map"
        auv_netcdf.args.noinput = self.args.noinput
        auv_netcdf.args.clobber = self.args.clobber
        auv_netcdf.args.noreprocess = self.args.noreprocess
        auv_netcdf.args.verbose = self.args.verbose
        auv_netcdf.set_portal()
        auv_netcdf.args.mission = mission
        auv_netcdf.download_process_logs()

    def process_command_line(self):
        examples = "Examples:" + "\n\n"
        examples += "  Write to local missionnetcdfs direcory:\n"
        examples += (
            "    poetry run " + sys.argv[0] + " --start_year 2017 --end_year 2017\n"
        )

        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter,
            description=(
                "Scan master/i2MAP directory on cifs://titan.shore.mbari.org/M3"
                " and process all missions found there."
            ),
            epilog=examples,
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


if __name__ == "__main__":
    proc = Processor()
    proc.process_command_line()
    for mission in proc.mission_list():
        proc.process_mission(mission)
