#!/usr/bin/env python
"""
Archive processed data to relevant repositories.

Use cifs://atlas.shore.mbari.org/AUVCTD for STOQS loading.
Use smb://titan.shore.mbari.org/M3 for paring with original data.
"""

__author__ = "Mike McCann"
__copyright__ = "Copyright 2022, Monterey Bay Aquarium Research Institute"

import argparse
import logging
import os
import platform
import sys
import time
from shutil import copyfile, copytree

from logs2netcdfs import BASE_PATH, MISSIONNETCDFS


class Archiver:
    logger = logging.getLogger(__name__)
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        "%(levelname)s %(asctime)s %(filename)s "
        "%(funcName)s():%(lineno)d %(message)s"
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    _log_levels = (logging.WARN, logging.INFO, logging.DEBUG)

    def copy_to_AUVTCD(self, resampled_nc_file: str) -> None:
        "Copy the resampled netCDF file(s) to appropriate AUVCTD directory"
        if platform.system() == "Darwin":
            auvctd_dir = "/Volumes/AUVCTD/surveys"
        else:
            auvctd_dir = "/mbari/AUVCTD/surveys"
        year = self.args.mission.split(".")[0]
        auvctd_dir = os.path.join(auvctd_dir, year, "netcdf")
        self.logger.info(f"Copying {resampled_nc_file} to {auvctd_dir}")
        copyfile(resampled_nc_file, auvctd_dir)

    def copy_to_M3(self, resampled_nc_file: str) -> None:
        pass

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
        ),
        parser.add_argument(
            "--auv_name",
            action="store",
            default="Dorado389",
            help="Dorado389 (default), i2map, or Multibeam",
        )
        parser.add_argument(
            "--mission",
            action="store",
            help="Mission directory, e.g.: 2020.064.10",
        ),
        parser.add_argument(
            "--M3",
            action="store_true",
            help="Copy reampled netCDF file(s) to appropriate place on M3",
        ),
        parser.add_argument(
            "--AUVCTD",
            action="store_true",
            help="Copy reampled netCDF file(s) to appropriate place on AUVCTD",
        ),
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
    archive = Archiver()
    archive.process_command_line()
    file_name = f"{archive.args.auv_name}_{archive.args.mission}_align.nc"
    nc_file = os.path.join(
        BASE_PATH,
        archive.args.auv_name,
        MISSIONNETCDFS,
        archive.args.mission,
        file_name,
    )
    p_start = time.time()
    if archive.args.M3:
        archive.copy_to_M3(nc_file)
    if archive.args.AUVCTD:
        archive.copy_to_AUVTCD(nc_file)
    archive.logger.info(f"Time to process: {(time.time() - p_start):.2f} seconds")
