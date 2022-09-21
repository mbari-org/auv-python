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
from resample import FREQ

LOG_NAME = "processing.log"


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

    def copy_to_AUVTCD(self, nc_file_base: str, freq: str = FREQ) -> None:
        "Copy the resampled netCDF file(s) to appropriate AUVCTD directory"
        auvctd_dir = "/Volumes/AUVCTD/surveys"
        try:
            os.stat(auvctd_dir)
        except FileNotFoundError:
            self.logger.error(
                f"{auvctd_dir} not found. Is cifs://atlas.shore.mbari.org/AUVCTD mounted?"
            )
            return
        year = self.args.mission.split(".")[0]
        auvctd_dir = os.path.join(auvctd_dir, year, "netcdf")
        self.logger.info(f"Copying {nc_file_base} files to {auvctd_dir}")
        # To avoid "fchmod failed: Permission denied" message use rsync instead cp
        # https://apple.stackexchange.com/a/206251
        for ftype in (f"{freq}.nc", "cal.nc", "align.nc", LOG_NAME):
            src_file = f"{nc_file_base}_{ftype}"
            if os.path.exists(src_file):
                os.system(f"rsync {src_file} {auvctd_dir}")
                self.logger.info(f"rsync {src_file} {auvctd_dir}")
            else:
                self.logger.warning(f"{src_file} not found")

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
            "--freq",
            action="store",
            default=FREQ,
            help="Resample freq",
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
    arch = Archiver()
    arch.process_command_line()
    file_name_base = f"{arch.args.auv_name}_{arch.args.mission}"
    nc_file_base = os.path.join(
        BASE_PATH,
        arch.args.auv_name,
        MISSIONNETCDFS,
        arch.args.mission,
        file_name_base,
    )
    p_start = time.time()
    if arch.args.M3:
        arch.copy_to_M3(nc_file_base)
    if arch.args.AUVCTD:
        arch.copy_to_AUVTCD(nc_file_base)
    arch.logger.info(f"Time to process: {(time.time() - p_start):.2f} seconds")
