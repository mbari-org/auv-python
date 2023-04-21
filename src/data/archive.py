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
from pathlib import Path

from logs2netcdfs import BASE_PATH, LOG_FILES, MISSIONNETCDFS, AUV_NetCDF
from resample import FREQ

LOG_NAME = "processing.log"
AUVCTD_VOL = "/Volumes/AUVCTD"


class Archiver:
    logger = logging.getLogger(__name__)
    _handler = logging.StreamHandler()
    _handler.setFormatter(AUV_NetCDF._formatter)
    logger.addHandler(_handler)
    _log_levels = (logging.WARN, logging.INFO, logging.DEBUG)

    def copy_to_AUVTCD(self, nc_file_base: str, freq: str = FREQ) -> None:
        "Copy the resampled netCDF file(s) to appropriate AUVCTD directory"
        surveys_dir = os.path.join(AUVCTD_VOL, "surveys")
        try:
            os.stat(surveys_dir)
        except FileNotFoundError:
            self.logger.error(f"{surveys_dir} not found")
            self.logger.info("Is cifs://atlas.shore.mbari.org/AUVCTD mounted?")
            sys.exit(1)
        year = self.args.mission.split(".")[0]
        surveys_dir = os.path.join(surveys_dir, year, "netcdf")
        self.logger.info(f"Copying {nc_file_base} files to {surveys_dir}")
        # To avoid "fchmod failed: Permission denied" message use rsync instead  of cp
        # https://apple.stackexchange.com/a/206251
        for ftype in (f"{freq}.nc", "cal.nc", "align.nc", LOG_NAME):
            src_file = f"{nc_file_base}_{ftype}"
            if os.path.exists(src_file):
                os.system(f"rsync {src_file} {surveys_dir}")
                self.logger.info(f"rsync {src_file} {surveys_dir} done.")
            else:
                self.logger.error(f"{src_file} not found")

        # Copy intermediate files to AUVCTD/missionnetcdfs/YYYY/YYYYJJJ
        YYYYJJJ = "".join(self.args.mission.split(".")[:2])
        missionnetcdfs_dir = os.path.join(
            AUVCTD_VOL, MISSIONNETCDFS, year, YYYYJJJ, self.args.mission
        )
        Path(missionnetcdfs_dir).mkdir(parents=True, exist_ok=True)
        src_dir = "/".join(nc_file_base.split("/")[:-1])
        for log in LOG_FILES:
            src_file = os.path.join(src_dir, f"{log.replace('.log', '')}.nc")
            if os.path.exists(src_file):
                os.system(f"rsync {src_file} {missionnetcdfs_dir}")
                self.logger.info(f"rsync {src_file} {missionnetcdfs_dir} done.")
            else:
                self.logger.debug(f"{src_file} not found")

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
