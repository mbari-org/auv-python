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
    _log_levels = (logging.WARN, logging.INFO, logging.DEBUG)

    def __init__(self, add_handlers=True):
        if add_handlers:
            self.logger.addHandler(self._handler)

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
        surveynetcdfs_dir = os.path.join(surveys_dir, year, "netcdf")

        # To avoid "fchmod failed: Permission denied" message use rsync instead
        # of cp: https://apple.stackexchange.com/a/206251

        if not self.args.archive_only_products:
            self.logger.info(f"Archiving {nc_file_base} files to {surveynetcdfs_dir}")
            # Rsync netCDF files to AUVCTD/surveys/YYYY/netcdf
            for ftype in (f"{freq}.nc", "cal.nc", "align.nc"):
                src_file = f"{nc_file_base}_{ftype}"
                dst_file = (
                    f"{os.path.join(surveynetcdfs_dir, os.path.basename(src_file))}"
                )
                if self.args.clobber and os.path.exists(dst_file):
                    self.logger.info(f"Removing {dst_file}")
                    os.remove(dst_file)
                if os.path.exists(src_file):
                    os.system(f"rsync {src_file} {surveynetcdfs_dir}")
                    self.logger.info(f"rsync {src_file} {surveynetcdfs_dir} done.")
                else:
                    self.logger.debug(f"{src_file} not found")

            # Rsync intermediate files to AUVCTD/missionnetcdfs/YYYY/YYYYJJJ
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

        # Rsync files created by create_products.py
        self.logger.info(f"Archiving product files")
        for src_dir, dst_dir in (("missionodvs", "odv"), ("missionimages", "images")):
            src_dir = os.path.join(
                BASE_PATH, self.args.auv_name, src_dir, self.args.mission
            )
            if os.path.exists(src_dir):
                dst_dir = os.path.join(surveys_dir, year, dst_dir)
                Path(dst_dir).mkdir(parents=True, exist_ok=True)
                os.system(f"rsync -r {src_dir}/* {dst_dir}")
                self.logger.info(f"rsync {src_dir}/* {dst_dir} done.")
            else:
                self.logger.debug(f"{src_dir} not found")
        if self.args.create_products:
            # Do not rsync processing.log file if only partial processing was done
            self.logger.info(
                f"Partial processing, not archiving {nc_file_base}_{LOG_NAME}"
            )
        else:
            # Rsync the processing.log file last so that we get everything
            src_file = f"{nc_file_base}_{LOG_NAME}"
            dst_file = f"{os.path.join(surveynetcdfs_dir, os.path.basename(src_file))}"
            if os.path.exists(src_file):
                self.logger.info(f"rsync {src_file} {surveynetcdfs_dir}")
                os.system(f"rsync {src_file} {surveynetcdfs_dir}")

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
            "--clobber",
            action="store_true",
            help="Remove existing netCDF files before rsyncing to the AUVCTD directory",
        )
        parser.add_argument(
            "--archive_only_products",
            action="store_true",
            help="Rsync to AUVCTD directory only the products, not the netCDF files",
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
