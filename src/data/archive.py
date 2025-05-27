#!/usr/bin/env python
"""
Archive processed data to relevant repositories.

Use smb://atlas.shore.mbari.org/AUVCTD for STOQS loading.
Use smb://titan.shore.mbari.org/M3 for paring with original data.
"""

__author__ = "Mike McCann"
__copyright__ = "Copyright 2022, Monterey Bay Aquarium Research Institute"

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

from create_products import MISSIONIMAGES, MISSIONODVS
from logs2netcdfs import BASE_PATH, LOG_FILES, MISSIONNETCDFS, AUV_NetCDF
from resample import FREQ

LOG_NAME = "processing.log"
AUVCTD_VOL = "/Volumes/AUVCTD"


class Archiver:
    logger = logging.getLogger(__name__)
    _handler = logging.StreamHandler()
    _handler.setFormatter(AUV_NetCDF._formatter)
    _log_levels = (logging.WARN, logging.INFO, logging.DEBUG)

    def __init__(self, add_handlers=True):  # noqa: FBT002
        if add_handlers:
            self.logger.addHandler(self._handler)

    def copy_to_AUVTCD(self, nc_file_base: Path, freq: str = FREQ) -> None:  # noqa: C901, PLR0912, PLR0915
        "Copy the resampled netCDF file(s) to appropriate AUVCTD directory"
        surveys_dir = Path(AUVCTD_VOL) / "surveys"
        try:
            Path(surveys_dir).stat()
        except FileNotFoundError:
            self.logger.exception("%s not found", surveys_dir)
            self.logger.info("Is smb://atlas.shore.mbari.org/AUVCTD mounted?")
            sys.exit(1)
        year = self.args.mission.split(".")[0]
        surveynetcdfs_dir = Path(surveys_dir, year, "netcdf")

        # To avoid "fchmod failed: Permission denied" message use shutil.copyfile

        if not self.args.archive_only_products:
            self.logger.info("Archiving %s files to %s", nc_file_base, surveynetcdfs_dir)
            # Copy netCDF files to AUVCTD/surveys/YYYY/netcdf
            if hasattr(self.args, "flash_threshold"):
                if self.args.flash_threshold and self.args.resample:
                    ft_ending = f"{freq}_ft{self.args.flash_threshold:.0E}.nc".replace(
                        "E+",
                        "E",
                    )
                    ftypes = (ft_ending,)
                else:
                    ftypes = (f"{freq}.nc", "cal.nc", "align.nc")
            else:
                ftypes = (f"{freq}.nc", "cal.nc", "align.nc")
            for ftype in ftypes:
                src_file = Path(f"{nc_file_base}_{ftype}")
                dst_file = Path(surveynetcdfs_dir, src_file.name)
                if self.args.clobber:
                    if dst_file.exists():
                        self.logger.info("Removing %s", dst_file)
                        dst_file.unlink()
                    if src_file.exists():
                        shutil.copyfile(src_file, dst_file)
                        self.logger.info("copyfile %s %s done.", src_file, surveynetcdfs_dir)
                else:
                    self.logger.info(
                        "%26s exists, but is not being archived because --clobber is not specified.",  # noqa: E501
                        src_file.name,
                    )

            if not hasattr(self.args, "resample") or not self.args.resample:
                # Copy intermediate files to AUVCTD/missionnetcdfs/YYYY/YYYYJJJ
                YYYYJJJ = "".join(self.args.mission.split(".")[:2])
                missionnetcdfs_dir = Path(
                    AUVCTD_VOL,
                    MISSIONNETCDFS,
                    year,
                    YYYYJJJ,
                    self.args.mission,
                )
                Path(missionnetcdfs_dir).mkdir(parents=True, exist_ok=True)
                src_dir = Path(nc_file_base).parent
                # The original lopc.bin file is logged out of band from the MVC, add it to LOG_FILES
                # so that lopc.nc is archived along with the other netcdf versions of the log files.
                for log in [*LOG_FILES, "lopc.log"]:
                    src_file = Path(src_dir, f"{log.replace('.log', '')}.nc")
                    if self.args.clobber:
                        if src_file.exists():
                            shutil.copyfile(src_file, missionnetcdfs_dir / src_file.name)
                            self.logger.info("copyfile %s %s done.", src_file, missionnetcdfs_dir)
                    else:
                        self.logger.info(
                            "%26s exists, but is not being archived because --clobber is not specified.",  # noqa: E501
                            src_file.name,
                        )

        # Copy files created by create_products.py
        self.logger.info("Archiving product files")
        for src_dir, dst_dir in ((MISSIONODVS, "odv"), (MISSIONIMAGES, "images")):
            src_dir = Path(  # noqa: PLW2901
                BASE_PATH,
                self.args.auv_name,
                src_dir,
                self.args.mission,
            )
            if Path(src_dir).exists():
                dst_dir = Path(surveys_dir, year, dst_dir)  # noqa: PLW2901
                Path(dst_dir).mkdir(parents=True, exist_ok=True)
                if self.args.clobber:
                    # Copy files individually to avoid permission issues with copytree.
                    # This will not copy subdirectories, but we don't expect any.
                    for src_file in src_dir.glob("*"):
                        dst_file = Path(dst_dir, src_file.name)
                        if dst_file.exists():
                            self.logger.info("Removing %s", dst_file)
                            dst_file.unlink()
                        shutil.copyfile(src_file, dst_file)
                        self.logger.info("copyfile %s %s done.", src_file, dst_dir)

                    # shutil.copytree(
                    #     src_dir, dst_dir, dirs_exist_ok=True, copy_function=shutil.copy
                    # )
                    # self.logger.info("copytree %s/* %s done.", src_dir, dst_dir)
                else:
                    self.logger.info(
                        "%26s exists, but is not being archived because --clobber is not specified.",  # noqa: E501
                        src_dir.name,
                    )
            else:
                self.logger.debug("%s not found", src_dir)
        if self.args.create_products or (hasattr(self.args, "resample") and self.args.resample):
            # Do not copy processing.log file if only partial processing was done
            self.logger.info(
                "Partial processing, not archiving %s",
                f"{nc_file_base}_{LOG_NAME}",
            )
        else:
            # Copy the processing.log file last so that we get everything
            src_file = Path(f"{nc_file_base}_{LOG_NAME}")
            dst_file = Path(surveynetcdfs_dir, src_file.name)
            if src_file.exists():
                if self.args.clobber:
                    self.logger.info("copyfile %s %s", src_file, surveynetcdfs_dir)
                    shutil.copyfile(src_file, dst_file)
                    self.logger.info("copyfile %s %s done.", src_file, surveynetcdfs_dir)
                else:
                    self.logger.info(
                        "%26s exists, but is not being archived because --clobber is not specified.",  # noqa: E501
                        src_file.name,
                    )

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
            help="Base directory for missionlogs and missionnetcdfs, default: auv_data",
        )
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
        )
        parser.add_argument(
            "--freq",
            action="store",
            default=FREQ,
            help="Resample freq",
        )
        parser.add_argument(
            "--M3",
            action="store_true",
            help="Copy reampled netCDF file(s) to appropriate place on M3",
        )
        parser.add_argument(
            "--AUVCTD",
            action="store_true",
            help="Copy reampled netCDF file(s) to appropriate place on AUVCTD",
        )
        parser.add_argument(
            "--clobber",
            action="store_true",
            help="Remove existing netCDF files before copying to the AUVCTD directory",
        )
        parser.add_argument(
            "--archive_only_products",
            action="store_true",
            help="Copy to AUVCTD directory only the products, not the netCDF files",
        )
        parser.add_argument(
            "--create_products",
            action="store_true",
            help="Create products from the resampled netCDF file(s)",
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
                [f"{i}: {v}" for i, v in enumerate(("WARN", "INFO", "DEBUG"))],
            ),
        )
        self.args = parser.parse_args()
        self.logger.setLevel(self._log_levels[self.args.verbose])
        self.commandline = " ".join(sys.argv)


if __name__ == "__main__":
    arch = Archiver()
    arch.process_command_line()
    file_name_base = f"{arch.args.auv_name}_{arch.args.mission}"
    nc_file_base = Path(
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
    arch.logger.info("Time to process: %.2f seconds", (time.time() - p_start))
