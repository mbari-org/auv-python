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

import argparse
from datetime import datetime
import logging
import os
import platform
import subprocess
import sys

from align import Align_NetCDF
from calibrate import Calibrate_NetCDF
from logs2netcdfs import BASE_PATH, AUV_NetCDF, MISSIONNETCDFS
from resample import Resampler, FREQ
from archive import Archiver


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
        FIND_CMD = f'find -E {I2MAP_DIR} -regex "{REGEX}"'
    else:
        FIND_CMD = f'find {I2MAP_DIR} -regex "{REGEX}"'

    def mission_list(self, start_year: int, end_year: int) -> dict:
        missions = {}
        self.logger.debug("Executing %s", self.FIND_CMD)
        if self.args.last_n_days:
            self.FIND_CMD += f" -mtime -{self.args.last_n_days}"
        self.logger.info("Collecting missions from %s to %s", start_year, end_year)
        lines = subprocess.getoutput(f"{self.FIND_CMD} | sort").split("\n")
        for line in lines:
            self.logger.debug(line)
            if "No such file or directory" in line:
                self.logger.error("%s", line)
                self.logger.info("Is smb://titan.shore.mbari.org/M3 mounted?")
                return missions
            mission = line.split("/")[-1]
            if not mission:
                continue
            try:
                year = int(mission.split(".")[0])
                if start_year <= year and year <= end_year:
                    missions[mission] = line.rstrip()
            except ValueError:
                self.logger.warning("Cannot parse year from %s", mission)
        return missions

    def download_process(self, mission: str, src_dir: str) -> None:
        self.logger.info("Download and processing steps for %s", mission)
        auv_netcdf = AUV_NetCDF()
        auv_netcdf.args = argparse.Namespace()
        auv_netcdf.args.base_path = self.args.base_path
        auv_netcdf.args.local = self.args.local
        auv_netcdf.args.noinput = self.args.noinput
        auv_netcdf.args.clobber = self.args.clobber
        auv_netcdf.args.noreprocess = self.args.noreprocess
        auv_netcdf.args.auv_name = self.VEHICLE
        auv_netcdf.args.mission = mission
        auv_netcdf.args.use_m3 = True
        auv_netcdf.set_portal()
        auv_netcdf.args.verbose = self.args.verbose
        auv_netcdf.logger.setLevel(self._log_levels[self.args.verbose])
        auv_netcdf.commandline = self.commandline
        auv_netcdf.download_process_logs(src_dir=src_dir)

    def calibrate(self, mission: str) -> None:
        self.logger.info("Calibration steps for %s", mission)
        cal_netcdf = Calibrate_NetCDF()
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
        except (FileNotFoundError, EOFError) as e:
            cal_netcdf.logger.error("%s %s", mission, e)

    def align(self, mission: str) -> None:
        self.logger.info("Alignment steps for %s", mission)
        align_netcdf = Align_NetCDF()
        align_netcdf.args = argparse.Namespace()
        align_netcdf.args.base_path = self.args.base_path
        align_netcdf.args.auv_name = self.VEHICLE
        align_netcdf.args.mission = mission
        align_netcdf.args.plot = None
        align_netcdf.args.verbose = self.args.verbose
        align_netcdf.logger.setLevel(self._log_levels[self.args.verbose])
        align_netcdf.commandline = self.commandline
        try:
            netcdf_dir = align_netcdf.process_cal()
            align_netcdf.write_netcdf(netcdf_dir)
        except (FileNotFoundError, EOFError) as e:
            align_netcdf.logger.error("%s %s", mission, e)

    def resample(self, mission: str) -> None:
        self.logger.info("Resampling steps for %s", mission)
        resamp = Resampler()
        resamp.args = argparse.Namespace()
        resamp.args.auv_name = self.VEHICLE
        resamp.args.mission = mission
        resamp.args.plot = None
        resamp.args.freq = self.args.freq
        resamp.commandline = self.commandline
        resamp.args.verbose = self.args.verbose
        resamp.logger.setLevel(self._log_levels[self.args.verbose])
        file_name = f"{resamp.args.auv_name}_{resamp.args.mission}_align.nc"
        nc_file = os.path.join(
            self.args.base_path,
            resamp.args.auv_name,
            MISSIONNETCDFS,
            resamp.args.mission,
            file_name,
        )
        try:
            resamp.resample_mission(nc_file)
        except FileNotFoundError as e:
            self.logger.error("%s %s", mission, e)

    def archive(self, mission: str) -> None:
        self.logger.info("Archiving steps for %s", mission)
        arch = Archiver()
        arch.args = argparse.Namespace()
        arch.args.auv_name = self.VEHICLE
        arch.args.mission = mission
        arch.commandline = self.commandline
        arch.args.verbose = self.args.verbose
        arch.logger.setLevel(self._log_levels[self.args.verbose])
        file_name = f"{arch.args.auv_name}_{arch.args.mission}.nc"
        nc_file = os.path.join(
            BASE_PATH,
            arch.args.auv_name,
            MISSIONNETCDFS,
            arch.args.mission,
            file_name,
        )
        arch.copy_to_AUVTCD(nc_file, self.args.freq)

    def process_mission(self, mission: str, src_dir: str = None) -> None:
        self.logger.info("Processing mission %s", mission)
        if self.args.download_process:
            self.download_process(mission, src_dir)
        elif self.args.calibrate:
            self.calibrate(mission)
        elif self.args.align:
            self.align(mission)
        elif self.args.resample:
            self.resample(mission)
        elif self.args.archive:
            self.archive(mission)
        else:
            self.download_process(mission, src_dir)
            self.calibrate(mission)
            self.align(mission)
            self.resample(mission)
            self.archive(mission)

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
            default=2017,
            help="Begin processing at this year",
        )
        parser.add_argument(
            "--end_year",
            action="store",
            type=int,
            default=datetime.now().year,
            help="End processing at this year",
        )
        parser.add_argument(
            "--start_yd",
            action="store",
            type=int,
            default=1,
            help="Begin processing at this year day - start_year and end_year should be the same",
        )
        parser.add_argument(
            "--end_yd",
            action="store",
            type=int,
            default=366,
            help="End processing before this year day - start_year and end_year should be the same",
        )
        parser.add_argument(
            "--last_n_days",
            action="store",
            type=int,
            help="Process mission directories modified in the last n days",
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
            "--align",
            action="store_true",
            help="Align corrected coordinates onto calibrated instrument data",
        )
        parser.add_argument(
            "--resample",
            action="store_true",
            help="Resample all instrument data to common times",
        )
        parser.add_argument(
            "--archive",
            action="store_true",
            help="Archive the resampled netCDF file(s)",
        )
        parser.add_argument(
            "--mission",
            action="store",
            help="Process only this mission",
        )
        parser.add_argument(
            "--freq",
            action="store",
            default=FREQ,
            help="Resample freq",
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
