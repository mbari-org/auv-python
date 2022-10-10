#!/usr/bin/env python
"""
Base module for data processing.

Run the data through standard science data processing to calibrated,
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
import logging
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from lopcToNetCDF import LOPC_Processor, UnexpectedAreaOfCode
from align import Align_NetCDF, InvalidCalFile
from archive import LOG_NAME, Archiver
from calibrate import Calibrate_NetCDF
from getpass import getuser
from logs2netcdfs import BASE_PATH, MISSIONLOGS, MISSIONNETCDFS, AUV_NetCDF
from numpy.core._exceptions import UFuncTypeError
from resample import FREQ, Resampler
from socket import gethostname


class Processor:
    """
    Base class for data processing. Run the data through standard science data
    processing to calibrated, aligned, and resampled netCDF files.
    """

    logger = logging.getLogger(__name__)
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        "%(levelname)s %(asctime)s %(filename)s "
        "%(funcName)s():%(lineno)d %(message)s"
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    _log_levels = (logging.WARN, logging.INFO, logging.DEBUG)

    def __init__(self, vehicle, vehicle_dir, mount_dir) -> None:
        # Variables to be set by subclasses, e.g.:
        # vehicle = "i2map"
        # vehicle_dir = "/Volumes/M3/master/i2MAP"
        # mount_dir = "smb://titan.shore.mbari.org/M3"
        self.vehicle = vehicle
        self.vehicle_dir = vehicle_dir
        self.mount_dir = mount_dir

    def mission_list(self, start_year: int, end_year: int) -> dict:
        """Return a dictionary of source directories keyed by mission name."""
        missions = {}
        REGEX = r".*\/[0-9][0-9][0-9][0-9]\.[0-9][0-9][0-9]\.[0-9][0-9]"
        if platform.system() == "Darwin":
            find_cmd = f'find -E {self.vehicle_dir} -regex "{REGEX}"'
        else:
            find_cmd = f'find {self.vehicle_dir} -regex "{REGEX}"'
        self.logger.debug("Executing %s", find_cmd)
        if self.args.last_n_days:
            self.logger.info(
                f"Will be looking back {self.args.last_n_days} days for new missions..."
            )
            find_cmd += f" -mtime -{self.args.last_n_days}"
        self.logger.info("Finding missions from %s to %s", start_year, end_year)
        # Can be time consuming - use to discover missions
        lines = subprocess.getoutput(f"{find_cmd} | sort").split("\n")
        for line in lines:
            self.logger.debug(line)
            if "No such file or directory" in line:
                self.logger.error("%s", line)
                self.logger.info(f"Is {self.mount_dir} mounted?")
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

    def get_mission_dir(self, mission: str) -> str:
        """Return the mission directory."""
        yearyd = "".join(mission.split(".")[:2])
        path = os.path.join(self.vehicle_dir, yearyd, mission)
        if not os.path.exists(path):
            self.logger.error("%s does not exist.", path)
            self.logger.info(f"Is {self.mount_dir} mounted?")
            raise FileNotFoundError(path)
        return path

    def download_process(self, mission: str, src_dir: str) -> None:
        self.logger.info("Download and processing steps for %s", mission)
        auv_netcdf = AUV_NetCDF()
        auv_netcdf.args = argparse.Namespace()
        auv_netcdf.args.base_path = self.args.base_path
        auv_netcdf.args.local = self.args.local
        auv_netcdf.args.noinput = self.args.noinput
        auv_netcdf.args.clobber = self.args.clobber
        auv_netcdf.args.noreprocess = self.args.noreprocess
        auv_netcdf.args.auv_name = self.vehicle
        auv_netcdf.args.mission = mission
        auv_netcdf.args.use_portal = self.args.use_portal
        auv_netcdf.set_portal()
        auv_netcdf.args.verbose = self.args.verbose
        auv_netcdf.logger.setLevel(self._log_levels[self.args.verbose])
        auv_netcdf.logger.addHandler(self.log_handler)
        auv_netcdf.commandline = self.commandline
        auv_netcdf.download_process_logs(src_dir=src_dir)
        auv_netcdf.logger.removeHandler(self.log_handler)

        # Run lopcToNetCDF.py - mimic log message from logs2netcdfs.py
        lopc_bin = os.path.join(
            self.args.base_path, self.vehicle, MISSIONLOGS, mission, "lopc.bin"
        )
        try:
            file_size = os.path.getsize(lopc_bin)
        except FileNotFoundError:
            self.logger.warning("No lopc.bin file for %s", mission)
            return
        self.logger.info(f"Processing file {lopc_bin} ({file_size} bytes)")
        lopc_processor = LOPC_Processor()
        lopc_processor.args = argparse.Namespace()
        lopc_processor.args.bin_fileName = lopc_bin
        lopc_processor.args.netCDF_fileName = os.path.join(
            self.args.base_path, self.vehicle, MISSIONNETCDFS, mission, "lopc.nc"
        )
        lopc_processor.args.text_fileName = ""
        lopc_processor.args.trans_AIcrit = 0.4
        lopc_processor.args.LargeCopepod_AIcrit = 0.6
        lopc_processor.args.LargeCopepod_ESDmin = 1100.0
        lopc_processor.args.LargeCopepod_ESDmax = 1700.0
        lopc_processor.args.verbose = self.args.verbose
        lopc_processor.args.debugLevel = 0
        lopc_processor.args.force = self.args.clobber
        lopc_processor.logger.setLevel(self._log_levels[self.args.verbose])
        lopc_processor.logger.addHandler(self.log_handler)
        try:
            lopc_processor.main()
        except UnexpectedAreaOfCode as e:
            self.logger.error(e)
        lopc_processor.logger.removeHandler(self.log_handler)

    def calibrate(self, mission: str) -> None:
        self.logger.info("Calibration steps for %s", mission)
        cal_netcdf = Calibrate_NetCDF()
        cal_netcdf.args = argparse.Namespace()
        cal_netcdf.args.base_path = self.args.base_path
        cal_netcdf.args.local = self.args.local
        cal_netcdf.args.noinput = self.args.noinput
        cal_netcdf.args.clobber = self.args.clobber
        cal_netcdf.args.noreprocess = self.args.noreprocess
        cal_netcdf.args.auv_name = self.vehicle
        cal_netcdf.args.mission = mission
        cal_netcdf.args.plot = None
        cal_netcdf.args.verbose = self.args.verbose
        cal_netcdf.logger.setLevel(self._log_levels[self.args.verbose])
        cal_netcdf.logger.addHandler(self.log_handler)
        cal_netcdf.commandline = self.commandline
        try:
            netcdf_dir = cal_netcdf.process_logs()
            cal_netcdf.write_netcdf(netcdf_dir)
        except (FileNotFoundError, EOFError) as e:
            cal_netcdf.logger.error("%s %s", mission, e)
        cal_netcdf.logger.removeHandler(self.log_handler)

    def align(self, mission: str) -> None:
        self.logger.info("Alignment steps for %s", mission)
        align_netcdf = Align_NetCDF()
        align_netcdf.args = argparse.Namespace()
        align_netcdf.args.base_path = self.args.base_path
        align_netcdf.args.auv_name = self.vehicle
        align_netcdf.args.mission = mission
        align_netcdf.args.plot = None
        align_netcdf.args.verbose = self.args.verbose
        align_netcdf.logger.setLevel(self._log_levels[self.args.verbose])
        align_netcdf.logger.addHandler(self.log_handler)
        align_netcdf.commandline = self.commandline
        try:
            netcdf_dir = align_netcdf.process_cal()
            align_netcdf.write_netcdf(netcdf_dir)
        except (FileNotFoundError, EOFError, UFuncTypeError) as e:
            # UFuncTypeError seen in dorado 2008.010.10
            align_netcdf.logger.error("%s %s", mission, e)
            raise InvalidCalFile(f"{mission} {e}")
        finally:
            align_netcdf.logger.removeHandler(self.log_handler)

    def resample(self, mission: str) -> None:
        self.logger.info("Resampling steps for %s", mission)
        resamp = Resampler()
        resamp.args = argparse.Namespace()
        resamp.args.auv_name = self.vehicle
        resamp.args.mission = mission
        resamp.args.plot = None
        resamp.args.freq = self.args.freq
        resamp.commandline = self.commandline
        resamp.args.verbose = self.args.verbose
        resamp.logger.setLevel(self._log_levels[self.args.verbose])
        resamp.logger.addHandler(self.log_handler)
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
        finally:
            resamp.logger.removeHandler(self.log_handler)

    def archive(self, mission: str) -> None:
        self.logger.info("Archiving steps for %s", mission)
        arch = Archiver()
        arch.args = argparse.Namespace()
        arch.args.auv_name = self.vehicle
        arch.args.mission = mission
        arch.commandline = self.commandline
        arch.args.verbose = self.args.verbose
        arch.logger.setLevel(self._log_levels[self.args.verbose])
        arch.logger.addHandler(self.log_handler)
        file_name_base = f"{arch.args.auv_name}_{arch.args.mission}"
        nc_file_base = os.path.join(
            BASE_PATH,
            arch.args.auv_name,
            MISSIONNETCDFS,
            arch.args.mission,
            file_name_base,
        )
        arch.copy_to_AUVTCD(nc_file_base, self.args.freq)
        arch.logger.removeHandler(self.log_handler)

    def process_mission(self, mission: str, src_dir: str = None) -> None:
        netcdfs_dir = os.path.join(
            self.args.base_path, self.vehicle, MISSIONNETCDFS, mission
        )
        Path(netcdfs_dir).mkdir(parents=True, exist_ok=True)
        self.log_handler = logging.FileHandler(
            os.path.join(netcdfs_dir, f"{self.vehicle}_{mission}_{LOG_NAME}"), mode="w+"
        )
        self.log_handler.setLevel(self._log_levels[self.args.verbose])
        self.log_handler.setFormatter(self._formatter)
        self.logger.info("=" * 80)
        self.logger.addHandler(self.log_handler)
        self.logger.info(
            "Processing mission %s by user %s on host %s",
            mission,
            getuser(),
            gethostname(),
        )
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

    def process_missions(self, start_year: int) -> None:
        if not self.args.start_year:
            self.args.start_year = start_year
        if self.args.mission:
            # mission is string like: 2021.062.01 and is assumed to exist
            year = int(self.args.mission.split(".")[0])
            # missions = self.mission_list(start_year=year, end_year=year)
            try:
                t_start = time.time()
                self.process_mission(
                    self.args.mission,
                    src_dir=self.get_mission_dir(self.args.mission),
                )
            except (InvalidCalFile, FileNotFoundError, EOFError) as e:
                self.logger.error("%s %s", self.args.mission, e)
                self.logger.error("Cannot continue without a valid _cal.nc file")
            finally:
                self.logger.info(
                    "Mission %s took %.1f seconds to process",
                    self.args.mission,
                    time.time() - t_start,
                )
                if hasattr(self, "log_handler"):
                    self.logger.removeHandler(self.log_handler)
        elif self.args.start_year and self.args.end_year:
            missions = self.mission_list(
                start_year=self.args.start_year, end_year=self.args.end_year
            )
            # TODO: Parallelize this with asyncio
            for mission in missions:
                if (
                    int(mission.split(".")[1]) < self.args.start_yd
                    or int(mission.split(".")[1]) > self.args.end_yd
                ):
                    continue
                try:
                    t_start = time.time()
                    self.process_mission(mission, src_dir=missions[mission])
                except TimeoutError:
                    try:
                        self.logger.warning(
                            "Timeout error processing mission %s, trying again", mission
                        )
                        t_start = time.time()
                        self.process_mission(mission, src_dir=missions[mission])
                    except TimeoutError:
                        self.logger.exception(
                            "Timeout error processing mission %s", mission
                        )
                    except (InvalidCalFile, FileNotFoundError, EOFError) as e:
                        self.logger.error("%s %s", self.args.mission, e)
                        self.logger.error(
                            "Cannot continue without a valid _cal.nc file"
                        )
                    finally:
                        self.logger.info(
                            "Mission %s took %.1f seconds to process",
                            self.args.mission,
                            time.time() - t_start,
                        )
                        if hasattr(self, "log_handler"):
                            self.logger.removeHandler(self.log_handler)
                except (InvalidCalFile, FileNotFoundError, EOFError) as e:
                    self.logger.error("%s %s", mission, e)
                    self.logger.error("Cannot continue without a valid _cal.nc file")
                finally:
                    self.logger.info(
                        "Mission %s took %.1f seconds to process",
                        mission,
                        time.time() - t_start,
                    )
                    if hasattr(self, "log_handler"):
                        self.logger.removeHandler(self.log_handler)

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
            "--use_portal",
            action="store_true",
            help=(
                "Download data using portal (much faster than copy over"
                " remote connection), otherwise copy from mount point"
            ),
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

        # Append year to vehicle_dir if --start_year and --end_year identical
        if self.args.start_year == self.args.end_year:
            self.vehicle_dir = os.path.join(self.vehicle_dir, str(self.args.start_year))
        # Append year if processing a single mission
        if self.args.mission:
            self.vehicle_dir = os.path.join(
                self.vehicle_dir, self.args.mission.split(".")[0]
            )

        self.logger.setLevel(self._log_levels[self.args.verbose])
        self.commandline = " ".join(sys.argv)


if __name__ == "__main__":
    VEHICLE = "i2map"
    VEHICLE_DIR = "/Volumes/M3/master/i2MAP"
    MOUNT_DIR = "smb://titan.shore.mbari.org/M3"
    proc = Processor()
    proc.process_command_line()
    proc.process_missions()
