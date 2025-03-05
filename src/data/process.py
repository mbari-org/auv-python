#!/usr/bin/env python
"""
Base module for data processing.

Run the data through standard science data processing to calibrated,
aligned, and resampled netCDF files.  Use a standard set of processing options;
more flexibility is available via the inndividual processing modules.

Limit processing to specific steps by providing arugments:
    --download_process
    --calibrate
    --align
    --resample
    --archive
    --create_products
    --email_to
    --cleanup
If none provided then perform all steps.

Uses command line arguments from logs2netcdf.py and calibrate.py.
"""

__author__ = "Mike McCann"
__copyright__ = "Copyright 2021, Monterey Bay Aquarium Research Institute"

import argparse
import logging
import multiprocessing
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from getpass import getuser
from pathlib import Path
from socket import gethostname

from align import Align_NetCDF, InvalidCalFile
from archive import LOG_NAME, Archiver
from calibrate import EXPECTED_SENSORS, Calibrate_NetCDF
from create_products import CreateProducts
from dorado_info import FAILED, TEST, dorado_info
from emailer import NOTIFICATION_EMAIL, Emailer
from logs2netcdfs import BASE_PATH, MISSIONLOGS, MISSIONNETCDFS, AUV_NetCDF
from lopcToNetCDF import LOPC_Processor, UnexpectedAreaOfCode
from resample import (
    AUVCTD_OPENDAP_BASE,
    FLASH_THRESHOLD,
    FREQ,
    MF_WIDTH,
    InvalidAlignFile,
    Resampler,
)


class MissingDoradoInfo(Exception):
    pass


class TestMission(Exception):
    pass


class FailedMission(Exception):
    pass


class Processor:
    """
    Base class for data processing. Run the data through standard science data
    processing to calibrated, aligned, and resampled netCDF files.
    """

    logger = logging.getLogger(__name__)
    _handler = logging.StreamHandler()
    _handler.setFormatter(AUV_NetCDF._formatter)
    logger.addHandler(_handler)
    _log_levels = (logging.WARN, logging.INFO, logging.DEBUG)

    def __init__(self, vehicle, vehicle_dir, mount_dir, calibration_dir) -> None:
        # Variables to be set by subclasses, e.g.:
        # vehicle = "i2map"
        # vehicle_dir = "/Volumes/M3/master/i2MAP"
        # mount_dir = "smb://titan.shore.mbari.org/M3"
        self.vehicle = vehicle
        self.vehicle_dir = vehicle_dir
        self.mount_dir = mount_dir
        self.calibration_dir = calibration_dir

    def mission_list(self, start_year: int, end_year: int) -> dict:
        """Return a dictionary of source directories keyed by mission name."""
        missions = {}
        REGEX = r".*\/[0-9][0-9][0-9][0-9]\.[0-9][0-9][0-9]\.[0-9][0-9]"
        safe_vehicle_dir = Path(self.vehicle_dir).resolve()
        if not safe_vehicle_dir.exists():
            self.logger.error("%s does not exist.", safe_vehicle_dir)
            self.logger.info("Is %s mounted?", self.mount_dir)
            sys.exit(1)
        if platform.system() == "Darwin":
            find_cmd = f'find -E {safe_vehicle_dir} -regex "{REGEX}"'
        else:
            find_cmd = f'find {safe_vehicle_dir} -regex "{REGEX}"'
        self.logger.debug("Executing %s", find_cmd)
        if self.args.last_n_days:
            self.logger.info(
                "Will be looking back %d days for new missions...", self.args.last_n_days
            )
            find_cmd += f" -mtime -{self.args.last_n_days}"
        self.logger.info("Finding missions from %s to %s", start_year, end_year)
        # Can be time consuming - use to discover missions
        lines = subprocess.getoutput(f"{find_cmd} | sort").split("\n")  # noqa: S605
        for line in lines:
            self.logger.debug(line)
            if "No such file or directory" in line:
                self.logger.error("%s", line)
                self.logger.info("Is %s mounted?", self.mount_dir)
                return missions
            mission = line.split("/")[-1]
            if not mission:
                continue
            try:
                year = int(mission.split(".")[0])
                if start_year <= year <= end_year:
                    missions[mission] = line.rstrip()
            except ValueError:
                self.logger.warning("Cannot parse year from %s", mission)
        return missions

    def get_mission_dir(self, mission: str) -> str:
        """Return the mission directory."""
        if not Path(self.vehicle_dir).exists():
            self.logger.error("%s does not exist.", self.vehicle_dir)
            self.logger.info("Is %s mounted?", self.mount_dir)
            sys.exit(1)
        if self.vehicle.lower() == "dorado":
            year = mission.split(".")[0]
            yearyd = "".join(mission.split(".")[:2])
            path = Path(self.vehicle_dir, year, yearyd, mission)
        elif self.vehicle.lower() == "i2map":
            year = int(mission.split(".")[0])
            # Could construct the YYYY/MM/YYYYMMDD path on M3/Master
            # but use the mission_list() method to find the mission dir instead
            missions = self.mission_list(start_year=year, end_year=year)
            if mission in missions:
                path = missions[mission]
            else:
                self.logger.error("Cannot find %s in %s", mission, self.vehicle_dir)
                error_message = f"Cannot find {mission} in {self.vehicle_dir}"
                raise FileNotFoundError(error_message)
        elif self.vehicle == "Dorado389":
            # The Dorado389 vehicle is a special case used for testing locally and in CI
            path = self.vehicle_dir
        if not Path(path).exists():
            self.logger.error("%s does not exist.", path)
            error_message = f"{path} does not exist."
            raise FileNotFoundError(error_message)
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
        lopc_bin = Path(
            self.args.base_path,
            self.vehicle,
            MISSIONLOGS,
            mission,
            "lopc.bin",
        )
        try:
            file_size = Path(lopc_bin).stat().st_size
        except FileNotFoundError:
            if "lopc" in EXPECTED_SENSORS[self.vehicle]:
                self.logger.warning("No lopc.bin file for %s", mission)
            return
        self.logger.info("Processing file %s (%d bytes)", lopc_bin, file_size)
        lopc_processor = LOPC_Processor()
        lopc_processor.args = argparse.Namespace()
        lopc_processor.args.bin_fileName = lopc_bin
        lopc_processor.args.netCDF_fileName = Path(
            self.args.base_path,
            self.vehicle,
            MISSIONNETCDFS,
            mission,
            "lopc.nc",
        )
        lopc_processor.args.text_fileName = ""
        lopc_processor.args.trans_AIcrit = 0.4
        lopc_processor.args.LargeCopepod_AIcrit = 0.6
        lopc_processor.args.LargeCopepod_ESDmin = 1100.0
        lopc_processor.args.LargeCopepod_ESDmax = 1700.0
        lopc_processor.args.verbose = self.args.verbose
        lopc_processor.args.debugLevel = 0
        lopc_processor.args.force = self.args.clobber
        lopc_processor.args.noinput = self.args.noinput
        lopc_processor.logger.setLevel(self._log_levels[self.args.verbose])
        lopc_processor.logger.addHandler(self.log_handler)
        try:
            lopc_processor.main()
        except UnexpectedAreaOfCode as e:
            self.logger.error(e)  # noqa: TRY400
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
        cal_netcdf.calibration_dir = self.calibration_dir
        cal_netcdf.args.verbose = self.args.verbose
        cal_netcdf.logger.setLevel(self._log_levels[self.args.verbose])
        cal_netcdf.logger.addHandler(self.log_handler)
        cal_netcdf.commandline = self.commandline
        try:
            netcdf_dir = cal_netcdf.process_logs()
            cal_netcdf.write_netcdf(netcdf_dir)
        except (FileNotFoundError, EOFError) as e:
            cal_netcdf.logger.error("%s %s", mission, e)  # noqa: TRY400
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
        except (FileNotFoundError, EOFError) as e:
            align_netcdf.logger.error("%s %s", mission, e)  # noqa: TRY400
            error_message = f"{mission} {e}"
            raise InvalidCalFile(error_message) from e
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
        resamp.args.mf_width = self.args.mf_width
        resamp.args.flash_threshold = self.args.flash_threshold
        resamp.commandline = self.commandline
        resamp.args.verbose = self.args.verbose
        resamp.logger.setLevel(self._log_levels[self.args.verbose])
        resamp.logger.addHandler(self.log_handler)
        file_name = f"{resamp.args.auv_name}_{resamp.args.mission}_align.nc"
        nc_file = Path(
            self.args.base_path,
            resamp.args.auv_name,
            MISSIONNETCDFS,
            resamp.args.mission,
            file_name,
        )
        if self.args.flash_threshold and self.args.resample:
            self.logger.info(
                "Executing only resample step to produce netCDF file with flash_threshold = %s",
                f"{self.args.flash_threshold:.0e}",
            )
            dap_file_str = os.path.join(  # noqa: PTH118
                AUVCTD_OPENDAP_BASE.replace("opendap/", ""),
                "surveys",
                resamp.args.mission.split(".")[0],
                "netcdf",
                file_name,
            )
            self.logger.info("Copying file %s", dap_file_str)
            wget_path = shutil.which("wget")
            if not wget_path:
                error_message = "wget not found"
                raise LookupError(error_message)
            nc_file_str = str(nc_file)
            if not dap_file_str.startswith("http"):
                error_message = f"Invalid URL for dap_file: {dap_file_str}"
                raise ValueError(error_message)
            subprocess.run([wget_path, dap_file_str, "-O", nc_file_str], check=True)  # noqa: S603
        try:
            resamp.resample_mission(nc_file)
        except FileNotFoundError as e:
            self.logger.error("%s %s", mission, e)  # noqa: TRY400
        finally:
            resamp.logger.removeHandler(self.log_handler)

    def archive(self, mission: str, add_logger_handlers: bool = True) -> None:  # noqa: FBT001, FBT002
        arch = Archiver(add_logger_handlers)
        arch.args = argparse.Namespace()
        arch.args.auv_name = self.vehicle
        arch.args.mission = mission
        arch.commandline = self.commandline
        arch.args.create_products = self.args.create_products
        arch.args.archive_only_products = self.args.archive_only_products
        arch.args.clobber = self.args.clobber
        arch.args.resample = self.args.resample
        arch.args.flash_threshold = self.args.flash_threshold
        arch.args.verbose = self.args.verbose
        arch.logger.setLevel(self._log_levels[self.args.verbose])
        if add_logger_handlers:
            self.logger.info("Archiving steps for %s", mission)
            arch.logger.addHandler(self.log_handler)
        file_name_base = f"{arch.args.auv_name}_{arch.args.mission}"
        nc_file_base = Path(
            BASE_PATH,
            arch.args.auv_name,
            MISSIONNETCDFS,
            arch.args.mission,
            file_name_base,
        )
        self.logger.info("nc_file_base = %s, BASE_PATH = %s", nc_file_base, BASE_PATH)
        if str(BASE_PATH).startswith(("/home/runner/", "/root")):
            arch.logger.info(
                "Not archiving %s %s to AUVCTD as it's likely CI testing",
                arch.args.auv_name,
                arch.args.mission,
            )
        else:
            arch.copy_to_AUVTCD(nc_file_base, self.args.freq)
        arch.logger.removeHandler(self.log_handler)

    def create_products(self, mission: str) -> None:
        cp = CreateProducts()
        cp.args = argparse.Namespace()
        cp.args.auv_name = self.vehicle
        cp.args.mission = mission
        cp.args.local = self.args.local
        cp.args.start_esecs = None
        cp.args.verbose = self.args.verbose
        cp.logger.setLevel(self._log_levels[self.args.verbose])
        cp.logger.addHandler(self.log_handler)

        # cp.plot_biolume()
        # cp.plot_2column()
        if "dorado" in cp.args.auv_name.lower():
            cp.gulper_odv()
        cp.logger.removeHandler(self.log_handler)

    def email(self, mission: str) -> None:
        self.logger.info("Sending notification email for %s", mission)
        email = Emailer()
        email.args = argparse.Namespace()
        email.args.auv_name = self.vehicle
        email.args.mission = mission
        email.commandline = self.commandline
        email.args.clobber = self.args.clobber
        email.args.verbose = self.args.verbose
        email.logger.setLevel(self._log_levels[self.args.verbose])
        email.logger.addHandler(self.log_handler)

    def cleanup(self, mission: str) -> None:
        self.logger.info(
            "Removing %s files from %s and %s",
            mission,
            MISSIONNETCDFS,
            MISSIONLOGS,
        )
        try:
            shutil.rmtree(
                Path(self.args.base_path, self.vehicle, MISSIONLOGS, mission),
            )
            shutil.rmtree(
                Path(self.args.base_path, self.vehicle, MISSIONNETCDFS, mission),
            )
            self.logger.info("Done removing %s work files", mission)
        except FileNotFoundError as e:
            self.logger.info("File not found: %s", e)

    def process_mission(self, mission: str, src_dir: str = "") -> None:  # noqa: C901, PLR0912, PLR0915
        netcdfs_dir = Path(
            self.args.base_path,
            self.vehicle,
            MISSIONNETCDFS,
            mission,
        )
        if self.args.clobber and (
            self.args.noinput
            or input("Do you want to remove all work files? [y/N] ").lower() == "y"
        ):
            self.cleanup(mission)
        Path(netcdfs_dir).mkdir(parents=True, exist_ok=True)
        self.log_handler = logging.FileHandler(
            Path(netcdfs_dir, f"{self.vehicle}_{mission}_{LOG_NAME}"),
            mode="w+",
        )
        self.log_handler.setLevel(self._log_levels[self.args.verbose])
        self.log_handler.setFormatter(AUV_NetCDF._formatter)
        self.logger.info(
            "=====================================================================================================================",
        )
        self.logger.addHandler(self.log_handler)
        self.logger.info("commandline = %s", self.commandline)
        try:
            program = ""
            if self.vehicle.lower() == "dorado":
                program = dorado_info[mission]["program"]
                self.logger.info(
                    'dorado_info[mission]["comment"] = %s', dorado_info[mission]["comment"]
                )
            elif self.vehicle.lower() == "i2map":
                program = "i2map"
            if program == TEST:
                error_message = (
                    f"{TEST} program specified in dorado_info.py. Not processing {mission}"
                )
                raise TestMission(error_message)
            if program == FAILED:
                error_message = (
                    f"{FAILED} program specified in dorado_info.py. Not processing {mission}"
                )
                raise FailedMission(error_message)
            self.logger.info(
                "Processing %s mission %s by user %s on host %s",
                program,
                mission,
                getuser(),
                gethostname(),
            )
        except KeyError:
            error_message = f"{mission} not in dorado_info"
            raise MissingDoradoInfo(error_message) from None
        if self.args.download_process:
            self.download_process(mission, src_dir)
        elif self.args.calibrate:
            self.calibrate(mission)
        elif self.args.align:
            self.align(mission)
        elif self.args.resample:
            self.resample(mission)
        elif self.args.resample and self.args.archive:
            self.resample(mission)
            self.archive(mission, add_logger_handlers=False)
        elif self.args.create_products and self.args.archive:
            self.create_products(mission)
            self.archive(mission, add_logger_handlers=False)
        elif self.args.create_products:
            self.create_products(mission)
        elif self.args.archive:
            self.archive(mission)
        elif self.args.email_to:
            self.email(mission)
        elif self.args.cleanup:
            self.cleanup(mission)
        else:
            if not self.args.skip_download_process:
                self.download_process(mission, src_dir)
            self.calibrate(mission)
            self.align(mission)
            self.resample(mission)
            self.create_products(mission)
            # self.archive() is called in finally: blocks in process_missions()

    def process_mission_job(self, mission: str, src_dir: str = "") -> None:
        try:
            t_start = time.time()
            self.process_mission(mission, src_dir)
        except (
            InvalidCalFile,
            InvalidAlignFile,
            FileNotFoundError,
            EOFError,
            MissingDoradoInfo,
        ) as e:
            self.logger.error(repr(e))  # noqa: TRY400
            self.logger.error("Failed to process to completion: %s", mission)  # noqa: TRY400
        except (TestMission, FailedMission) as e:
            self.logger.info(str(e))
        finally:
            # Still need to archive the mission, especially the processing.log file
            self.archive(mission)
            if not self.args.no_cleanup:
                self.cleanup(mission)
            self.logger.info(
                "Mission %s took %.1f seconds to process",
                mission,
                time.time() - t_start,
            )
            if hasattr(self, "log_handler"):
                self.logger.removeHandler(self.log_handler)
        return f"[{os.getpid()}] {mission}: {time.time() - t_start:.1f} seconds"

    def process_mission_exception_wrapper(
        self,
        mission: str,
        src_dir: str = "",
    ) -> None:
        try:
            t_start = time.time()
            self.process_mission(mission, src_dir=src_dir)
        except (
            InvalidCalFile,
            InvalidAlignFile,
            FileNotFoundError,
            EOFError,
            MissingDoradoInfo,
        ) as e:
            self.logger.error(repr(e))  # noqa: TRY400
            self.logger.error("Failed to process to completion: %s", mission)  # noqa: TRY400
        except (TestMission, FailedMission) as e:
            self.logger.info(str(e))
        finally:
            if hasattr(self, "log_handler"):
                # If no log_handler then process_mission() failed, likely due to missing mount
                # Always archive the mission, especially the processing.log file
                if self.vehicle == "Dorado389" and mission == "2011.256.02":
                    self.logger.info(
                        "Not archiving %s %s as it's likely CI testing",
                        self.vehicle,
                        mission,
                    )
                else:
                    self.archive(mission)
                if not self.args.no_cleanup:
                    self.cleanup(mission)
                self.logger.info(
                    "Mission %s took %.1f seconds to process",
                    mission,
                    time.time() - t_start,
                )
                self.logger.removeHandler(self.log_handler)

    def process_missions(self, start_year: int) -> None:
        if not self.args.start_year:
            self.args.start_year = start_year
        if self.args.mission:
            # mission is string like: 2021.062.01 and is assumed to exist
            self.process_mission_exception_wrapper(
                self.args.mission,
                src_dir=self.get_mission_dir(self.args.mission),
            )
        elif self.args.start_year and self.args.end_year:
            missions = self.mission_list(
                start_year=self.args.start_year,
                end_year=self.args.end_year,
            )
            if self.args.start_year == self.args.end_year:
                # Subselect missions by year day, has effect if --start_yd & --end_yd
                # are specified and --start_year & --end_year are the same
                missions = {
                    mission: missions[mission]
                    for mission in missions
                    if (
                        int(mission.split(".")[1]) >= self.args.start_yd
                        and int(mission.split(".")[1]) <= self.args.end_yd
                    )
                }

            # https://pythonspeed.com/articles/python-multiprocessing/ - Swimming with sharks!
            ncores = self.args.num_cores if self.args.num_cores else multiprocessing.cpu_count()
            missions = dict(sorted(missions.items()))
            if ncores > 1:
                self.logger.info(
                    "Using %d cores for %d missions",
                    ncores,
                    len(missions),
                )
                with multiprocessing.get_context("spawn").Pool(
                    processes=ncores,
                ) as pool:
                    overall_start = time.time()
                    # TODO: Fix logger to include process.py messages from subprocesses
                    results = pool.starmap(
                        self.process_mission_job,
                        [[mission, missions[mission]] for mission in missions],
                    )
                    self.logger.info(
                        "Finished processing %d missions in %.1f seconds",
                        len(missions),
                        time.time() - overall_start,
                    )
                    self.logger.info("Results:")
                    for result in results:
                        self.logger.info(result)
            else:
                # Don't use multiprocessing - we get all logger messages with --num_cores 1
                for mission in missions:
                    self.process_mission_exception_wrapper(
                        mission,
                        src_dir=self.get_mission_dir(mission),
                    )

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
            "--local",
            action="store_true",
            help="Specify if files are local in the MISSION directory",
        )
        parser.add_argument(
            "--clobber",
            action="store_true",
            help="Use with --noinput to overwrite existing downloaded"
            " log files and to remove existing netCDF files before"
            " rsyncing to the AUVCTD directory",
        )
        parser.add_argument(
            "--noinput",
            action="store_true",
            help="Execute without asking for a response, e.g.  to not ask to re-download file",
        )
        parser.add_argument(
            "--noreprocess",
            action="store_true",
            help="Use with --noinput to not re-process existing downloaded log files",
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
            default=datetime.now().astimezone(timezone.utc).year,
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
            "--create_products",
            action="store_true",
            help="Create products from the resampled netCDF file(s)",
        )
        parser.add_argument(
            "--email_to",
            action="store",
            help=(
                f"Send email to this address when processing is complete, use "
                f"{NOTIFICATION_EMAIL} for everyone who cares"
            ),
        )
        parser.add_argument(
            "--cleanup",
            action="store_true",
            help=(
                f"Remove {MISSIONLOGS} and {MISSIONNETCDFS} files following "
                "archive of processed mission"
            ),
        )
        parser.add_argument(
            "--no_cleanup",
            action="store_true",
            help="Do not perform cleanup after processing mission",
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
            "--mf_width",
            action="store",
            default=MF_WIDTH,
            type=int,
            help="Median filter width",
        )
        parser.add_argument(
            "--use_portal",
            action="store_true",
            help=(
                "Download data using portal (much faster than copy over"
                " remote connection), otherwise copy from mount point"
            ),
        )
        (
            parser.add_argument(
                "--skip_download_process",
                action="store_true",
                help="Skip download_process() step - start with original .nc files",
            ),
        )
        (
            parser.add_argument(
                "--archive_only_products",
                action="store_true",
                help="Rsync to AUVCTD directory only the products, not the netCDF files",
            ),
        )
        parser.add_argument(
            "--flash_threshold",
            action="store",
            type=float,
            help=(
                f"Override the default flash_threshold value of {FLASH_THRESHOLD:.0E} "
                "and append to the netCDF file name"
            ),
        )
        parser.add_argument(
            "--num_cores",
            action="store",
            type=int,
            help="Number of core processors to use",
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

        # Append year to vehicle_dir if --start_year and --end_year identical
        if self.args.start_year == self.args.end_year:
            self.logger.debug(
                "start_yd and end_yd will be honored as start_year and end_year are identical",
            )
        # Warn that --start_yd and --end_yd will be ignored
        elif self.args.start_yd != 1 or self.args.end_yd != 366:  # noqa: PLR2004
            self.logger.warning(
                "start_yd and end_yd will be ignored as start_year and end_year are different",
            )

        self.logger.setLevel(self._log_levels[self.args.verbose])
        self.commandline = " ".join(sys.argv)


if __name__ == "__main__":
    VEHICLE = "i2map"
    VEHICLE_DIR = "/Volumes/M3/master/i2MAP"
    CALIBRATION_DIR = "/Volumes/DMO/MDUC_CORE_CTD_200103/Calibration Files"
    MOUNT_DIR = "smb://titan.shore.mbari.org/M3"

    # Initialize for i2MAP processing, meant to be subclassed for other vehicles
    proc = Processor(VEHICLE, VEHICLE_DIR, MOUNT_DIR, CALIBRATION_DIR)
    proc.process_command_line()
    proc.process_missions()
