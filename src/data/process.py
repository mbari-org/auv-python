#!/usr/bin/env python
"""
Base module for data processing for Dorado class and LRAUV class data.

Run the data through standard science data processing to calibrated,
aligned, and resampled netCDF files.  Use a standard set of processing options;
more flexibility is available via the inndividual processing modules.

The desire is to reuse as much code as possible between Dorado class
and LRAUV class data processing. The initial steps of creating the _cal.nc
files differ because Dorado class data are raw binary log files that need to be
processed to _nc files, while LRAUV class data are NetCDF4 log files that
already contain much of the necessary information. The initial step for Dorado
class data are: download_process and calibrate, while for LRAUV class data
are: extract and combine. After that, the processing steps are similar with
the data in a local directory organized similarly to their institutional
archives.

Dorado class data processing:
=============================

Limit processing to specific steps by providing arugments:
    --download_process (logs2netcdf.py & lopcToNetCDF.py)
    --calibrate
    --align
    --resample
    --archive
    --create_products
    --email_to
    --cleanup
If none provided then perform all steps.

Uses command line arguments from logs2netcdf.py and calibrate.py.


LRAUV class data processing:
============================

Limit processing to specific steps by providing arugments:
    --extract (nc42netcdfs.py)
    --combine
    --align
    --resample
    --archive
    --create_products
    --email_to
    --cleanup
If none provided then perform all steps.
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
from datetime import UTC, datetime
from getpass import getuser
from pathlib import Path
from socket import gethostname

from align import Align_NetCDF, InvalidCalFile, InvalidCombinedFile
from archive import LOG_NAME, Archiver
from calibrate import EXPECTED_SENSORS, Calibrate_NetCDF
from combine import Combine_NetCDF
from create_products import CreateProducts
from dorado_info import FAILED, TEST, dorado_info
from emailer import NOTIFICATION_EMAIL, Emailer
from logs2netcdfs import BASE_PATH, MISSIONLOGS, MISSIONNETCDFS, AUV_NetCDF
from lopcToNetCDF import LOPC_Processor, UnexpectedAreaOfCode
from nc42netcdfs import BASE_LRAUV_PATH, BASE_LRAUV_WEB, Extract
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


def log_file_processor(func):
    """Decorator to handle LRAUV log_file processing exceptions and cleanup."""

    def wrapper(self, log_file: str):
        t_start = time.time()
        try:
            return func(self, log_file)
        except (TestMission, FailedMission, EOFError) as e:
            self.logger.info(str(e))
        finally:
            if hasattr(self, "log_handler"):
                # Cleanup and archiving logic
                self.archive(mission=None, log_file=log_file)
                if not self.config.get("no_cleanup"):
                    self.cleanup(log_file=log_file)
                self.logger.info(
                    "log_file %s took %.1f seconds to process", log_file, time.time() - t_start
                )
                self.logger.removeHandler(self.log_handler)

    return wrapper


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

    def __init__(self, auv_name, vehicle_dir, mount_dir, calibration_dir, config=None) -> None:  # noqa: PLR0913
        # Variables to be set by subclasses, e.g.:
        # auv_name = "i2map"
        # vehicle_dir = "/Volumes/M3/master/i2MAP"
        # mount_dir = "smb://thalassa.shore.mbari.org/M3"
        self.auv_name = auv_name
        self.vehicle_dir = vehicle_dir
        self.mount_dir = mount_dir
        self.calibration_dir = calibration_dir
        self.config = config or {}

    # Configuration schema with defaults - shared between from_args and common_config
    _CONFIG_SCHEMA = {
        # Core configuration
        "base_path": BASE_PATH,
        "local": False,
        "noinput": False,
        "clobber": False,
        "noreprocess": False,
        "use_portal": False,
        "add_seconds": None,
        "verbose": 0,
        "freq": FREQ,
        "mf_width": MF_WIDTH,
        "flash_threshold": None,
        "log_file": None,
        # Processing control
        "download_process": False,
        "calibrate": False,
        "align": False,
        "resample": False,
        "archive": False,
        "create_products": False,
        "email_to": None,
        "cleanup": False,
        "no_cleanup": False,
        "skip_download_process": False,
        "archive_only_products": False,
        "num_cores": None,
        # Filtering/processing params (only used in from_args, not common_config)
        "start_year": None,
        "end_year": None,
        "start_yd": None,
        "end_yd": None,
        "last_n_days": None,
        "mission": None,
        "start": None,  # LRAUV datetime filtering
        "end": None,  # LRAUV datetime filtering
        "auv_name": None,  # LRAUV AUV name filtering
    }

    # Subset of config schema that should be passed to child processes
    _CHILD_CONFIG_KEYS = {
        "base_path",
        "local",
        "noinput",
        "clobber",
        "noreprocess",
        "use_portal",
        "add_seconds",
        "verbose",
        "freq",
        "mf_width",
        "flash_threshold",
        "log_file",
        "download_process",
        "calibrate",
        "align",
        "resample",
        "archive",
        "create_products",
        "email_to",
        "cleanup",
        "no_cleanup",
        "skip_download_process",
        "archive_only_products",
        "num_cores",
    }

    @property
    def common_config(self):
        """Get common configuration used by all child processes"""
        return {
            key: self.config.get(key, self._CONFIG_SCHEMA[key]) for key in self._CHILD_CONFIG_KEYS
        }

    def _create_child_namespace(self, **overrides):
        """Create args namespace for child processes with config overrides"""
        config = {**self.common_config, **overrides}

        namespace = argparse.Namespace()
        for key, value in config.items():
            setattr(namespace, key, value)
        return namespace

    @classmethod
    def from_args(cls, auv_name, vehicle_dir, mount_dir, calibration_dir, args):  # noqa: PLR0913
        """Factory method to create Processor from argparse namespace"""
        config = {}
        for key, default_value in cls._CONFIG_SCHEMA.items():
            # Handle special cases for args that might not exist or have different names
            if key == "add_seconds":
                config[key] = getattr(args, "add_seconds", default_value)
            else:
                config[key] = getattr(args, key, default_value)

        instance = cls(auv_name, vehicle_dir, mount_dir, calibration_dir, config)
        instance.args = args  # Keep reference for compatibility
        instance.commandline = " ".join(sys.argv)  # Set commandline attribute
        instance.logger.setLevel(instance._log_levels[args.verbose])  # Set logger level
        return instance

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
        if self.config.get("last_n_days"):
            self.logger.info(
                "Will be looking back %d days for new missions...", self.config["last_n_days"]
            )
            find_cmd += f" -mtime -{self.config['last_n_days']}"
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

    def _parse_datetime_string(self, datetime_str: str) -> datetime | None:
        """Parse datetime string in YYYYMMDDTHHMMSS format."""
        try:
            return datetime.strptime(datetime_str, "%Y%m%dT%H%M%S").replace(tzinfo=UTC)
        except ValueError:
            return None

    def _normalize_datetime_dir(self, dir_datetime_str: str) -> str:
        """Normalize datetime directory name to YYYYMMDDTHHMMSS format."""
        if "T" not in dir_datetime_str:
            return ""

        PARTIAL_DATETIME_LEN = 13  # YYYYMMDDTHHNN format
        SHORT_DATETIME_LEN = 11  # YYYYMMDDTHH format

        if len(dir_datetime_str) == PARTIAL_DATETIME_LEN:
            return dir_datetime_str + "00"  # Add seconds
        if len(dir_datetime_str) == SHORT_DATETIME_LEN:
            return dir_datetime_str + "0000"  # Add minutes and seconds
        return dir_datetime_str

    def _find_log_files_in_datetime_dir(
        self, datetime_dir: Path, start_dt: datetime, end_dt: datetime
    ) -> list:
        """Find log files in a datetime directory if it's in range."""
        import re

        log_files = []

        # Normalize and parse directory datetime
        normalized_str = self._normalize_datetime_dir(datetime_dir.name)
        if not normalized_str:
            return log_files

        dir_dt = self._parse_datetime_string(normalized_str)
        if not dir_dt:
            return log_files

        # Check if directory datetime is in range
        if start_dt <= dir_dt <= end_dt:
            # Look for main log file (*.nc4 file) but exclude _combined.nc4 and _align.nc4
            # Pattern matches: YYYYMMDDHHMMSS_YYYYMMDDHHMMSS.nc4
            # Example: 202506072219_202506072336.nc4
            log_pattern = re.compile(r"^\d{12}_\d{12}\.nc4$")
            for nc4_file in datetime_dir.glob("*.nc4"):
                if log_pattern.match(nc4_file.name):
                    relative_path = str(nc4_file.relative_to(Path(self.vehicle_dir)))
                    log_files.append(relative_path)
                    self.logger.debug("Found log file: %s", relative_path)

        return log_files

    def _should_process_auv_dir(self, auv_dir: Path, auv_name: str) -> bool:
        """Check if an AUV directory should be processed based on auv_name filter."""
        if auv_name and auv_dir.name.lower() != auv_name.lower():
            return False

        missionlogs_dir = auv_dir / "missionlogs"
        return missionlogs_dir.exists()

    def log_file_list(self, start_datetime: str, end_datetime: str, auv_name: str = None) -> list:
        """Return a list of LRAUV log files within the specified datetime range.

        Args:
            start_datetime: Start datetime in YYYYMMDDTHHMMSS format
            end_datetime: End datetime in YYYYMMDDTHHMMSS format
            auv_name: Optional AUV name to filter results (e.g., 'brizo', 'ahi')

        Returns:
            Sorted list of log file paths relative to base_path
        """
        log_files = []
        vehicle_dir = Path(self.vehicle_dir).resolve()

        # Parse datetime strings
        start_dt = self._parse_datetime_string(start_datetime)
        end_dt = self._parse_datetime_string(end_datetime)

        if not start_dt or not end_dt:
            self.logger.exception("Invalid datetime format. Use YYYYMMDDTHHMMSS")
            return log_files

        if auv_name:
            self.logger.info(
                "Finding log files from %s to %s for AUV: %s",
                start_datetime,
                end_datetime,
                auv_name,
            )
        else:
            self.logger.info(
                "Finding log files from %s to %s for all AUVs",
                start_datetime,
                end_datetime,
            )

        # Search through each AUV directory
        for auv_dir in vehicle_dir.glob("*/"):
            if not self._should_process_auv_dir(auv_dir, auv_name):
                continue

            missionlogs_dir = auv_dir / "missionlogs"

            # Search through years
            for year_dir in sorted(missionlogs_dir.glob("*/")):
                try:
                    year = int(year_dir.name)
                    # Skip if year is clearly outside our range
                    if year < start_dt.year or year > end_dt.year:
                        continue
                except ValueError:
                    continue

                # Search through date range directories and datetime directories
                for date_range_dir in year_dir.glob("*/"):
                    for datetime_dir in date_range_dir.glob("*/"):
                        files_found = self._find_log_files_in_datetime_dir(
                            datetime_dir, start_dt, end_dt
                        )
                        log_files.extend(files_found)

        self.logger.info("Found %d log files in date range", len(log_files))
        return sorted(log_files)

    def get_mission_dir(self, mission: str) -> str:
        """Return the mission directory."""
        if not Path(self.vehicle_dir).exists():
            self.logger.error("%s does not exist.", self.vehicle_dir)
            self.logger.info("Is %s mounted?", self.mount_dir)
            sys.exit(1)
        if self.auv_name.lower() == "dorado" or self.auv_name == "Dorado389":
            if self.config.get("local"):
                path = Path(self.vehicle_dir, mission)
            else:
                year = mission.split(".")[0]
                yearyd = "".join(mission.split(".")[:2])
                path = Path(self.vehicle_dir, year, yearyd, mission)
        elif self.auv_name.lower() == "i2map":
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
        elif self.auv_name == "Dorado389":
            # The Dorado389 auv_name is a special case used for testing locally and in CI
            path = self.vehicle_dir
        if not Path(path).exists():
            self.logger.error("%s does not exist.", path)
            error_message = f"{path} does not exist."
            raise FileNotFoundError(error_message)
        return path

    def download_process(self, mission: str, src_dir: str) -> None:
        self.logger.info("Download and processing steps for %s", mission)
        auv_netcdf = AUV_NetCDF(
            auv_name=self.auv_name,
            mission=mission,
            base_path=str(self.config["base_path"]),
            local=self.config["local"],
            noinput=self.config["noinput"],
            clobber=self.config["clobber"],
            noreprocess=self.config["noreprocess"],
            use_portal=self.config["use_portal"],
            add_seconds=self.config["add_seconds"],
            verbose=self.config["verbose"],
            commandline=self.commandline,
        )
        auv_netcdf.set_portal()
        auv_netcdf.logger.setLevel(self._log_levels[self.config["verbose"]])
        auv_netcdf.logger.addHandler(self.log_handler)
        auv_netcdf.download_process_logs(src_dir=src_dir)
        auv_netcdf.logger.removeHandler(self.log_handler)

        # Run lopcToNetCDF.py - mimic log message from logs2netcdfs.py
        lopc_bin = Path(
            self.config["base_path"],
            self.auv_name,
            MISSIONLOGS,
            mission,
            "lopc.bin",
        )
        try:
            file_size = Path(lopc_bin).stat().st_size
        except FileNotFoundError:
            if "lopc" in EXPECTED_SENSORS[self.auv_name]:
                self.logger.warning("No lopc.bin file for %s", mission)
            return
        self.logger.info("Processing file %s (%d bytes)", lopc_bin, file_size)
        lopc_processor = LOPC_Processor()
        lopc_processor.args = self._create_child_namespace(
            bin_fileName=lopc_bin,
            netCDF_fileName=os.path.join(  # noqa: PTH118 This is an arg, keep it a string
                self.config["base_path"],
                self.auv_name,
                MISSIONNETCDFS,
                mission,
                "lopc.nc",
            ),
            text_fileName="",
            trans_AIcrit=0.4,
            LargeCopepod_AIcrit=0.6,
            LargeCopepod_ESDmin=1100.0,
            LargeCopepod_ESDmax=1700.0,
            debugLevel=0,
            force=self.config["clobber"],
        )
        lopc_processor.logger.setLevel(self._log_levels[self.config["verbose"]])
        lopc_processor.logger.addHandler(self.log_handler)
        try:
            lopc_processor.main()
        except UnexpectedAreaOfCode as e:
            self.logger.error(e)  # noqa: TRY400
        lopc_processor.logger.removeHandler(self.log_handler)

    def calibrate(self, mission: str) -> None:
        self.logger.info("Calibration steps for %s", mission)
        cal_netcdf = Calibrate_NetCDF(
            auv_name=self.auv_name,
            mission=mission,
            base_path=self.config["base_path"],
            calibration_dir=self.calibration_dir,
            plot=None,
            verbose=self.config["verbose"],
            commandline=self.commandline,
            local=self.config["local"],
            noinput=self.config["noinput"],
            clobber=self.config["clobber"],
            noreprocess=self.config["noreprocess"],
        )
        cal_netcdf.logger.addHandler(self.log_handler)
        try:
            netcdf_dir = cal_netcdf.process_logs()
            cal_netcdf.write_netcdf(netcdf_dir)
        except (FileNotFoundError, EOFError) as e:
            cal_netcdf.logger.error("%s %s", mission, e)  # noqa: TRY400
        cal_netcdf.logger.removeHandler(self.log_handler)

    def align(self, mission: str = "", log_file: str = "") -> None:
        self.logger.info("Alignment steps for %s", mission or log_file)
        align_netcdf = Align_NetCDF(
            auv_name=self.auv_name,
            mission=mission,
            base_path=self.config["base_path"],
            log_file=log_file,
            plot=None,
            verbose=self.config["verbose"],
            commandline=self.commandline,
        )
        align_netcdf.logger.addHandler(self.log_handler)
        try:
            if log_file:
                netcdf_dir = align_netcdf.process_combined()
                align_netcdf.write_combined_netcdf(netcdf_dir)
            else:
                netcdf_dir = align_netcdf.process_cal()
                align_netcdf.write_combined_netcdf(netcdf_dir)
        except (FileNotFoundError, EOFError) as e:
            align_netcdf.logger.error("%s %s", mission, e)  # noqa: TRY400
            error_message = f"{mission} {e}"
            raise InvalidCalFile(error_message) from e
        finally:
            align_netcdf.logger.removeHandler(self.log_handler)

    def resample(self, mission: str = "", log_file: str = "") -> None:
        self.logger.info("Resampling steps for %s", mission)
        resamp = Resampler(
            auv_name=self.auv_name,
            mission=mission,
            log_file=log_file,
            freq=self.config["freq"],
            mf_width=self.config["mf_width"],
            flash_threshold=self.config["flash_threshold"],
            verbose=self.config["verbose"],
            plot=None,
            commandline=self.commandline,
        )
        resamp.logger.setLevel(self._log_levels[self.config["verbose"]])
        resamp.logger.addHandler(self.log_handler)
        file_name = f"{resamp.auv_name}_{resamp.mission}_align.nc4"
        if resamp.log_file:
            netcdfs_dir = Path(BASE_LRAUV_PATH, Path(resamp.log_file).parent)
            nc_file = Path(netcdfs_dir, f"{Path(resamp.log_file).stem}_align.nc4")
        else:
            nc_file = Path(
                self.config["base_path"],
                resamp.auv_name,
                MISSIONNETCDFS,
                resamp.mission,
                file_name,
            )
        if self.config["flash_threshold"] and self.config["resample"]:
            self.logger.info(
                "Executing only resample step to produce netCDF file with flash_threshold = %s",
                f"{self.config['flash_threshold']:.0e}",
            )
            dap_file_str = os.path.join(  # noqa: PTH118
                AUVCTD_OPENDAP_BASE.replace("opendap/", ""),
                "surveys",
                resamp.mission.split(".")[0],
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
        except (FileNotFoundError, InvalidAlignFile) as e:
            self.logger.error("%s %s", nc_file, e)  # noqa: TRY400
        finally:
            resamp.logger.removeHandler(self.log_handler)

    def archive(
        self,
        mission: str = None,
        log_file: Path = None,
        add_logger_handlers: bool = True,  # noqa: FBT001, FBT002
    ) -> None:
        """Archiving steps for mission or log_file.

        If mission is provided, archive the processed data for Dorado class vehicles.
        If log_file is provided, archive the processed data for LRAUV class vehicles."""
        arch = Archiver(
            add_handlers=add_logger_handlers,
            auv_name=self.auv_name,
            mission=mission,
            clobber=self.config["clobber"],
            resample=self.config["resample"],
            flash_threshold=self.config["flash_threshold"],
            archive_only_products=self.config["archive_only_products"],
            create_products=self.config["create_products"],
            verbose=self.config["verbose"],
            commandline=self.commandline,
        )
        arch.mount_dir = self.mount_dir
        arch.logger.setLevel(self._log_levels[self.config["verbose"]])
        if add_logger_handlers:
            arch.logger.addHandler(self.log_handler)
        if mission:
            # Dorado class vehicle archiving
            self.logger.info("Archiving steps for %s", mission)
            file_name_base = f"{arch.auv_name}_{arch.mission}"
            nc_file_base = Path(
                BASE_PATH,
                arch.auv_name,
                MISSIONNETCDFS,
                arch.mission,
                file_name_base,
            )
            self.logger.info("nc_file_base = %s, BASE_PATH = %s", nc_file_base, BASE_PATH)
            if str(BASE_PATH).startswith(("/home/runner/", "/root")):
                arch.logger.info(
                    "Not archiving %s %s to AUVCTD as it's likely CI testing",
                    arch.auv_name,
                    arch.mission,
                )
            else:
                arch.copy_to_AUVTCD(nc_file_base, self.config["freq"])
        elif log_file:
            # LRAUV class vehicle archiving
            self.logger.info("Archiving steps for %s", log_file)
            arch.copy_to_LRAUV(log_file, freq=self.config["freq"])
        else:
            arch.logger.error("Either mission or log_file must be provided for archiving.")
        arch.logger.removeHandler(self.log_handler)

    def create_products(self, mission: str) -> None:
        cp = CreateProducts(
            auv_name=self.auv_name,
            mission=mission,
            base_path=str(self.config["base_path"]),
            start_esecs=None,
            local=self.config["local"],
            verbose=self.config["verbose"],
            commandline=self.commandline,
        )
        cp.logger.setLevel(self._log_levels[self.config["verbose"]])
        cp.logger.addHandler(self.log_handler)

        # cp.plot_biolume()
        # cp.plot_2column()
        if "dorado" in cp.auv_name.lower():
            cp.gulper_odv()
        cp.logger.removeHandler(self.log_handler)

    def email(self, mission: str) -> None:
        self.logger.info("Sending notification email for %s", mission)
        email = Emailer()
        email.args = self._create_child_namespace(auv_name=self.auv_name, mission=mission)
        email.commandline = self.commandline
        email.logger.setLevel(self._log_levels[self.config["verbose"]])
        email.logger.addHandler(self.log_handler)

    def _remove_empty_parents(self, path: Path, stop_at: Path) -> None:
        """Remove empty parent directories up to stop_at path."""
        parent = path.parent
        while parent != stop_at:
            try:
                ds_store = parent / ".DS_Store"
                if ds_store.exists():
                    ds_store.unlink()  # Remove .DS_Store file so that the directory is empty
                if parent.exists() and not any(parent.iterdir()):
                    self.logger.debug("Removing empty directory: %s", parent)
                    parent.rmdir()
                    parent = parent.parent
                else:
                    break
            except OSError as e:
                self.logger.debug("Could not remove directory %s: %s", parent, e)
                break

    def cleanup(self, mission: str = None, log_file: str = None) -> None:
        if mission:
            self.logger.info(
                "Removing mission %s files from %s and %s",
                mission,
                MISSIONNETCDFS,
                MISSIONLOGS,
            )
            try:
                shutil.rmtree(
                    Path(self.config["base_path"], self.auv_name, MISSIONLOGS, mission),
                )
                shutil.rmtree(
                    Path(self.config["base_path"], self.auv_name, MISSIONNETCDFS, mission),
                )
                self.logger.info("Done removing %s work files", mission)
            except FileNotFoundError as e:
                self.logger.info("File not found: %s", e)
        elif log_file:
            self.logger.info("Removing work files from local directory for %s", log_file)
            try:
                log_path = Path(BASE_LRAUV_PATH, log_file).resolve()
                for item in log_path.parent.iterdir():
                    if item.is_file():
                        self.logger.debug("Removing file %s", item)
                        item.unlink()
                    elif item.is_dir():
                        self.logger.debug("Removing directory %s", item)
                        shutil.rmtree(item)
                self._remove_empty_parents(log_path, Path(BASE_LRAUV_PATH))
                self.logger.info("Done removing work files for %s", log_file)
            except FileNotFoundError as e:
                self.logger.info("File not found: %s", e)
        else:
            self.logger.error("Either mission or log_file must be provided for cleanup.")

    def process_mission(self, mission: str, src_dir: str = "") -> None:  # noqa: C901, PLR0912, PLR0915
        netcdfs_dir = Path(
            self.config["base_path"],
            self.auv_name,
            MISSIONNETCDFS,
            mission,
        )
        if self.config["clobber"] and (
            self.config["noinput"]
            or input("Do you want to remove all work files? [y/N] ").lower() == "y"
        ):
            self.cleanup(mission)
        Path(netcdfs_dir).mkdir(parents=True, exist_ok=True)
        self.log_handler = logging.FileHandler(
            Path(netcdfs_dir, f"{self.auv_name}_{mission}_{LOG_NAME}"),
            mode="w+",
        )
        self.log_handler.setLevel(self._log_levels[self.config["verbose"]])
        self.log_handler.setFormatter(AUV_NetCDF._formatter)
        self.logger.info(
            "=====================================================================================================================",
        )
        self.logger.addHandler(self.log_handler)
        self.logger.info("commandline = %s", self.commandline)
        try:
            program = ""
            if self.auv_name.lower() == "dorado":
                program = dorado_info[mission]["program"]
                self.logger.info(
                    'dorado_info[mission]["comment"] = %s', dorado_info[mission]["comment"]
                )
            elif self.auv_name.lower() == "i2map":
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
            # Try to get actual host name, fall back to container name
            actual_hostname = os.getenv("HOST_NAME", gethostname())
            self.logger.info(
                "Processing %s mission %s by user %s on host %s",
                program,
                mission,
                getuser(),
                actual_hostname,
            )
        except KeyError:
            error_message = f"{mission} not in dorado_info"
            raise MissingDoradoInfo(error_message) from None
        if self.config["download_process"]:
            self.download_process(mission, src_dir)
        elif self.config["calibrate"]:
            self.calibrate(mission)
        elif self.config["align"]:
            self.align(mission)
        elif self.config["resample"]:
            self.resample(mission)
        elif self.config["resample"] and self.config["archive"]:
            self.resample(mission)
            self.archive(mission, add_logger_handlers=False)
        elif self.config["create_products"] and self.config["archive"]:
            self.create_products(mission)
            self.archive(mission, add_logger_handlers=False)
        elif self.config["create_products"]:
            self.create_products(mission)
        elif self.config["archive"]:
            self.archive(mission)
        elif self.config["email_to"]:
            self.email(mission)
        elif self.config["cleanup"]:
            self.cleanup(mission)
        else:
            if not self.config["skip_download_process"]:
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
            if self.config["download_process"]:
                self.logger.info("Not archiving %s as --download_process is set", mission)
            else:
                # Still need to archive the mission, especially the processing.log file
                self.archive(mission)
            if not self.config["no_cleanup"]:
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
        ):
            self.logger.exception("An exception occurred")
            self.logger.error("Failed to process to completion: %s", mission)  # noqa: TRY400
        except (TestMission, FailedMission) as e:
            self.logger.info(str(e))
        finally:
            if hasattr(self, "log_handler"):
                # If no log_handler then process_mission() failed, likely due to missing mount
                # Always archive the mission, especially the processing.log file
                if self.auv_name == "Dorado389" and mission == "2011.256.02":
                    self.logger.info(
                        "Not archiving %s %s as it's likely CI testing",
                        self.auv_name,
                        mission,
                    )
                if self.config["download_process"]:
                    self.logger.info("Not archiving %s as --download_process is set", mission)
                else:
                    self.archive(mission)
                if not self.config["no_cleanup"]:
                    self.cleanup(mission)
                self.logger.info(
                    "Mission %s took %.1f seconds to process",
                    mission,
                    time.time() - t_start,
                )
                self.logger.removeHandler(self.log_handler)

    def process_missions(self, start_year: int = None) -> None:
        if not self.config.get("start_year"):
            self.config["start_year"] = start_year
        if self.config.get("mission"):
            # mission is string like: 2021.062.01 and is assumed to exist
            self.process_mission_exception_wrapper(
                self.config["mission"],
                src_dir=self.get_mission_dir(self.config["mission"]),
            )
        elif self.config.get("start_year") and self.config.get("end_year"):
            missions = self.mission_list(
                start_year=self.config["start_year"],
                end_year=self.config["end_year"],
            )
            if self.config["start_year"] == self.config["end_year"]:
                # Subselect missions by year day, has effect if --start_yd & --end_yd
                # are specified and --start_year & --end_year are the same
                missions = {
                    mission: missions[mission]
                    for mission in missions
                    if (
                        int(mission.split(".")[1]) >= self.config["start_yd"]
                        and int(mission.split(".")[1]) <= self.config["end_yd"]
                    )
                }

            # https://pythonspeed.com/articles/python-multiprocessing/ - Swimming with sharks!
            ncores = self.config.get("num_cores") or multiprocessing.cpu_count()
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

    # ====================== LRAUV data specific processing ======================
    # The command line arument --log_file distinguishes LRAUV data from Dorado data.
    # Dorado class data uses --mission instead.  Also, start and end specifications
    # are different for LRAUV data: --start and --end instead of --start_year,
    # --start_yd, --end_year, and --end_yd. If --start and --end are spcified then
    # --auv_name is required to look up the individual log files to process.

    def extract(self, log_file: str) -> None:
        self.logger.info("Extracting log file: %s", log_file)
        extract = Extract(
            log_file=log_file,
            plot_time=False,
            filter_monotonic_time=True,
            verbose=self.config["verbose"],
            commandline=self.commandline,
        )
        extract.logger.setLevel(self._log_levels[self.config["verbose"]])
        extract.logger.addHandler(self.log_handler)

        url = os.path.join(BASE_LRAUV_WEB, log_file)  # noqa: PTH118
        output_dir = Path(BASE_LRAUV_PATH, Path(log_file).parent)
        extract.logger.info("Downloading %s", url)
        input_file = extract.download_with_pooch(url, output_dir)
        return extract.extract_groups_to_files_netcdf4(input_file)

    def combine(self, log_file: str) -> None:
        self.logger.info("Combining netCDF files for log file: %s", log_file)
        self.logger.info(
            "Equivalent to the calibrate step for Dorado class vehicles. "
            "Adds nudge positions and more layers of quality control."
        )
        combine = Combine_NetCDF(
            log_file=log_file,
            verbose=self.config["verbose"],
            plot=None,
            commandline=self.commandline,
        )
        combine.logger.setLevel(self._log_levels[self.config["verbose"]])
        combine.logger.addHandler(self.log_handler)

        combine.combine_groups()
        combine.write_netcdf()

    @log_file_processor
    def process_log_file(self, log_file: str) -> None:
        netcdfs_dir = Path(BASE_LRAUV_PATH, Path(log_file).parent)
        Path(netcdfs_dir).mkdir(parents=True, exist_ok=True)
        self.log_handler = logging.FileHandler(
            Path(netcdfs_dir, f"{Path(log_file).stem}_processing.log"), mode="w+"
        )
        self.log_handler.setLevel(self._log_levels[self.config["verbose"]])
        self.log_handler.setFormatter(AUV_NetCDF._formatter)
        self.logger.info(
            "=====================================================================================================================",
        )
        self.logger.addHandler(self.log_handler)
        self.logger.info("commandline = %s", self.commandline)
        # Try to get actual host name, fall back to container name
        actual_hostname = os.getenv("HOST_NAME", gethostname())
        self.logger.info(
            "Processing log_file %s by user %s on host %s",
            log_file,
            getuser(),
            actual_hostname,
        )

        netcdfs_dir = self.extract(log_file)
        self.combine(log_file=log_file)
        self.align(log_file=log_file)
        self.resample(log_file=log_file)
        # self.create_products(log_file)
        self.logger.info("Finished processing log file: %s", log_file)

    def process_log_files(self) -> None:
        if self.config.get("log_file"):
            # log_file is string like:
            # brizo/missionlogs/2025/20250909_20250915/20250914T080941/202509140809_202509150109.nc4
            self.auv_name = self.config["log_file"].split("/")[0].lower()
            self.process_log_file(self.config["log_file"])
        elif self.config.get("start") and self.config.get("end"):
            # Process multiple log files within datetime range
            log_files = self.log_file_list(
                self.config["start"], self.config["end"], self.config.get("auv_name")
            )
            if not log_files:
                self.logger.warning(
                    "No log files found in datetime range %s to %s",
                    self.config["start"],
                    self.config["end"],
                )
                return

            self.logger.info("Processing %d log files in datetime range", len(log_files))
            for log_file in log_files:
                # Extract AUV name from path
                self.auv_name = log_file.split("/")[0].lower()
                self.logger.info("Processing log file: %s", log_file)
                try:
                    self.process_log_file(log_file)
                except (InvalidCalFile, InvalidCombinedFile) as e:
                    self.logger.warning("%s", e)
        else:
            self.logger.error("Must provide either --log_file or both --start and --end arguments")
            return

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
            " copying to the AUVCTD directory",
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
            default=datetime.now().astimezone(UTC).year,
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
            help="For Doado class data - process only this mission",
        )
        parser.add_argument(
            "--log_file",
            action="store",
            help="For LRAUV class data - process only this log file",
        )
        parser.add_argument(
            "--start",
            action="store",
            help="For LRAUV class data - start processing from this datetime "
            "(YYYYMMDDTHHMMSS format)",
        )
        parser.add_argument(
            "--end",
            action="store",
            help="For LRAUV class data - end processing at this datetime (YYYYMMDDTHHMMSS format)",
        )
        parser.add_argument(
            "--auv_name",
            action="store",
            help="For LRAUV class data - restrict log file search to this AUV name "
            "(e.g., brizo, ahi). If not specified, all AUVs will be searched.",
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
            "--add_seconds",
            action="store",
            type=int,
            help=(
                "Add seconds to time variables. Used to correct Dorado log files "
                "saved with GPS Week Rollover Bug."
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
        return self.args


if __name__ == "__main__":
    AUV_NAME = "i2map"
    VEHICLE_DIR = "/Volumes/M3/master/i2MAP"
    CALIBRATION_DIR = "/Volumes/DMO/MDUC_CORE_CTD_200103/Calibration Files"
    MOUNT_DIR = "smb://thalassa.shore.mbari.org/M3"

    # Parse command line and initialize with config pattern
    temp_proc = Processor(AUV_NAME, VEHICLE_DIR, MOUNT_DIR, CALIBRATION_DIR)
    args = temp_proc.process_command_line()

    # Create configured processor instance
    proc = Processor.from_args(AUV_NAME, VEHICLE_DIR, MOUNT_DIR, CALIBRATION_DIR, args)

    # Process based on arguments
    if args.log_file:
        proc.process_log_files()
    elif args.start and args.end:
        # Process LRAUV log files in datetime range
        proc.process_log_files()
    else:
        proc.process_missions(2020)
