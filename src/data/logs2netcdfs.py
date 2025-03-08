#!/usr/bin/env python
"""
Convert AUV log files to NetCDF files

Parse logged data from AUV MVC .log files and translate to NetCDF including
all of the available metadata from associated .cfg and .xml files.
"""

__author__ = "Mike McCann"
__copyright__ = "Copyright 2020, Monterey Bay Aquarium Research Institute"

import argparse
import asyncio
import concurrent
import logging
import struct
import subprocess
import sys
import time
from http import HTTPStatus
from pathlib import Path

import aiofiles
import numpy as np
import requests
from aiohttp import ClientSession
from aiohttp.client_exceptions import ClientConnectorError
from AUV import AUV, monotonic_increasing_time_indices
from netCDF4 import Dataset
from readauvlog import log_record

LOG_FILES = (
    "ctdDriver.log",
    "ctdDriver2.log",
    "gps.log",
    "hydroscatlog.log",
    "navigation.log",
    "isuslog.log",
    "parosci.log",
    "seabird25p.log",
    "FLBBCD2K.log",
    "tailCone.log",
    "biolume.log",
)
BASE_PATH = Path(__file__).parent.joinpath("../../data/auv_data").resolve()

MISSIONLOGS = "missionlogs"
MISSIONNETCDFS = "missionnetcdfs"
PORTAL_BASE = "http://portal.shore.mbari.org:8080/auvdata/v1"
TIME = "time"
TIME60HZ = "time60hz"
TIMEOUT = 240
SUMMARY_SOURCE = "Original log files copied from {}"


class CustomException(Exception):
    pass


class AUV_NetCDF(AUV):
    logger = logging.getLogger(__name__)
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        "%(levelname)s %(asctime)s %(filename)s "
        "%(funcName)s():%(lineno)d [%(process)d] %(message)s",
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    _log_levels = (logging.WARN, logging.INFO, logging.DEBUG)

    def read(self, file: Path) -> list[log_record]:
        """Reads and parses an AUV log and returns a list of `log_records`"""
        byte_offset = 0
        records = []
        (byte_offset, records) = self._read_header(file)
        if "biolume" in str(file):
            (byte_offset, records) = self._read_biolume_header(file)
            self._read_biolume_data(file, records, byte_offset)
        else:
            (byte_offset, records) = self._read_header(file)
            self._read_data(file, records, byte_offset)

        return records

    def _read_header(self, file: Path):
        """Parses the ASCII header of the log file"""
        with file.open(encoding="ISO-8859-15") as f:
            byte_offset = 0
            records = []
            instrument_name = Path(f.name).name

            # Yes, read 2 lines here.
            line = f.readline()
            line = f.readline()
            while line:
                if "begin" in line:
                    break
                # parse line
                ssv = line.split(" ")
                data_type = ssv[1]
                short_name = ssv[2]

                csv = line.split(",")
                if len(csv) == 1:
                    # Likely early missions before commas were used
                    error_message = "Failed to parse long_name & units from header"
                    raise ValueError(error_message)
                long_name = csv[1].strip()
                units = csv[2].strip()
                if short_name == "time":
                    units = "seconds since 1970-01-01 00:00:00Z"
                r = log_record(
                    data_type,
                    short_name,
                    long_name,
                    units,
                    instrument_name,
                    [],
                )
                records.append(r)

                line = f.readline()
                byte_offset = f.tell()

            return (byte_offset, records)

    def _read_biolume_header(self, file: Path):
        """Parses the ASCII header of the log file, collapse all the raw_
        variables into a single 60 hz raw variable"""
        with file.open(encoding="ISO-8859-15") as f:
            byte_offset = 0
            records = []
            instrument_name = Path(f.name).name

            # Yes, read 2 lines here.
            line = f.readline()
            line = f.readline()
            seen_raw = False
            while line:
                if "raw_" in line:
                    if not seen_raw:
                        # No need to read the raw_ variables. On first encounter
                        # create a single raw variable for all the 60 hz raw_ data
                        # and break out of the loop
                        r = log_record(
                            "integer",
                            "raw",
                            "Original 60 hz raw data",
                            "photons s^-1",
                            instrument_name,
                            [],
                        )
                        records.append(r)
                        seen_raw = True
                elif "begin" in line:
                    # Breaking here results in byte_offset being correct
                    break
                else:
                    # parse non-raw_ line
                    _, data_type, short_name, _ = line.split(" ", maxsplit=3)

                    csv = line.split(",")
                    if len(csv) == 1:
                        # Likely early missions before commas were used
                        error_message = "Failed to parse long_name & units from header"
                        raise ValueError(error_message)
                    long_name = csv[1].strip()
                    units = csv[2].strip()
                    if short_name == "time":
                        units = "seconds since 1970-01-01 00:00:00Z"
                    r = log_record(
                        data_type,
                        short_name,
                        long_name,
                        units,
                        instrument_name,
                        [],
                    )
                    records.append(r)

                line = f.readline()
                byte_offset = f.tell()

            return (byte_offset, records)

    def _read_data(self, file: Path, records: list[log_record], byte_offset: int):
        """Parse the binary section of the log file"""
        if byte_offset == 0:
            error_message = f"{file}: 0 sized file"
            raise EOFError(error_message)
        file_size = Path(file).stat().st_size

        ok = True
        rec_count = 0
        len_sum = 0
        with file.open("rb") as f:
            f.seek(byte_offset)
            while ok:
                for r in records:
                    b = f.read(r.length())
                    len_sum += r.length()
                    if not b:
                        ok = False
                        len_sum -= r.length()
                        break
                    s = "<d"
                    if r.data_type == "float":
                        s = "<f"
                    elif r.data_type == "integer":
                        s = "<i"
                    elif r.data_type == "short":
                        s = "<h"
                    try:
                        v = struct.unpack(s, b)[0]
                    except struct.error as e:
                        self.logger.warning(
                            "%s, b = %s at record %d, for %s in file %s",
                            e,
                            b,
                            rec_count,
                            r.short_name,
                            file,
                        )
                        self.logger.info(
                            "bytes read = %d file size = %d",
                            byte_offset + len_sum,
                            file_size,
                        )
                        self.logger.info(
                            "Tried to read %d bytes, but only %d bytes remaining",
                            r.length(),
                            byte_offset + len_sum - file_size,
                        )
                        if rec_count > 0:
                            self.logger.info(
                                "Successfully unpacked %d records",
                                rec_count,
                            )
                        else:
                            self.logger.exception("No records unpacked")
                            raise
                    r.data.append(v)
                rec_count += 1

        self.logger.debug(
            "bytes read = %d file size = %d",
            byte_offset + len_sum,
            file_size,
        )

    def _read_biolume_data(  # noqa: C901, PLR0912, PLR0915
        self,
        file: Path,
        records: list[log_record],
        byte_offset: int,
    ):
        """Parse the binary section of the log file, collecting the 60 hz
        raw values into a 60 hz time series.  Subtract 1/2 second from the
        1 second biolume time stamps as it's recorded at the end of the
        sampling period.
        """
        if byte_offset == 0:
            error_message = f"{file}: 0 sized file"
            raise EOFError(error_message)
        file_size = Path(file).stat().st_size

        ok = True
        rec_count = 0
        len_sum = 0
        with file.open("rb") as f:
            f.seek(byte_offset)
            while ok:
                for r in records:
                    try:
                        if r.short_name == "raw":
                            if records[2].short_name != "cal_striing":
                                error_message = "Expected 'cal_striing' in records[2].short_name"
                                raise ValueError(error_message)
                            for _ in range(60):
                                b = f.read(r.length())
                                len_sum += r.length()
                                if not b:
                                    ok = False
                                    len_sum -= r.length()
                                    break
                                v = struct.unpack("<i", b)[0]
                                # raw data is multiplied by cal_striing
                                r.data.append(v * records[2].data[-1])
                        else:
                            b = f.read(r.length())
                            len_sum += r.length()
                            if not b:
                                ok = False
                                len_sum -= r.length()
                                break
                            s = "<d"
                            if r.data_type == "float":
                                s = "<f"
                            elif r.data_type == "integer":
                                s = "<i"
                            elif r.data_type == "short":
                                s = "<h"
                            v = struct.unpack(s, b)[0]
                            r.data.append(v)
                    except struct.error as e:
                        self.logger.warning(
                            "%s, b = %s at record %d, for %s in file %s",
                            e,
                            b,
                            rec_count,
                            r.short_name,
                            file,
                        )
                        self.logger.info(
                            "bytes read = %d file size = %d",
                            byte_offset + len_sum,
                            file_size,
                        )
                        self.logger.info(
                            "Tried to read %d bytes, but only %d bytes remaining",
                            r.length(),
                            byte_offset + len_sum - file_size,
                        )
                        if rec_count > 0:
                            self.logger.info(
                                "Successfully unpacked %d records",
                                rec_count,
                            )
                        else:
                            self.logger.error("No records uppacked")  # noqa: TRY400
                            raise
                rec_count += 1

        self.logger.debug("bytes read = %d file size = %d", byte_offset + len_sum, file_size)

    def _unique_vehicle_names(self):
        self.logger.debug("Getting deployments from %s", self.deployments_url)
        with requests.get(self.deployments_url, timeout=TIMEOUT) as resp:
            if resp.status_code != HTTPStatus.OK:
                self.logger.error(
                    "Cannot read %s, status_code = %d",
                    self.deployments_url,
                    resp.status_code,
                )
                return None

            return {d["vehicle"] for d in resp.json()}

    def _deployments_between(self):
        start = f"{self.args.start}T000000Z"
        end = f"{self.args.end}T235959Z"
        url = f"{self.deployments_url}?from={start}&to={end}"
        self.logger.debug("Getting missions from %s", url)
        with requests.get(url, timeout=TIMEOUT) as resp:
            if resp.status_code != HTTPStatus.OK:
                error_message = f"Cannot read {url}, status_code = {resp.status_code}"
                raise LookupError(error_message)
            if not resp.json():
                error_message = f"No missions from {url}"
                raise LookupError(error_message)
            for item in resp.json():
                if self.args.preview:
                    self.logger.setLevel(self._log_levels[max(1, self.args.verbose)])
                    self.logger.info("%s %s", item["vehicle"], item["name"])
                else:
                    if self.args.auv_name and item["vehicle"].upper() != self.args.auv_name.upper():
                        self.logger.debug(
                            "%s != %s",
                            item["vehicle"],
                            self.args.auv_name,
                        )
                        continue
                    try:
                        self.download_process_logs(item["vehicle"], item["name"])
                    except asyncio.exceptions.TimeoutError:
                        self.logger.warning(
                            "TimeoutError for self.download_process_logs('%s', '%s')",
                            item["vehicle"],
                            item["name"],
                        )
                        self.logger.info("Sleeping for 60 seconds...")
                        time.sleep(60)
                        self.logger.info(
                            "Trying to download_process_logs('%s', '%s') again...",
                            item["vehicle"],
                            item["name"],
                        )
                        self.download_process_logs(item["vehicle"], item["name"])

    def _files_from_mission(self, name=None, vehicle=None):
        name = name or self.args.mission
        vehicle = vehicle or self.args.auv_name
        files_url = f"{self.portal_base}/files/list/{name}/{vehicle}"
        self.logger.debug("Getting files list from %s", files_url)
        with requests.get(files_url, timeout=TIMEOUT) as resp:
            if resp.status_code != HTTPStatus.OK:
                self.logger.error(
                    "Cannot read %s, status_code = %d",
                    files_url,
                    resp.status_code,
                )
                return None
            if names := resp.json()["names"]:
                return names
            error_message = f"Nothing in names from {files_url}"
            raise LookupError(error_message)

    async def _get_file(self, download_url, local_filename, session):
        try:
            async with session.get(download_url, timeout=TIMEOUT) as resp:
                if resp.status != HTTPStatus.OK:
                    self.logger.warning(
                        "Cannot read %s, status = %d",
                        download_url,
                        resp.status,
                    )
                else:
                    async with aiofiles.open(local_filename, "wb") as handle:
                        async for chunk in resp.content.iter_chunked(1024):
                            await handle.write(chunk)
                            handle.write(chunk)
                        if self.args.verbose > 1:
                            print(  # noqa: T201
                                f"{Path(local_filename).name}(done) ",
                                end="",
                                flush=True,
                            )

        except (ClientConnectorError, concurrent.futures._base.TimeoutError):
            self.logger.exception()

    async def _download_files(self, logs_dir, name=None, vehicle=None):
        name = name or self.args.mission
        vehicle = vehicle or self.args.auv_name
        tasks = []
        async with ClientSession(timeout=TIMEOUT) as session:
            for ffm in self._files_from_mission(name, vehicle):
                download_url = f"{self.portal_base}/files/download/{name}/{vehicle}/{ffm}"
                self.logger.debug("Getting file contents from %s", download_url)
                Path(logs_dir).mkdir(parents=True, exist_ok=True)
                local_filename = Path(logs_dir, ffm)
                task = asyncio.ensure_future(
                    self._get_file(download_url, local_filename, session),
                )
                tasks.append(task)

            await asyncio.gather(*tasks)

    def _portal_download(self, logs_dir, name=None, vehicle=None):
        self.logger.debug("Getting logs from %s", self.portal_base)
        self.logger.info("Downloading mission: %s %s", vehicle, name)
        d_start = time.time()
        loop = asyncio.get_event_loop()
        try:
            future = asyncio.ensure_future(
                self._download_files(logs_dir, name, vehicle),
            )
        except asyncio.exceptions.TimeoutError as e:
            self.logger.warning("%s", e)
        try:
            loop.run_until_complete(future)
        except LookupError:
            self.logger.exception()
            self.logger.info("Perhaps use '--update' option?")
            return
        self.logger.info("Time to download: %.2f seconds", time.time() - d_start)

    def _correct_dup_short_names(self, log_data):
        short_names = [v.short_name for v in log_data]
        dupes = {x for n, x in enumerate(short_names) if x in short_names[:n]}
        if len(dupes) > 1:
            error_message = f"Found more than one duplicate: {dupes}"
            raise ValueError(error_message)
        if len(dupes) == 1:
            count = 0
            for i, variable in enumerate(log_data):
                if variable.short_name in dupes:
                    count += 1
                    log_data[i].short_name = f"{variable.short_name}{count}"

        return log_data

    def _get_standard_name(self, short_name, long_name):
        standard_name = ""
        if short_name.lower() == "time" or short_name.lower() == "time60hz":
            standard_name = "time"
        elif short_name.lower() == "temperature":
            standard_name = "sea_water_temperature"
        if standard_name:
            self.logger.debug(
                "Setting standard_name = %s for %s",
                standard_name,
                long_name,
            )

        return standard_name

    def _create_variable(  # noqa: PLR0913
        self,
        data_type,
        short_name,
        long_name,
        units,
        data,
        time_axis=TIME,
    ):
        if data_type == "short":
            nc_data_type = "h"
        elif data_type == "integer":
            nc_data_type = "i"
        elif data_type in {"float", "timeTag", "double", "angle"}:
            nc_data_type = "f8"
        else:
            error_message = f"No conversion for data_type = {data_type}"
            raise ValueError(error_message)

        self.logger.debug("createVariable %s", short_name)
        setattr(
            self,
            short_name,
            self.nc_file.createVariable(short_name, nc_data_type, (time_axis,)),
        )
        if standard_name := self._get_standard_name(short_name, long_name):
            getattr(self, short_name).standard_name = standard_name
        getattr(self, short_name).long_name = long_name
        getattr(self, short_name).units = units
        try:
            self.logger.debug(
                "%s.shape[0] (%d) should equal len(data) (%d)",
                short_name,
                getattr(self, short_name).shape[0],
                len(data),
            )
            getattr(self, short_name)[:] = data
        except ValueError as e:
            self.logger.warning("%s: %s", short_name, e)
            self.logger.info(
                "len(data) (%d) does not match shape of %s.shape[0] (%d)",
                len(data),
                short_name,
                getattr(self, short_name).shape[0],
            )
            if getattr(self, short_name).shape[0] - len(data) == 1:
                self.logger.warning(
                    "%s data is short by one, appending the last value: %s",
                    short_name,
                    data[-1],
                )
                data.append(data[-1])
                getattr(self, short_name)[:] = data
            else:
                self.logger.error("data seriously does not match shape")  # noqa: TRY400
                raise

    def write_variables(self, log_data, netcdf_filename):
        log_data = self._correct_dup_short_names(log_data)
        self.nc_file.createDimension(TIME, len(log_data[0].data))
        for variable in log_data:
            self.logger.debug(
                "Creating Variable %s: %s (%s)",
                variable.short_name,
                variable.long_name,
                variable.units,
            )
            if "biolume" in str(netcdf_filename):
                if variable.short_name == "raw":
                    # The "raw" log is the last one in the list, and time is the first
                    if log_data[-1].short_name != "raw":
                        error_message = "Expected the last log_data short_name to be 'raw'"
                        raise ValueError(error_message)
                    self.nc_file.createDimension(TIME60HZ, len(log_data[0].data) * 60)
                    if log_data[0].data_type != "timeTag":
                        error_message = "Expected data_type to be 'timeTag'"
                        raise ValueError(error_message)
                    self.logger.info(
                        "Expanding original timeTag to time60Hz variable for raw data",
                    )
                    self._create_variable(
                        "timeTag",
                        TIME60HZ,
                        "60Hz time",
                        "seconds since 1970-01-01 00:00:00Z",
                        [tv + frac for tv in log_data[0].data for frac in np.arange(0, 1, 1 / 60)],
                        time_axis=TIME60HZ,
                    )
                    self._create_variable(
                        "float",
                        variable.short_name,
                        variable.long_name,
                        variable.units,
                        variable.data,
                        time_axis=TIME60HZ,
                    )
                elif variable.short_name == "timeTag":
                    # The biolume time value needs to have 1/2 second subtracted
                    self.logger.info("Subtracting 1/2 second from avg_biolume timeTag")
                    self._create_variable(
                        "timeTag",
                        TIME,
                        "avg_biolume time",
                        "seconds since 1970-01-01 00:00:00Z",
                        [tv - 0.5 for tv in log_data[0].data],
                        time_axis=TIME,
                    )
                    self._create_variable(
                        "float",
                        variable.short_name,
                        variable.long_name,
                        variable.units,
                        variable.data,
                        time_axis=TIME,
                    )
                else:
                    self._create_variable(
                        variable.data_type,
                        variable.short_name,
                        variable.long_name,
                        variable.units,
                        variable.data,
                    )
            else:
                self._create_variable(
                    variable.data_type,
                    variable.short_name,
                    variable.long_name,
                    variable.units,
                    variable.data,
                )

    def _remove_bad_values(self, netcdf_filename):
        """Loop through all variables in self.nc_file,
        remove `bad_indices`, and write back out"""
        ds_orig = Dataset(netcdf_filename)
        #  Finding < -1.0e20 works for 2010.172.01's navigation.log
        #                          and 2010.265.00's ctdDriver.log
        bad_indices = np.where(ds_orig[TIME][:] < -1.0e20)[0]  # noqa: PLR2004
        self.logger.warning(
            "Removing %s bad_indices from %s: %s",
            len(bad_indices),
            netcdf_filename,
            bad_indices,
        )
        netcdf_filename.rename(f"{netcdf_filename}.orig")
        self.logger.info("Renamed original file to  %s", f"{netcdf_filename}.orig")
        self.nc_file = Dataset(netcdf_filename, "w")
        clean_time_values = np.delete(ds_orig[TIME][:], bad_indices)

        # Copy all orig file data (without bad_indices) and attributes
        # Thanks for this! https://stackoverflow.com/a/49592545/1281657
        # copy global attributes all at once via dictionary
        self.nc_file.setncatts(ds_orig.__dict__)
        # copy dimensions
        for name in ds_orig.dimensions:
            self.nc_file.createDimension(name, len(clean_time_values))
        # copy all file data except for the excluded
        for name, variable in ds_orig.variables.items():
            self.nc_file.createVariable(name, variable.datatype, variable.dimensions)
            # copy variable attributes all at once via dictionary
            self.nc_file[name].setncatts(ds_orig[name].__dict__)
            self.nc_file[name][:] = np.delete(ds_orig[name][:], bad_indices)

        self.nc_file.close()
        self.logger.info("Wrote (without bad values) %s", netcdf_filename)

    def _process_log_file(self, log_filename, netcdf_filename, src_dir=None):
        log_data = self.read(log_filename)
        if Path(netcdf_filename).exists():
            # xarray's Dataset raises permission denied error if file exists
            Path(netcdf_filename).unlink()
        self.nc_file = Dataset(netcdf_filename, "w")
        self.write_variables(log_data, netcdf_filename)

        # Add the global metadata, overriding with command line options provided
        self.add_global_metadata()
        vehicle = self.args.auv_name
        self.nc_file.title = f"Original AUV {vehicle} data converted from {log_filename}"
        if hasattr(self.args, "title") and self.args.title:
            self.nc_file.title = self.args.title
        if src_dir:
            # The source attribute might make more sense for the location of
            # the source data, but the summary field is shown in STOQS metadata
            self.nc_file.summary = SUMMARY_SOURCE.format(src_dir)
        if hasattr(self.args, "summary") and self.args.summary:
            self.nc_file.summary = self.args.summary
        monotonic = monotonic_increasing_time_indices(self.nc_file["time"][:])
        if (~monotonic).any():
            self.logger.info(
                "Non-monotonic increasing time indices in %s: %s",
                log_filename,
                np.argwhere(~monotonic).flatten(),
            )
            # Write comment here, preserving original data - corrected in calibration
            self.nc_file.comment += "Non-monotonic increasing times detected."
        self.nc_file.close()

    def get_mission_dir(self, mission: str) -> str:
        """Return the mission directory. This method is nearly identical to the
        one in the Processor class, but it is used here to be explicit and to
        avoid the need to import the Processor class."""
        if not Path(self.args.vehicle_dir).exists():
            self.logger.error("%s does not exist.", self.args.vehicle_dir)
            self.logger.info("Is %s mounted?", self.mount_dir)
            sys.exit(1)
        if self.args.auv_name.lower() == "dorado":
            year = mission.split(".")[0]
            yearyd = "".join(mission.split(".")[:2])
            path = Path(self.args.vehicle_dir, year, yearyd, mission)
        elif self.args.auv_name.lower() == "i2map":
            year = int(mission.split(".")[0])
            # Could construct the YYYY/MM/YYYYMMDD path on M3/Master
            # but use the mission_list() method to find the mission dir instead
            missions = self.mission_list(start_year=year, end_year=year)
            if mission in missions:
                path = missions[mission]
            else:
                self.logger.error("Cannot find %s in %s", mission, self.args.vehicle_dir)
                error_message = f"Cannot find {mission} in {self.args.vehicle_dir}"
                raise FileNotFoundError(error_message)
        elif self.args.auv_name == "Dorado389":
            # The Dorado389 vehicle is a special case used for testing locally and in CI
            path = self.args.vehicle_dir
        if not Path(path).exists():
            self.logger.error("%s does not exist.", path)
            error_message = f"{path} does not exist."
            raise FileNotFoundError(error_message)
        return path

    def download_process_logs(  # noqa: C901, PLR0912, PLR0915
        self,
        vehicle: str = "",
        name: str = "",
        src_dir: Path = Path(),
    ) -> None:
        name = name or self.args.mission
        vehicle = vehicle or self.args.auv_name
        logs_dir = Path(self.args.base_path, vehicle, MISSIONLOGS, name)

        if src_dir:
            self.logger.info("src_dir = %s", src_dir)

        if not self.args.local:
            # As of 20 July 2023 this returns 404, which is dstracting
            # self.logger.debug(
            #   f"Unique vehicle names: {self._unique_vehicle_names()} seconds"
            # )
            yes_no = "Y"
            if Path(logs_dir, "vehicle.cfg").exists():
                if self.args.noinput:
                    if self.args.clobber:
                        self.logger.info("Clobbering existing %s files", logs_dir)
                    else:
                        self.logger.info("%s exists", logs_dir)
                        yes_no = "N"
                        if self.args.noreprocess:
                            self.logger.info("Not reprocessing %s", logs_dir)
                            return
                else:
                    yes_no = input(f"Directory {logs_dir} exists. Re-download? [Y/n]: ") or "Y"
            if yes_no.upper().startswith("Y"):
                if self.args.use_portal:
                    self._portal_download(logs_dir, name=name, vehicle=vehicle)
                elif src_dir:
                    safe_src_dir = Path(src_dir).resolve()
                    if not safe_src_dir.exists():
                        error_message = f"src_dir {safe_src_dir} does not exist"
                        raise FileNotFoundError(error_message)

                    self.logger.info("Rsyncing %s to %s", src_dir, logs_dir)
                    subprocess.run(  # noqa: S603
                        ["/usr/bin/rsync", "-av", str(safe_src_dir), str(logs_dir.parent)],
                        check=True,
                    )
                else:
                    self.logger.info(
                        "src_dir not provided, so downloading from portal",
                    )
                    self._portal_download(logs_dir, name=name, vehicle=vehicle)

        self.logger.info("Processing mission: %s %s", vehicle, name)
        netcdfs_dir = Path(self.args.base_path, vehicle, MISSIONNETCDFS, name)
        Path(netcdfs_dir).mkdir(parents=True, exist_ok=True)
        p_start = time.time()
        for log in LOG_FILES:
            log_filename = Path(logs_dir, log)
            netcdf_filename = Path(netcdfs_dir, log.replace(".log", ".nc"))
            try:
                file_size = Path(log_filename).stat().st_size
                self.logger.info("Processing file %s (%d bytes)", log_filename, file_size)
                if file_size == 0:
                    self.logger.warning("%s is empty", str(log_filename))
                self._process_log_file(log_filename, netcdf_filename, src_dir)
            except (FileNotFoundError, EOFError, struct.error, IndexError) as e:
                self.logger.debug("%s", e)
            except ValueError as e:
                self.logger.warning("%s in file %s", e, log_filename)

            if log == "navigation.log" and "2010.172.01" in str(log_filename):
                # Remove egregiously bad values as found in 2010.172.01's navigation.log - Comment
                # from processNav.m:
                # % For Mission 2010.172.01 the first part of the time array had really
                # % large negative epoch second values.
                # % Take only the positive time values in addition to the good depth values
                self._remove_bad_values(netcdf_filename)
            if log == "ctdDriver.log" and "2010.265.00" in str(log_filename):
                self._remove_bad_values(netcdf_filename)

        self.logger.info("Time to process: %.2f seconds", time.time() - p_start)

    def update(self):
        self.logger.setLevel(self._log_levels[max(1, self.args.verbose)])
        url = "http://portal.shore.mbari.org:8080/auvdata/v1/deployments/update"
        auv_netcdf.logger.info("Sending an 'update' request: %s", url)
        resp = requests.post(url, timeout=TIMEOUT)
        if resp.status_code != HTTPStatus.OK:
            self.logger.error(
                "Update failed for url = %s, status_code = %d",
                url,
                resp.status_code,
            )
        else:
            self.logger.info("Wait a few minutes for new missions to appear")

    def set_portal(self) -> None:
        self.portal_base = PORTAL_BASE
        self.deployments_url = Path(self.portal_base, "deployments")
        if hasattr(self.args, "portal") and self.args.portal:
            self.portal_base = self.args.portal
            self.deployments_url = Path(self.args.portal, "deployments")

    def process_command_line(self):
        examples = "Examples:" + "\n\n"
        examples += "  Write to local missionnetcdfs direcory:\n"
        examples += "    " + sys.argv[0] + " --mission 2020.064.10\n"
        examples += "    " + sys.argv[0] + " --auv_name i2map --mission 2020.055.01\n"

        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter,
            description=__doc__,
            epilog=examples,
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
            help=(
                "Dorado389, i2map, or multibeam. Will be saved in "
                "directory with this name no matter its portal entry"
            ),
        )
        parser.add_argument(
            "--mission",
            action="store",
            help="Mission directory, e.g.: 2020.064.10",
        )
        parser.add_argument(
            "--local",
            action="store_true",
            help="Specify if files are local in the MISSION directory",
        )

        parser.add_argument(
            "--title",
            action="store",
            help="A short description of the dataset",
        )
        parser.add_argument(
            "--summary",
            action="store",
            help="Additional information about the dataset",
        )

        parser.add_argument(
            "--noinput",
            action="store_true",
            help="Execute without asking for a response, e.g.  to not ask to re-download file",
        )
        parser.add_argument(
            "--clobber",
            action="store_true",
            help="Use with --noinput to overwrite existing downloaded log files",
        )
        parser.add_argument(
            "--noreprocess",
            action="store_true",
            help="Use with --noinput to not re-process existing downloaded log files",
        )
        parser.add_argument(
            "--start",
            action="store",
            help="Convert a range of missions wth start time in YYYYMMDD format",
        )
        parser.add_argument(
            "--end",
            action="store",
            help="Convert a range of missions wth end time in YYYYMMDD format",
        )
        parser.add_argument(
            "--preview",
            action="store_true",
            help="List missions that will be downloaded and processed with --start and --end",
        )
        parser.add_argument(
            "--update",
            action="store_true",
            help='Send an "update" POST request to the  auv-portal data service',
        )
        parser.add_argument(
            "--portal",
            action="store",
            help="Specify the base url for the auv-portal data"
            " service, e.g.:"
            " http://stoqs.mbari.org:8080/auvdata/v1",
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
            "--vehicle_dir",
            action="store",
            help="Directory for the vehicle's mission logs, e.g.: /Volumes/AUVCTD/missionlogs",
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
        self.set_portal()
        self.commandline = " ".join(sys.argv)


if __name__ == "__main__":
    auv_netcdf = AUV_NetCDF()
    auv_netcdf.process_command_line()

    p_start = time.time()
    if auv_netcdf.args.update:
        auv_netcdf.update()
    elif auv_netcdf.args.auv_name and auv_netcdf.args.mission:
        if auv_netcdf.args.vehicle_dir:
            path = auv_netcdf.get_mission_dir(auv_netcdf.args.mission)
            auv_netcdf.download_process_logs(src_dir=path)
        else:
            raise argparse.ArgumentError(
                None,
                "Must provide --src_dir with --auv_name & --mission",
            )

        auv_netcdf.download_process_logs(src_dir=Path())
    elif auv_netcdf.args.start and auv_netcdf.args.end:
        auv_netcdf._deployments_between()
    else:
        raise argparse.ArgumentError(
            None,
            "Must provide either (--auv_name & --mission) OR (--start & --end)",
        )

    auv_netcdf.logger.info(
        "Time to download and process: %.2f seconds",
        (time.time() - p_start),
    )
