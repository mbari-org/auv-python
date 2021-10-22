#!/usr/bin/env python
"""
Parse logged data from AUV MVC .log files and translate to NetCDF including
all of the available metadata from associated .cfg and .xml files.
"""

__author__ = "Mike McCann"
__copyright__ = "Copyright 2020, Monterey Bay Aquarium Research Institute"

import argparse
import asyncio
import concurrent
import os
import sys
import logging
import requests
import struct
import time
from aiohttp import ClientSession
from aiohttp.client_exceptions import ClientConnectorError
from AUV import AUV
from netCDF4 import Dataset
from pathlib import Path
from readauvlog import log_record
from typing import List

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
)
BASE_PATH = "auv_data"

MISSIONLOGS = "missionlogs"
MISSIONNETCDFS = "missionnetcdfs"
PORTAL_BASE = "http://portal.shore.mbari.org:8080/auvdata/v1"
TIME = "time"
TIMEOUT = 240


class AUV_NetCDF(AUV):

    logger = logging.getLogger(__name__)
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        "%(levelname)s %(asctime)s %(filename)s "
        "%(funcName)s():%(lineno)d %(message)s"
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    _log_levels = (logging.WARN, logging.INFO, logging.DEBUG)

    def read(self, file: str) -> List[log_record]:
        """Reads and parses an AUV log and returns a list of `log_records`"""
        byte_offset = 0
        records = []
        (byte_offset, records) = self._read_header(file)
        self._read_data(file, records, byte_offset)

        return records

    def _read_header(self, file: str):
        """Parses the ASCII header of the log file"""
        with open(file, "r", encoding="ISO-8859-15") as f:
            byte_offset = 0
            records = []
            instrument_name = os.path.basename(f.name)

            # Yes, read 2 lines here.
            line = f.readline()
            line = f.readline()
            while line:
                if "begin" in line:
                    break
                else:
                    # parse line
                    ssv = line.split(" ")
                    data_type = ssv[1]
                    short_name = ssv[2]

                    csv = line.split(",")
                    long_name = csv[1].strip()
                    units = csv[2].strip()
                    if short_name == "time":
                        units = "seconds since 1970-01-01 00:00:00Z"
                    r = log_record(
                        data_type, short_name, long_name, units, instrument_name, []
                    )
                    records.append(r)

                line = f.readline()
                byte_offset = f.tell()

            return (byte_offset, records)

    def _read_data(self, file: str, records: List[log_record], byte_offset: int):
        """Parse the binary section of the log file"""
        if byte_offset == 0:
            raise EOFError(f"{file}: 0 sized file")
        file_size = os.path.getsize(file)

        ok = True
        rec_count = 0
        len_sum = 0
        with open(file, "rb") as f:
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
                            f"{e}, b = {b} at record {rec_count},"
                            f" for {r.short_name} in file {file}"
                        )
                        self.logger.info(
                            f"bytes read = {byte_offset + len_sum}"
                            f" file size = {file_size}"
                        )
                        self.logger.info(
                            f"Tried to read {r.length()} bytes, but"
                            f" only {byte_offset+len_sum-file_size}"
                            f" bytes remaining"
                        )
                        raise
                    r.data.append(v)
                rec_count += 1

        self.logger.debug(
            f"bytes read = {byte_offset + len_sum}" f" file size = {file_size}"
        )

    def _unique_vehicle_names(self):
        self.logger.debug(f"Getting deployments from {self.deployments_url}")
        with requests.get(self.deployments_url) as resp:
            if resp.status_code != 200:
                self.logger.error(
                    f"Cannot read {self.deployments_url},"
                    f" status_code = {resp.status_code}"
                )
                return

            return set([d["vehicle"] for d in resp.json()])

    def _deployments_between(self):
        start = f"{self.args.start}T000000Z"
        end = f"{self.args.end}T235959Z"
        url = f"{self.deployments_url}?from={start}&to={end}"
        self.logger.debug(f"Getting missions from {url}")
        with requests.get(url) as resp:
            if resp.status_code != 200:
                raise Exception(f"Cannot read {url}, status_code = {resp.status_code}")
            if not resp.json():
                raise LookupError(f"No missions from {url}")
            for item in resp.json():
                if self.args.preview:
                    self.logger.setLevel(self._log_levels[max(1, self.args.verbose)])
                    self.logger.info(f"{item['vehicle']} {item['name']}")
                else:
                    if self.args.auv_name:
                        if item["vehicle"].upper() != self.args.auv_name.upper():
                            self.logger.debug(
                                f"{item['vehicle']} != {self.args.auv_name}"
                            )
                            continue
                    try:
                        self.download_process_logs(item["vehicle"], item["name"])
                    except asyncio.exceptions.TimeoutError:
                        self.logger.warning(
                            f"TimeoutError for self.download_process_logs("
                            f"'{item['vehicle']}'', '{item['name']}')"
                        )
                        self.logger.info("Sleeping for 60 seconds...")
                        time.sleep(60)
                        self.logger.info(
                            f"Trying to download_process_logs("
                            f"'{item['vehicle']}'', '{item['name']}') again..."
                        )
                        self.download_process_logs(item["vehicle"], item["name"])

    def _files_from_mission(self, name=None, vehicle=None):
        name = name or self.args.mission
        vehicle = vehicle or self.args.auv_name
        files_url = f"{self.portal_base}/files/list/{name}/{vehicle}"
        self.logger.debug(f"Getting files list from {files_url}")
        with requests.get(files_url) as resp:
            if resp.status_code != 200:
                self.logger.error(
                    f"Cannot read {files_url}, status_code = {resp.status_code}"
                )
                return
            if names := resp.json()["names"]:
                return names
            else:
                raise LookupError(f"Nothing in names from {files_url}")

    async def _get_file(self, download_url, local_filename, session):
        try:
            async with session.get(download_url, timeout=TIMEOUT) as resp:
                if resp.status != 200:
                    self.logger.warning(
                        f"Cannot read {download_url}, status = {resp.status}"
                    )
                else:
                    self.logger.info(f"Started download to {local_filename}...")
                    with open(local_filename, "wb") as handle:
                        async for chunk in resp.content.iter_chunked(1024):
                            handle.write(chunk)
                        if self.args.verbose > 1:
                            print(
                                f"{os.path.basename(local_filename)}(done) ",
                                end="",
                                flush=True,
                            )

        except (ClientConnectorError, concurrent.futures._base.TimeoutError) as e:
            self.logger.error(f"{e}")

    async def _download_files(self, logs_dir, name=None, vehicle=None):
        name = name or self.args.mission
        vehicle = vehicle or self.args.auv_name
        tasks = []
        async with ClientSession(timeout=TIMEOUT) as session:
            for ffm in self._files_from_mission(name, vehicle):
                download_url = (
                    f"{self.portal_base}/files/download/{name}/{vehicle}/{ffm}"
                )
                self.logger.debug(f"Getting file contents from {download_url}")
                Path(logs_dir).mkdir(parents=True, exist_ok=True)
                local_filename = os.path.join(logs_dir, ffm)
                task = asyncio.ensure_future(
                    self._get_file(download_url, local_filename, session)
                )
                tasks.append(task)

            await asyncio.gather(*tasks)

    def _correct_dup_short_names(self, log_data):
        short_names = [v.short_name for v in log_data]
        dupes = set([x for n, x in enumerate(short_names) if x in short_names[:n]])
        if len(dupes) > 1:
            raise ValueError(f"Found more than one duplicate: {dupes}")
        elif len(dupes) == 1:
            count = 0
            for i, variable in enumerate(log_data):
                if variable.short_name in dupes:
                    count += 1
                    log_data[i].short_name = f"{log_data[i].short_name}{count}"

        return log_data

    def _get_standard_name(self, short_name, long_name):
        standard_name = ""
        self.logger.debug(
            f"Using a rough heuristic to set a standard_name for {long_name}"
        )
        if short_name.lower() == "time":
            standard_name = "time"
        elif short_name.lower() == "temperature":
            standard_name = "sea_water_temperature"

        return standard_name

    def _create_variable(self, data_type, short_name, long_name, units, data):
        if data_type == "short":
            nc_data_type = "h"
        elif data_type == "integer":
            nc_data_type = "i"
        elif (
            data_type == "float"
            or data_type == "timeTag"
            or data_type == "double"
            or data_type == "angle"
        ):
            nc_data_type = "f8"
        else:
            raise ValueError(f"No conversion for data_type = {data_type}")

        self.logger.debug(f"createVariable {short_name}")
        setattr(
            self,
            short_name,
            self.nc_file.createVariable(short_name, nc_data_type, (TIME,)),
        )
        if standard_name := self._get_standard_name(short_name, long_name):
            setattr(getattr(self, short_name), "standard_name", standard_name)
        setattr(getattr(self, short_name), "long_name", long_name)
        setattr(getattr(self, short_name), "units", units)
        try:
            self.logger.debug(
                f"{short_name}.shape[0] ({getattr(self, short_name).shape[0]})"
                f" should equal len(data) ({len(data)})"
            )
            getattr(self, short_name)[:] = data
        except ValueError as e:
            self.logger.warning(f"{e}")
            self.logger.info(
                f"len(data) ({len(data)}) does not match shape of"
                f" {short_name}.shape[0] ({getattr(self, short_name).shape[0]})"
            )
            if getattr(self, short_name).shape[0] - len(data) == 1:
                self.logger.warning(
                    f"{short_name} data is short by one,"
                    f" appending the last value: {data[-1]}"
                )
                data.append(data[-1])
                getattr(self, short_name)[:] = data
            else:
                self.logger.error("data seriously does not match shape")
                raise

    def write_variables(self, log_data, netcdf_filename):
        log_data = self._correct_dup_short_names(log_data)
        for variable in log_data:
            self.logger.debug(
                f"Creating Variable {variable.short_name}:"
                f" {variable.long_name} ({variable.units})"
            )
            self._create_variable(
                variable.data_type,
                variable.short_name,
                variable.long_name,
                variable.units,
                variable.data,
            )

    def _process_log_file(self, log_filename, netcdf_filename):
        log_data = self.read(log_filename)
        if os.path.exists(netcdf_filename):
            # xarray's Dataset raises permission denied error if file exists
            os.remove(netcdf_filename)
        self.nc_file = Dataset(netcdf_filename, "w")
        self.nc_file.createDimension(TIME, len(log_data[0].data))
        self.write_variables(log_data, netcdf_filename)

        # Add the global metadata, overriding with command line options provided
        self.add_global_metadata()
        vehicle = self.args.auv_name
        self.nc_file.title = (
            f"Original AUV {vehicle} data converted straight from the .log file"
        )
        if self.args.title:
            self.nc_file.title = self.args.title
        if self.args.summary:
            self.nc_file.summary = self.args.summary

        self.nc_file.close()

    def download_process_logs(self, vehicle=None, name=None):
        name = name or self.args.mission
        vehicle = vehicle or self.args.auv_name
        logs_dir = os.path.join(self.args.base_path, vehicle, MISSIONLOGS, name)

        if not self.args.local:
            # Download logs via portal service
            self.logger.debug(
                f"Unique vehicle names: {self._unique_vehicle_names()} seconds"
            )
            yes_no = "Y"
            if os.path.exists(os.path.join(logs_dir, "vehicle.cfg")):
                if self.args.noinput:
                    if self.args.clobber:
                        self.logger.info(f"Clobbering existing {logs_dir} files")
                    else:
                        self.logger.info(f"{logs_dir} exists")
                        yes_no = "N"
                        if self.args.noreprocess:
                            self.logger.info(f"Not reprocessing {logs_dir}")
                            return
                else:
                    yes_no = (
                        input(f"Directory {logs_dir} exists. Re-download? [Y/n]: ")
                        or "Y"
                    )
            if yes_no.upper().startswith("Y"):
                self.logger.info(f"Downloading mission: {vehicle} {name}")
                d_start = time.time()
                loop = asyncio.get_event_loop()
                try:
                    future = asyncio.ensure_future(
                        self._download_files(logs_dir, name, vehicle)
                    )
                except asyncio.exceptions.TimeoutError as e:
                    self.logger.warning(f"{e}")
                try:
                    loop.run_until_complete(future)
                except LookupError as e:
                    self.logger.error(f"{e}")
                    self.logger.info(f"Perhaps use '--update' option?")
                    return
                self.logger.info(
                    f"Time to download: {(time.time() - d_start):.2f} seconds"
                )

        self.logger.info(f"Processing mission: {vehicle} {name}")
        netcdfs_dir = os.path.join(self.args.base_path, vehicle, MISSIONNETCDFS, name)
        Path(netcdfs_dir).mkdir(parents=True, exist_ok=True)
        p_start = time.time()
        for log in LOG_FILES:
            log_filename = os.path.join(logs_dir, log)
            netcdf_filename = os.path.join(netcdfs_dir, log.replace(".log", ".nc"))
            try:
                self.logger.info(f"Processing file {log_filename}")
                self._process_log_file(log_filename, netcdf_filename)
            except (FileNotFoundError, EOFError, struct.error) as e:
                self.logger.debug(f"{e}")
        self.logger.info(f"Time to process: {(time.time() - p_start):.2f} seconds")

    def update(self):
        self.logger.setLevel(self._log_levels[max(1, self.args.verbose)])
        url = "http://portal.shore.mbari.org:8080/auvdata/v1/deployments/update"
        auv_netcdf.logger.info(f"Sending an 'update' request: {url}")
        resp = requests.post(url)
        if resp.status_code != 200:
            self.logger.error(
                f"Update failed for url = {url}," f" status_code = {resp.status_code}"
            )
        else:
            self.logger.info("Wait a few minutes for new missions to appear")

    def process_command_line(self):
        examples = "Examples:" + "\n\n"
        examples += "  Write to local missionnetcdfs direcory:\n"
        examples += "    " + sys.argv[0] + " --mission 2020.064.10\n"
        examples += "    " + sys.argv[0] + " --auv_name i2map --mission 2020.055.01\n"

        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter,
            description="Convert AUV log files to NetCDF files",
            epilog=examples,
        )

        parser.add_argument(
            "--base_path",
            action="store",
            default=BASE_PATH,
            help="Base directory for missionlogs and"
            " missionnetcdfs, default: auv_data",
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
            "--mission", action="store", help="Mission directory, e.g.: 2020.064.10"
        )
        parser.add_argument(
            "--local",
            action="store_true",
            help="Specify if files are local in the MISSION directory",
        )

        parser.add_argument(
            "--title", action="store", help="A short description" " of the dataset"
        )
        parser.add_argument(
            "--summary", action="store", help="Additional information about the dataset"
        )

        parser.add_argument(
            "--noinput",
            action="store_true",
            help="Execute without asking for a response, e.g. "
            " to not ask to re-download file",
        )
        parser.add_argument(
            "--clobber",
            action="store_true",
            help="Use with --noinput to overwrite existing" " downloaded log files",
        )
        parser.add_argument(
            "--noreprocess",
            action="store_true",
            help="Use with --noinput to not re-process existing"
            " downloaded log files",
        )
        parser.add_argument(
            "--start",
            action="store",
            help="Convert a range of missions wth start time in" " YYYYMMDD format",
        )
        parser.add_argument(
            "--end",
            action="store",
            help="Convert a range of missions wth end time in" " YYYYMMDD format",
        )
        parser.add_argument(
            "--preview",
            action="store_true",
            help="List missions that will be downloaded and"
            " processed with --start and --end",
        )
        parser.add_argument(
            "--update",
            action="store_true",
            help='Send an "update" POST request to the ' " auv-portal data service",
        )
        parser.add_argument(
            "--portal",
            action="store",
            help="Specify the base url for the auv-portal data"
            " service, e.g.:"
            " http://stoqs.mbari.org:8080/auvdata/v1",
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

        if self.args.portal:
            self.portal_base = self.args.portal
            self.deployments_url = os.path.join(self.args.portal, "deployments")
        else:
            self.portal_base = PORTAL_BASE
            self.deployments_url = os.path.join(self.portal_base, "deployments")

        self.commandline = " ".join(sys.argv)


if __name__ == "__main__":

    auv_netcdf = AUV_NetCDF()
    auv_netcdf.process_command_line()

    p_start = time.time()
    if auv_netcdf.args.update:
        auv_netcdf.update()
    elif auv_netcdf.args.auv_name and auv_netcdf.args.mission:
        auv_netcdf.download_process_logs()
    elif auv_netcdf.args.start and auv_netcdf.args.end:
        auv_netcdf._deployments_between()
    else:
        raise argparse.ArgumentError(
            None, "Must provide either (--auv_name &" " --mission) OR (--start & --end)"
        )

    auv_netcdf.logger.info(
        f"Time to download and process:" f" {(time.time() - p_start):.2f} seconds"
    )
