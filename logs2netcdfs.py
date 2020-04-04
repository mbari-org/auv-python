#!/usr/bin/env python
'''
Parse logged data from AUV MVC .log files and translate to NetCDF including
all of the available metadata from associated .cfg and .xml files.
'''

__author__ = "Mike McCann"
__copyright__ = "Copyright 2020, Monterey Bay Aquarium Research Institute"

import asyncio
import concurrent
import os
import sys
import logging
import readauvlog
import requests
import time
from aiohttp import ClientSession
from aiohttp.client_exceptions import ClientConnectorError
from AUV import AUV
from pathlib import Path
from netCDF4 import Dataset

LOG_FILES = ('ctdDriver.log', 'ctdDriver2.log', 'gps.log', 'hydroscatlog.log', 
             'navigation.log', 'isuslog.log', 'parosci.log')

MISSIONLOGS = 'missionlogs'
MISSIONNETCDFS = 'missionnetcdfs'
DEPLOYMENTS_URL = 'http://portal.shore.mbari.org:8080/auvdata/v1/deployments'
TIME = 'time'

class AUV_NetCDF(AUV):

    logger = logging.getLogger(__name__)
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter('%(levelname)s %(asctime)s %(filename)s '
                                  '%(funcName)s():%(lineno)d %(message)s')
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    _log_levels = (logging.WARN, logging.INFO, logging.DEBUG)

    def __init__(self):
        return super(AUV_NetCDF, self).__init__()

    def _unique_vehicle_names(self):
        self.logger.debug(f"Getting deplolments from {DEPLOYMENTS_URL}")
        with requests.get(DEPLOYMENTS_URL) as resp:
            if resp.status_code != 200:
                self.logger.error(f"Cannot read {DEPLOYMENTS_URL}, status_code = {resp.status_code}")
                return

            return set([d['vehicle'] for d in resp.json()])

    def _files_from_mission(self):
        name = self.args.mission
        vehicle = self.args.auv_name
        files_url = f"http://portal.shore.mbari.org:8080/auvdata/v1/files/list/{name}/{vehicle}"
        self.logger.debug(f"Getting files list from {files_url}")
        with requests.get(files_url) as resp:
            if resp.status_code != 200:
                self.logger.error(f"Cannot read {files_url}, status_code = {resp.status_code}")
                return

            return resp.json()['names']

    async def _get_file(self, download_url, local_filename, session):
        try:
            async with session.get(download_url, timeout=60) as resp:
                if resp.status != 200:
                    self.logger.warning(f"Cannot read {download_url}, status = {resp.status}")
                else:
                    self.logger.info(f"Started download to {local_filename}...")
                    with open(local_filename, 'wb') as handle:
                        async for chunk in resp.content.iter_chunked(1024):
                            handle.write(chunk)
                        if self.args.verbose > 1:
                            print(f"{os.path.basename(local_filename)}(done) ", end='', flush=True)

        except (ClientConnectorError, concurrent.futures._base.TimeoutError) as e:
            self.logger.error(f"{e}")

    async def _download_files(self, logs_dir):
        name = self.args.mission
        vehicle = self.args.auv_name
        tasks = []
        async with ClientSession() as session:
            for ffm in self._files_from_mission():
                if 'syslog' in ffm:
                    continue
                download_url = f"http://portal.shore.mbari.org:8080/auvdata/v1/files/download/{name}/{vehicle}/{ffm}"
                self.logger.debug(f"Getting file contents from {download_url}")
                Path(logs_dir).mkdir(parents=True, exist_ok=True)
                local_filename = os.path.join(logs_dir, ffm)
                task = asyncio.ensure_future(self._get_file(download_url, local_filename, session))
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
        standard_name = ''
        self.logger.debug(f"Using a rough heuristic to set a standard_name for {long_name}")
        if short_name.lower() == 'time':
            standard_name = 'time'
        elif short_name.lower() == 'temperature':
            standard_name = 'sea_water_temperature'

        return standard_name

    def _create_variable(self, data_type, short_name, long_name, units, data):
        if data_type == 'short':
            nc_data_type = 'h'
        elif data_type == 'integer':
            nc_data_type = 'i'
        elif (data_type == 'float' or data_type == 'timeTag' or data_type == 'double' or
              data_type == 'angle'):
            nc_data_type = 'f8'
        else:
            raise ValueError(f"No conversion for data_type = {data_type}")

        self.logger.debug(f"createVariable {short_name}")
        setattr(self, short_name, self.nc_file.createVariable(short_name, nc_data_type, (TIME,)))
        if (standard_name := self._get_standard_name(short_name, long_name)):
            setattr(getattr(self, short_name), 'standard_name', standard_name)
        setattr(getattr(self, short_name), 'long_name', long_name)
        setattr(getattr(self, short_name), 'units', units)
        getattr(self, short_name)[:] = data

    def write_variables(self, log_data, netcdf_filename):
        name = self.args.mission
        vehicle = self.args.auv_name
        log_data = self._correct_dup_short_names(log_data)
        for variable in log_data:
            self.logger.debug(f"Creating Variable {variable.short_name}: {variable.long_name} ({variable.units})")
            self._create_variable(variable.data_type, variable.short_name, 
                                  variable.long_name, variable.units, variable.data)

    def _process_log_file(self, log_filename, netcdf_filename):
        log_data = readauvlog.read(log_filename)
        self.nc_file = Dataset(netcdf_filename, 'w')
        self.nc_file.createDimension(TIME, len(log_data[0].data))
        self.write_variables(log_data, netcdf_filename)

        # Add the global metadata, overriding with command line options provided
        self.add_global_metadata()
        vehicle = self.args.auv_name
        self.nc_file.title = f"Original AUV {vehicle} data converted straight from the .log file"
        if self.args.title:
            self.nc_file.title = self.args.title
        if self.args.summary:
            self.nc_file.summary = self.args.summary

        self.nc_file.close()

    def process_logs(self):
        name = self.args.mission
        vehicle = self.args.auv_name
        logs_dir = os.path.join(self.args.base_dir, vehicle, MISSIONLOGS, name)

        if not self.args.local:
            self.logger.debug(f"Unique vehicle names: {self._unique_vehicle_names()} seconds")
            if os.path.exists(logs_dir) and not self.args.noinput:
                yes_no = input(f"Directory {logs_dir} exists. Re-download? [Y/n]: ") or 'Y'
                if yes_no.upper().startswith('Y'):
                    d_start = time.time()
                    loop = asyncio.get_event_loop()
                    future = asyncio.ensure_future(self._download_files(logs_dir))
                    loop.run_until_complete(future)
                    self.logger.info(f"Time to download: {(time.time() - d_start):.2f}")

        logs_dir = os.path.join(self.args.base_dir, vehicle, MISSIONLOGS, name)
        netcdfs_dir = os.path.join(self.args.base_dir, vehicle, MISSIONNETCDFS, name)
        Path(netcdfs_dir).mkdir(parents=True, exist_ok=True)
        for log in LOG_FILES:
            log_filename = os.path.join(logs_dir, log)
            netcdf_filename = os.path.join(netcdfs_dir, log.replace('.log', '.nc'))
            self.logger.info(f"Processing {log_filename}")
            self._process_log_file(log_filename, netcdf_filename)

    def process_command_line(self):

        import argparse
        from argparse import RawTextHelpFormatter

        examples = 'Examples:' + '\n\n'
        examples += '  Write to local missionnetcdfs direcory:\n'
        examples += '    ' + sys.argv[0] + " --mission 2020.064.10 \n"

        parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter,
                                         description='Convert AUV log file to a NetCDF files',
                                         epilog=examples)

        parser.add_argument('--base_dir', action='store', default='.', help="Base directory for missionlogs and missionnetcdfs, default: .")
        parser.add_argument('--auv_name', action='store', default='Dorado389', help="Dorado389 (default), i2MAP, or Multibeam")
        parser.add_argument('--mission', action='store', required=True, help="Mission directory, e.g.: 2020.064.10")
        parser.add_argument('--local', action='store_true', help="Specify if files are local in the MISSION directory")

        parser.add_argument('--title', action='store', help='A short description of the dataset')
        parser.add_argument('--summary', action='store', help='Additional information about the dataset')

        parser.add_argument('--noinput', action='store_true', help='Execute without asking for a response, e.g. to not ask to re-download file')        
        parser.add_argument('-v', '--verbose', type=int, choices=range(3), action='store', default=0, 
                            help="verbosity level: " + ', '.join([f"{i}: {v}" for i, v, in enumerate(('WARN', 'INFO', 'DEBUG'))]))

        self.args = parser.parse_args()
        self.logger.setLevel(self._log_levels[self.args.verbose])

        self.commandline = ' '.join(sys.argv)

    
if __name__ == '__main__':

    auv_netcdf = AUV_NetCDF()
    auv_netcdf.process_command_line()
    p_start = time.time()
    auv_netcdf.process_logs()
    auv_netcdf.logger.info(f"Time to process: {(time.time() - p_start):.2f} seconds")
