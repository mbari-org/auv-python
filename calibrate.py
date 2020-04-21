#!/usr/bin/env python
'''
Read original data from netCDF files created by logs2netcdfs.py, apply
calibration information in .cfg and .xml files assiaciated with the 
original .log files and write out a single netCDF file with the 
variables at original sampling intervals.  The file will be analogous
to the original netCDF4 files produced by MBARI's LRAUVs.
'''

__author__ = "Mike McCann"
__copyright__ = "Copyright 2020, Monterey Bay Aquarium Research Institute"

import os
import sys
import logging
import requests
import time
from AUV import AUV
from pathlib import Path
from netCDF4 import Dataset
from logs2netcdfs import LOG_FILES, BASE_PATH, MISSIONLOGS, MISSIONNETCDFS

TIME = 'time'

class Calibrated_NetCDF(AUV):

    logger = logging.getLogger(__name__)
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter('%(levelname)s %(asctime)s %(filename)s '
                                  '%(funcName)s():%(lineno)d %(message)s')
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    _log_levels = (logging.WARN, logging.INFO, logging.DEBUG)

    def __init__(self):
        return super(Calibrated_NetCDF, self).__init__()


    def _get_file(self, download_url, local_filename, session):
        try:
            with session.get(download_url, timeout=60) as resp:
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

    def _apply_calibration(self, base_filename):
        orig_netcdf_filename = f"{base_filename}.nc"
        cfg_filename = f"{base_filename}.cfg"
        xml_filename = f"{base_filename}.xml"

        self.nc_file = Dataset(netcdf_filename, 'w')
        self.nc_file.createDimension(f"{TIME}_{base_filename}", len(log_data[0].data))
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

    def calibrate_and_write(self):
        name = self.args.mission
        vehicle = self.args.auv_name
        logs_dir = os.path.join(self.args.base_path, vehicle, MISSIONLOGS, name)

        netcdfs_dir = os.path.join(self.args.base_path, vehicle, MISSIONNETCDFS, name)
        for log in LOG_FILES:
            log_filename = os.path.join(logs_dir, log)
            base_filename = log_filename.replace('.log', '')

            self.logger.info(f"Applying calibrations for {orig_netcdf_filename}")
            self._apply_calibration(base_filename)

    def process_command_line(self):

        import argparse
        from argparse import RawTextHelpFormatter

        examples = 'Examples:' + '\n\n'
        examples += '  Calibrate original data for some missions:\n'
        examples += '    ' + sys.argv[0] + " --mission 2020.064.10\n"
        examples += '    ' + sys.argv[0] + " --auv_name i2map --mission 2020.055.01\n"

        parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter,
                                         description='Calibrate original data and produce NetCDF file for mission - all data locally stored',
                                         epilog=examples)

        parser.add_argument('--base_path', action='store', default=BASE_PATH, help="Base directory for missionlogs and missionnetcdfs, default: auv_data")
        parser.add_argument('--auv_name', action='store', default='Dorado389', help="Dorado389 (default), i2MAP, or Multibeam")
        parser.add_argument('--mission', action='store', required=True, help="Mission directory, e.g.: 2020.064.10")

        parser.add_argument('--title', action='store', help='A short description of the dataset to be written to the netCDF file')
        parser.add_argument('--summary', action='store', help='Additional information about the dataset to be written to the netCDF file')

        parser.add_argument('--noinput', action='store_true', help='Execute without asking for a response, e.g. to not ask to re-download file')        
        parser.add_argument('-v', '--verbose', type=int, choices=range(3), action='store', default=0, const=1, nargs='?',
                            help="verbosity level: " + ', '.join([f"{i}: {v}" for i, v, in enumerate(('WARN', 'INFO', 'DEBUG'))]))

        self.args = parser.parse_args()
        self.logger.setLevel(self._log_levels[self.args.verbose])

        self.commandline = ' '.join(sys.argv)

    
if __name__ == '__main__':

    cal_netcdf = Calibrate_NetCDF()
    cal_netcdf.process_command_line()
    cal_netcdf.calibrate_and_write()
    cal_netcdf.logger.info(f"Time to process: {(time.time() - p_start):.2f} seconds")
