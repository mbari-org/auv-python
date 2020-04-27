#!/usr/bin/env python
'''
Read original data from netCDF files created by logs2netcdfs.py, apply
calibration information in .cfg and .xml files assiaciated with the 
original .log files and write out a single netCDF file with the 
variables at original sampling intervals.  The file will be analogous
to the original netCDF4 files produced by MBARI's LRAUVs.

Note: The name "sensor" is used here, but it's really more aligned 
      with the concept of "instrument" in SSDS parlance
'''

__author__ = "Mike McCann"
__copyright__ = "Copyright 2020, Monterey Bay Aquarium Research Institute"

import os
import sys
import logging
import requests
import time
import numpy as np
import xarray as xr
from AUV import AUV
from collections import namedtuple
from datetime import datetime
from pathlib import Path
from netCDF4 import Dataset
from logs2netcdfs import LOG_FILES, BASE_PATH, MISSIONLOGS, MISSIONNETCDFS

TIME = 'time'

class Coeffs():
    pass


class SensorInfo():
    pass

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
                        for chunk in resp.content.iter_chunked(1024):
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

    def _define_sensor_info(self, start_datetime):
        '''
        % From defineAUVsensorinfo.m
        if strcmp(sensorname,'navigation');
            data_filename = 'navigation.nc'; cal_filename=[]; sensor_offsets=[]; return
        elseif strcmp(sensorname,'depth');
            data_filename = 'parosci.nc'; cal_filename=[]; sensor_offsets=[-0.927,-0.076]; return
        elseif strcmp(sensorname,'hs2');
            data_filename='hydroscatlog.nc'; sensor_offsets=[.1397,-.2794];
            cal_filename='hs2Calibration.dat';      % Read from missionlogs dir
        elseif strcmp(sensorname,'ctd');
            data_filename=('ctdDriver.nc'); cal_filename='ctdDriver.cfg';
            sensor_offsets=[1.003,0.0001]; %% Serves the purpose for T and OBS
            %% O2 and Nitrate (on CTD 1 circuit) are aligned to temperature by plumbing lag only
        elseif strcmp(sensorname,'ctd2');
            data_filename=('ctdDriver2.nc'); cal_filename='ctdDriver2.cfg';
            sensor_offsets=[1.003,0.0001];
        elseif strcmp(sensorname,'isus');
            data_filename = ('isuslog.nc'); cal_filename=[];
            sensor_offsets=[6];   % Estimated plumbing lag in seconds; needs improvement to use flow
        elseif strcmp(sensorname,'gps');
            data_filename='gps.nc'; cal_filename=[]; sensor_offsets=[];
        elseif strcmp(sensorname,'biolume');
            data_filename = ('biolume.nc');
            cal_filename=[];
            if (yr < 2003)
                disp(['Using sensor offsets for SPOKES 2002']);
            sensor_offsets = [-.889, -.0508];   % SPOKES 2002 (X = -35 in?, Y = -2 in)
            else
            sensor_offsets = [-.1778, -.0889];  % AOSNII, SPED, SPES (X = -7 in, Y = -3.5 in);
            end
        elseif strcmp(sensorname,'lopc');
            data_filename = ('lopc.nc'); cal_filename=[]; sensor_offsets=[]; return
        end
        '''
        # Horizontal and vertical distance from origin in meters
        SensorOffset = namedtuple('SensorOffset', 'x y')

        # Original configuration of Dorado389 - Modify below with changes over time
        self.sinfo =  {'navigation': {'data_filename': 'navigation.nc',
                                      'cal_filename':  None,
                                      'lag_secs':      None,
                                      'sensor_offset': None},
                       'depth':      {'data_filename': 'parosci.nc',
                                      'cal_filename':  None,
                                      'lag_secs':      None,
                                      'sensor_offset': SensorOffset(-0.927, -0.076)},
                       'hs2':        {'data_filename': 'hydroscatlog.nc',
                                      'cal_filename':  'hs2Calibration.dat',
                                      'lag_secs':      None,
                                      'sensor_offset': SensorOffset(.1397, -.2794)},
                       'ctd':        {'data_filename': 'ctdDriver.nc',
                                      'cal_filename':  'ctdDriver.cfg',
                                      'lag_secs':      None,
                                      'sensor_offset': SensorOffset(1.003, 0.0001)},
                       'ctd2':       {'data_filename': 'ctdDriver2.nc',
                                      'cal_filename':  'ctdDriver2.cfg',
                                      'lag_secs':      None,
                                      'sensor_offset': SensorOffset(1.003, 0.0001)},
                       'seabird25p': {'data_filename': 'seabird25p.nc',
                                      'cal_filename':  'seabird25p.cfg',
                                      'lag_secs':      None,
                                      'sensor_offset': SensorOffset(1.003, 0.0001)},
                       'isus':       {'data_filename': 'isuslog.nc',
                                      'cal_filename':  None,
                                      'lag_secs':      6,
                                      'sensor_offset': None},
                       'gps':        {'data_filename': 'gps.nc',
                                      'cal_filename':  None,
                                      'lag_secs':      None,
                                      'sensor_offset': None},
                       'biolume':    {'data_filename': 'biolume.nc',
                                      'cal_filename':  None,
                                      'lag_secs':      None,
                                      'sensor_offset': SensorOffset(-.889, -.0508)},
                       'lopc':       {'data_filename': 'lopc.nc',
                                      'cal_filename':  None,
                                      'lag_secs':      None,
                                      'sensor_offset': None}
                      }

        # Changes over time
        if start_datetime.year >= 2003:
            self.sinfo['biolume']['sensor_offset'] = SensorOffset(1.003, 0.0001)

    def _read_data(self, logs_dir, netcdfs_dir):
        '''Read in all the instrument data into member variables named by "sensor"
        Access xarray.Dataset like: self.ctd.data, self.navigation.data
        Access calibration coeeficients like: self.ctd.cals.t_f0
        '''
        for sensor, info in self.sinfo.items():
            sensor_info = SensorInfo()
            orig_netcdf_filename = os.path.join(netcdfs_dir, info['data_filename'])
            self.logger.debug(f"Reading data from {orig_netcdf_filename} into self.{sensor}.data")
            try:
                setattr(sensor_info, 'data', xr.open_dataset(orig_netcdf_filename))
            except FileNotFoundError as e:
                self.logger.debug(f"{e}")
            if info['cal_filename']:
                cal_filename = os.path.join(logs_dir, info['cal_filename'])
                self.logger.debug(f"Reading calibrations from {orig_netcdf_filename} into self.{sensor}.cals")
                try:
                    setattr(sensor_info, 'cals', self._read_cfg(cal_filename))
                except FileNotFoundError as e:
                    self.logger.debug(f"{e}")

            setattr(self, sensor, sensor_info)
    
    def _read_cfg(self, cfg_filename):
        '''Emulate what get_auv_cal.m and processCTD.m do in the Matlab doradosdp toolbox
        '''
        self.logger.debug(f"Opening {cfg_filename}")
        coeffs = Coeffs()
        with open(cfg_filename) as fh:
            for line in fh:
                self.logger.debug(line)
                # From get_auv_cal.m
                if line[:2] in ('t_','c_','ep','SO','BO','Vo','TC','PC','Sc','Da'):
                    coeff, value = [s.strip() for s in line.split('=')]
                    try:
                        setattr(coeffs, coeff, float(value.replace(';','')))
                    except ValueError as e:
                        self.logger.debug(f"{e}")

        return coeffs

    def _ctd_calibrate(self, sensor, cf):
        # From processCTD.m:
        # TC = 1./(t_a + t_b*(log(t_f0./temp_frequency)) + t_c*((log(t_f0./temp_frequency)).^2) + t_d*((log(t_f0./temp_frequency)).^3)) - 273.15;
        try:
            nc = getattr(self, sensor).data
        except FileNotFoundError as e:
            self.logger.error(f"{e}")
            return

        # Seabird specific calibration
        nc['temp_frequency'][nc['temp_frequency'] == 0.0] = np.nan
        K2C = 273.15
        TC = (1.0 / 
                (cf.t_a + 
                 cf.t_b * np.log(cf.t_f0 / nc['temp_frequency'].values) + 
                 cf.t_c * np.power(np.log(cf.t_f0 / nc['temp_frequency']),2) + 
                 cf.t_d * np.power(np.log(cf.t_f0 / nc['temp_frequency']),3)
                ) - K2C)
        breakpoint()
        pass

        '''
        %% Compute depth for temperature sensor with geometric correction
        cpitch=interp1(Nav.time,Nav.pitch,time);    %find the pitch(time)
        cdepth=interp1(Dep.time,Dep.data,time);     %find reference depth(time)
        zoffset=align_geom(sensor_offsets,cpitch);  %calculate offset from 0,0
        depth=cdepth-zoffset;                       %Find True depth of sensor

        % Output structured array
        CTD.temperature=TC;
        CTD.time=time;
        CTD.Tdepth=depth;  % depth of temperature sensor
        '''



    def _apply_calibration(self, sensor, netcdfs_dir):
        try:
            coeffs = getattr(self, sensor).cals
        except AttributeError as e:
            self.logger.debug(f"{sensor}: {e}")
        else:
            self.logger.info(f"Applying calibrations for {sensor}")
            if sensor in ('ctdDriver', 'seabird25p'):
                self._ctd_calibrate(sensor, coeffs)

    def _write_netcdf(self):
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
        start_datetime = datetime.strptime('.'.join(name.split('.')[:2]), "%Y.%j")
        self._define_sensor_info(start_datetime)
        self._read_data(logs_dir, netcdfs_dir)
        for sensor in self.sinfo.keys():
            self._apply_calibration(sensor, netcdfs_dir)

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

    cal_netcdf = Calibrated_NetCDF()
    cal_netcdf.process_command_line()
    p_start = time.time()
    cal_netcdf.calibrate_and_write()
    cal_netcdf.logger.info(f"Time to process: {(time.time() - p_start):.2f} seconds")
