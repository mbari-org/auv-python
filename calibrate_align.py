#!/usr/bin/env python
'''
Read original data from netCDF files created by logs2netcdfs.py, apply
calibration information in .cfg and .xml files associated with the 
original .log files and write out a single netCDF file with the important
variables at original sampling intervals.  The file will be analogous
to the original netCDF4 files produced by MBARI's LRAUVs.

Note: The name "sensor" is used here, but it's really more aligned 
      with the concept of "instrument" in SSDS parlance
'''

__author__ = "Mike McCann"
__copyright__ = "Copyright 2020, Monterey Bay Aquarium Research Institute"

import altair as ar
import coards
import logging
import requests
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import scipy
import sys
import xarray as xr
from collections import namedtuple, OrderedDict
from datetime import datetime
from pathlib import Path
from netCDF4 import Dataset
from scipy.interpolate import interp1d
from seawater import eos80
from socket import gethostname
from logs2netcdfs import BASE_PATH, MISSIONLOGS, MISSIONNETCDFS

TIME = 'time'

class Coeffs():
    pass


class SensorInfo():
    pass


class Calibrated_NetCDF():

    logger = logging.getLogger(__name__)
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter('%(levelname)s %(asctime)s %(filename)s '
                                  '%(funcName)s():%(lineno)d %(message)s')
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    _log_levels = (logging.WARN, logging.INFO, logging.DEBUG)

    def global_metadata(self):
        '''Use instance variables to return a dictionary of 
        metadata specific for the data that are written
        '''
        iso_now = datetime.utcnow().isoformat() + 'Z'

        metadata = {}
        metadata['netcdf_version'] = '4'
        metadata['Conventions'] = 'CF-1.6'
        metadata['date_created'] = iso_now
        metadata['date_update'] = iso_now
        metadata['date_modified'] = iso_now
        metadata['featureType'] = 'trajectory'

        metadata['time_coverage_start'] = str(self.combined_nc['depth_time']
                                                  .to_pandas()[0].isoformat())
        metadata['time_coverage_end'] = str(self.combined_nc['depth_time']
                                                .to_pandas()[-1].isoformat())
        metadata['distribution_statement'] = 'Any use requires prior approval from MBARI'
        metadata['license'] = metadata['distribution_statement']
        metadata['useconst'] = "Not intended for legal use. Data may contain inaccuracies."
        metadata['history'] = f"Created by {self.commandline} on {iso_now}"

        metadata['title'] = (f"Calibrated and aligned AUV sensor data from"
                             f" {self.args.auv_name} mission {self.args.mission}")
        metadata['summary'] = (f"Observational oceanographic data obtained from an Autonomous"
                               f" Underwater Vehicle mission with measurements at"
                               f" original sampling intervals. The data have been calibrated "
                               f" and aligned by MBARI's auv-python software.")
        metadata['comment'] = (f"MBARI Dorado-class AUV data produced from original data"
                               f" with execution of '{self.commandline}'' at {iso_now} on"
                               f" host {gethostname()}. Software available at"
                               f" 'https://bitbucket.org/mbari/auv-python'")

        return metadata

    def _get_file(self, download_url, local_filename, session):
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

    def _define_sensor_info(self, start_datetime):
        # Horizontal and vertical distance from origin in meters
        SensorOffset = namedtuple('SensorOffset', 'x y')

        # Original configuration of Dorado389 - Modify below with changes over time
        self.sinfo =  OrderedDict([
                       ('navigation', {'data_filename': 'navigation.nc',
                                      'cal_filename':  None,
                                      'lag_secs':      None,
                                      'sensor_offset': None}),
                       ('depth',      {'data_filename': 'parosci.nc',
                                      'cal_filename':  None,
                                      'lag_secs':      None,
                                      'sensor_offset': SensorOffset(-0.927, -0.076)}),
                       ('hs2',        {'data_filename': 'hydroscatlog.nc',
                                      'cal_filename':  'hs2Calibration.dat',
                                      'lag_secs':      None,
                                      'sensor_offset': SensorOffset(.1397, -.2794)}),
                       ('ctd',        {'data_filename': 'ctdDriver.nc',
                                      'cal_filename':  'ctdDriver.cfg',
                                      'lag_secs':      None,
                                      'sensor_offset': SensorOffset(1.003, 0.0001)}),
                       ('ctd2',       {'data_filename': 'ctdDriver2.nc',
                                      'cal_filename':  'ctdDriver2.cfg',
                                      'lag_secs':      None,
                                      'sensor_offset': SensorOffset(1.003, 0.0001)}),
                       ('seabird25p', {'data_filename': 'seabird25p.nc',
                                      'cal_filename':  'seabird25p.cfg',
                                      'lag_secs':      None,
                                      'sensor_offset': SensorOffset(1.003, 0.0001)}),
                       ('isus',       {'data_filename': 'isuslog.nc',
                                      'cal_filename':  None,
                                      'lag_secs':      6,
                                      'sensor_offset': None}),
                       ('gps',        {'data_filename': 'gps.nc',
                                      'cal_filename':  None,
                                      'lag_secs':      None,
                                      'sensor_offset': None}),
                       ('biolume',    {'data_filename': 'biolume.nc',
                                      'cal_filename':  None,
                                      'lag_secs':      None,
                                      'sensor_offset': SensorOffset(-.889, -.0508)}),
                       ('lopc',       {'data_filename': 'lopc.nc',
                                      'cal_filename':  None,
                                      'lag_secs':      None,
                                      'sensor_offset': None})
        ])

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
            self.logger.debug(f"Reading data from {orig_netcdf_filename} into self.{sensor}.orig_data")
            try:
                setattr(sensor_info, 'orig_data', xr.open_dataset(orig_netcdf_filename))
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

        # TODO: Warn if no data found and if logs2netcdfs.py should be run
    
    def _read_cfg(self, cfg_filename):
        '''Emulate what get_auv_cal.m and processCTD.m do in the Matlab doradosdp toolbox
        '''
        self.logger.debug(f"Opening {cfg_filename}")
        coeffs = Coeffs()
        with open(cfg_filename) as fh:
            for line in fh:
                ##self.logger.debug(line)
                # From get_auv_cal.m
                if line[:2] in ('t_','c_','ep','SO','BO','Vo','TC','PC','Sc','Da'):
                    coeff, value = [s.strip() for s in line.split('=')]
                    try:
                        self.logger.debug(f"Saving {line}")
                        setattr(coeffs, coeff, float(value.replace(';','')))
                    except ValueError as e:
                        self.logger.debug(f"{e}")

        return coeffs

    def _calibrated_temp_from_frequency(self, cf, nc):
        K2C = 273.15
        calibrated_temp = (1.0 / 
                (cf.t_a + 
                 cf.t_b * np.log(cf.t_f0 / nc['temp_frequency'].values) + 
                 cf.t_c * np.power(np.log(cf.t_f0 / nc['temp_frequency']),2) + 
                 cf.t_d * np.power(np.log(cf.t_f0 / nc['temp_frequency']),3)
                ) - K2C)

        return calibrated_temp

    def _calibrated_sal_from_cond_frequency(self, cf, nc, temp, depth):
        # Note that recalculation of conductivity and correction for thermal mass
        # are possible, however, their magnitude results in salinity differences
        # of less than 10^-4.  
        # In other regions where these corrections are more significant, the
        # corrections can be turned on.
        # conductivity at S=35 psu , T=15 C [ITPS 68] and P=0 db) ==> 42.914
        sw_c3515 = 42.914
        eps = np.spacing(1)

        ##p1 = 10 * (interp1(Dep.time,Dep.fltpres,time))    # pressure in db
        f_interp = interp1d(self.combined_nc['depth_time'].values.tolist(), 
                            self.combined_nc['filt_pres'].values,
                            fill_value="extrapolate")
        p1 = f_interp(nc['time'].values.tolist())
                    
        cfreq = nc['cond_frequency'].values / 1000.0

        c1 = (cf.c_a * np.power(cfreq, cf.c_m) +
              cf.c_b * np.power(cfreq, 2) +
              cf.c_c + 
              cf.c_d * temp.values) / (10 * (1 + eps * p1))

        cratio = c1 * 10 / sw_c3515
        calibrated_salinity = eos80.salt(cratio, temp, p1)

        return calibrated_salinity


    def _ctd_calibrate(self, sensor, cf):
        # From processCTD.m:
        # TC = 1./(t_a + t_b*(log(t_f0./temp_frequency)) + t_c*((log(t_f0./temp_frequency)).^2) + t_d*((log(t_f0./temp_frequency)).^3)) - 273.15;
        try:
            orig_nc = getattr(self, sensor).orig_data
        except FileNotFoundError as e:
            self.logger.error(f"{e}")
            return

        # Need to do this zeroth-level QC to calibrate temperature
        orig_nc['temp_frequency'][orig_nc['temp_frequency'] == 0.0] = np.nan

        # Seabird specific calibrations
        temperature = xr.DataArray(self._calibrated_temp_from_frequency(cf, orig_nc),
                                  coords=[orig_nc.get_index('time')],
                                  dims={f"{sensor}_time"},
                                  name="temperature")

        salinity = xr.DataArray(self._calibrated_sal_from_cond_frequency(cf, orig_nc,
                                temperature, self.combined_nc['filt_depth']),
                                coords=[orig_nc.get_index('time')],
                                dims={f"{sensor}_time"},
                                name="salinity")

        self.combined_nc['temperatue'] = temperature
        self.combined_nc['salinity'] = salinity
        
        # Other variables that may be in the original data

        # Salinity
        '''
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Salinity
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Note that recalculation of conductivity and correction for thermal mass
        %% are possible, however, their magnitude results in salinity differences
        %% of less than 10^-4.  
        %% In other regions where these corrections are more significant, the
        %% corrections can be turned on.
        p1=10*(interp1(Dep.time,Dep.fltpres,time));  %% pressure in db

        % Always calculate conductivity from cond_frequency
        do_thermal_mass_calc=0;		% Has a negligable effect
        if do_thermal_mass_calc;
            %% Conductivity Calculation
            cfreq=cond_frequency/1000;
            Cuncorrected = (c_a*(cfreq.^c_m)+c_b*(cfreq.^2)+c_c+c_d*TC)./(10*(1+eps*p1));
            
            % correct conductivity for cell thermal mass (see Seabird documentation for explanation of equations)
            sampint  = 0.25;  % Sampling interval for CT sensors
            Tmp.data = TC;
            alphac = 0.04;   % constant for conductivity thermal mass calculation
            betac  = 1/8.0;  % constant for conductivity thermal mass calculation
            ctm1(1) = 0;
            for i = 1:(length(Cuncorrected)-1)
                ctm1(i+1) = (-1.0*(1-(2*(2*alphac/(sampint*betac+2))/alphac))*ctm1(i)) + ...
                    (2*(alphac/(sampint*betac+2))*(0.1*(1+0.006*(Tmp.data(i)-20)))*(Tmp.data(i+1)-Tmp.data(i)));
            end
            c1 = Cuncorrected + ctm1'; % very, very small correction. +/-0.0005
        else
            %% Conductivity Calculation
            cfreq=cond_frequency/1000;
            c1 = (c_a*(cfreq.^c_m)+c_b*(cfreq.^2)+c_c+c_d*TC)./(10*(1+eps*p1));

            %%c1=conductivity;		% This uses conductivity as calculated on the vehicle with the cal
                        % params that were in the .cfg file at the time.  Not what we want.
        end

        % Calculate Salinty
        cratio = c1*10/sw_c3515; % sw_C is conductivity value at 35,15,0
        CTD.salinity = sw_salt(cratio,CTD.temperature,p1); % (psu)

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

    def _filter_depth(self, sensor, latitude=36, cutoff_freq=1):
        '''Depth data (from the Parosci) is 10 Hz - Use a butterworth window
        to filter recorded pressure to values that are appropriately sampled
        at 1 Hz (when matched with other sensor data).  cutoff_freq is in
        units of Hz.
        '''
        try:
            orig_nc = getattr(self, sensor).orig_data
        except (FileNotFoundError, AttributeError) as e:
            self.logger.error(f"{e}")
            return

        # From initial CVS commit in 2004 the processDepth.m file computed
        # pres from depth this way.  I don't know what is done on the vehicle 
        # side where a latitude of 36 is not appropriate: GoM, SoCal, etc.
        self.logger.debug(f"Converting depth to pressure using latitude = {latitude}")
        pres = eos80.pres(orig_nc['depth'], latitude)
        
        # See https://docs.scipy.org/doc/scipy-1.0.0/reference/generated/scipy.signal.filtfilt.html#scipy.signal.filtfilt
        # and https://docs.scipy.org/doc/scipy-1.0.0/reference/generated/scipy.signal.butter.html#scipy.signal.butter
        # Sample rate should be 10 - calcuate it to be sure
        sample_rate = 1. / \
            np.round(
                np.mean(np.diff(orig_nc['time'])) / np.timedelta64(1, 's'), decimals=2)
        if sample_rate != 10:
            self.logger.warning(f"Expected sample_rate to be 10 Hz, instead it's {sample_rate} Hz")

        # The Wn parameter for butter() is fraction of the Nyquist frequency
        Wn = cutoff_freq / (sample_rate / 2.)
        b, a = scipy.signal.butter(8, Wn)
        filt_pres_butter = scipy.signal.filtfilt(b, a, pres)
        filt_depth_butter = scipy.signal.filtfilt(b, a, orig_nc['depth'])

        # Use 10 points in boxcar as in processDepth.m
        a = 10
        b = scipy.signal.boxcar(a)
        filt_pres_boxcar = scipy.signal.filtfilt(b, a, pres)
        if self.args.plots:
            # Use Pandas to plot multiple columns of a subset (npts) of data
            # to validate that the filtering works as expected
            npts = 2000
            df_plot = pd.DataFrame(index=orig_nc.get_index('time')[:npts])
            df_plot['pres'] = pres[:npts]
            df_plot['filt_pres_butter'] = filt_pres_butter[:npts]
            df_plot['filt_pres_boxcar'] = filt_pres_boxcar[:npts]
            title = (f"First {npts} points from"
                     f" {self.args.mission}/{self.sinfo[sensor]['data_filename']}")
            ax = df_plot.plot(title=title)
            ax.grid('on')
            plt.show()

        filt_depth = xr.DataArray(filt_depth_butter,
                                  coords=[orig_nc.get_index('time')],
                                  dims={f"{sensor}_time"},
                                  name="filt_depth")
        filt_pres = xr.DataArray(filt_pres_butter,
                                 coords=[orig_nc.get_index('time')],
                                 dims={f"{sensor}_time"},
                                 name="filt_pres")

        self.combined_nc['filt_depth'] = filt_depth
        self.combined_nc['filt_pres'] = filt_pres

    def _apply_calibration(self, sensor, netcdfs_dir):
        coeffs = None
        try:
            coeffs = getattr(self, sensor).cals
        except AttributeError as e:
            self.logger.debug(f"No calibration information for {sensor}: {e}")

        if sensor in ('depth',):
            self._filter_depth(sensor)
        if sensor in ('ctd',) and coeffs:
            self._ctd_calibrate(sensor, coeffs)
            pass

    def _write_netcdf(self, netcdfs_dir):
        self.combined_nc.attrs = self.global_metadata()
        out_fn = os.path.join(netcdfs_dir, f"{self.args.auv_name}_{self.args.mission}.nc")
        self.logger.info(f"Writing calibrated and aligned data to file {out_fn}")
        self.combined_nc.to_netcdf(out_fn)

    def calibrate_and_write(self):
        name = self.args.mission
        vehicle = self.args.auv_name
        logs_dir = os.path.join(self.args.base_path, vehicle, MISSIONLOGS, name)
        netcdfs_dir = os.path.join(self.args.base_path, vehicle, MISSIONNETCDFS, name)
        start_datetime = datetime.strptime('.'.join(name.split('.')[:2]), "%Y.%j")
        self._define_sensor_info(start_datetime)
        self._read_data(logs_dir, netcdfs_dir)
        self.combined_nc = xr.Dataset()

        for sensor in self.sinfo.keys():
            setattr(getattr(self, sensor), 'cal_align_data', xr.Dataset())
            self._apply_calibration(sensor, netcdfs_dir)

        self._write_netcdf(netcdfs_dir)

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
        parser.add_argument('--noinput', action='store_true', help='Execute without asking for a response, e.g. to not ask to re-download file')        
        parser.add_argument('--plots', action='store_true', help='Create intermediate plots to validate data opaerations - program blocks upon show')        
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
