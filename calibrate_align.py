#!/usr/bin/env python
'''
Read original data from netCDF files created by logs2netcdfs.py, apply
calibration information in .cfg and .xml files associated with the 
original .log files and write out a single netCDF file with the important
variables at original sampling intervals. Alignment and plumbing lag
corrections are also done during this step. The file will contain combined
variables (the combined_nc member variable) and be analogous to the original
netCDF4 files produced by MBARI's LRAUVs.

Note: The name "sensor" is used here, but it's really more aligned 
      with the concept of "instrument" in SSDS parlance
'''

__author__ = "Mike McCann"
__copyright__ = "Copyright 2020, Monterey Bay Aquarium Research Institute"

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
from ctd_proc import (_calibrated_sal_from_cond_frequency, 
                      _calibrated_temp_from_frequency)
from datetime import datetime
from hs2_proc import hs2_read_cal_file, hs2_calc_bb
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


class CalAligned_NetCDF():

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
                       ('gps',        {'data_filename': 'gps.nc',
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
                       ('biolume',    {'data_filename': 'biolume.nc',
                                      'cal_filename':  None,
                                      'lag_secs':      None,
                                      'sensor_offset': SensorOffset(-.889, -.0508)}),
                       ('lopc',       {'data_filename': 'lopc.nc',
                                      'cal_filename':  None,
                                      'lag_secs':      None,
                                      'sensor_offset': None}),
                       ('ecopuck',    {'data_filename': 'FLBBCD2K.nc',
                                      'cal_filename':  'FLBBCD2K-3695.dev',
                                      'lag_secs':      None,
                                      'sensor_offset': None}),
        ])

        # Changes over time
        if start_datetime.year >= 2003:
            self.sinfo['biolume']['sensor_offset'] = SensorOffset(1.003, 0.0001)
        # ...


    def _read_data(self, logs_dir, netcdfs_dir):
        '''Read in all the instrument data into member variables named by "sensor"
        Access xarray.Dataset like: self.ctd.data, self.navigation.data, ...
        Access calibration coefficients like: self.ctd.cals.t_f0, or as a
        dictionary for hs2 data.
        '''
        for sensor, info in self.sinfo.items():
            sensor_info = SensorInfo()
            orig_netcdf_filename = os.path.join(netcdfs_dir, info['data_filename'])
            self.logger.debug(f"Reading data from {orig_netcdf_filename}"
                              f" into self.{sensor}.orig_data")
            try:
                setattr(sensor_info, 'orig_data', xr.open_dataset(orig_netcdf_filename))
            except FileNotFoundError as e:
                self.logger.warning(f"{sensor:10}: Cannot open file"
                                    f" {orig_netcdf_filename}")
            if info['cal_filename']:
                cal_filename = os.path.join(logs_dir, info['cal_filename'])
                self.logger.debug(f"Reading calibrations from {orig_netcdf_filename}"
                                  f" into self.{sensor}.cals")
                if cal_filename.endswith('.cfg'):
                    try:
                        setattr(sensor_info, 'cals', self._read_cfg(cal_filename))
                    except FileNotFoundError as e:
                        self.logger.debug(f"{e}")

            setattr(self, sensor, sensor_info)

        # TODO: Warn if no data found and if logs2netcdfs.py should be run
    
    def _read_cfg(self, cfg_filename):
        '''Emulate what get_auv_cal.m and processCTD.m do in the 
           Matlab doradosdp toolbox
        '''
        self.logger.debug(f"Opening {cfg_filename}")
        coeffs = Coeffs()
        with open(cfg_filename) as fh:
            for line in fh:
                ##self.logger.debug(line)
                # From get_auv_cal.m - Handle CTD calibration parameters
                if line[:2] in ('t_','c_','ep','SO','BO','Vo','TC','PC','Sc','Da'):
                    coeff, value = [s.strip() for s in line.split('=')]
                    try:
                        self.logger.debug(f"Saving {line}")
                        setattr(coeffs, coeff, float(value.replace(';','')))
                    except ValueError as e:
                        self.logger.debug(f"{e}")

        return coeffs

    def _navigation_process(self, sensor):
        # AUV navigation data, which comes from a process on the vehicle that
        # integrates data from several instruments.  We use it to grab the DVL
        # data to help determine vehicle position when it is below the surface.
        # 
        #  Nav.depth is used to compute pressure for salinity and oxygen computations
        #  Nav.latitude and Nav.longitude converted to degrees were added to
        #                                 the log file at end of 2004
        #  Nav.roll, Nav.pitch, Nav.yaw, Nav.Xpos and Nav.Ypos are extracted for
        #                                 3-D mission visualization
        try:
            orig_nc = getattr(self, sensor).orig_data
        except FileNotFoundError as e:
            self.logger.error(f"{e}")
            return

        # Nav.time  = time;
        # Nav.roll = mPhi;
        # Nav.pitch = mTheta;
        # Nav.yaw = mPsi;
        # Nav.depth = mDepth;
        # Nav.posx  = mPos_x-mPos_x(1);
        # Nav.posy  = mPos_y-mPos_y(1);
        # if ( exist('latitude') )
        #     Nav.latitude = latitude * 180 / pi;
        # end
        # if ( exist('longitude') )
        #     Nav.longitude = longitude * 180 / pi;
        # end
        source = self.sinfo[sensor]['data_filename']
        self.combined_nc['roll'] = xr.DataArray(orig_nc['mPhi'].values,
                                    coords=[orig_nc.get_index('time')],
                                    dims={f"{sensor}_time"},
                                    name="roll")
        self.combined_nc['roll'].attrs = {'long_name': 'Vehicle roll',
                                          'standard_name': 'platform_roll_angle',
                                          'units': 'degree',
                                          'comment': f"mPhi from {source}"}

        self.combined_nc['pitch'] = xr.DataArray(orig_nc['mTheta'].values,
                                    coords=[orig_nc.get_index('time')],
                                    dims={f"{sensor}_time"},
                                    name="pitch")
        self.combined_nc['pitch'].attrs = {'long_name': 'Vehicle pitch',
                                           'standard_name': 'platform_pitch_angle',
                                           'units': 'degree',
                                           'comment': f"mTheta from {source}"}

        self.combined_nc['yaw'] = xr.DataArray(orig_nc['mPsi'].values,
                                    coords=[orig_nc.get_index('time')],
                                    dims={f"{sensor}_time"},
                                    name="yaw")
        self.combined_nc['yaw'].attrs = {'long_name': 'Vehicle yaw',
                                         'standard_name': 'platform_yaw_angle',
                                         'units': 'degree',
                                         'comment': f"mPsi from {source}"}

        self.combined_nc['nav_depth'] = xr.DataArray(orig_nc['mDepth'].values,
                                    coords=[orig_nc.get_index('time')],
                                    dims={f"{sensor}_time"},
                                    name="nav_depth")
        self.combined_nc['nav_depth'].attrs = {'long_name': 'Depth from Nav',
                                         'standard_name': 'depth',
                                         'units': 'm',
                                         'comment': f"mDepth from {source}"}

        self.combined_nc['posx'] = xr.DataArray(orig_nc['mPos_x'].values
                                                - orig_nc['mPos_x'].values[0],
                                    coords=[orig_nc.get_index('time')],
                                    dims={f"{sensor}_time"},
                                    name="posx")
        self.combined_nc['posx'].attrs = {'long_name': 'Relative lateral easting',
                         'units': 'm',
                         'comment': f"mPos_x (minus first position) from {source}"}

        self.combined_nc['posy'] = xr.DataArray(orig_nc['mPos_y'].values
                                                - orig_nc['mPos_y'].values[0],
                                    coords=[orig_nc.get_index('time')],
                                    dims={f"{sensor}_time"},
                                    name="posy")
        self.combined_nc['posy'].attrs = {'long_name': 'Relative lateral northing',
                         'units': 'm',
                         'comment': f"mPos_y (minus first position) from {source}"}

        try:
            self.combined_nc['latitude'] = xr.DataArray(orig_nc['latitude'].values
                                                        * 180 / np.pi, 
                                        coords=[orig_nc.get_index('time')],
                                        dims={f"{sensor}_time"},
                                        name="latitude")
            self.combined_nc['latitude'].attrs = {'long_name': 'latitude',
                             'standard_name': 'latitude',
                             'units': 'degrees_north',
                             'comment': f"latitude (converted from radians) from {source}"}

        except KeyError:
            pass
        try:
            self.combined_nc['longitude'] = xr.DataArray(orig_nc['longitude'].values
                                                        * 180 / np.pi, 
                                        coords=[orig_nc.get_index('time')],
                                        dims={f"{sensor}_time"},
                                        name="longitude")
            self.combined_nc['longitude'].attrs = {'long_name': 'longitude',
                             'standard_name': 'longitude',
                             'units': 'degrees_east',
                             'comment': f"longitude (converted from radians) from {source}"}
        except KeyError:
            pass

        # % Remove obvious outliers that later disrupt the section plots.
        # % (First seen on mission 2008.281.03)
        # % In case we ever use this software for the D Allan B mapping vehicle determine
        # % the good depth range from the median of the depths
        # % From mission 2011.250.11 we need to first eliminate the near surface values before taking the
        # % median.
        # pdIndx = find(Nav.depth > 1);
        # posDepths = Nav.depth(pdIndx);
        pos_depths = np.where(self.combined_nc['nav_depth'].values > 1)
        if self.args.mission == '2013.301.02' or self.args.mission == '2009.111.00':
            print('Bypassing Nav QC depth check')
            maxGoodDepth = 1250
        else:
            maxGoodDepth = 7 * np.median(pos_depths)
            if maxGoodDepth < 0:
                maxGoodDepth = 100 # Fudge for the 2009.272.00 mission where median was -0.1347!
            if self.args.mission == '2010.153.01':
                maxGoodDepth = 1250    # Fudge for 2010.153.01 where the depth was bogus, about 1.3

        self.logger.debug(f"median of positive valued depths = {np.median(pos_depths)}")
        self.logger.debug(f"Finding depths less than '{maxGoodDepth}' and times > 0'")

    def _gps_process(self, sensor):
        try:
            orig_nc = getattr(self, sensor).orig_data
        except FileNotFoundError as e:
            self.logger.error(f"{e}")
            return

        if self.args.mission == '2010.151.04':
            # Gulf of Mexico mission - read from usbl.dat files
            self.logger.info('Cannot read latitude data using load command.  Just for the GoMx mission use USBL instead...');
            #-data_filename = 'usbl.nc'
            #-loaddata
            #-time = time(1:10:end);
            #-lat = latitude(1:10:end);	% Subsample usbl so that iit is like our gps data
            #-lon = longitude(1:10:end);

        lat = orig_nc['latitude'] * 180.0 / np.pi
        if orig_nc['longitude'][0] > 0:
            lon = -1 * orig_nc['longitude'] * 180.0 / np.pi
        else:
            lon = orig_nc['longitude'] * 180.0 / np.pi
        
        # Filter out positions outside of operational box
        if (self.args.mission == '2010.151.04' or 
            self.args.mission == '2010.153.01' or 
            self.args.mission == '2010.154.01'):
            lat_min = 26
            lat_max = 40
            lon_min = -124
            lon_max = -70
        else:
            lat_min = 30            
            lat_max = 40
            lon_min = -124
            lon_max = -114

        self.logger.debug(f"Finding positions outside of longitude: {lon_min},"
                          f" {lon_max} and latitide: {lat_min}, {lat_max}")
        mlat = np.ma.masked_invalid(lat)
        mlat = np.ma.masked_outside(mlat, lat_min, lat_max)
        mlon = np.ma.masked_invalid(lon)
        mlon = np.ma.masked_outside(mlon, lon_min, lon_max)
        pm = np.logical_and(mlat, mlon)
        bad_pos = [f"{lo}, {la}" for lo, la in zip(lon.values[:][pm.mask],
                                                   lat.values[:][pm.mask])]
        if bad_pos:
            self.logger.info(f"Number of bad {sensor} positions:"
                             f" {len(lat.values[:][pm.mask])}")
            self.logger.debug(f"Removing bad {sensor} positions (indices,"
                             f" (lon, lat)): {np.where(pm.mask)[0]}, {bad_pos}")
            self.combined_nc['gps_time'] = orig_nc['time'][:][~pm.mask]
            self.combined_nc['gps_latitude'] = lat[:][~pm.mask]
            self.combined_nc['gps_longitude'] = lon[:][~pm.mask]
        else:
            self.combined_nc['gps_time'] = orig_nc['time']
            self.combined_nc['gps_latitude'] = lat
            self.combined_nc['gps_longitude'] = lon

        if self.args.plot:
            pbeg = 0
            pend = len(self.combined_nc['gps_latitude'])
            if self.args.plot.startswith('first'):
                pend = int(self.args.plot.split('first')[1])
            fig, axes = plt.subplots(nrows=2, figsize=(18,6))            
            axes[0].plot(self.combined_nc['gps_latitude'][pbeg:pend], '-o')
            axes[0].set_ylabel('gps_latitude')
            axes[1].plot(self.combined_nc['gps_longitude'][pbeg:pend], '-o')
            axes[1].set_ylabel('gps_longitude')
            title = "GPS Positions"
            title += f" - First {pend} Points"
            fig.suptitle(title)
            axes[0].grid()
            axes[1].grid()
            self.logger.debug(f"Pausing with plot entitled: {title}."
                               " Close window to continue.")
            plt.show()

    def _depth_process(self, sensor, latitude=36, cutoff_freq=1):
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
        if self.args.plot:
            # Use Pandas to plot multiple columns of data
            # to validate that the filtering works as expected
            pbeg = 0
            pend = len(orig_nc.get_index('time'))
            if self.args.plot.startswith('first'):
                pend = int(self.args.plot.split('first')[1])
            df_plot = pd.DataFrame(index=orig_nc.get_index('time')[pbeg:pend])
            df_plot['pres'] = pres[pbeg:pend]
            df_plot['filt_pres_butter'] = filt_pres_butter[pbeg:pend]
            df_plot['filt_pres_boxcar'] = filt_pres_boxcar[pbeg:pend]
            title = (f"First {pend} points from"
                     f" {self.args.mission}/{self.sinfo[sensor]['data_filename']}")
            ax = df_plot.plot(title=title, figsize=(18,6))
            ax.grid('on')
            self.logger.debug(f"Pausing with plot entitled: {title}."
                               " Close window to continue.")
            plt.show()

        filt_depth = xr.DataArray(filt_depth_butter,
                                  coords=[orig_nc.get_index('time')],
                                  dims={f"{sensor}_time"},
                                  name="filt_depth")
        filt_depth.attrs = {'long_name': 'Filtered Depth',
                            'standard_name': 'depth',
                            'units': 'm'}

        filt_pres = xr.DataArray(filt_pres_butter,
                                 coords=[orig_nc.get_index('time')],
                                 dims={f"{sensor}_time"},
                                 name="filt_pres")
        filt_pres.attrs = {'long_name': 'Filtered Pressure',
                            'standard_name': 'sea_water_pressure',
                            'units': 'dbar'}

        self.combined_nc['filt_depth'] = filt_depth
        self.combined_nc['filt_pres'] = filt_pres

    def _hs2_process(self, sensor, logs_dir):
        try:
            orig_nc = getattr(self, sensor).orig_data
        except (FileNotFoundError, AttributeError) as e:
            self.logger.error(f"{e}")
            return

        try:
            cal_fn = os.path.join(logs_dir, self.sinfo['hs2']['cal_filename'])
            cals = hs2_read_cal_file(cal_fn)
        except FileNotFoundError as e:
            self.logger.error(f"{e}")
            raise()

        hs2 = hs2_calc_bb(orig_nc, cals)

        source = self.sinfo[sensor]['data_filename']

        # Blue backscatter
        if hasattr(hs2, 'bb420'):
            blue_bs = xr.DataArray(hs2.bb420.values,
                                   coords=[hs2.bb420.get_index('time')],
                                   dims={"hs2_time"},
                                   name="hs2_bb420")
            blue_bs.attrs = {'long_name': 'Backscatter at 420 nm',
                             'comment': (f"Computed by hs2_calc_bb()"
                                         f" from data in {source}")}
        if hasattr(hs2, 'bb470'):
            blue_bs = xr.DataArray(hs2.bb470.values,
                                   coords=[hs2.bb470.get_index('time')],
                                   dims={"hs2_time"},
                                   name="hs2_bb470")
            blue_bs.attrs = {'long_name': 'Backscatter at 470 nm',
                             'comment': (f"Computed by hs2_calc_bb()"
                                         f" from data in {source}")}

        # Red backscatter
        if hasattr(hs2, 'bb676'):
            red_bs = xr.DataArray(hs2.bb676.values,
                                  coords=[hs2.bb676.get_index('time')],
                                  dims={"hs2_time"},
                                  name="hs2_bb676")
            red_bs.attrs = {'long_name': 'Backscatter at 676 nm',
                            'comment': (f"Computed by hs2_calc_bb()"
                                        f" from data in {source}")}
        if hasattr(hs2, 'bb700'):
            red_bs = xr.DataArray(hs2.bb700.values,
                                  coords=[hs2.bb700.get_index('time')],
                                  dims={"hs2_time"},
                                  name="hs2_bb700")
            red_bs.attrs = {'long_name': 'Backscatter at 700 nm',
                            'comment': (f"Computed by hs2_calc_bb()"
                                        f" from data in {source}")}

        # Fluoresence
        if hasattr(hs2, 'bb676'):
            fl = xr.DataArray(hs2.bb676.values,
                                 coords=[hs2.bb676.get_index('time')],
                                 dims={"hs2_time"},
                                 name="hs2_bb676")
            fl.attrs = {'long_name': 'Fluoresence at 676 nm',
                           'comment': (f"Computed by hs2_calc_bb()"
                                       f" from data in {source}")}
            self.combined_nc['hs2_bb676'] = bb676
            fl = bb676
        if hasattr(hs2, 'fl700'):
            fl700 = xr.DataArray(hs2.fl700.values,
                                 coords=[hs2.fl700.get_index('time')],
                                 dims={"hs2_time"},
                                 name="hs2_fl700")
            fl700.attrs = {'long_name': 'Fluoresence at 700 nm',
                           'comment': (f"Computed by hs2_calc_bb()"
                                       f" from data in {source}")}
            fl = fl700

        # Zeroeth level quality control
        mblue = np.ma.masked_invalid(blue_bs)
        mblue = np.ma.masked_greater(mblue, 0.1)
        mred = np.ma.masked_invalid(red_bs)
        mred = np.ma.masked_greater(mred, 0.1)
        mfl = np.ma.masked_invalid(fl)
        mfl = np.ma.masked_greater(mfl, 0.02)
        mhs2 = np.logical_and(mblue, np.logical_and(mred, mfl))

        bad_hs2 = [f"{b}, {r}, {f}" for b, r, f in zip(blue_bs.values[:][mhs2.mask],
                                                   red_bs.values[:][mhs2.mask],
                                                   fl.values[:][mhs2.mask])]

        if bad_hs2:
            self.logger.info(f"Number of bad {sensor} points:"
                             f" {len(blue_bs.values[:][mhs2.mask])}"
                             f" of {len(blue_bs)}")
            self.logger.debug(f"Removing bad {sensor} points (indices,"
                             f" (blue, red, fl)): {np.where(mhs2.mask)[0]},"
                             f" {bad_hs2}")
            blue_bs = blue_bs[:][~mhs2.mask]
            red_bs = red_bs[:][~mhs2.mask]

        if self.args.plot:
            # Use Pandas to more easiily plot multiple columns of data
            pbeg = 0
            pend = len(blue_bs.get_index('hs2_time'))
            if self.args.plot.startswith('first'):
                pend = int(self.args.plot.split('first')[1])
            df_plot = pd.DataFrame(index=blue_bs.get_index('hs2_time')[pbeg:pend])
            df_plot['blue_bs'] = blue_bs[pbeg:pend]
            df_plot['red_bs'] = red_bs[pbeg:pend]
            df_plot['fl'] = fl[pbeg:pend]
            title = (f"First {pend} points from"
                     f" {self.args.mission}/{self.sinfo[sensor]['data_filename']}")
            ax = df_plot.plot(title=title, figsize=(18,6))
            ax.grid('on')
            self.logger.debug(f"Pausing with plot entitled: {title}."
                               " Close window to continue.")
            plt.show()

        if hasattr(hs2, 'bb420'):
            self.combined_nc['hs2_bb420'] = blue_bs
        if hasattr(hs2, 'bb470'):
            self.combined_nc['hs2_bb470'] = blue_bs
        if hasattr(hs2, 'bb676'):
            self.rombined_nc['hs2_bb676'] = red_bs
        if hasattr(hs2, 'bb700'):
            self.combined_nc['hs2_bb700'] = red_bs
        if hasattr(hs2, 'fl676'):
            self.combined_nc['hs2_fl676'] = fl
        if hasattr(hs2, 'fl700'):
            self.combined_nc['hs2_fl700'] = fl


        # For missions before 2009.055.05 hs2 will have attributes like bb470, bb676, and fl676
        # Hobilabs modified the instrument in 2009 to now give:      bb420, bb700, and fl700,
        # apparently giving a better measurement of chlorophyl.
        #
        # Detect the difference in this code and keep the mamber names descriptive in the survey data so 
        # the the end user knows the difference.


        #-% Align Geometry, correct for pitch
        p_interp = interp1d(self.combined_nc['navigation_time'].values.tolist(),
                            self.combined_nc['pitch'].values, 
                            fill_value="extrapolate")
        hs2.pitch    = p_interp(orig_nc['time'].values.tolist())
        #-hs2.RefDepth = interp1(Dep.time,Dep.data,hs2.time);     	%find reference depth(time)
        #-hs2.offs     = align_geom(HS2OffS,hs2.pitch);		      	%calculate offset from 0,0
        #-hs2.depth    = hs2.RefDepth-hs2.offs;		      		%Find true depth of sensor

        #-% 0th order Quality control, just set to NaN any unreasonable values
        #-% These limits (0.1 for backscatter & 0.02 for fl676 should be in the metadata for the instrument...)

        #-% Blue
        #-if isfield(hs2, 'bb470'),
        #-    ibad470 = (find(hs2.bbp470 > 0.1));
        #-    hs2.bbp470(ibad470) = NaN;
        #-elseif isfield(hs2, 'bb420'),
        #-    ibad420 = (find(hs2.bbp420 > 0.1));
        #-    hs2.bbp420(ibad420) = NaN;
        #-end

        #-% Red
        #-if isfield(hs2, 'bb676'),
        #-    ibad676 = (find(hs2.bbp676 > 0.1));
        #-    hs2.bbp676(ibad676) = NaN;
        #-elseif isfield(hs2, 'bb700'),
        #-    ibad700 = (find(hs2.bbp700 > 0.1));
        #-    hs2.bbp700(ibad700) = NaN;
        #-end

        #-% Fl
        #-if isfield(hs2, 'bb676'),
        #-    ibadfl= (find(hs2.fl676_uncorr > 0.02));
        #-    hs2.fl676_uncorr(ibadfl) = NaN;
        #-elseif isfield(hs2, 'bb700'),
        #-    ibadfl= (find(hs2.fl700_uncorr > 0.02));
        #-    hs2.fl700_uncorr(ibadfl) = NaN;
        #-end

        return

    def _ctd_process(self, sensor, cf):
        try:
            orig_nc = getattr(self, sensor).orig_data
        except FileNotFoundError as e:
            self.logger.error(f"{e}")
            return

        # Need to do this zeroth-level QC to calibrate temperature
        orig_nc['temp_frequency'][orig_nc['temp_frequency'] == 0.0] = np.nan
        source = self.sinfo[sensor]['data_filename']

        # Seabird specific calibrations
        temperature = xr.DataArray(_calibrated_temp_from_frequency(cf, orig_nc),
                                  coords=[orig_nc.get_index('time')],
                                  dims={f"{sensor}_time"},
                                  name="temperature")
        temperature.attrs = {'long_name': 'Temperature',
                            'standard_name': 'sea_water_temperature',
                            'units': 'degree_Celsius',
                            'comment': (f"Derived from temp_frequency from"
                                        f" {source} via calibration parms:"
                                        f" {cf.__dict__}")}

        salinity = xr.DataArray(_calibrated_sal_from_cond_frequency(self.args, 
                                self.combined_nc, self.logger, cf, orig_nc,
                                temperature, self.combined_nc['filt_depth']),
                                coords=[orig_nc.get_index('time')],
                                dims={f"{sensor}_time"},
                                name="salinity")
        salinity.attrs = {'long_name': 'Salinity',
                            'standard_name': 'sea_water_salinity',
                            'units': '',
                            'comment': (f"Derived from cond_frequency from"
                                        f" {source} via calibration parms:"
                                        f" {cf.__dict__}")}

        self.combined_nc['temperatue'] = temperature
        self.combined_nc['salinity'] = salinity

        self._ctd_depth_geometric_correction(sensor, cf)

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
        do_thermal_mass_calc=0;    % Has a negligable effect
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

            %%c1=conductivity;    % This uses conductivity as calculated on the vehicle with the cal
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
    def _ctd_depth_geometric_correction(self, sensor, cf):
        # %% Compute depth for temperature sensor with geometric correction
        # cpitch=interp1(Nav.time,Nav.pitch,time);    %find the pitch(time)
        # cdepth=interp1(Dep.time,Dep.data,time);     %find reference depth(time)
        # zoffset=align_geom(sensor_offsets,cpitch);  %calculate offset from 0,0
        # depth=cdepth-zoffset;                       %Find True depth of sensor

        ## f_interp = interp1d(self.combined_nc['filt_depth'
        pass

    def _process(self, sensor, logs_dir, netcdfs_dir):
        coeffs = None
        try:
            coeffs = getattr(self, sensor).cals
        except AttributeError as e:
            self.logger.debug(f"No calibration information for {sensor}: {e}")

        if sensor == 'navigation':
            self._navigation_process(sensor)
        elif sensor == 'gps':
            self._gps_process(sensor)
        elif sensor == 'depth':
            self._depth_process(sensor)
        elif sensor == 'hs2':
            self._hs2_process(sensor, logs_dir)
        elif (sensor == 'ctd' or sensor == 'ctd2') and coeffs:
            self._ctd_process(sensor, coeffs)
        else:
            self.logger.warning(f"No method to process {sensor}")

        return

    def write_netcdf(self, netcdfs_dir):
        self.combined_nc.attrs = self.global_metadata()
        out_fn = os.path.join(netcdfs_dir, f"{self.args.auv_name}_{self.args.mission}.nc")
        self.logger.info(f"Writing calibrated and aligned data to file {out_fn}")
        self.combined_nc.to_netcdf(out_fn)

    def process_logs(self):
        name = self.args.mission
        vehicle = self.args.auv_name
        logs_dir = os.path.join(self.args.base_path, vehicle, MISSIONLOGS, name)
        netcdfs_dir = os.path.join(self.args.base_path, vehicle, MISSIONNETCDFS, name)
        start_datetime = datetime.strptime('.'.join(name.split('.')[:2]), "%Y.%j")
        self._define_sensor_info(start_datetime)
        self._read_data(logs_dir, netcdfs_dir)
        self.combined_nc = xr.Dataset()

        try:
            for sensor in self.sinfo.keys():
                setattr(getattr(self, sensor), 'cal_align_data', xr.Dataset())
                self._process(sensor, logs_dir, netcdfs_dir)
        except AttributeError as e:
            # Likely: 'SensorInfo' object has no attribute 'orig_data'
            # - meaning netCDF file not loaded
            raise FileNotFoundError(f"orig_data not found for {sensor}:"
                                    f" refer to previous WARNING messages.")

        return netcdfs_dir
        self.write_netcdf(netcdfs_dir)

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
        parser.add_argument('--plot', action='store', help='Create intermediate plots'
                            ' to validate data operations. Use first<n> to plot <n>'
                            ' points, e.g. first2000. Program blocks upon show.')        
        parser.add_argument('-v', '--verbose', type=int, choices=range(3), 
                            action='store', default=0, const=1, nargs='?',
                            help="verbosity level: " + ', '.join([f"{i}: {v}"
                            for i, v, in enumerate(('WARN', 'INFO', 'DEBUG'))]))

        self.args = parser.parse_args()
        self.logger.setLevel(self._log_levels[self.args.verbose])

        self.commandline = ' '.join(sys.argv)

    
if __name__ == '__main__':

    cal_netcdf = CalAligned_NetCDF()
    cal_netcdf.process_command_line()
    p_start = time.time()
    netcdf_dir = cal_netcdf.process_logs()
    cal_netcdf.write_netcdf(netcdf_dir)
    cal_netcdf.logger.info(f"Time to process: {(time.time() - p_start):.2f} seconds")
