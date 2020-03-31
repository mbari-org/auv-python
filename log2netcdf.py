#!/usr/bin/env python
'''
Parse logged data from AUV MVC .log files and translate to NetCDF including
all of the available metadata from associated .cfg and .xml files.

--
Mike McCann
30 March 2020
'''

import os
import sys
import csv
import math
import coards
import datetime
import logging
import numpy as np
import numpy.ma as ma
import readauvlog
import requests
import shutil
from AUV import AUV
from scipy.interpolate import interp1d
from seawater import eos80
from netCDF4 import Dataset

LOG_FILES = ('ctdDriver.log', 'ctdDriver2.log', 'gps.log', 'hydroscatlog.log', 
             'navigation.log', 'isuslog.log', )

class AUV_NetCDF(AUV):

    logger = logging.getLogger(__name__)
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter('%(levelname)s %(asctime)s %(filename)s '
                                  '%(funcName)s():%(lineno)d %(message)s')
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    _log_levels = (logging.ERROR, logging.WARN, logging.INFO, logging.DEBUG)

    def __init__(self):
        return super(AUV_NetCDF, self).__init__()

    def _download_file(self, url):
        local_filename = url.split('/')[-1]
        self.logger.info(f"Dowloading {url}...")
        with requests.get(url, stream=True) as r:
            with open(local_filename, 'wb') as f:
                shutil.copyfileobj(r.raw, f)

        return local_filename

    def _url_from_mission(self, auv_name, mission):
        '''Folling MBARI's conventions return url for the missionlogs directory
        '''
        if auv_name == 'Dorado':
            # aka Gulper, e.g.: http://dods.mbari.org/data/auvctd/missionlogs/2020/2020064/2020.064.10/
            url = 'http://dods.mbari.org/data/auvctd/missionlogs/'
            url += f"{mission.split('.')[0]}/{mission.split('.')[0]}{mission.split('.')[1]}/"
            url += mission

        return url

    def _read_logs(self):
        base_url = self._url_from_mission('Dorado', self.args.mission)
        for log_file_name in LOG_FILES:
            local_filename = self._download_file(os.path.join(base_url, log_file_name))
            print(log_file_name)
            log_data = readauvlog.read(local_filename)
            breakpoint()

    def createNetCDFfromFile(self):
        '''Read data from log file and write to netCDf file with minimal
        modification ot original data
        '''
        if self.args.trajectory and self.args.beg_depth and self.args.end_depth:
            self.readTrajectory(self.args.beg_depth, self.args.end_depth)

        for file_name in LOG_FILES:
            self._download_file(log_url)
            # Make sure input file is openable
            try:
                with open(fileName): 
                    pass
            except IOError:
                raise Exception('Cannot open input file %s' % fileName)

            if self.sensorType == 'Invensense':
                try:
                    self.readBEDsFile(fileName)
                    self.processAccelerations()
                    self.processRotations(useMatlabCode=False)
                except NoPressureData as e:
                    continue
            else:
                raise Exception("No handler for sensorType = %s" % self.sensorType)

        if not hasattr(self, 's2013'):
            print('Could not read time (s2013) from input file(s)')
            exit(-1)

        if self.args.seconds_offset:
            self.s2013 = self.s2013 + self.args.seconds_offset
            self.ps2013 = self.ps2013 + self.args.seconds_offset

        if self.args.output:
            self.outFile = self.args.output
        elif len(self.inputFileNames) == 1:
            self.outFile = self.inputFileNames[0].split('.')[0]
            if '.EVT' in self.inputFileNames[0]:
                self.outFile += '_full'
            elif '.E00' in self.inputFileNames[0]:
                self.outFile += '_decim'
            if self.args.trajectory:
                self.outFile += '_traj'
            self.outFile += '.nc'
        else:
            raise Exception("Must specify --output if more than one input file.")

        if self.args.trajectory and self.args.bed_name:
            # Expect we have well-calibrated and tide-corrected depths in bed_depth[] array
            self.readTrajectory(self.bed_depth[0], self.bed_depth[-1])

        if (not self.traj_lat or not self.traj_lon) and self.args.trajectory:
            raise Exception('Could not exctract trajectory between {} and {}.'
                            ' Consider processing as a stationary event.'.format(
                            self.bed_depth[0], self.bed_depth[-1]))

        if self.args.trajectory:
            self.featureType = 'trajectory'

        # Interpolate data to regularly spaced time values - may need to do this to improve accuracy
        # (See http://www.freescale.com/files/sensors/doc/app_note/AN3397.pdf)
        ##si = linspace(self.s2013[0], self.s2013[-1], len(self.s2013))
        ##axi = interp(si, self.s2013, self.ax)

        # TODO: Review this calculation - may need to rotate these to the absolute (not rotating) frame
        # Double integrate accelerations to get position and construct X3D position values string
        # (May need to high-pass filter the data to remove noise that can give unreasonably large positions.)
        ##x = self.cumtrapz(self.s2013, self.cumtrapz(self.s2013, self.ax))
        ##y = self.cumtrapz(self.s2013, self.cumtrapz(self.s2013, self.ay))
        ##z = self.cumtrapz(self.s2013, self.cumtrapz(self.s2013, self.az))

        dateCreated = datetime.datetime.now().strftime("%d %B %Y %H:%M:%S")
        yearCreated = datetime.datetime.now().strftime("%Y")

        # Create the NetCDF file
        self.ncFile = Dataset(self.outFile, 'w')

        # Time dimensions for both trajectory and timeSeries datasets - IMU and pressure have different times
        self.ncFile.createDimension('time', len(self.s2013))
        self.time = self.ncFile.createVariable('time', 'float64', ('time',))
        self.time.standard_name = 'time'
        self.time.long_name = 'Time(GMT)'
        self.time.units = 'seconds since 2013-01-01 00:00:00'
        self.time[:] = self.s2013


        if self.featureType == 'trajectory':
            ifmt = '{var} linearly intepolated onto thalweg data from file {traj_file} using formula {formula}'
            # Coordinate variables for trajectory 
            # Interpolate trajectory lat and lon onto the times of the data
            self.latitude = self.ncFile.createVariable('latitude', 'float64', ('time',))
            self.latitude.long_name = 'LATITUDE'
            self.latitude.standard_name = 'latitude'
            self.latitude.units = 'degree_north'
            self.latitude.comment = ifmt.format(var='Latitude', traj_file=self.args.trajectory, formula=
                    'np.interp(np.linspace(0,1,len(self.s2013)), np.linspace(0,1,len(self.traj_lat)), self.traj_lat)')
            self.latitude[:] = np.interp(np.linspace(0,1,len(self.s2013)), np.linspace(0,1,len(self.traj_lat)), self.traj_lat)
    
            self.longitude = self.ncFile.createVariable('longitude', 'float64', ('time',))
            self.longitude.long_name = 'LONGITUDE'
            self.longitude.standard_name = 'longitude'
            self.longitude.units = 'degree_east'
            self.longitude.comment = ifmt.format(var='Longitude', traj_file=self.args.trajectory, formula=
                    'np.interp(np.linspace(0,1,len(self.s2013)), np.linspace(0,1,len(self.traj_lon)), self.traj_lon)')
            self.longitude[:] = np.interp(np.linspace(0,1,len(self.s2013)), np.linspace(0,1,len(self.traj_lon)), self.traj_lon)
    
            self.depth = self.ncFile.createVariable('depth', 'float64', ('time',))
            self.depth.long_name = 'DEPTH'
            self.depth.standard_name = 'depth'
            self.depth.units = 'm'
            self.depth.comment = "{} Linearly interpolated to IMU samples.".format(self.bed_depth_comment)
            self.depth[:] = np.interp(self.s2013, self.ps2013, self.bed_depth)

            # Record Variables - Accelerations
            xa = self.ncFile.createVariable('XA', 'float64', ('time',))
            xa.long_name = 'Acceleration along X-axis'
            xa.comment = 'Recorded by instrument'
            xa.coordinates = 'time depth latitude longitude'
            xa.units = 'g'
            xa[:] = self.ax

            ya = self.ncFile.createVariable('YA', 'float64', ('time',))
            ya.long_name = 'Acceleration along Y-axis'
            ya.comment = 'Recorded by instrument'
            ya.coordinates = 'time depth latitude longitude'
            ya.units = 'g'
            ya[:] = self.ay

            za = self.ncFile.createVariable('ZA', 'float64', ('time',))
            za.long_name = 'Acceleration along X-axis'
            za.comment = 'Recorded by instrument'
            za.coordinates = 'time depth latitude longitude'
            za.units = 'g'
            za[:] = self.az

            a = self.ncFile.createVariable('A', 'float64', ('time',))
            a.long_name = 'Acceleration Magnitude'
            a.comment = 'Computed with: np.sqrt(self.ax**2 + self.ay**2 + self.az**2)'
            a.coordinates = 'time depth latitude longitude'
            a.units = 'g'
            a[:] = self.a

            # Record Variables - Rotations
            # Nose of model points to -Z (north) and Up is +Y
            xr = self.ncFile.createVariable('XR', 'float64', ('time',))
            xr.long_name = 'Rotation about X-axis'
            xr.standard_name = 'platform_pitch_angle'
            xr.comment = self.euler_comment
            xr.coordinates = 'time depth latitude longitude'
            xr.units = 'degree'
            xr[:] = (self.rx * 180 / np.pi)

            yr = self.ncFile.createVariable('YR', 'float64', ('time',))
            yr.long_name = 'Rotation about Y-axis'
            yr.standard_name = 'platform_yaw_angle'
            yr.comment = self.euler_comment
            yr.coordinates = 'time depth latitude longitude'
            yr.units = 'degree'
            if self.args.yaw_offset:
                yr.comment = yr.comment + '. Added {} degrees to original values.'.format(self.args.yaw_offset)
                yawl = []
                for y in (self.ry * 180 / np.pi) + self.args.yaw_offset:
                    if y > 360.0:
                        yawl.append(y - 360.0)
                    else:
                        yawl.append(y)

                yaw = np.array(yawl)
            else:
                yaw = (self.ry * 180 / np.pi)

            yr[:] = yaw
    
            zr = self.ncFile.createVariable('ZR', 'float64', ('time',))
            zr.long_name = 'Rotation about Z-axis'
            zr.standard_name = 'platform_roll_angle'
            zr.comment = self.euler_comment
            zr.coordinates = 'time depth latitude longitude'
            zr.units = 'degree'
            zr[:] = (self.rz * 180 / np.pi)

            # Axis coordinates & angle for angle_axis form of the quaternion
            # Note: STOQS UI has preference for AXIS_X, AXIS_Y, AXIS_Z, ANGLE over roll, pitch, and yaw
            axis_x = self.ncFile.createVariable('AXIS_X', 'float64', ('time',))
            axis_x.long_name = 'X-component of rotation vector'
            axis_x.comment = self.p_angle_axis_comment
            axis_x.coordinates = 'time depth latitude longitude'
            axis_x.units = ''
            axis_x[:] = self.px

            axis_y = self.ncFile.createVariable('AXIS_Y', 'float64', ('time',))
            axis_y.long_name = 'Y-component of rotation vector'
            axis_y.comment = self.p_angle_axis_comment
            axis_y.coordinates = 'time depth latitude longitude'
            axis_y.units = ''
            axis_y[:] = self.py

            axis_z = self.ncFile.createVariable('AXIS_Z', 'float64', ('time',))
            axis_z.long_name = 'Z-component of rotation vector'
            axis_z.comment = self.p_angle_axis_comment
            axis_z.coordinates = 'time depth latitude longitude'
            axis_z.units = ''
            axis_z[:] = self.pz

            angle = self.ncFile.createVariable('ANGLE', 'float64', ('time',))
            angle.long_name = 'Angle rotated about rotation vector'
            angle.comment = self.p_angle_axis_comment
            angle.coordinates = 'time depth latitude longitude'
            angle.units = 'radian'
            angle[:] = self.angle

            # Axis about which platform is rotating - derived from dividing quaternions
            rot_x = self.ncFile.createVariable('ROT_X', 'float64', ('time',))
            rot_x.long_name = 'X-component of platform rotation vector'
            rot_x.comment = self.m_angle_axis_comment
            rot_x.coordinates = 'time depth latitude longitude'
            rot_x.units = ''
            rot_x[:] = self.mx

            rot_y = self.ncFile.createVariable('ROT_Y', 'float64', ('time',))
            rot_y.long_name = 'Y-component of platform rotation vector'
            rot_y.comment = self.m_angle_axis_comment
            rot_y.coordinates = 'time depth latitude longitude'
            rot_y.units = ''
            rot_y[:] = self.my

            rot_z = self.ncFile.createVariable('ROT_Z', 'float64', ('time',))
            rot_z.long_name = 'Z-component of platform rotation vector'
            rot_z.comment = self.m_angle_axis_comment
            rot_z.coordinates = 'time depth latitude longitude'
            rot_z.units = ''
            rot_z[:] = self.mz

            # Rotation rate & count
            rot_rate = self.ncFile.createVariable('ROT_RATE', 'float64', ('time',))
            rot_rate.long_name = 'Absolute rotation rate about rotation vector'
            rot_rate.comment = 'Computed from angle output from Quaternion.get_euler() and the angle difference from one time step to the next'
            rot_rate.coordinates = 'time depth latitude longitude'
            rot_rate.units = 'degree/second'
            rot_rate[:] = self.rotrate

            rot_count = self.ncFile.createVariable('ROT_COUNT', 'float64', ('time', ))
            rot_count.long_name = 'Rotation Count - Cumulative Sum of ROT_RATE * dt / 360 deg'
            rot_count.comment = 'Computed with: np.cumsum(np.absolute(self.diffrot)) / 2. / np.pi'
            rot_count.coordinates = 'time depth latitude longitude'
            rot_count[:] = (self.rotcount)

            # Pressure sensor data linearly interpolated to IMU samples
            p = self.ncFile.createVariable('P', 'float64', ('time',))
            p.long_name = 'Pressure'
            p.comment = 'Recorded pressure linearly interpolated to IMU samples with np.interp(self.s2013, self.ps2013, self.pr)'
            p.coordinates = 'time depth latitude longitude'
            p.units = 'dbar'
            p[:] = np.interp(self.s2013, self.ps2013, self.pr)

            p_adj = self.ncFile.createVariable('P_ADJUSTED', 'float64', ('time',))
            p_adj.long_name = 'Adjusted Pressure'
            p_adj.coordinates = 'time depth latitude longitude'
            p_adj.units = 'dbar'
            p_adj.comment = self.pr_adj_comment
            p_adj[:] = np.interp(self.s2013, self.ps2013, self.pr_adj)

            # bed depth at pressure sample intervals
            bed_depth_li = self.ncFile.createVariable('BED_DEPTH_LI', 'float64', ('time',))
            bed_depth_li.long_name = 'Depth of BED - Linearly Interpolated to IMU samples'
            bed_depth_li.units = 'm'
            bed_depth_li.coordinates = 'time depth latitude longitude'
            bed_depth_li.comment = self.bed_depth_comment
            bed_depth_li[:] = np.interp(self.s2013, self.ps2013, self.bed_depth)

            # Avoid memory problems, see http://stackoverflow.com/questions/21435648/cubic-spline-memory-error
            if len(self.ps2013) < 6000:
                # Pressure sensor data linearly interpolated to IMU samples
                p_spline = self.ncFile.createVariable('P_SPLINE', 'float64', ('time',), fill_value=1.e20)
                p_spline.long_name = 'Pressure'
                p_spline.comment = ("Recorded pressure cubic spline interpolated to IMU samples with"
                                    " spline_func = scipy.interpolate.interp1d(self.ps2013, self.pr, kind='cubic');"
                                    " p_mask = ma.masked_less(ma.masked_greater(self.s2013, np.max(self.ps2013)), np.min(self.ps2013));"
                                    " inside_spline = spline_func(ma.compressed(p_mask));"
                                    " p_spline = spline_func(self.s2013); p_spline[ma.clump_unmasked(p_mask)] = inside_spline")
                p_spline.coordinates = 'time depth latitude longitude'
                p_spline.units = 'dbar'
                spline_func = interp1d(self.ps2013, self.pr_adj, kind='cubic')
                # Mask IMU points outside of pressure time, interpolate, then put back into filled array
                p_mask = ma.masked_less(ma.masked_greater(self.s2013, np.max(self.ps2013)), np.min(self.ps2013))
                inside_spline = spline_func(ma.compressed(p_mask))
                p_spline[ma.clump_unmasked(p_mask)] = inside_spline

                # First difference of splined pressure sensor data interpolated to IMU samples
                p_spline_rate = self.ncFile.createVariable('P_SPLINE_RATE', 'float64', ('time',), fill_value=1.e20)
                p_spline_rate.long_name = 'Rate of change of spline fit of pressure'
                p_spline_rate.comment = 'Pressure rate of change interpolated to IMU samples with p_spline_rate[ma.clump_unmasked(p_mask)] = np.append([0], np.diff(inside_spline)) * self.rateHz'
                p_spline_rate.coordinates = 'time depth latitude longitude'
                p_spline_rate.units = 'dbar/s'
                p_spline_rate[ma.clump_unmasked(p_mask)] = np.append([0], np.diff(inside_spline)) * self.rateHz

                # Spline interpolated bed depth
                bed_depth_csi = self.ncFile.createVariable('BED_DEPTH_CSI', 'float64', ('time',), fill_value=1.e20)
                bed_depth_csi.long_name = 'Depth of BED - Cubic Spline Interpolated to IMU Samples'
                bed_depth_csi.units = 'm'
                bed_depth_csi.coordinates = 'time depth latitude longitude'
                bed_depth_csi.comment = self.bed_depth_csi_comment
                bed_depth_csi[ma.clump_unmasked(self.p_mask)] = self.bed_depth_inside_spline


            # First difference of pressure sensor data interpolated to IMU samples
            p_rate = self.ncFile.createVariable('P_RATE', 'float64', ('time',))
            p_rate.long_name = 'Rate of change of pressure'
            p_rate.comment = 'Pressure rate of change interpolated to IMU samples with np.append([0], np.diff(np.interp(self.s2013, self.ps2013, self.pr))) * self.rateHz'
            p_rate.coordinates = 'time depth latitude longitude'
            p_rate.units = 'dbar/s'
            p_rate[:] = np.append([0], np.diff(np.interp(self.s2013, self.ps2013, self.pr_adj))) * self.rateHz

            # Compute implied distance and velocity based on 147 cm BED housing circumference
            rot_dist = self.ncFile.createVariable('ROT_DIST', 'float64', ('time', ))
            rot_dist.long_name = 'Implied distance traveled assuming pure rolling motion'
            rot_dist.comment = 'Computed with: ROT_COUNT * 1.47 m'
            rot_dist.coordinates = 'time depth latitude longitude'
            rot_dist.units = 'm'
            rot_dist[:] = rot_count[:] * 1.47

            implied_velocity = self.ncFile.createVariable('IMPLIED_VELOCITY', 'float64', ('time', ))
            implied_velocity.long_name = 'Implied BED velocity assuming pure rolling motion'
            implied_velocity.comment = 'Computed with: ROT_RATE * 1.47 / 360.0'
            implied_velocity.coordinates = 'time depth latitude longitude'
            implied_velocity.units = 'm/s'
            implied_velocity[:] = rot_rate[:] * 1.47 / 360.0

            if self.traj_dist_topo:
                # Distance over topo from mbgrdviz generated trajectory thalweg trace file
                self.dist_topo = self.ncFile.createVariable('DIST_TOPO', 'float64', ('time',))
                self.dist_topo.long_name = 'Distance over topography along thalweg'
                self.dist_topo.units = 'm'
                self.dist_topo.comment = ifmt.format(var='dist_topo', traj_file=self.args.trajectory, formula=
                        'np.interp(np.linspace(0,1,len(self.s2013)), np.linspace(0,1,len(self.traj_dist_topo)), self.traj_dist_topo)')
                self.dist_topo.coordinates = 'time depth latitude longitude'
                self.dist_topo[:] = np.interp(np.linspace(0,1,len(self.s2013)), np.linspace(0,1,len(self.traj_dist_topo)), self.traj_dist_topo)
    
            # Tumble rate & count
            tumble_rate = self.ncFile.createVariable('TUMBLE_RATE', 'float64', ('time', ))
            tumble_rate.long_name = 'Angle change of axis (vec) in axis-angle representation of BED rotation'
            tumble_rate.comment = 'Computed with: abs(last_vec.angle(vec))'
            tumble_rate.coordinates = 'time depth latitude longitude'
            tumble_rate.units = 'degree/second'
            tumble_rate[:] = self.tumblerate.reshape(len(self.tumblerate), 1, 1, 1)

            tumble_count = self.ncFile.createVariable('TUMBLE_COUNT', 'float64', ('time', ))
            tumble_count.long_name = 'Tumble Count - Cumulative Sum of TUMBLE_RATE * dt / 360 deg'
            tumble_count.comment = 'Computed with: np.cumsum(np.absolute(self.difftumble)) / 2. / np.pi'
            tumble_count.coordinates = 'time depth latitude longitude'
            tumble_count[:] = self.tumblecount

            # Compute tumble distance
            tumble_dist = self.ncFile.createVariable('TUMBLE_DIST', 'float64', ('time', ))
            tumble_dist.long_name = 'Implied distance traveled assuming tumbling translates to horizontal motion'
            tumble_dist.comment = 'Computed with: TUMBLE_COUNT * 1.47 m'
            tumble_dist.coordinates = 'time depth latitude longitude'
            tumble_dist.units = 'm'
            tumble_dist[:] = tumble_count[:] * 1.47

            # Sum of rotation and tumbling distances
            rot_plus_tumble_dist = self.ncFile.createVariable('ROT_PLUS_TUMBLE_DIST', 'float64', ('time', ))
            rot_plus_tumble_dist.long_name = 'Implied distance traveled assuming pure rolling motion'
            rot_plus_tumble_dist.comment = 'Computed with: ROT_DIST + TUMBLE_DIST'
            rot_plus_tumble_dist.coordinates = 'time depth latitude longitude'
            rot_plus_tumble_dist.units = 'm'
            rot_plus_tumble_dist[:] = rot_dist[:] + tumble_dist[:]

            # Tide data from OSTP Software calculation
            tide = self.ncFile.createVariable('TIDE', 'float64', ('time'))
            tide.long_name = 'OSTP2 Tide model height'
            tide.coordinates = 'time depth latitude longitude'
            tide.comment = self.tide_comment
            tide.units = 'm'
            tide[:] = np.interp(self.s2013, self.ps2013, self.tide)

        # Add the global metadata, overriding with command line options provided
        self.add_global_metadata()
        self.ncFile.title = 'Orientation and acceleration data from Benthic Event Detector'
        if self.args.title:
            self.ncFile.title = self.args.title
        if self.args.summary:
            self.ncFile.summary = self.args.summary

        self.ncFile.close()

    def process_command_line(self):

        import argparse
        from argparse import RawTextHelpFormatter

        examples = 'Examples:' + '\n\n'
        examples += '  Write to local missionnetcdfs direcory:\n'
        examples += '    ' + sys.argv[0] + " --mission 2020.064.10 --output_dir missionnetcdfs\n"

        parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter,
                                         description='Convert BED event file(s) to a NetCDF file',
                                         epilog=examples)

        parser.add_argument('--mission', action='store', required=True, help="Mission directory, e.g.: 2020.064.10")
        parser.add_argument('--output_dir', action='store', required=True, help="Location to write netCDF files, e.g.: missionnetcdfs")

        parser.add_argument('--title', action='store', help='A short description of the dataset')
        parser.add_argument('--summary', action='store', help='Additional information about the dataset')
        
        parser.add_argument('-v', '--verbose', type=int, choices=range(3), action='store', default=0, 
                            help="verbosity level: " + ','.join([f"{i}: {v}" for i, v, in enumerate(self._log_levels)]))

        self.args = parser.parse_args()
        self.logger.setLevel(self._log_levels[self.args.verbose])

        self.commandline = ' '.join(sys.argv)

    
if __name__ == '__main__':

    auv_netcdf = AUV_NetCDF()
    auv_netcdf.process_command_line()

    auv_netcdf._read_logs()

    auv_netcdf.createNetCDFfromFile()


