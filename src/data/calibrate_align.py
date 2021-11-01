#!/usr/bin/env python
"""
Calibrate original data and produce NetCDF file for mission

Read original data from netCDF files created by logs2netcdfs.py, apply
calibration information in .cfg and .xml files associated with the 
original .log files and write out a single netCDF file with the important
variables at original sampling intervals. Alignment and plumbing lag
corrections are also done during this step. The file will contain combined
variables (the combined_nc member variable) and be analogous to the original
netCDF4 files produced by MBARI's LRAUVs. Rather than using groups in netCDF-4
the data will be written in classic netCDF-4C with a naming syntax that mimics
the LRAUV group naming convention with the coordinates for each sensor:
```
    <sensor>_<variable>
    <sensor>_time
    <sensor>_depth
    <sensor>_latitude
    <sensor>_longitude
```
Note: The name "sensor" is used here, but it's really more aligned 
with the concept of "instrument" in SSDS parlance.
"""

__author__ = "Mike McCann"
__copyright__ = "Copyright 2020, Monterey Bay Aquarium Research Institute"

import cf_xarray
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import requests
import scipy
import sys
import time
import xarray as xr
from collections import namedtuple, OrderedDict
from ctd_proc import (
    _calibrated_sal_from_cond_frequency,
    _calibrated_temp_from_frequency,
)
from datetime import datetime
from hs2_proc import hs2_read_cal_file, hs2_calc_bb
from pathlib import Path
from netCDF4 import Dataset
from scipy.interpolate import interp1d
from seawater import eos80
from socket import gethostname
from logs2netcdfs import BASE_PATH, MISSIONLOGS, MISSIONNETCDFS

TIME = "time"


def align_geom(sensor_offset, pitches):
    """Use x & y sensor_offset values in meters from sensor_info and
    pitch in degrees to compute and return actual depths of the sensor
    based on the geometry relative to the vehicle's depth sensor.
    """
    # See https://en.wikipedia.org/wiki/Rotation_matrix
    #
    #                        * instrument location with pitch applied
    #                      / |
    #                     /  |
    #                    /   |
    #                   /    |
    #                  /     |
    #                 /      |
    #                /       |
    #               /        |
    #              /         |
    #             /
    #            /
    #           /            y
    #          /             _
    #         /              o
    #        /               f
    #       /                f
    #      /                                 *  instrument location
    #     /                                  |
    #    / \                 |               |
    #   /   \                |               y
    #  / pitch (theta)       |               |
    # /        \             |               |
    # --------------------x------------------+    --> nose
    #
    # [ cos(pitch) -sin(pitch) ]    [x]   [x']
    #                             X     =
    # [ sin(pitch)  cos(pitch) ]    [y]   [y']
    offsets = []
    for pitch in pitches:
        theta = pitch * np.pi / 180.0
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        x_off, y_off = np.matmul(R, sensor_offset)
        offsets.append(y_off)

    return offsets


class Coeffs:
    pass


class SensorInfo:
    pass


class CalAligned_NetCDF:

    logger = logging.getLogger(__name__)
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        "%(levelname)s %(asctime)s %(filename)s "
        "%(funcName)s():%(lineno)d %(message)s"
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    _log_levels = (logging.WARN, logging.INFO, logging.DEBUG)

    def global_metadata(self):
        """Use instance variables to return a dictionary of
        metadata specific for the data that are written
        """
        iso_now = datetime.utcnow().isoformat() + "Z"

        metadata = {}
        metadata["netcdf_version"] = "4"
        metadata["Conventions"] = "CF-1.6"
        metadata["date_created"] = iso_now
        metadata["date_update"] = iso_now
        metadata["date_modified"] = iso_now
        metadata["featureType"] = "trajectory"

        metadata["time_coverage_start"] = str(
            self.combined_nc["depth_time"].to_pandas()[0].isoformat()
        )
        metadata["time_coverage_end"] = str(
            self.combined_nc["depth_time"].to_pandas()[-1].isoformat()
        )
        metadata[
            "distribution_statement"
        ] = "Any use requires prior approval from MBARI"
        metadata["license"] = metadata["distribution_statement"]
        metadata[
            "useconst"
        ] = "Not intended for legal use. Data may contain inaccuracies."
        metadata["history"] = f"Created by {self.commandline} on {iso_now}"

        metadata["title"] = (
            f"Calibrated and aligned AUV sensor data from"
            f" {self.args.auv_name} mission {self.args.mission}"
        )
        metadata["summary"] = (
            f"Observational oceanographic data obtained from an Autonomous"
            f" Underwater Vehicle mission with measurements at"
            f" original sampling intervals. The data have been calibrated "
            f" and aligned by MBARI's auv-python software."
        )
        metadata["comment"] = (
            f"MBARI Dorado-class AUV data produced from original data"
            f" with execution of '{self.commandline}'' at {iso_now} on"
            f" host {gethostname()}. Software available at"
            f" 'https://bitbucket.org/mbari/auv-python'"
        )

        return metadata

    def _get_file(self, download_url, local_filename, session):
        with session.get(download_url, timeout=60) as resp:
            if resp.status != 200:
                self.logger.warning(
                    f"Cannot read {download_url}, status = {resp.status}"
                )
            else:
                self.logger.info(f"Started download to {local_filename}...")
                with open(local_filename, "wb") as handle:
                    for chunk in resp.content.iter_chunked(1024):
                        handle.write(chunk)
                    if self.args.verbose > 1:
                        print(
                            f"{os.path.basename(local_filename)}(done) ",
                            end="",
                            flush=True,
                        )

    def _define_sensor_info(self, start_datetime):
        # Horizontal and vertical distance from origin in meters
        SensorOffset = namedtuple("SensorOffset", "x y")

        # Original configuration of Dorado389 - Modify below with changes over time
        self.sinfo = OrderedDict(
            [
                (
                    "navigation",
                    {
                        "data_filename": "navigation.nc",
                        "cal_filename": None,
                        "lag_secs": None,
                        "sensor_offset": None,
                    },
                ),
                (
                    "gps",
                    {
                        "data_filename": "gps.nc",
                        "cal_filename": None,
                        "lag_secs": None,
                        "sensor_offset": None,
                    },
                ),
                (
                    "depth",
                    {
                        "data_filename": "parosci.nc",
                        "cal_filename": None,
                        "lag_secs": None,
                        "sensor_offset": SensorOffset(-0.927, -0.076),
                    },
                ),
                (
                    "hs2",
                    {
                        "data_filename": "hydroscatlog.nc",
                        "cal_filename": "hs2Calibration.dat",
                        "lag_secs": None,
                        "sensor_offset": SensorOffset(0.1397, -0.2794),
                    },
                ),
                (
                    "ctd",
                    {
                        "data_filename": "ctdDriver.nc",
                        "cal_filename": "ctdDriver.cfg",
                        "lag_secs": None,
                        "sensor_offset": SensorOffset(1.003, 0.0001),
                    },
                ),
                (
                    "ctd2",
                    {
                        "data_filename": "ctdDriver2.nc",
                        "cal_filename": "ctdDriver2.cfg",
                        "lag_secs": None,
                        "sensor_offset": SensorOffset(1.003, 0.0001),
                    },
                ),
                (
                    "seabird25p",
                    {
                        "data_filename": "seabird25p.nc",
                        "cal_filename": "seabird25p.cfg",
                        "lag_secs": None,
                        "sensor_offset": SensorOffset(1.003, 0.0001),
                    },
                ),
                (
                    "isus",
                    {
                        "data_filename": "isuslog.nc",
                        "cal_filename": None,
                        "lag_secs": 6,
                        "sensor_offset": None,
                    },
                ),
                (
                    "biolume",
                    {
                        "data_filename": "biolume.nc",
                        "cal_filename": None,
                        "lag_secs": None,
                        "sensor_offset": SensorOffset(-0.889, -0.0508),
                    },
                ),
                (
                    "lopc",
                    {
                        "data_filename": "lopc.nc",
                        "cal_filename": None,
                        "lag_secs": None,
                        "sensor_offset": None,
                    },
                ),
                (
                    "ecopuck",
                    {
                        "data_filename": "FLBBCD2K.nc",
                        "cal_filename": "FLBBCD2K-3695.dev",
                        "lag_secs": None,
                        "sensor_offset": None,
                    },
                ),
            ]
        )

        # Changes over time
        if start_datetime.year >= 2003:
            self.sinfo["biolume"]["sensor_offset"] = SensorOffset(1.003, 0.0001)
        # ...

    def _read_data(self, logs_dir, netcdfs_dir):
        """Read in all the instrument data into member variables named by "sensor"
        Access xarray.Dataset like: self.ctd.data, self.navigation.data, ...
        Access calibration coefficients like: self.ctd.cals.t_f0, or as a
        dictionary for hs2 data.
        """
        for sensor, info in self.sinfo.items():
            sensor_info = SensorInfo()
            orig_netcdf_filename = os.path.join(netcdfs_dir, info["data_filename"])
            self.logger.debug(
                f"Reading data from {orig_netcdf_filename}"
                f" into self.{sensor}.orig_data"
            )
            try:
                setattr(sensor_info, "orig_data", xr.open_dataset(orig_netcdf_filename))
            except FileNotFoundError as e:
                self.logger.warning(
                    f"{sensor:10}: Cannot open file" f" {orig_netcdf_filename}"
                )
            if info["cal_filename"]:
                cal_filename = os.path.join(logs_dir, info["cal_filename"])
                self.logger.debug(
                    f"Reading calibrations from {orig_netcdf_filename}"
                    f" into self.{sensor}.cals"
                )
                if cal_filename.endswith(".cfg"):
                    try:
                        setattr(sensor_info, "cals", self._read_cfg(cal_filename))
                    except FileNotFoundError as e:
                        self.logger.debug(f"{e}")

            setattr(self, sensor, sensor_info)

        # TODO: Warn if no data found and if logs2netcdfs.py should be run

    def _read_cfg(self, cfg_filename):
        """Emulate what get_auv_cal.m and processCTD.m do in the
        Matlab doradosdp toolbox
        """
        self.logger.debug(f"Opening {cfg_filename}")
        coeffs = Coeffs()
        with open(cfg_filename) as fh:
            for line in fh:
                ##self.logger.debug(line)
                # From get_auv_cal.m - Handle CTD calibration parameters
                if line[:2] in (
                    "t_",
                    "c_",
                    "ep",
                    "SO",
                    "BO",
                    "Vo",
                    "TC",
                    "PC",
                    "Sc",
                    "Da",
                ):
                    coeff, value = [s.strip() for s in line.split("=")]
                    try:
                        self.logger.debug(f"Saving {line}")
                        setattr(coeffs, coeff, float(value.replace(";", "")))
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

        source = self.sinfo[sensor]["data_filename"]
        coord_str = f"{sensor}_time {sensor}_depth {sensor}_latitude {sensor}_longitude"
        self.combined_nc["navigation_roll"] = xr.DataArray(
            orig_nc["mPhi"].values,
            coords=[orig_nc.get_index("time")],
            dims={f"{sensor}_time"},
            name=f"{sensor}_roll",
        )
        self.combined_nc["navigation_roll"].attrs = {
            "long_name": "Vehicle roll",
            "standard_name": "platform_roll_angle",
            "units": "degree",
            "coordinates": coord_str,
            "comment": f"mPhi from {source}",
        }

        self.combined_nc["navigation_pitch"] = xr.DataArray(
            orig_nc["mTheta"].values,
            coords=[orig_nc.get_index("time")],
            dims={f"navigation_time"},
            name="pitch",
        )
        self.combined_nc["navigation_pitch"].attrs = {
            "long_name": "Vehicle pitch",
            "standard_name": "platform_pitch_angle",
            "units": "degree",
            "coordinates": coord_str,
            "comment": f"mTheta from {source}",
        }

        self.combined_nc["navigation_yaw"] = xr.DataArray(
            orig_nc["mPsi"].values,
            coords=[orig_nc.get_index("time")],
            dims={f"navigation_time"},
            name="yaw",
        )
        self.combined_nc["navigation_yaw"].attrs = {
            "long_name": "Vehicle yaw",
            "standard_name": "platform_yaw_angle",
            "units": "degree",
            "coordinates": coord_str,
            "comment": f"mPsi from {source}",
        }

        self.combined_nc["navigation_posx"] = xr.DataArray(
            orig_nc["mPos_x"].values - orig_nc["mPos_x"].values[0],
            coords=[orig_nc.get_index("time")],
            dims={f"navigation_time"},
            name="posx",
        )
        self.combined_nc["navigation_posx"].attrs = {
            "long_name": "Relative lateral easting",
            "units": "m",
            "coordinates": coord_str,
            "comment": f"mPos_x (minus first position) from {source}",
        }

        self.combined_nc["navigation_posy"] = xr.DataArray(
            orig_nc["mPos_y"].values - orig_nc["mPos_y"].values[0],
            coords=[orig_nc.get_index("time")],
            dims={f"navigation_time"},
            name="posy",
        )
        self.combined_nc["navigation_posy"].attrs = {
            "long_name": "Relative lateral northing",
            "units": "m",
            "coordinates": coord_str,
            "comment": f"mPos_y (minus first position) from {source}",
        }

        self.combined_nc["navigation_depth"] = xr.DataArray(
            orig_nc["mDepth"].values,
            coords=[orig_nc.get_index("time")],
            dims={f"navigation_time"},
            name="navigation_depth",
        )
        self.combined_nc["navigation_depth"].attrs = {
            "long_name": "Depth from Nav",
            "standard_name": "depth",
            "units": "m",
            "comment": f"mDepth from {source}",
        }

        try:
            self.combined_nc["navigation_latitude"] = xr.DataArray(
                orig_nc["latitude"].values * 180 / np.pi,
                coords=[orig_nc.get_index("time")],
                dims={f"navigation_time"},
                name="latitude",
            )
            self.combined_nc["navigation_latitude"].attrs = {
                "long_name": "latitude",
                "standard_name": "latitude",
                "units": "degrees_north",
                "comment": f"latitude (converted from radians) from {source}",
            }

        except KeyError:
            self.logger.debug("Likely before late 2004 when latitude was added")
        try:
            self.combined_nc["navigation_longitude"] = xr.DataArray(
                orig_nc["longitude"].values * 180 / np.pi,
                coords=[orig_nc.get_index("time")],
                dims={f"navigation_time"},
                name="longitude",
            )
            # Setting standard_name attribute here once sets it for all variables
            self.combined_nc["navigation_longitude"].coords[f"{sensor}_time"].attrs = {
                "standard_name": "time"
            }
            self.combined_nc["navigation_longitude"].attrs = {
                "long_name": "longitude",
                "standard_name": "longitude",
                "units": "degrees_east",
                "comment": f"longitude (converted from radians) from {source}",
            }
        except KeyError:
            # Setting standard_name attribute here once sets it for all variables
            self.combined_nc["navigation_depth"].coords[f"{sensor}_time"].attrs = {
                "standard_name": "time"
            }
            self.logger.debug("Likely before late 2004 when longitude was added")

        # % Remove obvious outliers that later disrupt the section plots.
        # % (First seen on mission 2008.281.03)
        # % In case we ever use this software for the D Allan B mapping vehicle determine
        # % the good depth range from the median of the depths
        # % From mission 2011.250.11 we need to first eliminate the near surface values before taking the
        # % median.
        # pdIndx = find(Nav.depth > 1);
        # posDepths = Nav.depth(pdIndx);
        pos_depths = np.where(self.combined_nc["navigation_depth"].values > 1)
        if self.args.mission == "2013.301.02" or self.args.mission == "2009.111.00":
            self.logger.info("Bypassing Nav QC depth check")
            maxGoodDepth = 1250
        else:
            maxGoodDepth = 7 * np.median(pos_depths)
            if maxGoodDepth < 0:
                maxGoodDepth = (
                    100  # Fudge for the 2009.272.00 mission where median was -0.1347!
                )
            if self.args.mission == "2010.153.01":
                maxGoodDepth = (
                    1250  # Fudge for 2010.153.01 where the depth was bogus, about 1.3
                )

        self.logger.debug(f"median of positive valued depths = {np.median(pos_depths)}")
        self.logger.debug(f"Finding depths less than '{maxGoodDepth}' and times > 0'")

    def _nudge_pos(self, max_sec_diff_at_end=10):
        """Apply linear nudges to underwater latitudes and longitudes so that
        the match the surface gps positions.
        """
        self.segment_count = None
        self.segment_minsum = None

        lon = self.combined_nc["navigation_longitude"]
        lat = self.combined_nc["navigation_latitude"]
        lon_fix = self.combined_nc["gps_longitude"]
        lat_fix = self.combined_nc["gps_latitude"]

        self.logger.info(
            f"{'seg#':4s}  {'end_sec_diff':12s} {'end_lon_diff':12s} {'end_lat_diff':12s} {'len(segi)':9s} {'seg_min':>9s} {'u_drift (cm/s)':14s} {'v_drift (cm/s)':14s} {'start datetime of segment':>29}"
        )

        # Any dead reckoned points before first GPS fix - usually empty as GPS fix happens before dive
        segi = np.where(lat.cf["T"].data < lat_fix.cf["T"].data[0])[0]
        if lon[:][segi].any():
            lon_nudged = lon[segi]
            lat_nudged = lat[segi]
            dt_nudged = lon.index[segi]
            self.logger.debug(
                f"Filled _nudged arrays with {len(segi)} values starting at {lat.index[0]} which were before the first GPS fix at {lat_fix.index[0]}"
            )
        else:
            lon_nudged = np.array([])
            lat_nudged = np.array([])
            dt_nudged = np.array([], dtype="datetime64[ns]")
        if segi.any():
            seg_min = (lat.index[segi][-1] - lat.index[segi][0]).total_seconds() / 60
        else:
            seg_min = 0
        self.logger.info(
            f"{' ':4}  {'-':>12} {'-':>12} {'-':>12} {len(segi):-9d} {seg_min:9.2f} {'-':>14} {'-':>14} {'-':>29}"
        )

        seg_count = 0
        seg_minsum = 0
        for i in range(len(lat_fix) - 1):
            # Segment of dead reckoned (under water) positions, each surrounded by GPS fixes
            segi = np.where(
                np.logical_and(
                    lat.cf["T"].data > lat_fix.cf["T"].data[i],
                    lat.cf["T"].data < lat_fix.cf["T"].data[i + 1],
                )
            )[0]
            if not segi.any():
                self.logger.debug(
                    f"No dead reckoned values found between GPS times of {lat_fix.cf['T'].data[i]} and {lat_fix.cf['T'].data[i+1]}"
                )
                continue

            end_sec_diff = (
                float(lat_fix.cf["T"].data[i + 1] - lat.cf["T"].data[segi[-1]]) / 1.0e9
            )
            if abs(end_sec_diff) > max_sec_diff_at_end:
                self.logger.warning(
                    f"abs(end_sec_diff) ({end_sec_diff}) > max_sec_diff_at_end ({max_sec_diff_at_end})"
                )

            end_lon_diff = float(lon_fix[i + 1]) - float(lon[segi[-1]])
            end_lat_diff = float(lat_fix[i + 1]) - float(lat[segi[-1]])
            seg_min = (
                float(lat.cf["T"].data[segi][-1] - lat.cf["T"].data[segi][0])
                / 1.0e9
                / 60
            )
            seg_minsum += seg_min

            # Compute approximate horizontal drift rate as a sanity check
            u_drift = (
                end_lat_diff
                * float(np.cos(lat_fix[i + 1]))
                * 60
                * 185300
                / float(lat.cf["T"].data[segi][-1] - lat.cf["T"].data[segi][0])
                / 1.0e9
            )
            v_drift = (
                end_lat_diff
                * 60
                * 185300
                / float(lat.cf["T"].data[segi][-1] - lat.cf["T"].data[segi][0])
                / 1.0e9
            )
            if len(segi) > 10:
                self.logger.info(
                    f"{i:4d}: {end_sec_diff:12.3f} {end_lon_diff:12.7f}"
                    f" {end_lat_diff:12.7f} {len(segi):-9d} {seg_min:9.2f}"
                    f" {u_drift:14.2f} {v_drift:14.2f} {lat.cf['T'].data[segi][-1]}"
                )

            # Start with zero adjustment at begining and linearly ramp up to the diff at the end
            lon_nudge = np.interp(
                lon.cf["T"].data[segi].astype(np.int64),
                [
                    lon.cf["T"].data[segi].astype(np.int64)[0],
                    lon.cf["T"].data[segi].astype(np.int64)[-1],
                ],
                [0, end_lon_diff],
            )
            lat_nudge = np.interp(
                lat.cf["T"].data[segi].astype(np.int64),
                [
                    lat.cf["T"].data[segi].astype(np.int64)[0],
                    lat.cf["T"].data[segi].astype(np.int64)[-1],
                ],
                [0, end_lat_diff],
            )

            # Sanity checks
            if (
                np.max(np.abs(lon[segi] + lon_nudge)) > 180
                or np.max(np.abs(lat[segi] + lon_nudge)) > 90
            ):
                self.logger.warning(
                    f"Nudged coordinate is way out of reasonable range - segment {seg_count}"
                )
                self.logger.warning(
                    f" max(abs(lon)) = {np.max(np.abs(lon[segi] + lon_nudge))}"
                )
                self.logger.warning(
                    f" max(abs(lat)) = {np.max(np.abs(lat[segi] + lat_nudge))}"
                )

            lon_nudged = np.append(lon_nudged, lon[segi] + lon_nudge)
            lat_nudged = np.append(lat_nudged, lat[segi] + lat_nudge)
            dt_nudged = np.append(dt_nudged, lon.cf["T"].data[segi])
            seg_count += 1

        # Any dead reckoned points after first GPS fix - not possible to nudge, just copy in
        segi = np.where(lat.cf["T"].data > lat_fix.cf["T"].data[-1])[0]
        seg_min = 0
        if segi.any():
            lon_nudged = np.append(lon_nudged, lon[segi])
            lat_nudged = np.append(lat_nudged, lat[segi])
            dt_nudged = np.append(dt_nudged, lon.cf["T"].data[segi])
            seg_min = (
                float(lat.cf["T"].data[segi][-1] - lat.cf["T"].data[segi][0])
                / 1.0e9
                / 60
            )

        self.logger.info(
            f"{seg_count+1:4d}: {'-':>12} {'-':>12} {'-':>12} {len(segi):-9d} {seg_min:9.2f} {'-':>14} {'-':>14}"
        )
        self.segment_count = seg_count
        self.segment_minsum = seg_minsum

        self.logger.info(f"Points in final series = {len(dt_nudged)}")

        lon_series = pd.Series(lon_nudged, index=dt_nudged)
        lat_series = pd.Series(lat_nudged, index=dt_nudged)
        if self.args.plot:
            pbeg = 0
            pend = len(self.combined_nc["gps_latitude"])
            if self.args.plot.startswith("first"):
                pend = int(self.args.plot.split("first")[1])
            fig, axes = plt.subplots(nrows=2, figsize=(18, 6))
            ##axes[0].plot(lat_series[pbeg:pend], '-o')
            axes[0].plot(lat[pbeg:pend], "-")
            ##axes[0].plot(lat_fix[pbeg:pend], '*')
            axes[0].set_ylabel("Latitude")
            ##axes[1].plot(lon_series[pbeg:pend], '-o')
            axes[1].plot(lon[pbeg:pend], "-")
            ##axes[1].plot(lon_fix[pbeg:pend], '*')
            axes[1].set_ylabel("Longitude")
            title = "Corrected nav from _nudge_pos()"
            title += f" - First {pend} Points"
            fig.suptitle(title)
            axes[0].grid()
            axes[1].grid()
            self.logger.debug(
                f"Pausing with plot entitled: {title}." " Close window to continue."
            )
            plt.show()

        return pd.Series(lon_nudged, index=dt_nudged), pd.Series(
            lat_nudged, index=dt_nudged
        )

    def _gps_process(self, sensor):
        try:
            orig_nc = getattr(self, sensor).orig_data
        except FileNotFoundError as e:
            self.logger.error(f"{e}")
            return

        if self.args.mission == "2010.151.04":
            # Gulf of Mexico mission - read from usbl.dat files
            self.logger.info(
                "Cannot read latitude data using load command.  Just for the GoMx mission use USBL instead..."
            )
            self.logger.info("TODO: Implement this fix in auv-python")
            # -data_filename = 'usbl.nc'
            # -loaddata
            # -time = time(1:10:end);
            # -lat = latitude(1:10:end);	% Subsample usbl so that iit is like our gps data
            # -lon = longitude(1:10:end);

        lat = orig_nc["latitude"] * 180.0 / np.pi
        if orig_nc["longitude"][0] > 0:
            lon = -1 * orig_nc["longitude"] * 180.0 / np.pi
        else:
            lon = orig_nc["longitude"] * 180.0 / np.pi

        # Filter out positions outside of operational box
        if (
            self.args.mission == "2010.151.04"
            or self.args.mission == "2010.153.01"
            or self.args.mission == "2010.154.01"
        ):
            lat_min = 26
            lat_max = 40
            lon_min = -124
            lon_max = -70
        else:
            lat_min = 30
            lat_max = 40
            lon_min = -124
            lon_max = -114

        self.logger.debug(
            f"Finding positions outside of longitude: {lon_min},"
            f" {lon_max} and latitide: {lat_min}, {lat_max}"
        )
        mlat = np.ma.masked_invalid(lat)
        mlat = np.ma.masked_outside(mlat, lat_min, lat_max)
        mlon = np.ma.masked_invalid(lon)
        mlon = np.ma.masked_outside(mlon, lon_min, lon_max)
        pm = np.logical_and(mlat, mlon)
        bad_pos = [
            f"{lo}, {la}"
            for lo, la in zip(lon.values[:][pm.mask], lat.values[:][pm.mask])
        ]

        gps_time_to_save = orig_nc.get_index("time")
        lat_to_save = lat
        lon_to_save = lon
        if bad_pos:
            self.logger.info(
                f"Number of bad {sensor} positions:" f" {len(lat.values[:][pm.mask])}"
            )
            self.logger.debug(
                f"Removing bad {sensor} positions (indices,"
                f" (lon, lat)): {np.where(pm.mask)[0]}, {bad_pos}"
            )
            gps_time_to_save = orig_nc.get_index("time")[:][~pm.mask]
            lat_to_save = lat[:][~pm.mask]
            lon_to_save = lon[:][~pm.mask]

        source = self.sinfo[sensor]["data_filename"]
        self.combined_nc["gps_latitude"] = xr.DataArray(
            lat_to_save.values,
            coords=[gps_time_to_save],
            dims={f"gps_time"},
            name="gps_latitude",
        )
        self.combined_nc["gps_latitude"].attrs = {
            "long_name": "GPS Latitude",
            "standard_name": "latitude",
            "units": "degrees_north",
            "comment": f"latitude from {source}",
        }

        self.combined_nc["gps_longitude"] = xr.DataArray(
            lon_to_save.values,
            coords=[gps_time_to_save],
            dims={f"gps_time"},
            name="gps_longitude",
        )
        # Setting standard_name attribute here once sets it for all variables
        self.combined_nc["gps_longitude"].coords[f"{sensor}_time"].attrs = {
            "standard_name": "time"
        }
        self.combined_nc["gps_longitude"].attrs = {
            "long_name": "GPS Longitude",
            "standard_name": "longitude",
            "units": "degrees_east",
            "comment": f"longitude from {source}",
        }

        # With navigation dead reckoned positions available in self.combined_nc
        # and the gps positions added we can now match the underwater inertial
        # (dead reckoned) positions to the surface gps positions.
        self._nudge_pos()

        gps_plot = True  # Set to False for debugging other plots
        if self.args.plot and gps_plot:
            pbeg = 0
            pend = len(self.combined_nc["gps_latitude"])
            if self.args.plot.startswith("first"):
                pend = int(self.args.plot.split("first")[1])
            fig, axes = plt.subplots(nrows=2, figsize=(18, 6))
            axes[0].plot(self.combined_nc["gps_latitude"][pbeg:pend], "-o")
            axes[0].set_ylabel("gps_latitude")
            axes[1].plot(self.combined_nc["gps_longitude"][pbeg:pend], "-o")
            axes[1].set_ylabel("gps_longitude")
            title = "GPS Positions"
            title += f" - First {pend} Points"
            fig.suptitle(title)
            axes[0].grid()
            axes[1].grid()
            self.logger.debug(
                f"Pausing with plot entitled: {title}." " Close window to continue."
            )
            plt.show()

    def _depth_process(self, sensor, latitude=36, cutoff_freq=1):
        """Depth data (from the Parosci) is 10 Hz - Use a butterworth window
        to filter recorded pressure to values that are appropriately sampled
        at 1 Hz (when matched with other sensor data).  cutoff_freq is in
        units of Hz.
        """
        try:
            orig_nc = getattr(self, sensor).orig_data
        except (FileNotFoundError, AttributeError) as e:
            self.logger.error(f"{e}")
            return

        # From initial CVS commit in 2004 the processDepth.m file computed
        # pres from depth this way.  I don't know what is done on the vehicle
        # side where a latitude of 36 is not appropriate: GoM, SoCal, etc.
        self.logger.debug(f"Converting depth to pressure using latitude = {latitude}")
        pres = eos80.pres(orig_nc["depth"], latitude)

        # See https://docs.scipy.org/doc/scipy-1.0.0/reference/generated/scipy.signal.filtfilt.html#scipy.signal.filtfilt
        # and https://docs.scipy.org/doc/scipy-1.0.0/reference/generated/scipy.signal.butter.html#scipy.signal.butter
        # Sample rate should be 10 - calcuate it to be sure
        sample_rate = 1.0 / np.round(
            np.mean(np.diff(orig_nc["time"])) / np.timedelta64(1, "s"), decimals=2
        )
        if sample_rate != 10:
            self.logger.warning(
                f"Expected sample_rate to be 10 Hz, instead it's {sample_rate} Hz"
            )

        # The Wn parameter for butter() is fraction of the Nyquist frequency
        Wn = cutoff_freq / (sample_rate / 2.0)
        b, a = scipy.signal.butter(8, Wn)
        depth_filtpres_butter = scipy.signal.filtfilt(b, a, pres)
        depth_filtdepth_butter = scipy.signal.filtfilt(b, a, orig_nc["depth"])

        # Use 10 points in boxcar as in processDepth.m
        a = 10
        b = scipy.signal.boxcar(a)
        depth_filtpres_boxcar = scipy.signal.filtfilt(b, a, pres)
        pres_plot = True  # Set to False for debugging other plots
        if self.args.plot and pres_plot:
            # Use Pandas to plot multiple columns of data
            # to validate that the filtering works as expected
            pbeg = 0
            pend = len(orig_nc.get_index("time"))
            if self.args.plot.startswith("first"):
                pend = int(self.args.plot.split("first")[1])
            df_plot = pd.DataFrame(index=orig_nc.get_index("time")[pbeg:pend])
            df_plot["pres"] = pres[pbeg:pend]
            df_plot["depth_filtpres_butter"] = depth_filtpres_butter[pbeg:pend]
            df_plot["depth_filtpres_boxcar"] = depth_filtpres_boxcar[pbeg:pend]
            title = (
                f"First {pend} points from"
                f" {self.args.mission}/{self.sinfo[sensor]['data_filename']}"
            )
            ax = df_plot.plot(title=title, figsize=(18, 6))
            ax.grid("on")
            self.logger.debug(
                f"Pausing with plot entitled: {title}." " Close window to continue."
            )
            plt.show()

        depth_filtdepth = xr.DataArray(
            depth_filtdepth_butter,
            coords=[orig_nc.get_index("time")],
            dims={f"depth_time"},
            name="depth_filtdepth",
        )
        depth_filtdepth.attrs = {
            "long_name": "Filtered Depth",
            "standard_name": "depth",
            "units": "m",
        }

        depth_filtpres = xr.DataArray(
            depth_filtpres_butter,
            coords=[orig_nc.get_index("time")],
            dims={f"depth_time"},
            name="depth_filtpres",
        )
        depth_filtpres.attrs = {
            "long_name": "Filtered Pressure",
            "standard_name": "sea_water_pressure",
            "units": "dbar",
        }

        self.combined_nc["depth_filtdepth"] = depth_filtdepth
        self.combined_nc["depth_filtpres"] = depth_filtpres

    def _hs2_process(self, sensor, logs_dir):
        try:
            orig_nc = getattr(self, sensor).orig_data
        except (FileNotFoundError, AttributeError) as e:
            self.logger.error(f"{e}")
            return

        try:
            cal_fn = os.path.join(logs_dir, self.sinfo["hs2"]["cal_filename"])
            cals = hs2_read_cal_file(cal_fn)
        except FileNotFoundError as e:
            self.logger.error(f"{e}")
            raise ()

        hs2 = hs2_calc_bb(orig_nc, cals)

        source = self.sinfo[sensor]["data_filename"]
        coord_str = f"{sensor}_time {sensor}_depth {sensor}_latitude {sensor}_longitude"

        # Blue backscatter
        if hasattr(hs2, "bb420"):
            blue_bs = xr.DataArray(
                hs2.bb420.values,
                coords=[hs2.bb420.get_index("time")],
                dims={"hs2_time"},
                name="hs2_bb420",
            )
            blue_bs.attrs = {
                "long_name": "Backscatter at 420 nm",
                "coordinates": coord_str,
                "comment": (f"Computed by hs2_calc_bb()" f" from data in {source}"),
            }
        if hasattr(hs2, "bb470"):
            blue_bs = xr.DataArray(
                hs2.bb470.values,
                coords=[hs2.bb470.get_index("time")],
                dims={"hs2_time"},
                name="hs2_bb470",
            )
            blue_bs.attrs = {
                "long_name": "Backscatter at 470 nm",
                "coordinates": coord_str,
                "comment": (f"Computed by hs2_calc_bb()" f" from data in {source}"),
            }

        # Red backscatter
        if hasattr(hs2, "bb676"):
            red_bs = xr.DataArray(
                hs2.bb676.values,
                coords=[hs2.bb676.get_index("time")],
                dims={"hs2_time"},
                name="hs2_bb676",
            )
            red_bs.attrs = {
                "long_name": "Backscatter at 676 nm",
                "coordinates": coord_str,
                "comment": (f"Computed by hs2_calc_bb()" f" from data in {source}"),
            }
        if hasattr(hs2, "bb700"):
            red_bs = xr.DataArray(
                hs2.bb700.values,
                coords=[hs2.bb700.get_index("time")],
                dims={"hs2_time"},
                name="hs2_bb700",
            )
            red_bs.attrs = {
                "long_name": "Backscatter at 700 nm",
                "coordinates": coord_str,
                "comment": (f"Computed by hs2_calc_bb()" f" from data in {source}"),
            }

        # Fluoresence
        if hasattr(hs2, "bb676"):
            fl = xr.DataArray(
                hs2.bb676.values,
                coords=[hs2.bb676.get_index("time")],
                dims={"hs2_time"},
                name="hs2_bb676",
            )
            fl.attrs = {
                "long_name": "Fluoresence at 676 nm",
                "coordinates": coord_str,
                "comment": (f"Computed by hs2_calc_bb()" f" from data in {source}"),
            }
            self.combined_nc["hs2_bb676"] = bb676
            fl = bb676
        if hasattr(hs2, "fl700"):
            fl700 = xr.DataArray(
                hs2.fl700.values,
                coords=[hs2.fl700.get_index("time")],
                dims={"hs2_time"},
                name="hs2_fl700",
            )
            fl700.attrs = {
                "long_name": "Fluoresence at 700 nm",
                "coordinates": coord_str,
                "comment": (f"Computed by hs2_calc_bb()" f" from data in {source}"),
            }
            fl = fl700

        # Zeroth level quality control - same as in legacy Matlab
        mblue = np.ma.masked_invalid(blue_bs)
        mblue = np.ma.masked_greater(mblue, 0.1)
        mred = np.ma.masked_invalid(red_bs)
        mred = np.ma.masked_greater(mred, 0.1)
        mfl = np.ma.masked_invalid(fl)
        mfl = np.ma.masked_greater(mfl, 0.02)
        mhs2 = np.logical_and(mblue, np.logical_and(mred, mfl))

        bad_hs2 = [
            f"{b}, {r}, {f}"
            for b, r, f in zip(
                blue_bs.values[:][mhs2.mask],
                red_bs.values[:][mhs2.mask],
                fl.values[:][mhs2.mask],
            )
        ]

        if bad_hs2:
            self.logger.info(
                f"Number of bad {sensor} points:"
                f" {len(blue_bs.values[:][mhs2.mask])}"
                f" of {len(blue_bs)}"
            )
            self.logger.debug(
                f"Removing bad {sensor} points (indices,"
                f" (blue, red, fl)): {np.where(mhs2.mask)[0]},"
                f" {bad_hs2}"
            )
            blue_bs = blue_bs[:][~mhs2.mask]
            red_bs = red_bs[:][~mhs2.mask]

        red_blue_plot = True  # Set to False for debugging other plots
        if self.args.plot and red_blue_plot:
            # Use Pandas to more easiily plot multiple columns of data
            pbeg = 0
            pend = len(blue_bs.get_index("hs2_time"))
            if self.args.plot.startswith("first"):
                pend = int(self.args.plot.split("first")[1])
            df_plot = pd.DataFrame(index=blue_bs.get_index("hs2_time")[pbeg:pend])
            df_plot["blue_bs"] = blue_bs[pbeg:pend]
            df_plot["red_bs"] = red_bs[pbeg:pend]
            df_plot["fl"] = fl[pbeg:pend]
            title = (
                f"First {pend} points from"
                f" {self.args.mission}/{self.sinfo[sensor]['data_filename']}"
            )
            ax = df_plot.plot(title=title, figsize=(18, 6))
            ax.grid("on")
            self.logger.debug(
                f"Pausing with plot entitled: {title}." " Close window to continue."
            )
            plt.show()

        # Save blue, red, & fl to combined_nc, alsoe
        if hasattr(hs2, "bb420"):
            self.combined_nc["hs2_bb420"] = blue_bs
        if hasattr(hs2, "bb470"):
            self.combined_nc["hs2_bb470"] = blue_bs
        if hasattr(hs2, "bb676"):
            self.rombined_nc["hs2_bb676"] = red_bs
        if hasattr(hs2, "bb700"):
            self.combined_nc["hs2_bb700"] = red_bs
        if hasattr(hs2, "fl676"):
            self.combined_nc["hs2_fl676"] = fl
        if hasattr(hs2, "fl700"):
            self.combined_nc["hs2_fl700"] = fl

        # For missions before 2009.055.05 hs2 will have attributes like bb470, bb676, and fl676
        # Hobilabs modified the instrument in 2009 to now give:         bb420, bb700, and fl700,
        # apparently giving a better measurement of chlorophyl.
        #
        # Detect the difference in this code and keep the mamber names descriptive in the survey data so
        # the the end user knows the difference.

        # Align Geometry, correct for pitch
        self.combined_nc[f"{sensor}_depth"] = self._geometric_depth_correction(
            sensor, orig_nc
        )
        # TODO: Add latitude & longitude coordinates

    def _ctd_process(self, sensor, cf):
        try:
            orig_nc = getattr(self, sensor).orig_data
        except FileNotFoundError as e:
            self.logger.error(f"{e}")
            return

        # Need to do this zeroth-level QC to calibrate temperature
        orig_nc["temp_frequency"][orig_nc["temp_frequency"] == 0.0] = np.nan
        source = self.sinfo[sensor]["data_filename"]
        coord_str = f"{sensor}_time {sensor}_depth {sensor}_latitude {sensor}_longitude"

        # Seabird specific calibrations
        temperature = xr.DataArray(
            _calibrated_temp_from_frequency(cf, orig_nc),
            coords=[orig_nc.get_index("time")],
            dims={f"{sensor}_time"},
            name="temperature",
        )
        temperature.attrs = {
            "long_name": "Temperature",
            "standard_name": "sea_water_temperature",
            "units": "degree_Celsius",
            "coordinates": coord_str,
            "comment": (
                f"Derived from temp_frequency from"
                f" {source} via calibration parms:"
                f" {cf.__dict__}"
            ),
        }

        salinity = xr.DataArray(
            _calibrated_sal_from_cond_frequency(
                self.args,
                self.combined_nc,
                self.logger,
                cf,
                orig_nc,
                temperature,
                self.combined_nc["depth_filtdepth"],
            ),
            coords=[orig_nc.get_index("time")],
            dims={f"{sensor}_time"},
            name="salinity",
        )
        salinity.attrs = {
            "long_name": "Salinity",
            "standard_name": "sea_water_salinity",
            "units": "",
            "coordinates": coord_str,
            "comment": (
                f"Derived from cond_frequency from"
                f" {source} via calibration parms:"
                f" {cf.__dict__}"
            ),
        }

        self.combined_nc[f"{sensor}_temperatue"] = temperature
        self.combined_nc[f"{sensor}_salinity"] = salinity
        self.combined_nc[f"{sensor}_depth"] = self._geometric_depth_correction(
            sensor, orig_nc
        )
        # TODO: Save nudged latitude & longitude coordinates
        """
        self.combined_nc[f"{sensor}_latitude"] = 
        la_interp = interp1d(self.combined_nc['navigation_time'].values.tolist(),
                            self.combined_nc['navigation_pitch'].values, 
                            fill_value="extrapolate")
        pitch = p_interp(orig_nc['time'].values.tolist())
        """
        pass

        # Other variables that may be in the original data

        # Salinity
        """
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
        """

    def _geometric_depth_correction(self, sensor, orig_nc):
        """Performs the align_geom() function from the legacy Matlab.
        Works for any sensor, but requires navigation being processed first
        as its variables in combined_nc are required. Returns corrected depth
        array.
        """
        p_interp = interp1d(
            self.combined_nc["navigation_time"].values.tolist(),
            self.combined_nc["navigation_pitch"].values,
            fill_value="extrapolate",
        )
        pitch = p_interp(orig_nc["time"].values.tolist())

        d_interp = interp1d(
            self.combined_nc["depth_time"].values.tolist(),
            self.combined_nc["depth_filtdepth"].values,
            fill_value="extrapolate",
        )
        orig_depth = d_interp(orig_nc["time"].values.tolist())
        offs_depth = align_geom(self.sinfo[sensor]["sensor_offset"], pitch)

        return orig_depth - offs_depth

    def _process(self, sensor, logs_dir, netcdfs_dir):
        coeffs = None
        try:
            coeffs = getattr(self, sensor).cals
        except AttributeError as e:
            self.logger.debug(f"No calibration information for {sensor}: {e}")

        if sensor == "navigation":
            self._navigation_process(sensor)
        elif sensor == "gps":
            self._gps_process(sensor)
        elif sensor == "depth":
            self._depth_process(sensor)
        elif sensor == "hs2":
            self._hs2_process(sensor, logs_dir)
        elif (sensor == "ctd" or sensor == "ctd2") and coeffs:
            self._ctd_process(sensor, coeffs)
        else:
            self.logger.warning(f"No method (yet) to process {sensor}")

        return

    def write_netcdf(self, netcdfs_dir):
        self.combined_nc.attrs = self.global_metadata()
        out_fn = os.path.join(
            netcdfs_dir, f"{self.args.auv_name}_{self.args.mission}.nc"
        )
        self.logger.info(f"Writing calibrated and aligned data to file {out_fn}")
        if os.path.exists(out_fn):
            os.remove(out_fn)
        self.combined_nc.to_netcdf(out_fn)

    def process_logs(self, vehicle: str, name: str) -> None:
        name = name or self.args.mission
        vehicle = vehicle or self.args.auv_name
        logs_dir = os.path.join(self.args.base_path, vehicle, MISSIONLOGS, name)
        netcdfs_dir = os.path.join(self.args.base_path, vehicle, MISSIONNETCDFS, name)
        start_datetime = datetime.strptime(".".join(name.split(".")[:2]), "%Y.%j")
        self._define_sensor_info(start_datetime)
        self._read_data(logs_dir, netcdfs_dir)
        self.combined_nc = xr.Dataset()

        try:
            for sensor in self.sinfo.keys():
                setattr(getattr(self, sensor), "cal_align_data", xr.Dataset())
                self._process(sensor, logs_dir, netcdfs_dir)
        except AttributeError as e:
            # Likely: 'SensorInfo' object has no attribute 'orig_data'
            # - meaning netCDF file not loaded
            raise FileNotFoundError(
                f"orig_data not found for {sensor}:"
                f" refer to previous WARNING messages."
            )

        return netcdfs_dir

    def process_command_line(self):

        import argparse
        from argparse import RawTextHelpFormatter

        examples = "Examples:" + "\n\n"
        examples += "  Calibrate original data for some missions:\n"
        examples += "    " + sys.argv[0] + " --mission 2020.064.10\n"
        examples += "    " + sys.argv[0] + " --auv_name i2map --mission 2020.055.01\n"

        parser = argparse.ArgumentParser(
            formatter_class=RawTextHelpFormatter,
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
            default="Dorado389",
            help="Dorado389 (default), i2MAP, or Multibeam",
        )
        parser.add_argument(
            "--mission",
            action="store",
            help="Mission directory, e.g.: 2020.064.10",
        )
        parser.add_argument(
            "--noinput",
            action="store_true",
            help="Execute without asking for a response, e.g. to not ask to re-download file",
        )
        parser.add_argument(
            "--plot",
            action="store",
            help="Create intermediate plots"
            " to validate data operations. Use first<n> to plot <n>"
            " points, e.g. first2000. Program blocks upon show.",
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

        self.commandline = " ".join(sys.argv)


if __name__ == "__main__":

    cal_netcdf = CalAligned_NetCDF()
    cal_netcdf.process_command_line()
    p_start = time.time()
    netcdf_dir = cal_netcdf.process_logs()
    cal_netcdf.write_netcdf(netcdf_dir)
    cal_netcdf.logger.info(f"Time to process: {(time.time() - p_start):.2f} seconds")
