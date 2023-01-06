#!/usr/bin/env python
"""
Calibrate original data and produce NetCDF file for mission

Read original data from netCDF files created by logs2netcdfs.py, apply
calibration information in .cfg and .xml files associated with the 
original .log files and write out a single netCDF file with the important
variables at original sampling intervals. Geometric alignment and plumbing lag
corrections are also done during this step. The file will contain combined
variables (the combined_nc member variable) and be analogous to the original
netCDF4 files produced by MBARI's LRAUVs. Rather than using groups in netCDF-4
the data will be written in classic netCDF-CF with a naming syntax that mimics
the LRAUV group naming convention with the coordinates for each sensor:
```
    <sensor>_<variable_1>
    <sensor>_<..........>
    <sensor>_<variable_n>
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

import argparse
import logging
import os
import sys
import time
from argparse import RawTextHelpFormatter
from collections import OrderedDict, namedtuple
from datetime import datetime
from socket import gethostname
from typing import List, Tuple

try:
    import cartopy.crs as ccrs
    from shapely.geometry import LineString
except ModuleNotFoundError:
    # cartopy is not installed, will not be able to plot maps
    pass

import cf_xarray  # Needed for the .cf accessor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from AUV import monotonic_increasing_time_indices
from ctd_proc import (
    _beam_transmittance_from_volts,
    _calibrated_O2_from_volts,
    _calibrated_sal_from_cond_frequency,
    _calibrated_temp_from_frequency,
)
from hs2_proc import hs2_calc_bb, hs2_read_cal_file
from logs2netcdfs import BASE_PATH, MISSIONLOGS, MISSIONNETCDFS
from matplotlib import patches
from scipy import signal
from scipy.interpolate import interp1d
from seawater import eos80

TIME = "time"
Range = namedtuple("Range", "min max")


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


class Calibrate_NetCDF:
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
        try:
            metadata["time_coverage_start"] = str(
                self.combined_nc["depth_time"].to_pandas()[0].isoformat()
            )
        except KeyError:
            raise EOFError("No depth_time variable in combined_nc")
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
            f"Calibrated AUV sensor data from"
            f" {self.args.auv_name} mission {self.args.mission}"
        )
        metadata["summary"] = (
            f"Observational oceanographic data obtained from an Autonomous"
            f" Underwater Vehicle mission with measurements at"
            f" original sampling intervals. The data have been calibrated"
            f" by MBARI's auv-python software."
        )
        if self.summary_fields:
            # Should be just one item in set, but just in case join them
            metadata["summary"] += " " + ". ".join(self.summary_fields)
        metadata["comment"] = (
            f"MBARI Dorado-class AUV data produced from original data"
            f" with execution of '{self.commandline}'' at {iso_now} on"
            f" host {gethostname()}. Software available at"
            f" 'https://github.com/mbari-org/auv-python'"
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
        # TODO: Refactor this to use for multiple vehicles & changes over time
        # Horizontal and vertical distance from origin in meters
        # The origin of the x, y coordinate system is location of the
        # vehicle's paroscientific depth sensor in the tailcone.
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
                    "ctd1",
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
                        "sensor_offset": SensorOffset(4.04, 0.0),
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
                        "sensor_offset": SensorOffset(4.04, 0.0),
                        # From https://bitbucket.org/messiem/matlab_libraries/src/master/data_access/donnees_insitu/MBARI/AUV/charge_Dorado.m
                        # % UBAT flow conversion
                        # if time>=datenum(2010,6,29), flow_conversion=4.49E-04;
                        # else, flow_conversion=4.5E-04;			% calibration on 2/2/2009 but unknown before
                        # end
                        # flow_conversion=flow_conversion*1E3;	% using flow in mL/s
                        # flow1Hz=rpm*flow_conversion;
                        "flow_conversion": 4.5e-4 * 1e3,  # conversion to mL/s
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
                (
                    "tailcone",
                    {
                        "data_filename": "tailCone.nc",
                        "cal_filename": None,
                        "lag_secs": None,
                        "sensor_offset": None,
                    },
                ),
            ]
        )

        # Changes over time
        if self.args.auv_name.lower().startswith("dorado"):
            self.sinfo["depth"]["sensor_offset"] = None
            if start_datetime >= datetime(2007, 4, 30):
                # First missions with 10 Gulpers: 2007.120.00 & 2007.120.01
                for instr in ("ctd1", "ctd2", "hs2", "lopc", "ecopuck", "isus"):
                    # TODO: Verify the length of the 10-Gulper midsection
                    self.sinfo[instr]["sensor_offset"] = SensorOffset(4.5, 0.0)
            if start_datetime >= datetime(2014, 9, 21):
                # First mission with 20 Gulpers: 2014.265.03
                for instr in ("ctd1", "ctd2", "hs2", "lopc", "ecopuck", "isus"):
                    self.sinfo[instr]["sensor_offset"] = SensorOffset(4.5, 0.0)
            if start_datetime >= datetime(2010, 6, 29):
                self.sinfo["biolume"]["flow_conversion"] = 4.49e-4 * 1e3

    def _range_qc_combined_nc(
        self, instrument: str, variables: List[str], ranges: dict
    ) -> None:
        """For variables in combined_nc remove values that fall outside
        of specified min, max range.  Meant to be called by instrument so
        that the union of bad values from a set of variables can be removed."""
        out_of_range_indices = np.array([], dtype=int)
        vars_checked = []
        for var in variables:
            if var in self.combined_nc.variables:
                if var in ranges:
                    out_of_range = np.where(
                        (self.combined_nc[var] < ranges[var].min)
                        | (self.combined_nc[var] > ranges[var].max)
                    )[0]
                    self.logger.debug(
                        "%s: %d out of range values = %s",
                        var,
                        len(self.combined_nc[var][out_of_range].values),
                        self.combined_nc[var][out_of_range].values,
                    )
                    out_of_range_indices = np.union1d(
                        out_of_range_indices,
                        out_of_range,
                    )
                    vars_checked.append(var)
                else:
                    self.logger.debug(f"No Ranges set for {var}")

            else:
                self.logger.warning(f"{var} not in self.combined_nc")
        inst_vars = [
            str(var)
            for var in self.combined_nc.variables
            if str(var).startswith(f"{instrument}_")
        ]
        for var in inst_vars:
            self.logger.info(
                "Checked for data outside of these variables and ranges: %s",
                [(v, ranges[v]) for v in vars_checked],
            )
            self.logger.info(
                "%s: deleting %d values found outside of above ranges: %s",
                var,
                len(self.combined_nc[var][out_of_range_indices].values),
                self.combined_nc[var][out_of_range_indices].values,
            )
            self.logger.debug(
                f"{var}: deleting values {self.combined_nc[var][out_of_range_indices].values}"
            )
            coord = [k for k in self.combined_nc[var].coords][0]
            self.combined_nc[f"{var}_qced"] = (
                self.combined_nc[var]
                .drop_isel({coord: out_of_range_indices})
                .rename({f"{instrument}_time": f"{instrument}_time_qced"})
            )
        self.combined_nc = self.combined_nc.drop_vars(inst_vars)
        for var in inst_vars:
            self.logger.debug(f"Renaming {var}_qced to {var}")
            self.combined_nc[var] = self.combined_nc[f"{var}_qced"].rename(
                {f"{coord}_qced": coord}
            )
        qced_vars = [f"{var}_qced" for var in inst_vars]
        self.combined_nc = self.combined_nc.drop_vars(qced_vars)
        self.logger.info(f"Done QC'ing {instrument}")

    def _read_data(self, logs_dir, netcdfs_dir):
        """Read in all the instrument data into member variables named by "sensor"
        Access xarray.Dataset like: self.ctd.data, self.navigation.data, ...
        Access calibration coefficients like: self.ctd.cals.t_f0, or as a
        dictionary for hs2 data.  Collect summary metadata fields that should
        describe the source of the data if copied from M3.
        """
        self.summary_fields = set()
        for sensor, info in self.sinfo.items():
            sensor_info = SensorInfo()
            orig_netcdf_filename = os.path.join(netcdfs_dir, info["data_filename"])
            self.logger.debug(
                f"Reading data from {orig_netcdf_filename}"
                f" into self.{sensor}.orig_data"
            )
            try:
                setattr(sensor_info, "orig_data", xr.open_dataset(orig_netcdf_filename))
            except (FileNotFoundError, ValueError) as e:
                self.logger.debug(
                    f"{sensor:10}: Cannot open file" f" {orig_netcdf_filename}: {e}"
                )
            except OverflowError as e:
                self.logger.error(
                    f"{sensor:10}: Cannot open file" f" {orig_netcdf_filename}: {e}"
                )
                self.logger.info(
                    "Perhaps _remove_bad_values() needs to be called for it in logs2netcdfs.py"
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
            if hasattr(sensor_info, "orig_data"):
                try:
                    self.summary_fields.add(
                        getattr(self, sensor).orig_data.attrs["summary"]
                    )
                except KeyError:
                    self.logger.warning(f"{orig_netcdf_filename}: No summary field")

        # TODO: Warn if no data found and if logs2netcdfs.py should be run

    def _read_cfg(self, cfg_filename):
        """Emulate what get_auv_cal.m and processCTD.m do in the
        Matlab doradosdp toolbox
        """
        self.logger.debug(f"Opening {cfg_filename}")
        coeffs = Coeffs()
        # Default for non-i2map data
        coeffs.t_coefs = "A"
        coeffs.c_coefs = "A"
        with open(cfg_filename) as fh:
            for line in fh:
                ##self.logger.debug(line)
                if line.startswith("//"):
                    continue
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
                        # Like in Seabird25p.cc use ?_coefs to determine which
                        # calibration scheme to in ctd_proc.py for i2map data
                        if coeff == "t_coefs":
                            setattr(coeffs, coeff, str(value.split(";")[0]))
                        elif coeff == "c_coefs":
                            setattr(coeffs, coeff, str(value.split(";")[0]))
                        else:
                            setattr(coeffs, coeff, float(value.split(";")[0]))
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
        except AttributeError:
            raise EOFError(
                f"{sensor} has no orig_data - likely a missing or zero-sized .log file"
                f" in {os.path.join(MISSIONLOGS, self.args.mission)}"
            )

        # Remove non-monotonic times
        self.logger.debug("Checking for non-monotonic increasing times")
        monotonic = monotonic_increasing_time_indices(orig_nc.get_index("time"))
        if (~monotonic).any():
            self.logger.debug(
                "Removing non-monotonic increasing times at indices: %s",
                np.argwhere(~monotonic).flatten(),
            )
        orig_nc = orig_nc.sel(time=monotonic)

        source = self.sinfo[sensor]["data_filename"]
        coord_str = f"{sensor}_time {sensor}_depth {sensor}_latitude {sensor}_longitude"
        vars_to_qc = []
        # Units of these angles are radians in the original files, we want degrees
        vars_to_qc.append("navigation_roll")
        self.combined_nc["navigation_roll"] = xr.DataArray(
            orig_nc["mPhi"].values * 180 / np.pi,
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

        vars_to_qc.append("navigation_pitch")
        self.combined_nc["navigation_pitch"] = xr.DataArray(
            orig_nc["mTheta"].values * 180 / np.pi,
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

        vars_to_qc.append("navigation_yaw")
        self.combined_nc["navigation_yaw"] = xr.DataArray(
            orig_nc["mPsi"].values * 180 / np.pi,
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

        vars_to_qc.append("navigation_depth")
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

        self.combined_nc["navigation_mWaterSpeed"] = xr.DataArray(
            orig_nc["mWaterSpeed"].values,
            coords=[orig_nc.get_index("time")],
            dims={f"navigation_time"},
            name="navigation_mWaterSpeed",
        )
        self.combined_nc["navigation_mWaterSpeed"].attrs = {
            "long_name": "Current speed based upon DVL data",
            "standard_name": "platform_speed_wrt_sea_water",
            "units": "m/s",
            "comment": f"mWaterSpeed from {source}",
        }

        if "latitude" in orig_nc:
            navlat_var = "latitude"
        elif "latitudeNav" in orig_nc:
            # Starting with 2022.243.00 the latitude variable name was changed
            navlat_var = "latitudeNav"
        else:
            navlat_var = None
            self.logger.debug(
                "Likely before 2004.167.04 when latitude was added to navigation.log"
            )

        if "longitude" in orig_nc:
            navlon_var = "longitude"
        elif "longitudeNav" in orig_nc:
            # Starting with 2022.243.00 the longitude variable name was changed
            navlon_var = "longitudeNav"
        else:
            navlon_var = None
            self.logger.error(f"Cannot find latitude & longitude variables in {source}")
            self.logger.error(
                "Likely before 2004.167.04 when latitude & longitude were added to navigation.log"
            )

        if navlat_var is not None:
            vars_to_qc.append("navigation_latitude")
            self.combined_nc["navigation_latitude"] = xr.DataArray(
                orig_nc[navlat_var].values * 180 / np.pi,
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

        if navlon_var is not None:
            vars_to_qc.append("navigation_longitude")
            self.combined_nc["navigation_longitude"] = xr.DataArray(
                orig_nc[navlon_var].values * 180 / np.pi,
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
        else:
            # Setting standard_name attribute here once sets it for all variables
            self.combined_nc["navigation_depth"].coords[f"{sensor}_time"].attrs = {
                "standard_name": "time"
            }

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

        if self.args.mission == "2010.172.01":
            self.logger.info(
                f"Performing special QC for {self.args.mission}/navigation.nc"
            )
            self._range_qc_combined_nc(
                instrument="navigation",
                variables=vars_to_qc,
                ranges={
                    "navigation_depth": Range(0, 1000),
                    "navigation_roll": Range(-180, 180),
                    "navigation_pitch": Range(-180, 180),
                    "navigation_yaw": Range(-360, 360),
                    "navigation_longitude": Range(-360, 360),
                    "navigation_latitude": Range(-90, 90),
                },
            )

        if (
            self.args.mission == "2004.345.00"
            or self.args.mission == "2005.240.00"
            or self.args.mission == "2007.134.09"
            or self.args.mission == "2010.293.00"
            or self.args.mission == "2011.116.00"
            or self.args.mission == "2013.227.00"
            or self.args.mission == "2016.348.00"
            or self.args.mission == "2017.121.00"
            or self.args.mission == "2017.269.01"
            or self.args.mission == "2017.297.00"
            or self.args.mission == "2017.347.00"
            or self.args.mission == "2017.304.00"
        ):
            self.logger.info(
                f"Removing points outside of Monterey Bay for {self.args.mission}/navigation.nc"
            )
            self._range_qc_combined_nc(
                instrument="navigation",
                variables=vars_to_qc,
                ranges={
                    "navigation_longitude": Range(-122.1, -121.7),
                    "navigation_latitude": Range(36, 37),
                },
            )

    def _nudge_pos(self, max_sec_diff_at_end=10):
        """Apply linear nudges to underwater latitudes and longitudes so that
        they match the surface gps positions.
        """
        self.segment_count = None
        self.segment_minsum = None

        try:
            lon = self.combined_nc["navigation_longitude"]
        except KeyError:
            raise EOFError("No navigation_longitude data in combined_nc")
        lat = self.combined_nc["navigation_latitude"]
        lon_fix = self.combined_nc["gps_longitude"]
        lat_fix = self.combined_nc["gps_latitude"]

        self.logger.info(
            f"{'seg#':5s}  {'end_sec_diff':12s} {'end_lon_diff':12s} {'end_lat_diff':12s} {'len(segi)':9s} {'seg_min':>9s} {'u_drift (cm/s)':14s} {'v_drift (cm/s)':14s} {'start datetime of segment':>29}"
        )

        # Any dead reckoned points before first GPS fix - usually empty as GPS fix happens before dive
        segi = np.where(lat.cf["T"].data < lat_fix.cf["T"].data[0])[0]
        if lon[:][segi].any():
            lon_nudged_array = lon[segi]
            lat_nudged_array = lat[segi]
            dt_nudged = lon.get_index("navigation_time")[segi]
            self.logger.debug(
                f"Filled _nudged arrays with {len(segi)} values starting at {lat.get_index('navigation_time')[0]} which were before the first GPS fix at {lat_fix.get_index('navigation_time')[0]}"
            )
        else:
            lon_nudged_array = np.array([])
            lat_nudged_array = np.array([])
            dt_nudged = np.array([], dtype="datetime64[ns]")
        if segi.any():
            seg_min = (
                lat.get_index("navigation_time")[segi][-1]
                - lat.get_index("navigation_time")[segi][0]
            ).total_seconds() / 60
        else:
            seg_min = 0
        self.logger.info(
            f"{' ':5}  {'-':>12} {'-':>12} {'-':>12} {len(segi):-9d} {seg_min:9.2f} {'-':>14} {'-':>14} {'-':>29}"
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

            end_lon_diff = float(lon_fix[i + 1]) - float(lon[segi[-1]])
            end_lat_diff = float(lat_fix[i + 1]) - float(lat[segi[-1]])
            if abs(end_lon_diff) > 1 or abs(end_lat_diff) > 1:
                # It's a problem if we have more than 1 degree difference at the end of the segment.
                # This is usually because the GPS fix is bad, but sometimes it's because the
                # dead reckoned position is bad.  Or sometimes it's both as in dorado 2016.384.00.
                # Early QC by calling _range_qc_combined_nc() can remove the bad points.
                self.logger.error(
                    "End of underwater segment dead reckoned position is too different from GPS fix: "
                    f"abs(end_lon_diff) ({end_lon_diff}) > 1 or abs(end_lat_diff) ({end_lat_diff}) > 1"
                )
                self.logger.info(
                    "Fix this error by calling _range_qc_combined_nc() for gps and/or navigation variables for %s %s",
                    self.args.auv_name,
                    self.args.mission,
                )
                raise ValueError(
                    f"abs(end_lon_diff) ({end_lon_diff}) > 1 or abs(end_lat_diff) ({end_lat_diff}) > 1"
                )
            if abs(end_sec_diff) > max_sec_diff_at_end:
                # Happens in dorado 2016.348.00 because of a bad GPS fixes being removed
                self.logger.warning(
                    f"abs(end_sec_diff) ({end_sec_diff}) > max_sec_diff_at_end ({max_sec_diff_at_end})"
                )
                self.logger.info(
                    f"Overriding end_lon_diff ({end_lon_diff}) and end_lat_diff ({end_lat_diff}) by setting them to 0"
                )
                end_lon_diff = 0
                end_lat_diff = 0

            seg_min = (
                float(lat.cf["T"].data[segi][-1] - lat.cf["T"].data[segi][0])
                / 1.0e9
                / 60
            )
            seg_minsum += seg_min

            # Compute approximate horizontal drift rate as a sanity check
            try:
                u_drift = (
                    end_lon_diff
                    * float(np.cos(lat_fix[i + 1] * np.pi / 180))
                    * 60
                    * 185300
                    / (
                        float(lat.cf["T"].data[segi][-1] - lat.cf["T"].data[segi][0])
                        / 1.0e9
                    )
                )
            except ZeroDivisionError:
                u_drift = 0
            try:
                v_drift = (
                    end_lat_diff
                    * 60
                    * 185300
                    / (
                        float(lat.cf["T"].data[segi][-1] - lat.cf["T"].data[segi][0])
                        / 1.0e9
                    )
                )
            except ZeroDivisionError:
                v_drift = 0
            if len(segi) > 10:
                self.logger.info(
                    f"{i:5d}: {end_sec_diff:12.3f} {end_lon_diff:12.7f}"
                    f" {end_lat_diff:12.7f} {len(segi):-9d} {seg_min:9.2f}"
                    f" {u_drift:14.3f} {v_drift:14.3f} {lat.cf['T'].data[segi][-1]}"
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

            lon_nudged_array = np.append(lon_nudged_array, lon[segi] + lon_nudge)
            lat_nudged_array = np.append(lat_nudged_array, lat[segi] + lat_nudge)
            dt_nudged = np.append(dt_nudged, lon.cf["T"].data[segi])
            seg_count += 1

        # Any dead reckoned points after first GPS fix - not possible to nudge, just copy in
        segi = np.where(lat.cf["T"].data > lat_fix.cf["T"].data[-1])[0]
        seg_min = 0
        if segi.any():
            lon_nudged_array = np.append(lon_nudged_array, lon[segi])
            lat_nudged_array = np.append(lat_nudged_array, lat[segi])
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

        lon_nudged = xr.DataArray(
            data=lon_nudged_array,
            dims=["time"],
            coords=dict(time=dt_nudged),
            name="longitude",
        )
        lat_nudged = xr.DataArray(
            data=lat_nudged_array,
            dims=["time"],
            coords=dict(time=dt_nudged),
            name="latitude",
        )
        if self.args.plot:
            fig, axes = plt.subplots(nrows=2, figsize=(18, 6))
            axes[0].plot(lat_nudged.coords["time"].data, lat_nudged, "-")
            axes[0].plot(lat.cf["T"].data, lat, "--")
            axes[0].plot(lat_fix.cf["T"].data, lat_fix, "*")
            axes[0].set_ylabel("Latitude")
            axes[0].legend(["Nudged", "Original", "GPS Fixes"])
            axes[1].plot(lon_nudged.coords["time"].data, lon_nudged, "-")
            axes[1].plot(lon.cf["T"].data, lon, "--")
            axes[1].plot(lon_fix.cf["T"].data, lon_fix, "*")
            axes[1].set_ylabel("Longitude")
            axes[1].legend(["Nudged", "Original", "GPS Fixes"])
            title = "Corrected nav from _nudge_pos()"
            fig.suptitle(title)
            axes[0].grid()
            axes[1].grid()
            self.logger.debug(
                f"Pausing with plot entitled: {title}." " Close window to continue."
            )
            plt.show()

            gps_plot = True
            if gps_plot:
                try:
                    ax = plt.axes(projection=ccrs.PlateCarree())
                except NameError:
                    self.logger.warning("No gps_plot, could not import cartopy")
                    return lon_nudged, lat_nudged
                nudged = LineString(zip(lon_nudged.values, lat_nudged.values))
                original = LineString(zip(lon.values, lat.values))
                ax.add_geometries(
                    [nudged],
                    crs=ccrs.PlateCarree(),
                    edgecolor="red",
                    facecolor="none",
                    label="Nudged",
                )
                ax.add_geometries(
                    [original],
                    crs=ccrs.PlateCarree(),
                    edgecolor="grey",
                    facecolor="none",
                    label="Original",
                )
                handle_gps = ax.scatter(
                    lon_fix.values,
                    lat_fix.values,
                    color="green",
                    label="GPS Fixes",
                )
                bounds = nudged.buffer(0.02).bounds
                extent = bounds[0], bounds[2], bounds[1], bounds[3]
                ax.set_extent(extent, crs=ccrs.PlateCarree())
                ax.coastlines()
                handle_nudged = patches.Rectangle((0, 0), 1, 0.1, facecolor="red")
                handle_original = patches.Rectangle((0, 0), 1, 0.1, facecolor="gray")
                ax.legend(
                    [handle_nudged, handle_original, handle_gps],
                    ["Nudged", "Original", "GPS Fixes"],
                )
                ax.gridlines(
                    crs=ccrs.PlateCarree(),
                    draw_labels=True,
                    linewidth=1,
                    color="gray",
                    alpha=0.5,
                )
                ax.set_title(f"{self.args.auv_name} {self.args.mission}")
                self.logger.debug(
                    "Pausing map plot (doesn't work well in VS Code debugger)."
                    " Close window to continue."
                )
                plt.show()

        return lon_nudged, lat_nudged

    def _gps_process(self, sensor):
        try:
            orig_nc = getattr(self, sensor).orig_data
        except FileNotFoundError as e:
            self.logger.error(f"{e}")
            return
        except AttributeError:
            raise EOFError(
                f"{sensor} has no orig_data - likely a missing or zero-sized .log file"
                f" in {os.path.join(MISSIONLOGS, self.args.mission)}"
            )

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

        gps_time_to_save = orig_nc.get_index("time")
        lat_to_save = lat
        lon_to_save = lon

        source = self.sinfo[sensor]["data_filename"]
        vars_to_qc = []
        vars_to_qc.append("gps_latitude")
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

        vars_to_qc.append("gps_longitude")
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
        if (
            self.args.mission == "2004.345.00"
            or self.args.mission == "2005.240.00"
            or self.args.mission == "2007.134.09"
            or self.args.mission == "2010.293.00"
            or self.args.mission == "2011.116.00"
            or self.args.mission == "2013.227.00"
            or self.args.mission == "2016.348.00"
            or self.args.mission == "2017.121.00"
            or self.args.mission == "2017.269.01"
            or self.args.mission == "2017.297.00"
            or self.args.mission == "2017.347.00"
            or self.args.mission == "2017.304.00"
        ):
            self.logger.info(
                f"Removing points outside of Monterey Bay for {self.args.mission}/gps.nc"
            )
            self._range_qc_combined_nc(
                instrument="gps",
                variables=vars_to_qc,
                ranges={
                    "gps_latitude": Range(36, 37),
                    "gps_longitude": Range(-122.1, -121.7),
                },
            )

        # TODO: Put this in a separate module like match_to_gps.py or something
        # With navigation dead reckoned positions available in self.combined_nc
        # and the gps positions added we can now match the underwater inertial
        # (dead reckoned) positions to the surface gps positions.
        nudged_longitude, nudged_latitude = self._nudge_pos()
        self.combined_nc["nudged_latitude"] = nudged_latitude
        self.combined_nc["nudged_latitude"].attrs = {
            "long_name": "Nudged Latitude",
            "standard_name": "latitude",
            "units": "degrees_north",
            "comment": "Dead reckoned latitude nudged to GPS positions",
        }
        self.combined_nc["nudged_longitude"] = nudged_longitude
        self.combined_nc["nudged_longitude"].attrs = {
            "long_name": "Nudged Longitude",
            "standard_name": "longitude",
            "units": "degrees_east",
            "comment": "Dead reckoned longitude nudged to GPS positions",
        }

    def _depth_process(self, sensor, latitude=36, cutoff_freq=1):
        """Depth data (from the Parosci) is 10 Hz - Use a butterworth window
        to filter recorded pressure to values that are appropriately sampled
        at 1 Hz (when matched with other sensor data).  cutoff_freq is in
        units of Hz.
        """
        try:
            orig_nc = getattr(self, sensor).orig_data
        except (FileNotFoundError, AttributeError) as e:
            self.logger.debug(f"Original data not found for {sensor}: {e}")
            return

        # Remove non-monotonic times
        self.logger.debug("Checking for non-monotonic increasing times")
        monotonic = monotonic_increasing_time_indices(orig_nc.get_index("time"))
        if (~monotonic).any():
            self.logger.debug(
                "Removing non-monotonic increasing times at indices: %s",
                np.argwhere(~monotonic).flatten(),
            )
        orig_nc = orig_nc.sel(time=monotonic)

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
        b, a = signal.butter(8, Wn)
        try:
            depth_filtpres_butter = signal.filtfilt(b, a, pres)
        except ValueError as e:
            raise EOFError(f"Likely short or empty file: {e}")
        depth_filtdepth_butter = signal.filtfilt(b, a, orig_nc["depth"])

        # Use 10 points in boxcar as in processDepth.m
        a = 10
        b = signal.boxcar(a)
        depth_filtpres_boxcar = signal.filtfilt(b, a, pres)
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
            "comment": (
                f"Original {sample_rate:.3f} Hz depth data filtered using"
                f" Butterworth window with cutoff frequency of {cutoff_freq} Hz"
            ),
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
            "comment": (
                f"Original {sample_rate:.3f} Hz pressure data filtered using"
                f" Butterworth window with cutoff frequency of {cutoff_freq} Hz"
            ),
        }

        self.combined_nc["depth_filtdepth"] = depth_filtdepth
        self.combined_nc["depth_filtpres"] = depth_filtpres

    def _hs2_process(self, sensor, logs_dir):
        try:
            orig_nc = getattr(self, sensor).orig_data
        except (FileNotFoundError, AttributeError) as e:
            self.logger.debug(f"Original data not found for {sensor}: {e}")
            return

        # Remove non-monotonic times
        self.logger.debug("Checking for non-monotonic increasing times")
        monotonic = monotonic_increasing_time_indices(orig_nc.get_index("time"))
        if (~monotonic).any():
            self.logger.debug(
                "Removing non-monotonic increasing times at indices: %s",
                np.argwhere(~monotonic).flatten(),
            )
        orig_nc = orig_nc.sel(time=monotonic)

        try:
            cal_fn = os.path.join(logs_dir, self.sinfo["hs2"]["cal_filename"])
            cals = hs2_read_cal_file(cal_fn)
        except FileNotFoundError as e:
            self.logger.error(f"Cannot process HS2 data: {e}")
            return

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
        if hasattr(hs2, "fl676"):
            fl676 = xr.DataArray(
                hs2.fl676.values,
                coords=[hs2.fl676.get_index("time")],
                dims={"hs2_time"},
                name="hs2_fl676",
            )
            fl676.attrs = {
                "long_name": "Fluoresence at 676 nm",
                "coordinates": coord_str,
                "comment": (f"Computed by hs2_calc_bb()" f" from data in {source}"),
            }
            fl = fl676
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
            self.combined_nc["hs2_bb676"] = red_bs
        if hasattr(hs2, "bb700"):
            self.combined_nc["hs2_bb700"] = red_bs
        if hasattr(hs2, "fl676"):
            self.combined_nc["hs2_fl676"] = fl
        if hasattr(hs2, "fl700"):
            self.combined_nc["hs2_fl700"] = fl

        # For missions before 2009.055.05 hs2 will have attributes like bb470, bb676, and fl676
        # Hobilabs modified the instrument in 2009 to now give:         bb420, bb700, and fl700,
        # apparently giving a better measurement of chlorophyll.
        #
        # Detect the difference in this code and keep the member names descriptive in the survey data so
        # the the end user knows the difference.

        # Align Geometry, correct for pitch
        self.combined_nc[f"{sensor}_depth"] = self._geometric_depth_correction(
            sensor, orig_nc
        )
        out_fn = f"{self.args.auv_name}_{self.args.mission}_cal.nc"
        self.combined_nc[f"{sensor}_depth"].attrs = {
            "long_name": "Depth",
            "units": "m",
            "comment": (
                f"Variable depth_filtdepth from {out_fn} linearly interpolated"
                f" to {sensor}_time and corrected for pitch using"
                f" {self.sinfo[sensor]['sensor_offset']}"
            ),
        }

        # Coordinates latitude & longitude are interpolated to the sensor time
        # in the align.py code.  Here we add the sensor depths as this is where
        # the sensor offset is applied with _geometric_depth_correction().

    def _calibrated_oxygen(
        self,
        sensor,
        cf,
        orig_nc,
        var_name,
        temperature,
        salinity,
        portstbd="",
    ) -> Tuple[xr.DataArray, xr.DataArray]:
        """Calibrate oxygen data, returning DataArrays."""
        oxy_mll, oxy_umolkg = _calibrated_O2_from_volts(
            self.combined_nc,
            cf,
            orig_nc,
            var_name,
            temperature,
            salinity,
        )
        oxygen_mll = xr.DataArray(
            oxy_mll,
            coords=[orig_nc.get_index("time")],
            dims={f"{sensor}_time"},
            name="oxygen_mll" + portstbd,
        )
        oxygen_mll.attrs = {
            "long_name": "Dissolved Oxygen",
            "units": "ml/l",
            "comment": (f"..."),
        }
        oxygen_umolkg = xr.DataArray(
            oxy_umolkg,
            coords=[orig_nc.get_index("time")],
            dims={f"{sensor}_time"},
            name="oxygen_umolkg" + portstbd,
        )
        oxygen_umolkg.attrs = {
            "long_name": "Dissolved Oxygen",
            "units": "umol/kg",
            "comment": (f"..."),
        }
        return oxygen_mll, oxygen_umolkg

    def _ctd_process(self, sensor, cf):
        # Don't be put off by the length of this method.
        # It's lengthy because of all the possible netCDF variables and
        # attribute metadata that need to be added to the combined_nc.
        try:
            orig_nc = getattr(self, sensor).orig_data
        except FileNotFoundError as e:
            self.logger.error(f"{e}")
            return
        except AttributeError:
            raise EOFError(
                f"{sensor} has no orig_data - likely a missing or zero-sized .log file"
                f" in {os.path.join(MISSIONLOGS, self.args.mission)}"
            )

        # Remove non-monotonic times
        self.logger.debug("Checking for non-monotonic increasing times")
        monotonic = monotonic_increasing_time_indices(orig_nc.get_index("time"))
        if (~monotonic).any():
            self.logger.debug(
                "Removing non-monotonic increasing times at indices: %s",
                np.argwhere(~monotonic).flatten(),
            )
        orig_nc = orig_nc.sel(time=monotonic)

        # Need to do this zeroth-level QC to calibrate temperature
        orig_nc["temp_frequency"][orig_nc["temp_frequency"] == 0.0] = np.nan
        source = self.sinfo[sensor]["data_filename"]

        # === Temperature and salinity variables ===
        # Seabird specific calibrations
        vars_to_qc = []
        self.logger.debug("Calling _calibrated_temp_from_frequency()")
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
            "comment": (
                f"Derived from temp_frequency from"
                f" {source} via calibration parms:"
                f" {cf.__dict__}"
            ),
        }
        self.combined_nc[f"{sensor}_temperature"] = temperature

        self.logger.debug("Calling _calibrated_sal_from_cond_frequency()")
        cal_conductivity, cal_salinity = _calibrated_sal_from_cond_frequency(
            self.args,
            self.combined_nc,
            self.logger,
            cf,
            orig_nc,
            temperature,
        )
        conductivity = xr.DataArray(
            cal_conductivity,
            coords=[orig_nc.get_index("time")],
            dims={f"{sensor}_time"},
            name="conductivity",
        )
        conductivity.attrs = {
            "long_name": "Conductivity",
            "standard_name": "sea_water_conductivity",
            "units": "Siemens/meter",
            "comment": (
                f"Derived from cond_frequency from"
                f" {source} via calibration parms:"
                f" {cf.__dict__}"
            ),
        }
        self.combined_nc[f"{sensor}_conductivity"] = conductivity
        vars_to_qc.append(f"{sensor}_salinity")
        salinity = xr.DataArray(
            cal_salinity,
            coords=[orig_nc.get_index("time")],
            dims={f"{sensor}_time"},
            name="salinity",
        )
        salinity.attrs = {
            "long_name": "Salinity",
            "standard_name": "sea_water_salinity",
            "units": "",
            "comment": (
                f"Derived from cond_frequency from"
                f" {source} via calibration parms:"
                f" {cf.__dict__}"
            ),
        }
        self.combined_nc[f"{sensor}_salinity"] = salinity

        # Variables computed onboard the vehicle that are recomputed here
        self.logger.debug("Collecting temperature_onboard")
        temperature_onboard = xr.DataArray(
            orig_nc["temperature"],
            coords=[orig_nc.get_index("time")],
            dims={f"{sensor}_time"},
            name="temperature_onboard",
        )
        # Onboard software sets bad values to absolute zero - replace with NaN
        temperature_onboard[temperature_onboard <= -273] = np.nan
        temperature_onboard.attrs = {
            "long_name": "Temperature computed onboard the vehicle",
            "units": "degree_Celsius",
            "comment": (
                f"Temperature computed onboard the vehicle from"
                f" calibration parameters installed on the vehicle"
                f" at the time of deployment."
            ),
        }
        self.combined_nc[f"{sensor}_temperature_onboard"] = temperature_onboard

        self.logger.debug("Collecting conductivity_onboard")
        conductivity_onboard = xr.DataArray(
            orig_nc["conductivity"],
            coords=[orig_nc.get_index("time")],
            dims={f"{sensor}_time"},
            name="conductivity_onboard",
        )
        conductivity_onboard.attrs = {
            "long_name": "Conductivity computed onboard the vehicle",
            "units": "Siemens/meter",
            "comment": (
                f"Temperature computed onboard the vehicle from"
                f" calibration parameters installed on the vehicle"
                f" at the time of deployment."
            ),
        }
        self.combined_nc[f"{sensor}_conductivity_onboard"] = conductivity_onboard

        if "salinity" in orig_nc:
            self.logger.debug("Collecting salinity_onboard")
            salinity_onboard = xr.DataArray(
                orig_nc["salinity"],
                coords=[orig_nc.get_index("time")],
                dims={f"{sensor}_time"},
                name="salinity_onboard",
            )
            salinity_onboard.attrs = {
                "long_name": "Salinity computed onboard the vehicle",
                "units": "",
                "comment": (
                    f"Salinity computed onboard the vehicle from"
                    f" calibration parameters installed on the vehicle"
                    f" at the time of deployment."
                ),
            }
            self.combined_nc[f"{sensor}_salinity_onboard"] = salinity_onboard

        # === Oxygen variables ===
        # original values in units of volts
        self.logger.debug("Collecting dissolvedO2")
        try:
            dissolvedO2 = xr.DataArray(
                orig_nc["dissolvedO2"],
                coords=[orig_nc.get_index("time")],
                dims={f"{sensor}_time"},
                name="dissolvedO2",
            )
            dissolvedO2.attrs = {
                "long_name": "Dissolved Oxygen sensor",
                "units": "Volts",
                "comment": (f"Analog Voltage Channel 6 - to be converted to umol/kg"),
            }
            self.combined_nc[f"{sensor}_dissolvedO2"] = dissolvedO2
            (
                self.combined_nc[f"{sensor}_oxygen_mll"],
                self.combined_nc[f"{sensor}_oxygen_umolkg"],
            ) = self._calibrated_oxygen(
                sensor,
                cf,
                orig_nc,
                "dissolvedO2",
                temperature,
                salinity,
                "",
            )
        except KeyError:
            self.logger.debug("No dissolvedO2 data in %s", self.args.mission)
        except ValueError as e:
            cfg_file = os.path.join(
                MISSIONLOGS,
                "".join(self.args.mission.split(".")[:2]),
                self.args.mission,
                self.sinfo["ctd"]["cal_filename"],
            )
            self.logger.error(f"Likely missing a calibration coefficient in {cfg_file}")
            self.logger.error(e)
        self.logger.debug("Collecting dissolvedO2_port")
        try:
            dissolvedO2_port = xr.DataArray(
                orig_nc["dissolvedO2_port"],
                coords=[orig_nc.get_index("time")],
                dims={f"{sensor}_time"},
                name="dissolvedO2_port",
            )
            dissolvedO2_port.attrs = {
                "long_name": "Dissolved Oxygen port side sensor",
                "units": "Volts",
                "comment": (f"Analog Voltage Channel 3 - to be converted to umol/kg"),
            }
            self.combined_nc[f"{sensor}_dissolvedO2_port"] = dissolvedO2_port
            (
                self.combined_nc[f"{sensor}_oxygen_mll_port"],
                self.combined_nc[f"{sensor}_oxygen_umolkg_port"],
            ) = self._calibrated_oxygen(
                sensor,
                cf,
                orig_nc,
                "dissolvedO2_port",
                temperature,
                salinity,
                "port",
            )
        except KeyError:
            self.logger.debug("No dissolvedO2_port data in %s", self.args.mission)
        self.logger.debug("Collecting dissolvedO2_port")
        try:
            dissolvedO2_stbd = xr.DataArray(
                orig_nc["dissolvedO2_stbd"],
                coords=[orig_nc.get_index("time")],
                dims={f"{sensor}_time"},
                name="dissolvedO2_stbd",
            )
            dissolvedO2_stbd.attrs = {
                "long_name": "Dissolved Oxygen stbd side sensor",
                "units": "Volts",
                "comment": (f"Analog Voltage Channel 5 - to be converted to umol/kg"),
            }
            self.combined_nc[f"{sensor}_dissolvedO2_stbd"] = dissolvedO2_stbd
            (
                self.combined_nc[f"{sensor}_oxygen_mll_stbd"],
                self.combined_nc[f"{sensor}_oxygen_umolkg_stbd"],
            ) = self._calibrated_oxygen(
                sensor,
                cf,
                orig_nc,
                "dissolvedO2_stbd",
                temperature,
                salinity,
                "stbd",
            )
        except KeyError:
            self.logger.debug("No dissolvedO2_port data in %s", self.args.mission)

        # === flow variables ===
        # A lot of 0.0 values in Dorado missions until about 2020.282.01
        self.logger.debug("Collecting flow1")
        try:
            flow1 = xr.DataArray(
                orig_nc["flow1"],
                coords=[orig_nc.get_index("time")],
                dims={f"{sensor}_time"},
                name="flow1",
            )
            flow1.attrs = {
                "long_name": "Flow sensor on ctd1",
                "units": "Volts",
                "comment": f"flow1 from {source}",
            }
            self.combined_nc[f"{sensor}_flow1"] = flow1
        except KeyError:
            self.logger.debug("No flow1 data in %s", self.args.mission)
        self.logger.debug("Collecting flow2")
        try:
            flow2 = xr.DataArray(
                orig_nc["flow2"],
                coords=[orig_nc.get_index("time")],
                dims={f"{sensor}_time"},
                name="flow2",
            )
            flow2.attrs = {
                "long_name": "Flow sensor on ctd1",
                "units": "Volts",
                "comment": f"flow2 from {source}",
            }
            self.combined_nc[f"{sensor}_flow2"] = flow2
        except KeyError:
            self.logger.debug("No flow2 data in %s", self.args.mission)

        try:
            beam_transmittance, _ = _beam_transmittance_from_volts(
                self.combined_nc, orig_nc
            )
            beam_transmittance = xr.DataArray(
                beam_transmittance * 100.0,
                coords=[orig_nc.get_index("time")],
                dims={f"{sensor}_time"},
                name="beam_transmittance",
            )
            beam_transmittance.attrs = {
                "long_name": "Beam Transmittance",
                "units": "%",
                "comment": f"Calibrated Beam Transmittance from {source}'s transmissometer variable",
            }
            self.combined_nc[f"{sensor}_beam_transmittance"] = beam_transmittance

        except KeyError:
            self.logger.debug(
                "No transmissometer data in %s/%s.nc", self.args.mission, sensor
            )

        self.combined_nc[f"{sensor}_depth"] = self._geometric_depth_correction(
            sensor, orig_nc
        )
        out_fn = f"{self.args.auv_name}_{self.args.mission}_cal.nc"
        self.combined_nc[f"{sensor}_depth"].attrs = {
            "long_name": "Depth",
            "units": "m",
            "comment": (
                f"Variable depth_filtdepth from {out_fn} linearly interpolated"
                f" to {sensor}_time and corrected for pitch using"
                f" {self.sinfo[sensor]['sensor_offset']}"
            ),
        }

        self.logger.info(
            f"Performing QC for {vars_to_qc} in {self.args.mission}/{sensor}.nc"
        )
        self._range_qc_combined_nc(
            instrument=sensor,
            variables=vars_to_qc,
            ranges={
                f"{sensor}_salinity": Range(30, 40),
            },
        )

    def _tailcone_process(self, sensor):
        # As requested by Rob Sherlock capture propRpm for comparison with
        # mWaterSpeed from navigation.log
        try:
            orig_nc = getattr(self, sensor).orig_data
        except FileNotFoundError as e:
            self.logger.error(f"{e}")
            return
        except AttributeError:
            raise EOFError(
                f"{sensor} has no orig_data - likely a missing or zero-sized .log file"
                f" in {os.path.join(MISSIONLOGS, self.args.mission)}"
            )

        # Remove non-monotonic times
        self.logger.debug("Checking for non-monotonic increasing times")
        monotonic = monotonic_increasing_time_indices(orig_nc.get_index("time"))
        if (~monotonic).any():
            self.logger.debug(
                "Removing non-monotonic increasing times at indices: %s",
                np.argwhere(~monotonic).flatten(),
            )
        orig_nc = orig_nc.sel(time=monotonic)

        source = self.sinfo[sensor]["data_filename"]
        coord_str = f"{sensor}_time {sensor}_depth {sensor}_latitude {sensor}_longitude"
        self.combined_nc["tailcone_propRpm"] = xr.DataArray(
            orig_nc["propRpm"].values,
            coords=[orig_nc.get_index("time")],
            dims={f"{sensor}_time"},
            name=f"{sensor}_propRpm",
        )
        self.combined_nc["tailcone_propRpm"].attrs = {
            "long_name": "Vehicle propeller speed",
            # Don't be confused by its name - propeller speed is logged in radians/sec.
            "units": "rad/s",
            "coordinates": coord_str,
            "comment": f"propRpm from {source} (convert to RPM by multiplying by 9.549297)",
        }

    def _ecopuck_process(self, sensor):
        # ecpouck's first mission 2020.245.00 - email dialog on 5 Dec 2022 discussing
        # using it for developing an HS2 transfer function and comparison with LRAUV data
        try:
            orig_nc = getattr(self, sensor).orig_data
        except FileNotFoundError as e:
            self.logger.error(f"{e}")
            return
        except AttributeError:
            raise EOFError(
                f"{sensor} has no orig_data - likely a missing or zero-sized .log file"
                f" in {os.path.join(MISSIONLOGS, self.args.mission)}"
            )

        # Remove non-monotonic times
        self.logger.debug("Checking for non-monotonic increasing times")
        monotonic = monotonic_increasing_time_indices(orig_nc.get_index("time"))
        if (~monotonic).any():
            self.logger.debug(
                "Removing non-monotonic increasing times at indices: %s",
                np.argwhere(~monotonic).flatten(),
            )
        orig_nc = orig_nc.sel(time=monotonic)

        source = self.sinfo[sensor]["data_filename"]
        coord_str = f"{sensor}_time {sensor}_depth {sensor}_latitude {sensor}_longitude"
        self.combined_nc["ecopuck_BB_Sig"] = xr.DataArray(
            orig_nc["BB_Sig"].values,
            coords=[orig_nc.get_index("time")],
            dims={f"{sensor}_time"},
            name=f"{sensor}_BB_Sig",
        )
        self.combined_nc["ecopuck_BB_Sig"].attrs = {
            "long_name": "Backscatter signal ??",
            "units": "",
            "coordinates": coord_str,
            "comment": f"BB_Sig from {source}",
        }

        self.combined_nc["ecopuck_CDOM_Sig"] = xr.DataArray(
            orig_nc["CDOM_Sig"].values,
            coords=[orig_nc.get_index("time")],
            dims={f"{sensor}_time"},
            name=f"{sensor}_CDOM_Sig",
        )
        self.combined_nc["ecopuck_CDOM_Sig"].attrs = {
            "long_name": "Colored Dissolved Organic Matter signal ??",
            "units": "",
            "coordinates": coord_str,
            "comment": f"CDOM_Sig from {source}",
        }

        self.combined_nc["ecopuck_Chl_Sig"] = xr.DataArray(
            orig_nc["Chl_Sig"].values,
            coords=[orig_nc.get_index("time")],
            dims={f"{sensor}_time"},
            name=f"{sensor}_Chl_Sig",
        )
        self.combined_nc["ecopuck_Chl_Sig"].attrs = {
            "long_name": "Chlorophyll signal ??",
            "units": "",
            "coordinates": coord_str,
            "comment": f"Chl_Sig from {source}",
        }

        """ NU_ variables likely mean that they are Not Used based N/U text in WETLabs docs
        self.combined_nc["ecopuck_NU_bb"] = xr.DataArray(
            orig_nc["NU_bb"].values,
            coords=[orig_nc.get_index("time")],
            dims={f"{sensor}_time"},
            name=f"{sensor}_NU_bb",
        )
        self.combined_nc["ecopuck_NU_bb"].attrs = {
            "long_name": "Null Backscatter signal ??",
            "units": "",
            "coordinates": coord_str,
            "comment": f"NU_bb from {source}",
        }

        self.combined_nc["ecopuck_NU_CDOM"] = xr.DataArray(
            orig_nc["NU_CDOM"].values,
            coords=[orig_nc.get_index("time")],
            dims={f"{sensor}_time"},
            name=f"{sensor}_NU_CDOM",
        )
        self.combined_nc["ecopuck_NU_CDOM"].attrs = {
            "long_name": "Null Colored Dissolved Organic Matter signal ??",
            "units": "",
            "coordinates": coord_str,
            "comment": f"NU_CDOM from {source}",
        }

        self.combined_nc["ecopuck_NU_Chl"] = xr.DataArray(
            orig_nc["NU_Chl"].values,
            coords=[orig_nc.get_index("time")],
            dims={f"{sensor}_time"},
            name=f"{sensor}_NU_Chl",
        )
        self.combined_nc["ecopuck_NU_Chl"].attrs = {
            "long_name": "Backscatter signal ??",
            "units": "",
            "coordinates": coord_str,
            "comment": f"NU_Chl from {source}",
        }

        # Values from 2020.245.00 seem stuck arounnd 538
        self.combined_nc["ecopuck_Thermistor"] = xr.DataArray(
            orig_nc["Thermistor"].values,
            coords=[orig_nc.get_index("time")],
            dims={f"{sensor}_time"},
            name=f"{sensor}_Thermistor",
        )
        self.combined_nc["ecopuck_Thermistor"].attrs = {
            "long_name": "Temperature ??",
            "units": "",
            "coordinates": coord_str,
            "comment": f"Thermistor from {source}",
        }
        """

    def _biolume_process(self, sensor):
        try:
            orig_nc = getattr(self, sensor).orig_data
        except FileNotFoundError as e:
            self.logger.error(f"{e}")
            return
        except AttributeError:
            raise EOFError(
                f"{sensor} has no orig_data - likely a missing or zero-sized .log file"
                f" in {os.path.join(MISSIONLOGS, self.args.mission)}"
            )

        # Remove non-monotonic times
        self.logger.debug("Checking for non-monotonic increasing times")
        monotonic = monotonic_increasing_time_indices(orig_nc.get_index("time"))
        if (~monotonic).any():
            self.logger.debug(
                "Removing non-monotonic increasing times at indices: %s",
                np.argwhere(~monotonic).flatten(),
            )
        orig_nc = orig_nc.sel(time=monotonic)

        # TODO: Check this
        self.combined_nc[f"{sensor}_depth"] = self._geometric_depth_correction(
            sensor, orig_nc
        )

        source = self.sinfo[sensor]["data_filename"]
        self.combined_nc["biolume_flow"] = xr.DataArray(
            orig_nc["flow"].values * self.sinfo["biolume"]["flow_conversion"],
            coords=[orig_nc.get_index("time")],
            dims={f"{sensor}_time"},
            name=f"{sensor}_flow",
        )
        self.combined_nc["biolume_flow"].attrs = {
            "long_name": "Bioluminesence pump flow rate",
            "units": "mL/s",
            "coordinates": f"{sensor}_time {sensor}_depth",
            "comment": f"flow from {source}",
        }

        self.combined_nc["biolume_avg_biolume"] = xr.DataArray(
            orig_nc["avg_biolume"].values,
            coords=[orig_nc.get_index("time")],
            dims={f"{sensor}_time"},
            name=f"{sensor}_avg_biolume",
        )
        self.combined_nc["biolume_avg_biolume"].attrs = {
            "long_name": "Bioluminesence Average of 60Hz data",
            "units": "photons s^-1",
            "coordinates": f"{sensor}_time {sensor}_depth",
            "comment": f"avg_biolume from {source}",
        }

        self.combined_nc["biolume_raw"] = xr.DataArray(
            orig_nc["raw"].values,
            coords=[orig_nc.get_index("time60hz")],
            dims={f"{sensor}_time60hz"},
            name=f"{sensor}_raw",
        )
        self.combined_nc["biolume_raw"].attrs = {
            "long_name": "Raw 60 hz biolume data",
            # xarray writes out its own units attribute
            "coordinates": f"{sensor}_time60hz {sensor}_depth60hz",
            "comment": f"raw values from {source}",
        }

    def _lopc_process(self, sensor):
        try:
            orig_nc = getattr(self, sensor).orig_data
        except FileNotFoundError as e:
            self.logger.error(f"{e}")
            return
        except AttributeError:
            raise EOFError(
                f"{sensor} has no orig_data - likely a missing or zero-sized .log file"
                f" in {os.path.join(MISSIONLOGS, self.args.mission)}"
            )

        source = self.sinfo[sensor]["data_filename"]
        coord_str = f"{sensor}_time {sensor}_depth {sensor}_latitude {sensor}_longitude"

        # A lopc.nc file without a time variable will return a RangeIndex object
        # from orig_nc.get_index('time') - test for presence of actual 'time' coordinate
        if "time" not in orig_nc.coords:
            raise EOFError(
                f"{sensor} has no time coordinate - likely an incomplete lopc.nc file"
                f" in {os.path.join(MISSIONLOGS, self.args.mission)}"
            )

        self.combined_nc["lopc_countListSum"] = xr.DataArray(
            orig_nc["countListSum"].values,
            coords=[orig_nc.get_index("time")],
            dims={f"{sensor}_time"},
            name=f"{sensor}_countListSum",
        )
        self.combined_nc["lopc_countListSum"].attrs = {
            "long_name": orig_nc["countListSum"].attrs["long_name"],
            "units": orig_nc["countListSum"].attrs["units"],
            "coordinates": coord_str,
            "comment": f"Sum of countListSum values by size class from {source}",
        }

        self.combined_nc["lopc_transCount"] = xr.DataArray(
            orig_nc["transCount"].values,
            coords=[orig_nc.get_index("time")],
            dims={f"{sensor}_time"},
            name=f"{sensor}_transCount",
        )
        self.combined_nc["lopc_transCount"].attrs = {
            "long_name": orig_nc["transCount"].attrs["long_name"],
            "units": orig_nc["transCount"].attrs["units"],
            "coordinates": coord_str,
            "comment": f"transCount from {source}",
        }

        self.combined_nc["lopc_nonTransCount"] = xr.DataArray(
            orig_nc["nonTransCount"].values,
            coords=[orig_nc.get_index("time")],
            dims={f"{sensor}_time"},
            name=f"{sensor}_nonTransCount",
        )
        self.combined_nc["lopc_nonTransCount"].attrs = {
            "long_name": orig_nc["nonTransCount"].attrs["long_name"],
            "units": orig_nc["nonTransCount"].attrs["units"],
            "coordinates": coord_str,
            "comment": f"nonTransCount from {source}",
        }

        self.combined_nc["lopc_LCcount"] = xr.DataArray(
            orig_nc["LCcount"].values,
            coords=[orig_nc.get_index("time")],
            dims={f"{sensor}_time"},
            name=f"{sensor}_LCcount",
        )
        self.combined_nc["lopc_LCcount"].attrs = {
            "long_name": orig_nc["LCcount"].attrs["long_name"],
            "units": orig_nc["LCcount"].attrs["units"],
            "coordinates": coord_str,
            "comment": f"LCcount from {source}",
        }

        self.combined_nc["lopc_flowSpeed"] = xr.DataArray(
            orig_nc["flowSpeed"].values,
            coords=[orig_nc.get_index("time")],
            dims={f"{sensor}_time"},
            name=f"{sensor}_flowSpeed",
        )
        self.combined_nc["lopc_flowSpeed"].attrs = {
            "long_name": orig_nc["flowSpeed"].attrs["long_name"],
            "units": orig_nc["flowSpeed"].attrs["units"],
            "coordinates": coord_str,
            "comment": f"flowSpeed from {source}",
        }

    def _geometric_depth_correction(self, sensor, orig_nc):
        """Performs the align_geom() function from the legacy Matlab.
        Works for any sensor, but requires navigation being processed first
        as its variables in combined_nc are required. Returns corrected depth
        array.
        """
        try:
            p_interp = interp1d(
                self.combined_nc["navigation_time"].values.tolist(),
                self.combined_nc["navigation_pitch"].values,
                fill_value="extrapolate",
            )
        except KeyError:
            raise EOFError("No navigation_time or navigation_pitch in combined_nc. ")
        pitch = p_interp(orig_nc["time"].values.tolist())

        d_interp = interp1d(
            self.combined_nc["depth_time"].values.tolist(),
            self.combined_nc["depth_filtdepth"].values,
            fill_value="extrapolate",
        )
        orig_depth = d_interp(orig_nc["time"].values.tolist())
        offs_depth = align_geom(self.sinfo[sensor]["sensor_offset"], pitch)

        corrected_depth = xr.DataArray(
            (orig_depth - offs_depth).astype(np.float64).tolist(),
            coords=[orig_nc.get_index("time")],
            dims={f"{sensor}_time"},
            name=f"{sensor}_depth",
        )
        if self.args.plot:
            plt.figure(figsize=(18, 6))
            plt.plot(
                orig_nc["time"].values,
                orig_depth,
                "-",
                orig_nc["time"].values,
                corrected_depth,
                "--",
                orig_nc["time"].values,
                pitch,
                ".",
            )
            plt.ylabel("Depth (m) & Pitch (deg)")
            plt.legend(("Original depth", "Pitch corrected depth", "Pitch"))
            plt.title(
                f"Original and pitch corrected depth for {self.args.auv_name} {self.args.mission}"
            )
            plt.show()

        return corrected_depth

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
        elif sensor == "ecopuck":
            self._ecopuck_process(sensor)
        elif sensor == "hs2":
            self._hs2_process(sensor, logs_dir)
        elif sensor == "tailcone":
            self._tailcone_process(sensor)
        elif sensor == "lopc":
            self._lopc_process(sensor)
        elif sensor in ("ctd1", "ctd2", "seabird25p"):
            if coeffs is not None:
                self._ctd_process(sensor, coeffs)
            else:
                if hasattr(getattr(self, sensor), "orig_data"):
                    self.logger.warning(f"No calibration information for {sensor}")
        elif sensor == "biolume":
            self._biolume_process(sensor)
        else:
            if hasattr(getattr(self, sensor), "orig_data"):
                self.logger.warning(f"No method (yet) to process {sensor}")

        return

    def write_netcdf(self, netcdfs_dir, vehicle: str = None, name: str = None) -> None:
        name = name or self.args.mission
        vehicle = vehicle or self.args.auv_name
        self.combined_nc.attrs = self.global_metadata()
        out_fn = os.path.join(netcdfs_dir, f"{vehicle}_{name}_cal.nc")
        self.logger.info(f"Writing calibrated instrument data to {out_fn}")
        if os.path.exists(out_fn):
            os.remove(out_fn)
        self.combined_nc.to_netcdf(out_fn)
        self.logger.info(
            "Data variables written: %s", ", ".join(sorted(self.combined_nc.variables))
        )

    def process_logs(
        self, vehicle: str = None, name: str = None, process_gps: bool = True
    ) -> None:
        name = name or self.args.mission
        vehicle = vehicle or self.args.auv_name
        logs_dir = os.path.join(self.args.base_path, vehicle, MISSIONLOGS, name)
        netcdfs_dir = os.path.join(self.args.base_path, vehicle, MISSIONNETCDFS, name)
        start_datetime = datetime.strptime(".".join(name.split(".")[:2]), "%Y.%j")
        self._define_sensor_info(start_datetime)
        self._read_data(logs_dir, netcdfs_dir)
        self.combined_nc = xr.Dataset()

        for sensor in self.sinfo.keys():
            if not process_gps:
                if sensor == "gps":
                    continue  # to skip gps processing in conftest.py fixture
            setattr(getattr(self, sensor), "cal_align_data", xr.Dataset())
            self.logger.debug(f"Processing {vehicle} {name} {sensor}")
            try:
                self._process(sensor, logs_dir, netcdfs_dir)
            except (EOFError, ValueError) as e:
                self.logger.error(f"Error processing {sensor}: {e}")
            except KeyError as e:
                self.logger.error(f"Error processing {sensor}: missing variable {e}")

        return netcdfs_dir

    def process_command_line(self):

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
            help=f"Base directory for missionlogs and missionnetcdfs, default: {BASE_PATH}",
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

    cal_netcdf = Calibrate_NetCDF()
    cal_netcdf.process_command_line()
    p_start = time.time()
    # Set process_gps=False to skip time consuming _nudge_pos() processing
    # netcdf_dir = cal_netcdf.process_logs(process_gps=False)
    netcdf_dir = cal_netcdf.process_logs()
    cal_netcdf.write_netcdf(netcdf_dir)
    cal_netcdf.logger.info(f"Time to process: {(time.time() - p_start):.2f} seconds")
