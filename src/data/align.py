#!/usr/bin/env python
"""
Align calibrated data producing a netCDF file with coordinates on each variable

Read calibrated data from netCDF files created by calibrate.py, use the
best available (e.g. filtered, nudged) coordinate variables to interpolate
onto each each measured (record) variable sampling interval. The original
instrument sampling interval is preserved with the coordinate varaibles
interpolated onto that original time axis.
"""

__author__ = "Mike McCann"
__copyright__ = "Copyright 2021, Monterey Bay Aquarium Research Institute"

import json  # noqa: I001
import logging
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from socket import gethostname

import git
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import xarray as xr

from common_args import get_standard_lrauv_parser
from logs2netcdfs import AUV_NetCDF, MISSIONNETCDFS, SUMMARY_SOURCE, TIME, TIME60HZ
from nc42netcdfs import BASE_LRAUV_PATH
from utils import get_deployment_name


class InvalidCalFile(Exception):
    pass


class InvalidCombinedFile(Exception):
    pass


class Align_NetCDF:
    logger = logging.getLogger(__name__)
    _handler = logging.StreamHandler()
    _handler.setFormatter(AUV_NetCDF._formatter)
    logger.addHandler(_handler)
    _log_levels = (logging.WARN, logging.INFO, logging.DEBUG)

    # noqa: PLR0913 - Many parameters needed for initialization
    def __init__(  # noqa: PLR0913
        self,
        auv_name: str,
        mission: str,
        base_path: str,
        log_file: str = "",
        plot: str = None,
        verbose: int = 0,
        commandline: str = "",
    ) -> None:
        """Initialize Align_NetCDF with explicit parameters.

        Args:
            auv_name: Name of the AUV (e.g., 'Dorado389', 'i2map', 'tethys')
            mission: Mission identifier (e.g., '2011.256.02')
            base_path: Base directory path for data
            log_file: Optional LRAUV log file path for log-based processing
            plot: Optional plot specification
            verbose: Verbosity level (0=WARN, 1=INFO, 2=DEBUG)
            commandline: Command line string for metadata
        """
        self.auv_name = auv_name
        self.mission = mission
        self.base_path = base_path
        self.log_file = log_file
        self.plot = plot
        self.verbose = verbose
        self.commandline = commandline
        self.logger.setLevel(self._log_levels[verbose])

    def global_metadata(self) -> dict:  # noqa: PLR0915
        """Use instance variables to return a dictionary of
        metadata specific for the data that are written
        """
        auv_name = self.auv_name
        mission = self.mission
        log_file = self.log_file
        # Try to get actual host name, fall back to container name
        actual_hostname = os.getenv("HOST_NAME", gethostname())
        repo = git.Repo(search_parent_directories=True)
        try:
            gitcommit = repo.head.object.hexsha
        except (ValueError, BrokenPipeError) as e:
            self.logger.warning(
                "could not get head commit sha for %s: %s",
                repo.remotes.origin.url,
                e,
            )
            gitcommit = "<failed to get git commit>"
        iso_now = datetime.now(UTC).isoformat() + "Z"

        metadata = {}
        metadata["netcdf_version"] = "4"
        metadata["Conventions"] = "CF-1.6"
        metadata["date_created"] = iso_now
        metadata["date_update"] = iso_now
        metadata["date_modified"] = iso_now
        metadata["featureType"] = "trajectory"

        metadata["time_coverage_start"] = str(self.min_time)
        metadata["time_coverage_end"] = str(self.max_time)
        try:
            metadata["time_coverage_duration"] = str(self.max_time - self.min_time)
        except AttributeError:
            # Likely AttributeError: 'datetime.datetime' object has no attribute 'astype'
            self.logger.warning(
                "Could not save time_coverage_duration - likely because all data are bad "
                "and min_time and max_time were not set",
            )
        metadata["geospatial_vertical_min"] = self.min_depth
        metadata["geospatial_vertical_max"] = self.max_depth
        metadata["geospatial_lat_min"] = self.min_lat
        metadata["geospatial_lat_max"] = self.max_lat
        metadata["geospatial_lon_min"] = self.min_lon
        metadata["geospatial_lon_max"] = self.max_lon
        metadata["distribution_statement"] = "Any use requires prior approval from MBARI"
        metadata["license"] = metadata["distribution_statement"]
        metadata["useconst"] = "Not intended for legal use. Data may contain inaccuracies."
        metadata["history"] = f"Created by {self.commandline} on {iso_now}"

        if auv_name and mission:
            metadata["title"] = (
                f"Calibrated and aligned AUV sensor data from {auv_name} mission {mission}"
            )
            from_data = "calibrated data"
            metadata["source"] = (
                f"MBARI Dorado-class AUV data produced from {from_data}"
                f" with execution of '{self.commandline}' at {iso_now} on"
                f" host {actual_hostname} using git commit {gitcommit} from"
                f" software at 'https://github.com/mbari-org/auv-python'"
            )
            metadata["summary"] = (
                "Observational oceanographic data obtained from an Autonomous"
                " Underwater Vehicle mission with measurements at"
                " original sampling intervals. The data have been calibrated"
                " and the coordinate variables aligned using MBARI's auv-python"
                " software."
            )
        elif log_file:
            # Build title with optional deployment name
            title = f"Combined and aligned LRAUV instrument data from log file {Path(log_file)}"
            deployment_name = get_deployment_name(log_file, BASE_LRAUV_PATH, self.logger)
            if deployment_name:
                title += f" - Deployment: {deployment_name}"
            metadata["title"] = title

            from_data = "combined data"
            metadata["source"] = (
                f"MBARI Long Range AUV data produced from {from_data}"
                f" with execution of '{self.commandline}' at {iso_now} on"
                f" host {actual_hostname} using git commit {gitcommit} from"
                f" software at 'https://github.com/mbari-org/auv-python'"
            )
            metadata["summary"] = self.combined_nc.attrs.get(
                "summary",
                (
                    "Observational oceanographic data obtained from an Autonomous"
                    " Underwater Vehicle mission with measurements at"
                    " original sampling intervals. The position variables have been"
                    " corrected to GPS positions and aligned with the data variables"
                    " using MBARI's auv-python software."
                ),
            )
        # Append location of original data files to summary
        if self.auv_name and self.mission:
            matches = re.search(
                "(" + SUMMARY_SOURCE.replace("{}", r".+$") + ")",
                self.calibrated_nc.attrs["summary"],
            )
            metadata["comment"] = (
                f"MBARI Dorado-class AUV data produced from calibrated data"
                f" with execution of '{self.commandline}' at {iso_now} on"
                f" host {gethostname()}. Software available at"
                f" 'https://github.com/mbari-org/auv-python'"
            )
        elif log_file:
            matches = re.search(
                "(" + SUMMARY_SOURCE.replace("{}", r".+$") + ")",
                self.combined_nc.attrs["summary"],
            )
            metadata["comment"] = (
                f"MBARI LRAUV-class AUV data produced from logged data"
                f" with execution of '{self.commandline}' at {iso_now} on"
                f" host {gethostname()}. Software available at"
                f" 'https://github.com/mbari-org/auv-python'"
            )
        if matches:
            metadata["summary"] += " " + matches.group(1)

        return metadata

    def process_cal(self) -> Path:  # noqa: C901, PLR0912, PLR0915
        """Process calibrated netCDF file using instance attributes."""
        if self.mission and self.auv_name:
            netcdfs_dir = Path(self.base_path, self.auv_name, MISSIONNETCDFS, self.mission)
            src_file = Path(netcdfs_dir, f"{self.auv_name}_{self.mission}_cal.nc")
        elif self.log_file:
            netcdfs_dir = Path(BASE_LRAUV_PATH, f"{Path(self.log_file).parent}")
            src_file = Path(netcdfs_dir, f"{Path(self.log_file).stem}_cal.nc")
        else:
            msg = "Must provide either mission and vehicle or log_file"
            raise ValueError(msg)
        self.calibrated_nc = xr.open_dataset(src_file)
        self.logger.info("Processing %s", src_file)
        self.aligned_nc = xr.Dataset()
        self.min_time = datetime.now(UTC)
        self.max_time = datetime(1970, 1, 1, tzinfo=UTC)
        self.min_depth = np.inf
        self.max_depth = -np.inf
        self.min_lat = np.inf
        self.max_lat = -np.inf
        self.min_lon = np.inf
        self.max_lon = -np.inf
        for variable in self.calibrated_nc:
            instr, *_ = variable.split("_")
            self.logger.debug("instr: %s", instr)
            if instr in ("gps", "depth", "nudged"):
                # Skip coordinate type variables
                continue
            if variable.startswith("navigation") and variable.split("_")[1] not in (
                "mWaterSpeed",
                "roll",
                "pitch",
                "yaw",
            ):
                self.logger.info("Skipping %s", variable)
                continue
            # Process mWaterSpeed, roll, pitch, & yaw and variables from
            # instruments: seabird25p, ctd1, ctd2, hs2, ...
            self.logger.debug("Processing %s", variable)
            self.aligned_nc[variable] = self.calibrated_nc[variable]
            # Interpolators for the non-time dimensions
            # Fix values to first and last points for interpolation to time
            # values outside the range of the pitch values.
            try:
                lat_interp = interp1d(
                    self.calibrated_nc["nudged_latitude"].get_index("time").view(np.int64).tolist(),
                    self.calibrated_nc["nudged_latitude"].values,
                    fill_value=(
                        self.calibrated_nc["nudged_latitude"][0],
                        self.calibrated_nc["nudged_latitude"][-1],
                    ),
                    bounds_error=False,
                )
            except KeyError:
                error_message = f"No nudged_latitude data in {src_file}"
                raise InvalidCalFile(error_message) from None
            lon_interp = interp1d(
                self.calibrated_nc["nudged_longitude"].get_index("time").view(np.int64).tolist(),
                self.calibrated_nc["nudged_longitude"].values,
                fill_value=(
                    self.calibrated_nc["nudged_longitude"][0],
                    self.calibrated_nc["nudged_longitude"][-1],
                ),
                bounds_error=False,
            )
            timevar = f"{instr}_{TIME}"
            if variable == "biolume_raw":
                # biolume_raw is unique with its own time variable
                timevar = f"{instr}_{TIME60HZ}"
            try:
                depth_interp = interp1d(
                    self.calibrated_nc[f"{instr}_depth"].get_index(timevar).view(np.int64).tolist(),
                    self.calibrated_nc[f"{instr}_depth"].values,
                    fill_value=(
                        self.calibrated_nc[f"{instr}_depth"][0],
                        self.calibrated_nc[f"{instr}_depth"][-1],
                    ),
                    bounds_error=False,
                )
                self.logger.info(
                    "Using pitch corrected %s_depth: %s",
                    instr,
                    self.calibrated_nc[f"{instr}_depth"].attrs["comment"],
                )
            except KeyError:
                # No SensorInfo offset for this instr
                depth_interp = interp1d(
                    self.calibrated_nc["depth_filtdepth"]
                    .get_index("depth_time")
                    .view(np.int64)
                    .tolist(),
                    self.calibrated_nc["depth_filtdepth"].values,
                    fill_value=(
                        self.calibrated_nc["depth_filtdepth"][0],
                        self.calibrated_nc["depth_filtdepth"][-1],
                    ),
                    bounds_error=False,
                )
            except ValueError as e:
                # Likely x and y arrays must have at least 2 entries
                error_message = "Cannot interpolate depth"
                raise InvalidCalFile(error_message) from e

            var_time = self.aligned_nc[variable].get_index(timevar).view(np.int64).tolist()

            # Create new DataArrays of all the variables, including "aligned"
            # (interpolated) depth, latitude, and longitude coordinates.
            # Use attributes from the calibrated data.
            sample_rate = np.round(
                1.0 / (np.mean(np.diff(self.calibrated_nc[timevar])) / np.timedelta64(1, "s")),
                decimals=2,
            )
            self.aligned_nc[variable] = xr.DataArray(
                self.calibrated_nc[variable].values,
                dims={timevar},
                coords=[self.calibrated_nc[variable].get_index(timevar)],
                name=variable,
            )
            self.aligned_nc[variable].attrs = self.calibrated_nc[variable].attrs
            self.aligned_nc[variable].attrs["coordinates"] = (
                f"{instr}_time {instr}_depth {instr}_latitude {instr}_longitude"
            )
            self.logger.info(
                "%s: instrument_sample_rate_hz = %.2f",
                variable,
                sample_rate,
            )
            self.aligned_nc[variable].attrs["instrument_sample_rate_hz"] = sample_rate
            self.aligned_nc[f"{instr}_depth"] = xr.DataArray(
                depth_interp(var_time).astype(np.float64).tolist(),
                dims={timevar},
                coords=[self.calibrated_nc[variable].get_index(timevar)],
                name=f"{instr}_depth",
            )
            try:
                self.aligned_nc[f"{instr}_depth"].attrs = self.calibrated_nc[f"{instr}_depth"].attrs
            except KeyError:
                self.logger.debug(
                    "%s: %s_depth not found in %s",
                    variable,
                    instr,
                    self.calibrated_nc,
                )
            self.aligned_nc[f"{instr}_depth"].attrs["long_name"] = "Depth"
            self.aligned_nc[f"{instr}_depth"].attrs["instrument_sample_rate_hz"] = sample_rate

            self.aligned_nc[f"{instr}_latitude"] = xr.DataArray(
                lat_interp(var_time).astype(np.float64).tolist(),
                dims={timevar},
                coords=[self.calibrated_nc[variable].get_index(timevar)],
                name=f"{instr}_latitude",
            )
            self.aligned_nc[f"{instr}_latitude"].attrs = self.calibrated_nc["nudged_latitude"].attrs
            self.aligned_nc[f"{instr}_latitude"].attrs["comment"] += (
                f". Variable nudged_latitude from {src_file} file linearly"
                f" interpolated onto {variable.split('_')[0]} time values."
            )
            self.aligned_nc[f"{instr}_latitude"].attrs["long_name"] = "Latitude"
            self.aligned_nc[f"{instr}_latitude"].attrs["instrument_sample_rate_hz"] = sample_rate

            self.aligned_nc[f"{instr}_longitude"] = xr.DataArray(
                lon_interp(var_time).astype(np.float64).tolist(),
                dims={timevar},
                coords=[self.calibrated_nc[variable].get_index(timevar)],
                name=f"{instr}_longitude",
            )
            self.aligned_nc[f"{instr}_longitude"].attrs = self.calibrated_nc[
                "nudged_longitude"
            ].attrs
            self.aligned_nc[f"{instr}_longitude"].attrs["comment"] += (
                f". Variable nudged_longitude from {src_file} file linearly"
                f" interpolated onto {variable.split('_')[0]} time values."
            )
            self.aligned_nc[f"{instr}_longitude"].attrs["long_name"] = "Longitude"
            self.aligned_nc[f"{instr}_longitude"].attrs["instrument_sample_rate_hz"] = sample_rate

            # Update spatial temporal bounds for the global metadata
            # https://github.com/pydata/xarray/issues/4917#issue-809708107
            if pd.to_datetime(self.aligned_nc[timevar][0].values).tz_localize(
                UTC,
            ) < pd.to_datetime(self.min_time):
                self.min_time = pd.to_datetime(self.aligned_nc[timevar][0].values).tz_localize(
                    UTC,
                )
            if pd.to_datetime(self.aligned_nc[timevar][-1].values).tz_localize(
                UTC,
            ) > pd.to_datetime(self.max_time):
                self.max_time = pd.to_datetime(self.aligned_nc[timevar][-1].values).tz_localize(
                    UTC,
                )
            if self.aligned_nc[f"{instr}_depth"].min() < self.min_depth:
                self.min_depth = self.aligned_nc[f"{instr}_depth"].min().to_numpy()
            if self.aligned_nc[f"{instr}_depth"].max() > self.max_depth:
                self.max_depth = self.aligned_nc[f"{instr}_depth"].max().to_numpy()
            if self.aligned_nc[f"{instr}_latitude"].min() < self.min_lat:
                self.min_lat = self.aligned_nc[f"{instr}_latitude"].min().to_numpy()
            if self.aligned_nc[f"{instr}_latitude"].max() > self.max_lat:
                self.max_lat = self.aligned_nc[f"{instr}_latitude"].max().to_numpy()
            if self.aligned_nc[f"{instr}_longitude"].min() < self.min_lon:
                self.min_lon = self.aligned_nc[f"{instr}_longitude"].min().to_numpy()
            if self.aligned_nc[f"{instr}_longitude"].max() > self.max_lon:
                self.max_lon = self.aligned_nc[f"{instr}_longitude"].max().to_numpy()

        return netcdfs_dir

    def process_combined(self) -> Path:  # noqa: C901, PLR0912, PLR0915
        """Process combined LRAUV data from *_combined.nc files created by combine.py"""
        netcdfs_dir = Path(BASE_LRAUV_PATH, f"{Path(self.log_file).parent}")
        src_file = Path(netcdfs_dir, f"{Path(self.log_file).stem}_combined.nc")

        self.combined_nc = xr.open_dataset(src_file)
        self.logger.info("Processing %s", src_file)
        self.aligned_nc = xr.Dataset()
        self.min_time = datetime.now(UTC)
        self.max_time = datetime(1970, 1, 1, tzinfo=UTC)
        self.min_depth = np.inf
        self.max_depth = -np.inf
        self.min_lat = np.inf
        self.max_lat = -np.inf
        self.min_lon = np.inf
        self.max_lon = -np.inf

        # Coordinates - use mapping from global variable_time_coord_mapping attribute
        variable_time_coord_mapping = json.loads(
            self.combined_nc.attrs.get("variable_time_coord_mapping", "{}")
        )
        # Find navigation coordinates from combined data - must be from universals group
        nav_coords = {}
        for coord_type in ["longitude", "latitude", "depth", "time"]:
            coord_var = f"universals_{coord_type}"
            if coord_var not in self.combined_nc:
                error_message = (
                    f"Required universals coordinate {coord_var} not found in {src_file}"
                )
                raise InvalidCombinedFile(error_message)
            nav_coords[coord_type] = coord_var
            self.logger.info("Found navigation coordinate: %s", coord_var)

        # Create interpolators for navigation coordinates
        try:
            lat_interp = interp1d(
                self.combined_nc[nav_coords["latitude"]]
                .get_index(variable_time_coord_mapping[nav_coords["latitude"]])
                .view(np.int64)
                .tolist(),
                self.combined_nc[nav_coords["latitude"]].values,
                fill_value=(
                    self.combined_nc[nav_coords["latitude"]][0],
                    self.combined_nc[nav_coords["latitude"]][-1],
                ),
                bounds_error=False,
            )

            lon_interp = interp1d(
                self.combined_nc[nav_coords["longitude"]]
                .get_index(variable_time_coord_mapping[nav_coords["longitude"]])
                .view(np.int64)
                .tolist(),
                self.combined_nc[nav_coords["longitude"]].values,
                fill_value=(
                    self.combined_nc[nav_coords["longitude"]][0],
                    self.combined_nc[nav_coords["longitude"]][-1],
                ),
                bounds_error=False,
            )

            depth_interp = interp1d(
                self.combined_nc[nav_coords["depth"]]
                .get_index(variable_time_coord_mapping[nav_coords["depth"]])
                .view(np.int64)
                .tolist(),
                self.combined_nc[nav_coords["depth"]].values,
                fill_value=(
                    self.combined_nc[nav_coords["depth"]][0],
                    self.combined_nc[nav_coords["depth"]][-1],
                ),
                bounds_error=False,
            )

        except KeyError as e:
            error_message = f"Missing navigation data in {src_file}: {e}"
            raise InvalidCombinedFile(error_message) from e
        except ValueError as e:
            error_message = f"Cannot interpolate navigation coordinates: {e}"
            raise InvalidCombinedFile(error_message) from e

        # Process group-based variables (skip coordinate variables)
        for variable in self.combined_nc:
            # Skip time coordinate variables
            if variable.endswith("_time"):
                continue

            # Skip the navigation coordinate variables themselves
            if variable in nav_coords.values():
                continue

            # Extract group name from variable following convention for LRAUV data
            # enforced in combine.py where first underscore separates group name
            # from the rest of the variable name
            var_parts = variable.split("_")
            if len(var_parts) < 2:  # noqa: PLR2004
                self.logger.debug("Skipping variable with unexpected name format: %s", variable)
                continue

            # Try to find the corresponding time coordinate
            # Look for pattern: group_name + "_time"
            possible_time_coords = []
            for i in range(len(var_parts)):
                group_candidate = "_".join(var_parts[: i + 1])
                time_coord_candidate = f"{group_candidate}_time"
                if time_coord_candidate in self.combined_nc:
                    possible_time_coords.append((group_candidate, time_coord_candidate))

            if not possible_time_coords:
                self.logger.warning("No time coordinate found for variable: %s", variable)
                continue

            # Use the longest matching group name (most specific)
            group_name, timevar = max(possible_time_coords, key=lambda x: len(x[0]))
            self.logger.debug(
                "Processing %s with group %s and time %s", variable, group_name, timevar
            )

            # Get the time index for this variable
            var_time = self.combined_nc[variable].get_index(timevar).view(np.int64).tolist()

            # Calculate sampling rate
            sample_rate = np.round(
                1.0 / (np.mean(np.diff(self.combined_nc[timevar])) / np.timedelta64(1, "s")),
                decimals=2,
            )

            # Create interpolated coordinate variables for this group
            coord_names = ["depth", "latitude", "longitude"]
            coord_interps = [depth_interp, lat_interp, lon_interp]
            coord_sources = [nav_coords["depth"], nav_coords["latitude"], nav_coords["longitude"]]

            for coord_name, coord_interp, coord_source in zip(
                coord_names, coord_interps, coord_sources, strict=True
            ):
                coord_var_name = f"{group_name}_{coord_name}"

                self.aligned_nc[coord_var_name] = xr.DataArray(
                    coord_interp(var_time).astype(np.float64).tolist(),
                    dims={timevar},
                    coords=[self.combined_nc[variable].get_index(timevar)],
                    name=coord_var_name,
                )

                # Copy attributes from source coordinate
                if coord_source in self.combined_nc:
                    self.aligned_nc[coord_var_name].attrs = self.combined_nc[coord_source].attrs

                # Update attributes
                self.aligned_nc[coord_var_name].attrs["long_name"] = coord_name.title()
                self.aligned_nc[coord_var_name].attrs["instrument_sample_rate_hz"] = sample_rate

                if coord_name in ["longitude", "latitude", "depth"]:
                    self.aligned_nc[coord_var_name].attrs["comment"] = (
                        self.aligned_nc[coord_var_name].attrs.get("comment", "")
                        + f". Variable {coord_source} from {src_file} file linearly"
                        f" interpolated onto {group_name} time values."
                    )

            # Update spatial temporal bounds for global metadata
            if pd.to_datetime(self.aligned_nc[timevar][0].values).tz_localize(UTC) < pd.to_datetime(
                self.min_time
            ):
                self.min_time = pd.to_datetime(self.aligned_nc[timevar][0].values).tz_localize(UTC)
            if pd.to_datetime(self.aligned_nc[timevar][-1].values).tz_localize(
                UTC
            ) > pd.to_datetime(self.max_time):
                self.max_time = pd.to_datetime(self.aligned_nc[timevar][-1].values).tz_localize(UTC)

            time_coord = variable_time_coord_mapping.get(variable)
            depth_coord = (
                time_coord[:-5] + "_depth"
                if time_coord and time_coord.endswith("_time")
                else f"{group_name}_depth"
            )
            lat_coord = (
                time_coord[:-5] + "_latitude"
                if time_coord and time_coord.endswith("_time")
                else f"{group_name}_latitude"
            )
            lon_coord = (
                time_coord[:-5] + "_longitude"
                if time_coord and time_coord.endswith("_time")
                else f"{group_name}_longitude"
            )

            # Add interpolated depth, latitude, and longitude variables
            if depth_coord in self.combined_nc:
                self.aligned_nc[depth_coord].attrs = self.combined_nc[depth_coord].attrs
            self.aligned_nc[depth_coord] = xr.DataArray(
                depth_interp(var_time).astype(np.float64).tolist(),
                dims={timevar},
                coords=[self.combined_nc[variable].get_index(timevar)],
                name=depth_coord,
            )
            self.aligned_nc[depth_coord].attrs["long_name"] = "Depth"
            self.aligned_nc[depth_coord].attrs["comment"] = "depth from Group_Universals.nc"
            self.aligned_nc[depth_coord].attrs["instrument_sample_rate_hz"] = sample_rate

            self.aligned_nc[lat_coord] = xr.DataArray(
                lat_interp(var_time).astype(np.float64).tolist(),
                dims={timevar},
                coords=[self.combined_nc[variable].get_index(timevar)],
                name=lat_coord,
            )
            self.aligned_nc[lat_coord].attrs = self.combined_nc["nudged_latitude"].attrs
            self.aligned_nc[lat_coord].attrs["comment"] += (
                f". Variable nudged_latitude from {src_file} file linearly"
                f" interpolated onto {variable.split('_')[0]} time values."
            )
            self.aligned_nc[lat_coord].attrs["long_name"] = "Latitude"
            self.aligned_nc[lat_coord].attrs["instrument_sample_rate_hz"] = sample_rate

            self.aligned_nc[lon_coord] = xr.DataArray(
                lon_interp(var_time).astype(np.float64).tolist(),
                dims={timevar},
                coords=[self.combined_nc[variable].get_index(timevar)],
                name=lon_coord,
            )
            self.aligned_nc[lon_coord].attrs = self.combined_nc["nudged_longitude"].attrs
            self.aligned_nc[lon_coord].attrs["comment"] += (
                f". Variable nudged_longitude from {src_file} file linearly"
                f" interpolated onto {variable.split('_')[0]} time values."
            )
            self.aligned_nc[lon_coord].attrs["long_name"] = "Longitude"
            self.aligned_nc[lon_coord].attrs["instrument_sample_rate_hz"] = sample_rate

            # Update bounds using the interpolated coordinates
            if self.aligned_nc[depth_coord].min() < self.min_depth:
                self.min_depth = self.aligned_nc[depth_coord].min().to_numpy()
            if self.aligned_nc[depth_coord].max() > self.max_depth:
                self.max_depth = self.aligned_nc[depth_coord].max().to_numpy()
            if self.aligned_nc[lat_coord].min() < self.min_lat:
                self.min_lat = self.aligned_nc[lat_coord].min().to_numpy()
            if self.aligned_nc[lat_coord].max() > self.max_lat:
                self.max_lat = self.aligned_nc[lat_coord].max().to_numpy()
            if self.aligned_nc[lon_coord].min() < self.min_lon:
                self.min_lon = self.aligned_nc[lon_coord].min().to_numpy()
            if self.aligned_nc[lon_coord].max() > self.max_lon:
                self.max_lon = self.aligned_nc[lon_coord].max().to_numpy()

            # Create aligned variable with proper attributes
            self.aligned_nc[variable] = xr.DataArray(
                self.combined_nc[variable].values,
                dims={timevar},
                coords=[self.combined_nc[variable].get_index(timevar)],
                name=variable,
            )
            self.aligned_nc[variable].attrs = self.combined_nc[variable].attrs
            if (
                time_coord in self.aligned_nc
                and depth_coord in self.aligned_nc
                and lat_coord in self.aligned_nc
                and lon_coord in self.aligned_nc
            ):
                self.aligned_nc[variable].attrs["coordinates"] = (
                    f"{time_coord} {depth_coord} {lat_coord} {lon_coord}"
                )
            else:
                self.logger.info("Skipping setting coordinates attribute for %s", variable)

            self.logger.info("%s: instrument_sample_rate_hz = %.2f", variable, sample_rate)
            self.aligned_nc[variable].attrs["instrument_sample_rate_hz"] = sample_rate

        return netcdfs_dir

    def write_combined_netcdf(self, netcdfs_dir: Path) -> None:
        """Write aligned combined data to NetCDF file"""
        if self.log_file:
            # For LRAUV log files, use the log file stem for output name
            out_fn = Path(netcdfs_dir, f"{Path(self.log_file).stem}_align.nc")
        else:
            out_fn = Path(netcdfs_dir, f"{self.auv_name}_{self.mission}_align.nc")

        self.aligned_nc.attrs = self.global_metadata()
        self.logger.info("Writing aligned combined data to %s", out_fn)
        if out_fn.exists():
            self.logger.debug("Removing existing file %s", out_fn)
            out_fn.unlink()
        self.aligned_nc.to_netcdf(out_fn)
        self.logger.debug(
            "Data variables written: %s",
            ", ".join(sorted(self.aligned_nc.variables)),
        )

    def write_netcdf(self, netcdfs_dir: Path) -> None:
        """Write aligned netCDF file using instance attributes."""
        self.aligned_nc.attrs = self.global_metadata()
        out_fn = Path(netcdfs_dir, f"{self.auv_name}_{self.mission}_align.nc")
        self.logger.info("Writing aligned data to %s", out_fn)
        if out_fn.exists():
            self.logger.debug("Removing file %s", out_fn)
            out_fn.unlink()
        self.aligned_nc.to_netcdf(out_fn)
        self.logger.debug(
            "Data variables written: %s",
            ", ".join(sorted(self.aligned_nc.variables)),
        )

    def process_command_line(self):
        """Process command line arguments using shared parser infrastructure."""
        examples = "Examples:" + "\n\n"
        examples += "  Align calibrated data for some missions:\n"
        examples += "    " + sys.argv[0] + " --mission 2020.064.10\n"
        examples += "    " + sys.argv[0] + " --auv_name i2map --mission 2020.055.01\n"
        examples += "  Align combined LRAUV data:\n"
        examples += (
            "    "
            + sys.argv[0]
            + " --log_file brizo/missionlogs/2025/20250909_20250915/20250914T080941/"
            + "202509140809_202509150109.nc4\n"
        )

        # Use shared LRAUV parser since align handles both Dorado and LRAUV
        parser = get_standard_lrauv_parser(
            description=__doc__,
            epilog=examples,
        )

        # Add align-specific arguments
        parser.add_argument(
            "--plot",
            action="store_true",
            help="Create intermediate plots to validate data operations.",
        )

        args = parser.parse_args()

        # Reinitialize object with parsed arguments
        self.__init__(
            auv_name=args.auv_name,
            mission=args.mission,
            base_path=args.base_path,
            log_file=args.log_file if hasattr(args, "log_file") else None,
            plot=args.plot if hasattr(args, "plot") else False,
            verbose=args.verbose,
            commandline=" ".join(sys.argv),
        )
        self.logger.setLevel(self._log_levels[args.verbose])


if __name__ == "__main__":
    # Create with default values for command-line usage
    align_netcdf = Align_NetCDF(auv_name="", mission="", base_path="")
    align_netcdf.process_command_line()
    p_start = time.time()

    if align_netcdf.log_file:
        # Process combined LRAUV data using log_file
        netcdf_dir = align_netcdf.process_combined()
        align_netcdf.write_combined_netcdf(netcdf_dir)
    elif align_netcdf.auv_name and align_netcdf.mission:
        # Process calibrated data using auv_name and mission
        netcdf_dir = align_netcdf.process_cal()
        align_netcdf.write_netcdf(netcdf_dir)
    else:
        align_netcdf.logger.error("Must provide either --log_file or both --auv_name and --mission")
        sys.exit(1)

    align_netcdf.logger.info("Time to process: %.2f seconds", (time.time() - p_start))
