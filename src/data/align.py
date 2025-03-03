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

import argparse
import logging
import re
import sys
import time
from argparse import RawTextHelpFormatter
from datetime import datetime, timezone
from pathlib import Path
from socket import gethostname

import git
import numpy as np
import pandas as pd
import xarray as xr
from logs2netcdfs import BASE_PATH, MISSIONNETCDFS, SUMMARY_SOURCE, TIME, TIME60HZ, AUV_NetCDF
from numpy.core._exceptions import UFuncTypeError
from scipy.interpolate import interp1d


class InvalidCalFile(Exception):
    pass


class Align_NetCDF:
    logger = logging.getLogger(__name__)
    _handler = logging.StreamHandler()
    _handler.setFormatter(AUV_NetCDF._formatter)
    logger.addHandler(_handler)
    _log_levels = (logging.WARN, logging.INFO, logging.DEBUG)

    def global_metadata(self):
        """Use instance variables to return a dictionary of
        metadata specific for the data that are written
        """
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
        iso_now = datetime.now(timezone.utc).isoformat() + "Z"

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
            metadata["time_coverage_duration"] = str(
                pd.to_datetime(self.max_time.astype("float64") / 1.0e9, unit="s")
                - pd.to_datetime(self.min_time.astype("float64") / 1.0e9, unit="s"),
            )
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

        metadata["title"] = (
            f"Calibrated and aligned AUV sensor data from"
            f" {self.args.auv_name} mission {self.args.mission}"
        )
        from_data = "calibrated data"
        metadata["source"] = (
            f"MBARI Dorado-class AUV data produced from {from_data}"
            f" with execution of '{self.commandline}' at {iso_now} on"
            f" host {gethostname()} using git commit {gitcommit} from"
            f" software at 'https://github.com/mbari-org/auv-python'"
        )
        metadata["summary"] = (
            "Observational oceanographic data obtained from an Autonomous"
            " Underwater Vehicle mission with measurements at"
            " original sampling intervals. The data have been calibrated"
            " and the coordinate variables aligned using MBARI's auv-python"
            " software."
        )
        # Append location of original data files to summary
        matches = re.search(
            "(" + SUMMARY_SOURCE.replace("{}", r".+$") + ")",
            self.calibrated_nc.attrs["summary"],
        )
        if matches:
            metadata["summary"] += " " + matches.group(1)
        metadata["comment"] = (
            f"MBARI Dorado-class AUV data produced from calibrated data"
            f" with execution of '{self.commandline}' at {iso_now} on"
            f" host {gethostname()}. Software available at"
            f" 'https://github.com/mbari-org/auv-python'"
        )

        return metadata

    def process_cal(self, vehicle: str = "", name: str = "") -> None:  # noqa: C901, PLR0912, PLR0915
        name = name or self.args.mission
        vehicle = vehicle or self.args.auv_name
        netcdfs_dir = Path(self.args.base_path, vehicle, MISSIONNETCDFS, name)
        in_fn = f"{vehicle}_{name}_cal.nc"
        try:
            self.calibrated_nc = xr.open_dataset(Path(netcdfs_dir, in_fn))
        except ValueError as e:
            raise InvalidCalFile(e) from e
        self.logger.info("Processing %s from %s", in_fn, netcdfs_dir)
        self.aligned_nc = xr.Dataset()
        self.min_time = datetime.now(timezone.utc)
        self.max_time = datetime(1970, 1, 1, tzinfo=timezone.utc)
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
                error_message = f"No nudged_latitude data in {in_fn}"
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
            try:
                sample_rate = np.round(
                    1.0 / (np.mean(np.diff(self.calibrated_nc[timevar])) / np.timedelta64(1, "s")),
                    decimals=2,
                )
            except UFuncTypeError as e:
                # Seen in dorado 2008.010.10 - caused by time variable missing from lopc.nc
                self.logger.warning("UFuncTypeError: %s", e)
                self.logger.debug(
                    f"type(type(self.calibrated_nc[variable].get_index(f'{instr}_time'))"  # noqa: G004
                    f" = {type(self.calibrated_nc[variable].get_index(f'{instr}_time'))}",
                )
                self.logger.warning(
                    f"{variable}: Failed to calculate sample_rate -"  # noqa: G004
                    f" xarray wrote {instr}_time as RangeIndex rather than actual time values -"
                    f" skipping it",
                )
                del self.aligned_nc[variable]
                continue
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
                f". Variable nudged_latitude from {in_fn} file linearly"
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
                f". Variable nudged_longitude from {in_fn} file linearly"
                f" interpolated onto {variable.split('_')[0]} time values."
            )
            self.aligned_nc[f"{instr}_longitude"].attrs["long_name"] = "Longitude"
            self.aligned_nc[f"{instr}_longitude"].attrs["instrument_sample_rate_hz"] = sample_rate

            # Update spatial temporal bounds for the global metadata
            # https://github.com/pydata/xarray/issues/4917#issue-809708107
            if pd.to_datetime(self.aligned_nc[timevar][0].values).tz_localize(
                timezone.utc,
            ) < pd.to_datetime(self.min_time):
                self.min_time = pd.to_datetime(self.aligned_nc[timevar][0].values).tz_localize(
                    timezone.utc,
                )
            if pd.to_datetime(self.aligned_nc[timevar][-1].values).tz_localize(
                timezone.utc,
            ) > pd.to_datetime(self.max_time):
                self.max_time = pd.to_datetime(self.aligned_nc[timevar][-1].values).tz_localize(
                    timezone.utc,
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

    def write_netcdf(self, netcdfs_dir, vehicle: str = "", name: str = "") -> None:
        name = name or self.args.mission
        vehicle = vehicle or self.args.auv_name
        self.aligned_nc.attrs = self.global_metadata()
        out_fn = Path(netcdfs_dir, f"{vehicle}_{name}_align.nc")
        self.logger.info("Writing aligned data to %s", out_fn)
        if out_fn.exists():
            self.logger.debug("Removing file %s", out_fn)
            out_fn.unlink()
        self.aligned_nc.to_netcdf(out_fn)
        self.logger.info(
            "Data variables written: %s",
            ", ".join(sorted(self.aligned_nc.variables)),
        )

    def process_command_line(self):
        examples = "Examples:" + "\n\n"
        examples += "  Align calibrated data for some missions:\n"
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
            "--plot",
            action="store_true",
            help="Create intermediate plots to validate data operations.",
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
                [f"{i}: {v}" for i, v in enumerate(("WARN", "INFO", "DEBUG"))],
            ),
        )
        self.args = parser.parse_args()
        self.logger.setLevel(self._log_levels[self.args.verbose])
        self.commandline = " ".join(sys.argv)


if __name__ == "__main__":
    align_netcdf = Align_NetCDF()
    align_netcdf.process_command_line()
    p_start = time.time()
    netcdf_dir = align_netcdf.process_cal()
    align_netcdf.write_netcdf(netcdf_dir)
    align_netcdf.logger.info("Time to process: %.2f seconds", (time.time() - p_start))
