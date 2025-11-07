#!/usr/bin/env python
"""
Combine original LRAUV data from separate .nc files and produce a single NetCDF
file that also contains corrected (nudged) latitudes and longitudes.

Read original data from netCDF files created by nc42netcdfs.py and write out a
single netCDF file with the important variables at original sampling intervals.
Geometric alignment and any plumbing lag corrections are also done during this
step. This script is similar to calibrate.py that is used for Dorado and i2map
data, but does not apply any sensor calibrations as those are done on the LRAUV
vehicles before the data is logged and unserialized to NetCDF-4 files. The QC
methods implemented in calibrate.py will be reused here.

The file will contain combined variables (the combined_nc member variable) and
be analogous to the original NetCDF-4. Rather than using groups in NetCDF-4 the
data will be written in classic NetCDF-CF with a naming convention that is
similar to Dorado data, with group names (any underscores removed) preceeding
the variable name with an underscore - all lower case characters:
```
    <group>_<variable_1>
    <group>_<..........>
    <group>_<variable_n>
    <group>_time
    <group>_depth
    <group>_latitude
    <group>_longitude
```
The file will be named with a "_cal.nc" suffix to be consistent with the Dorado
and i2map files, indicating the stage of processing.

"""

__author__ = "Mike McCann"
__copyright__ = "Copyright 2025, Monterey Bay Aquarium Research Institute"

import argparse  # noqa: I001
import logging
import sys
import time
from argparse import RawTextHelpFormatter
from datetime import UTC
from pathlib import Path
from socket import gethostname
from typing import NamedTuple
import cf_xarray  # Needed for the .cf accessor  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.interpolate import interp1d

import pandas as pd
from AUV import monotonic_increasing_time_indices, nudge_positions
from logs2netcdfs import AUV_NetCDF, TIME, TIME60HZ
from nc42netcdfs import BASE_LRAUV_PATH, GROUP

AVG_SALINITY = 33.6  # Typical value for upper 100m of Monterey Bay


class Range(NamedTuple):
    min: float
    max: float


# Using lower case vehicle names, modify in _define_sensor_info() for changes
# over time Used to reduce ERROR & WARNING log messages for expected missing
# sensor data. There are core data common to most all vehicles, whose groups
# are listed in BASE_GROUPS. EXPECTED_GROUPS contains additional groups for
# specific vehicles.
BASE_GROUPS = {
    "lrauv": [
        "CTDSeabird",
        "WetLabsBB2FL",
    ],
}

EXPECTED_GROUPS = {
    "dorado": [
        "navigation",
        "gps",
        "depth",
        "ecopuck",
        "hs2",
        "ctd1",
        "ctd2",
        "isus",
        "biolume",
        "lopc",
        "tailcone",
    ],
    "i2map": [
        "navigation",
        "gps",
        "depth",
        "seabird25p",
        "transmissometer",
        "tailcone",
    ],
}
# Used in test fixture in conftetst.py
EXPECTED_GROUPS["Dorado389"] = EXPECTED_GROUPS["dorado"]


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


class Combine_NetCDF:
    logger = logging.getLogger(__name__)
    _handler = logging.StreamHandler()
    _handler.setFormatter(AUV_NetCDF._formatter)
    logger.addHandler(_handler)
    _log_levels = (logging.WARN, logging.INFO, logging.DEBUG)

    def global_metadata(self):
        """Use instance variables to return a dictionary of
        metadata specific for the data that are written
        """
        from datetime import datetime

        iso_now = datetime.now(tz=UTC).isoformat() + "Z"

        metadata = {}
        metadata["netcdf_version"] = "4"
        metadata["Conventions"] = "CF-1.6"
        metadata["date_created"] = iso_now
        metadata["date_update"] = iso_now
        metadata["date_modified"] = iso_now
        metadata["featureType"] = "trajectory"
        try:
            metadata["time_coverage_start"] = str(
                self.combined_nc["depth_time"].to_pandas().iloc[0].isoformat(),
            )
        except KeyError:
            error_message = "No depth_time variable in combined_nc"
            raise EOFError(error_message) from None
        metadata["time_coverage_end"] = str(
            self.combined_nc["depth_time"].to_pandas().iloc[-1].isoformat(),
        )
        metadata["distribution_statement"] = "Any use requires prior approval from MBARI"
        metadata["license"] = metadata["distribution_statement"]
        metadata["useconst"] = "Not intended for legal use. Data may contain inaccuracies."
        metadata["history"] = f"Created by {self.commandline} on {iso_now}"
        log_file = self.args.log_file
        metadata["title"] = (
            f"Combined LRAUV data from {log_file} - relevant variables extracted for STOQS"
        )
        metadata["summary"] = (
            "Observational oceanographic data obtained from a Long Range Autonomous"
            " Underwater Vehicle mission with measurements at"
            " original sampling intervals. The data have been processed"
            " by MBARI's auv-python software."
        )
        if self.summary_fields:
            # Should be just one item in set, but just in case join them
            metadata["summary"] += " " + ". ".join(self.summary_fields)
        metadata["comment"] = (
            f"MBARI Long Range AUV data produced from original data"
            f" with execution of '{self.commandline}'' at {iso_now} on"
            f" host {gethostname()}. Software available at"
            f" 'https://github.com/mbari-org/auv-python'"
        )

        return metadata

    def _range_qc_combined_nc(  # noqa: C901, PLR0912
        self,
        instrument: str,
        variables: list[str],
        ranges: dict,
        set_to_nan: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """For variables in combined_nc remove values that fall outside
        of specified min, max range.  Meant to be called by instrument so
        that the union of bad values from a set of variables can be removed.
        Use set_to_nan=True to set values outside of range to NaN instead of
        removing all variables from the instrument.  Setting set_to_nan=True
        makes sense for record (data) variables - such as ctd1_salinity,
        but not for coordinate variables."""
        out_of_range_indices = np.array([], dtype=int)
        vars_checked = []
        for var in variables:
            if var in self.combined_nc.variables:
                if var in ranges:
                    out_of_range = np.where(
                        (self.combined_nc[var] < ranges[var].min)
                        | (self.combined_nc[var] > ranges[var].max),
                    )[0]
                    self.logger.debug(
                        "%s: %d out of range values = %s",
                        var,
                        len(self.combined_nc[var][out_of_range].to_numpy()),
                        self.combined_nc[var][out_of_range].to_numpy(),
                    )
                    out_of_range_indices = np.union1d(
                        out_of_range_indices,
                        out_of_range,
                    )
                    if len(out_of_range_indices) > 500:  # noqa: PLR2004
                        self.logger.warning(
                            "More than 500 (%d) %s values found outside of range. "
                            "This may indicate a problem with the %s data.",
                            len(self.combined_nc[var][out_of_range_indices].to_numpy()),
                            var,
                            instrument,
                        )
                    if set_to_nan and var not in self.combined_nc.coords:
                        self.logger.info(
                            "Setting %s %s values to NaN", len(out_of_range_indices), var
                        )
                        self.combined_nc[var][out_of_range_indices] = np.nan
                    vars_checked.append(var)
                else:
                    self.logger.debug("No Ranges set for %s", var)
            else:
                self.logger.warning("%s not in self.combined_nc", var)
        inst_vars = [
            str(var) for var in self.combined_nc.variables if str(var).startswith(f"{instrument}_")
        ]
        self.logger.info(
            "Checked for data outside of these variables and ranges: %s",
            [(v, ranges[v]) for v in vars_checked],
        )
        if not set_to_nan:
            for var in inst_vars:
                self.logger.info(
                    "%s: deleting %d values found outside of above ranges: %s",
                    var,
                    len(self.combined_nc[var][out_of_range_indices].to_numpy()),
                    self.combined_nc[var][out_of_range_indices].to_numpy(),
                )
                coord = next(iter(self.combined_nc[var].coords))
                self.combined_nc[f"{var}_qced"] = (
                    self.combined_nc[var]
                    .drop_isel({coord: out_of_range_indices})
                    .rename({f"{instrument}_time": f"{instrument}_time_qced"})
                )
            self.combined_nc = self.combined_nc.drop_vars(inst_vars)
            for var in inst_vars:
                self.logger.debug("Renaming %s_qced to %s", var, var)
                self.combined_nc[var] = self.combined_nc[f"{var}_qced"].rename(
                    {f"{coord}_qced": coord},
                )
            qced_vars = [f"{var}_qced" for var in inst_vars]
            self.combined_nc = self.combined_nc.drop_vars(qced_vars)
        self.logger.info("Done range checking %s", instrument)

    def _apply_plumbing_lag(
        self,
        sensor: str,
        time_index: pd.DatetimeIndex,
        time_name: str,
    ) -> tuple[xr.DataArray, str]:
        """
        Apply plumbing lag to a time index in the combined netCDF file.
        """
        # Convert lag_secs to milliseconds as np.timedelta64 neeeds an integer
        lagged_time = time_index - np.timedelta64(
            int(self.sinfo[sensor]["lag_secs"] * 1000),
            "ms",
        )
        # Need to update the sensor's time coordinate in the combined netCDF file
        # so that DataArrays created with lagged_time fit onto the coordinate
        self.combined_nc.coords[f"{sensor}_{time_name}"] = xr.DataArray(
            lagged_time,
            coords=[lagged_time],
            dims={f"{sensor}_{time_name}"},
            name=f"{sensor}_{time_name}",
        )
        lag_info = f"with plumbing lag correction of {self.sinfo[sensor]['lag_secs']} seconds"
        return lagged_time, lag_info

    def _biolume_process(self, sensor):
        try:
            orig_nc = getattr(self, sensor).orig_data
        except FileNotFoundError as e:
            self.logger.error("%s", e)  # noqa: TRY400
            return
        except AttributeError:
            error_message = f"{sensor} has no orig_data - likely a missing or zero-sized .log file"
            raise EOFError(error_message) from None

        # Remove non-monotonic times
        self.logger.debug("Checking for non-monotonic increasing time")
        monotonic = monotonic_increasing_time_indices(orig_nc.get_index("time"))
        if (~monotonic).any():
            self.logger.debug(
                "Removing non-monotonic increasing time at indices: %s",
                np.argwhere(~monotonic).flatten(),
            )
        orig_nc = orig_nc.sel({TIME: monotonic})

        self.logger.info("Checking for non-monotonic increasing %s", TIME60HZ)
        monotonic = monotonic_increasing_time_indices(orig_nc.get_index(TIME60HZ))
        if (~monotonic).any():
            self.logger.info(
                "Removing non-monotonic increasing %s at indices: %s",
                TIME60HZ,
                np.argwhere(~monotonic).flatten(),
            )
        orig_nc = orig_nc.sel({TIME60HZ: monotonic})

        self.combined_nc[f"{sensor}_depth"] = self._geometric_depth_correction(
            sensor,
            orig_nc,
        )

        source = self.sinfo[sensor]["data_filename"]
        self.combined_nc["biolume_flow"] = xr.DataArray(
            orig_nc["flow"].to_numpy() * self.sinfo["biolume"]["flow_conversion"],
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

        lagged_time, lag_info = self._apply_plumbing_lag(
            sensor,
            orig_nc.get_index(TIME),
            TIME,
        )
        self.combined_nc["biolume_avg_biolume"] = xr.DataArray(
            orig_nc["avg_biolume"].to_numpy(),
            coords=[lagged_time],
            dims={f"{sensor}_{TIME}"},
            name=f"{sensor}_avg_biolume",
        )
        self.combined_nc["biolume_avg_biolume"].attrs = {
            "long_name": "Bioluminesence Average of 60Hz data",
            "units": "photons s^-1",
            "coordinates": f"{sensor}_{TIME} {sensor}_depth",
            "comment": f"avg_biolume from {source} {lag_info}",
        }

        lagged_time, lag_info = self._apply_plumbing_lag(
            sensor,
            orig_nc.get_index(TIME60HZ),
            TIME60HZ,
        )
        self.combined_nc["biolume_raw"] = xr.DataArray(
            orig_nc["raw"].to_numpy(),
            coords=[lagged_time],
            dims={f"{sensor}_{TIME60HZ}"},
            name=f"{sensor}_raw",
        )
        self.combined_nc["biolume_raw"].attrs = {
            "long_name": "Raw 60 hz biolume data",
            # xarray writes out its own units attribute
            "coordinates": f"{sensor}_{TIME60HZ} {sensor}_depth60hz",
            "comment": f"raw values from {source} {lag_info}",
        }
        if self.args.mission == "2010.284.00":
            self.logger.info(
                "Removing points outside of time range for %s/biolume.nc", self.args.mission
            )
            for time_axis in (TIME, TIME60HZ):
                self._range_qc_combined_nc(
                    instrument=sensor,
                    variables=[
                        "biolume_time",
                        "biolume_time60hz",
                        "biolume_depth",
                        "biolume_flow",
                        "biolume_avg_biolume",
                        "biolume_raw",
                    ],
                    ranges={
                        f"{sensor}_{time_axis}": Range(
                            pd.Timestamp(2010, 10, 11, 20, 0, 0),
                            pd.Timestamp(2010, 10, 12, 3, 28, 0),
                        ),
                    },
                    set_to_nan=True,
                )

    def _geometric_depth_correction(self, sensor, orig_nc):
        """Performs the align_geom() function from the legacy Matlab.
        Works for any sensor, but requires navigation being processed first
        as its variables in combined_nc are required. Returns corrected depth
        array.
        """
        # Fix pitch values to first and last points for interpolation to time
        # values outside the range of the pitch values.
        # See https://stackoverflow.com/a/45446546
        # and https://github.com/scipy/scipy/issues/12707#issuecomment-672794335
        try:
            p_interp = interp1d(
                self.combined_nc["navigation_time"].to_numpy().tolist(),
                self.combined_nc["navigation_pitch"].to_numpy(),
                fill_value=(
                    self.combined_nc["navigation_pitch"].to_numpy()[0],
                    self.combined_nc["navigation_pitch"].to_numpy()[-1],
                ),
                bounds_error=False,
            )
        except KeyError:
            error_message = "No navigation_time or navigation_pitch in combined_nc."
            raise EOFError(error_message) from None
        pitch = p_interp(orig_nc["time"].to_numpy().tolist())

        d_interp = interp1d(
            self.combined_nc["depth_time"].to_numpy().tolist(),
            self.combined_nc["depth_filtdepth"].to_numpy(),
            fill_value=(
                self.combined_nc["depth_filtdepth"].to_numpy()[0],
                self.combined_nc["depth_filtdepth"].to_numpy()[-1],
            ),
            bounds_error=False,
        )
        orig_depth = d_interp(orig_nc["time"].to_numpy().tolist())
        offs_depth = align_geom(self.sinfo[sensor]["sensor_offset"], pitch)

        corrected_depth = xr.DataArray(
            (orig_depth - offs_depth).astype(np.float64).tolist(),
            coords=[orig_nc.get_index("time")],
            dims={f"{sensor}_time"},
            name=f"{sensor}_depth",
        )
        # 2008.289.03 has self.combined_nc["depth_time"][-1] (2008-10-16T15:42:32)
        # at lot less than               orig_nc["time"][-1] (2008-10-16T16:24:43)
        # which, with "extrapolate" causes wildly incorrect depths to -359 m
        # There may be other cases where this happens, in which case we'd like
        # a general solution. For now, we'll just correct this mission.
        d_beg_time_diff = (
            orig_nc["time"].to_numpy()[0] - self.combined_nc["depth_time"].to_numpy()[0]
        )
        d_end_time_diff = (
            orig_nc["time"].to_numpy()[-1] - self.combined_nc["depth_time"].to_numpy()[-1]
        )
        self.logger.info(
            "%s: d_beg_time_diff: %s, d_end_time_diff: %s",
            sensor,
            d_beg_time_diff.astype("timedelta64[s]"),
            d_end_time_diff.astype("timedelta64[s]"),
        )
        if self.args.mission in (
            "2008.289.03",
            "2010.259.01",
            "2010.259.02",
        ):
            # This could be a more general check for all missions, but let's restrict it
            # to known problematic missions for now.  The above info message can help
            # determine if this is needed for other missions.
            self.logger.info(
                "%s: Special QC for mission %s: Setting corrected_depth to NaN for times after %s",
                sensor,
                self.args.mission,
                self.combined_nc["depth_time"][-1].to_numpy(),
            )
            corrected_depth[
                np.where(
                    orig_nc.get_index("time") > self.combined_nc["depth_time"].to_numpy()[-1],
                )
            ] = np.nan
        if self.args.plot:
            plt.figure(figsize=(18, 6))
            plt.plot(
                orig_nc["time"].to_numpy(),
                orig_depth,
                "-",
                orig_nc["time"].to_numpy(),
                corrected_depth,
                "--",
                orig_nc["time"].to_numpy(),
                pitch,
                ".",
            )
            plt.ylabel("Depth (m) & Pitch (deg)")
            plt.legend(("Original depth", "Pitch corrected depth", "Pitch"))
            plt.title(
                f"Original and pitch corrected depth for {self.args.auv_name} {self.args.mission}",
            )
            plt.show()

        return corrected_depth

    def _nudge_pos(self, max_sec_diff_at_end=10):
        """Apply linear nudges to underwater latitudes and longitudes so that
        they match the surface gps positions.
        """
        try:
            lon = self.combined_nc["universals_longitude"]
        except KeyError:
            error_message = "No universals_longitude data in combined_nc"
            raise EOFError(error_message) from None
        lat = self.combined_nc["universals_latitude"]
        lon_fix = self.combined_nc["nal9602_longitude_fix"]
        lat_fix = self.combined_nc["nal9602_latitude_fix"]

        # Use the shared function from AUV module
        lon_nudged, lat_nudged, segment_count, segment_minsum = nudge_positions(
            nav_longitude=lon,
            nav_latitude=lat,
            gps_longitude=lon_fix,
            gps_latitude=lat_fix,
            logger=self.logger,
            auv_name="",
            mission="",
            max_sec_diff_at_end=max_sec_diff_at_end,
            create_plots=True,
        )

        # Store results in instance variables for compatibility
        self.segment_count = segment_count
        self.segment_minsum = segment_minsum

        return lon_nudged, lat_nudged

    def combine_groups(self):
        log_file = self.args.log_file
        src_dir = Path(BASE_LRAUV_PATH, Path(log_file).parent)
        group_files = sorted(src_dir.glob(f"{Path(log_file).stem}_{GROUP}_*.nc"))
        self.combined_nc = xr.Dataset()
        for group_file in group_files:
            self.logger.info("Group file: %s", group_file.name)
            # Loop through each variable in the group file and add it to the combined_nc member list
            with xr.open_dataset(group_file) as ds:
                for orig_var in ds.variables:
                    if orig_var.lower().endswith("time"):
                        self.logger.debug("Skipping time variable: %s", orig_var)
                        continue
                    new_group = group_file.stem.split(f"{GROUP}_")[1].replace("_", "").lower()
                    new_var = new_group + "_" + orig_var.lower()
                    self.logger.info("Adding variable %-65s %s", f"{orig_var} as", new_var)
                    self.combined_nc[new_var] = ds[orig_var]

        # Add nudged longitude and latitude variables to the combined_nc dataset
        nudged_longitude, nudged_latitude = self._nudge_pos()
        self.combined_nc["nudged_longitude"] = nudged_longitude
        self.combined_nc["nudged_longitude"].attrs = {
            "long_name": "Nudged Longitude",
            "standard_name": "longitude",
            "units": "degrees_east",
            "comment": "Dead reckoned longitude nudged to GPS positions",
        }
        self.combined_nc["nudged_latitude"] = nudged_latitude
        self.combined_nc["nudged_latitude"].attrs = {
            "long_name": "Nudged Latitude",
            "standard_name": "latitude",
            "units": "degrees_north",
            "comment": "Dead reckoned latitude nudged to GPS positions",
        }

    def write_netcdf(self) -> None:
        log_file = self.args.log_file
        netcdfs_dir = Path(BASE_LRAUV_PATH, Path(log_file).parent)
        out_fn = Path(netcdfs_dir, f"{Path(log_file).stem}_cal.nc")

        self.combined_nc.attrs = self.global_metadata()
        self.logger.info("Writing combined group data to %s", out_fn)
        if Path(out_fn).exists():
            Path(out_fn).unlink()
        self.combined_nc.to_netcdf(out_fn)
        self.logger.info(
            "Data variables written: %s",
            ", ".join(sorted(self.combined_nc.variables)),
        )

        return netcdfs_dir

    def process_command_line(self):
        examples = "Examples:" + "\n\n"
        examples += "  Combine original data from Group files for an LRAUV log file:\n"
        examples += (
            "    "
            + sys.argv[0]
            + " -v --log_file brizo/missionlogs/2025/"
            + "20250909_20250915/20250914T080941/"
            + "202509140809_202509150109.nc4\n"
        )

        parser = argparse.ArgumentParser(
            formatter_class=RawTextHelpFormatter,
            description=__doc__,
            epilog=examples,
        )

        parser.add_argument(
            "--noinput",
            action="store_true",
            help="Execute without asking for a response, e.g. to not ask to re-download file",
        )
        parser.add_argument(
            "--log_file",
            action="store",
            help=(
                "Path to the log file of original LRAUV data, e.g.: "
                "brizo/missionlogs/2025/20250903_20250909/"
                "20250905T072042/202509050720_202509051653.nc4"
            ),
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
                [f"{i}: {v}" for i, v in enumerate(("WARN", "INFO", "DEBUG"))],
            ),
        )

        self.args = parser.parse_args()
        self.logger.setLevel(self._log_levels[self.args.verbose])

        self.commandline = " ".join(sys.argv)


if __name__ == "__main__":
    combine = Combine_NetCDF()
    combine.process_command_line()
    start = time.time()
    combine.combine_groups()
    combine.write_netcdf()
    combine.logger.info("Time to process: %.2f seconds", (time.time() - start))
