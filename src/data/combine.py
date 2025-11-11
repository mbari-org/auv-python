#!/usr/bin/env python
"""
Combine original LRAUV data from separate *_Group_*.nc files and produce a
single NetCDF file that also contains corrected (nudged) latitudes and
longitudes.

Read original data from netCDF files created by nc42netcdfs.py and write out a
single netCDF file with the important variables at original sampling intervals.
Any geometric alignment and any plumbing lag corrections can also be done during
this step. This script is similar to calibrate.py that is used for Dorado and
i2map data, but does not apply any sensor calibrations as those are done on the
LRAUV vehicles before the data is logged and unserialized to NetCDF4 files. The
QC methods implemented in calibrate.py may also be reused here. The calbrate.py
code is wrapped around the concept of "sensor" which has an anaolog in this code
of "group", but is too different to easily reuse.

The file will contain combined variables (the combined_nc member variable) and
be analogous to the original NetCDF4. Rather than using groups in NetCDF4 the
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
The file will be named with a "_combined.nc" suffix. It is analogous to the
"_cal.nc" suffix used for Dorado and i2map files and will provide a clear
indication of the stage of processing. The data are suiable for input to the
align.py script.

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
import numpy as np
import xarray as xr

import pandas as pd
from AUV import monotonic_increasing_time_indices, nudge_positions
from logs2netcdfs import AUV_NetCDF, TIME, TIME60HZ
from nc42netcdfs import BASE_LRAUV_PATH, GROUP

AVG_SALINITY = 33.6  # Typical value for upper 100m of Monterey Bay


class Range(NamedTuple):
    min: float
    max: float


# There are core data common to most all vehicles, whose groups are listed in
# BASE_GROUPS. EXPECTED_GROUPS contains additional groups for specific vehicles.
BASE_GROUPS = {
    "lrauv": [
        "CTDSeabird",
        "WetLabsBB2FL",
    ],
}

EXPECTED_GROUPS = {
    "pontus": [
        "WetLabsUBAT",
    ],
}
# Combine the BASE_GROUPS into each EXPECTED_GROUPS entry
for vehicle, groups in EXPECTED_GROUPS.items():
    EXPECTED_GROUPS[vehicle] = groups + BASE_GROUPS["lrauv"]


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
                pd.to_datetime(self.combined_nc["universals_time"].values, unit="s")[0].isoformat(),
            )
        except KeyError:
            error_message = "No universals_time variable in combined_nc"
            raise EOFError(error_message) from None
        metadata["time_coverage_end"] = str(
            pd.to_datetime(self.combined_nc["universals_time"].values, unit="s")[-1].isoformat(),
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

    def _nudge_pos(self, max_sec_diff_at_end=10):
        """Match variables from lrauv processing to those needed by
        AUV.nudged_positions() so that linear nudges to underwater dead reckoned
        positions will match the GPS positions at the surface.
        """
        try:
            lon = self.combined_nc["universals_longitude"]
        except KeyError:
            error_message = "No universals_longitude data in combined_nc"
            raise EOFError(error_message) from None
        lat = self.combined_nc["universals_latitude"]
        lon_fix = self.combined_nc["nal9602_longitude_fix"]
        lat_fix = self.combined_nc["nal9602_latitude_fix"]

        # Use the shared nudge_positions() function from AUV module
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

        return lon_nudged, lat_nudged

    def _consolidate_group_time_coords(self, ds: xr.Dataset, group_name: str) -> dict:
        """Analyze and consolidate time coordinates for a group.

        Returns:
            dict: Contains consolidated time info with keys:
                - consolidated_time_name: name of consolidated coordinate (or None)
                - consolidated_time_data: the time coordinate data (or None)
                - time_coord_mapping: dict mapping original dims to consolidated dims
        """
        # Find all time variables in this group
        time_vars = {var: ds[var] for var in ds.variables if var.lower().endswith("time")}

        if not time_vars:
            return {
                "consolidated_time_name": None,
                "consolidated_time_data": None,
                "time_coord_mapping": {},
            }

        if len(time_vars) == 1:
            # Single time coordinate - use it as consolidated
            time_name = list(time_vars.keys())[0]
            consolidated_name = f"{group_name}_time"
            self.logger.info(
                "Group %s: Single time coordinate '%s' - using as '%s'",
                group_name,
                time_name,
                consolidated_name,
            )
            return {
                "consolidated_time_name": consolidated_name,
                "consolidated_time_data": ds[time_name],
                "time_coord_mapping": {time_name: consolidated_name},
            }

        # Multiple time coordinates - check if they're identical
        time_arrays = list(time_vars.values())
        first_time = time_arrays[0]
        first_time_name = list(time_vars.keys())[0]

        all_identical = True
        for i, (_name, time_array) in enumerate(time_vars.items()):
            if i == 0:
                continue  # Skip first one (reference)

            # Compare sizes first
            if len(time_array) != len(first_time):
                all_identical = False
                break

            # Compare values with tolerance
            try:
                if not np.allclose(time_array.values, first_time.values, atol=1e-6):
                    all_identical = False
                    break
            except TypeError:
                # Handle datetime arrays
                if not np.array_equal(time_array.values, first_time.values):
                    all_identical = False
                    break

        if all_identical:
            # All time coordinates are identical - consolidate them
            consolidated_name = f"{group_name}_time"
            time_coord_mapping = dict.fromkeys(time_vars, consolidated_name)

            self.logger.info(
                "Group %s: All %d time coordinates identical - consolidating to '%s'",
                group_name,
                len(time_vars),
                consolidated_name,
            )

            return {
                "consolidated_time_name": consolidated_name,
                "consolidated_time_data": ds[first_time_name],
                "time_coord_mapping": time_coord_mapping,
            }

        # Time coordinates differ - keep them separate
        time_coord_mapping = {name: f"{group_name}_{name.lower()}" for name in time_vars}

        self.logger.warning(
            "Group %s: Time coordinates differ - keeping separate: %s",
            group_name,
            list(time_vars.keys()),
        )

        return {
            "consolidated_time_name": None,
            "consolidated_time_data": None,
            "time_coord_mapping": time_coord_mapping,
        }

    def combine_groups(self):
        log_file = self.args.log_file
        src_dir = Path(BASE_LRAUV_PATH, Path(log_file).parent)
        group_files = sorted(src_dir.glob(f"{Path(log_file).stem}_{GROUP}_*.nc"))
        self.summary_fields = set()
        self.combined_nc = xr.Dataset()
        for group_file in group_files:
            self.logger.info("Group file: %s", group_file.name)
            # Open group file without decoding to have np.allclose work properly
            with xr.open_dataset(group_file, decode_cf=False) as ds:
                # Group name to prepend variable names is lowercase with underscores removed
                group_name = group_file.stem.split(f"{GROUP}_")[1].replace("_", "").lower()
                time_info = self._consolidate_group_time_coords(ds, group_name)

                for orig_var in ds.variables:
                    if orig_var.lower().endswith("time"):
                        continue
                    new_var = group_name + "_" + orig_var.lower()
                    dim_name = time_info["time_coord_mapping"][ds[orig_var].dims[0]]
                    self.logger.info("Adding variable %-65s %s", f"{orig_var} as", new_var)
                    if (
                        orig_var in ("latitude", "longitude")
                        and ds[orig_var].attrs.get("units") == "radians"
                    ):
                        self.combined_nc[new_var] = xr.DataArray(
                            ds[orig_var].to_numpy() * 180.0 / np.pi,
                            dims=[dim_name],
                            coords=[ds[orig_var].get_index(orig_var + "_time")],
                        )
                        self.combined_nc[new_var].attrs = ds[orig_var].attrs.copy()
                        self.combined_nc[new_var].attrs["units"] = "degrees"
                    else:
                        self.combined_nc[new_var] = xr.DataArray(
                            ds[orig_var].to_numpy(),
                            dims=[dim_name],
                            coords=[ds[orig_var].get_index(orig_var + "_time")],
                        )
                        self.combined_nc[new_var].attrs = ds[orig_var].attrs.copy()

                    # Add metadata required for cf_xarray decoding
                    self.combined_nc[new_var].coords[dim_name].attrs["units"] = (
                        "seconds since 1970-01-01T00:00:00Z"
                    )
                    self.combined_nc[new_var].coords[dim_name].attrs["standard_name"] = "time"

                # Construct useful comment for consolidated time coordinate
                if time_info["consolidated_time_name"] in self.combined_nc.variables:
                    mapping_info = ", ".join(
                        [
                            f"{orig} -> {new}"
                            for orig, new in time_info["time_coord_mapping"].items()
                        ]
                    )
                    self.combined_nc[time_info["consolidated_time_name"]].attrs["comment"] = (
                        f"Consolidated time coordinate from: {mapping_info}"
                    )

        # Write out an intermediate netCDF file so that cf_xarray can decode
        # the data properly for nudging positions
        intermediate_file = self._intermediate_write_netcdf()
        with xr.open_dataset(intermediate_file, decode_cf=True) as ds:
            self.combined_nc = ds.load()

        # Add nudged longitude and latitude variables to the combined_nc dataset
        try:
            nudged_longitude, nudged_latitude = self._nudge_pos()
        except ValueError as e:
            self.logger.error("Nudging positions failed: %s", e)  # noqa: TRY400
            return
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
        # Remove the intermediate file
        Path(intermediate_file).unlink()

    def _intermediate_write_netcdf(self) -> None:
        """Write out an intermediate combined netCDF file so that data can be
        read using decode_cf=True for nudge_positions() to work with cf accessors."""
        log_file = self.args.log_file
        netcdfs_dir = Path(BASE_LRAUV_PATH, Path(log_file).parent)
        out_fn = Path(netcdfs_dir, f"{Path(log_file).stem}_combined_intermediate.nc")

        self.combined_nc.attrs = self.global_metadata()
        self.logger.info("Writing intermediate combined group data to %s", out_fn)
        if Path(out_fn).exists():
            Path(out_fn).unlink()
        self.combined_nc.to_netcdf(out_fn)
        self.logger.info(
            "Data variables written: %s",
            ", ".join(sorted(self.combined_nc.variables)),
        )
        self.logger.info(
            "Wrote intermediate (_combined_intermediate.nc) netCDF file: %s",
            out_fn,
        )
        return out_fn

    def write_netcdf(self) -> None:
        log_file = self.args.log_file
        netcdfs_dir = Path(BASE_LRAUV_PATH, Path(log_file).parent)
        out_fn = Path(netcdfs_dir, f"{Path(log_file).stem}_combined.nc")

        self.combined_nc.attrs = self.global_metadata()
        self.logger.info("Writing combined group data to %s", out_fn)
        if Path(out_fn).exists():
            Path(out_fn).unlink()
        self.combined_nc.to_netcdf(out_fn)
        self.logger.info(
            "Data variables written: %s",
            ", ".join(sorted(self.combined_nc.variables)),
        )
        self.logger.info("Wrote combined (_combined.nc) netCDF file: %s", out_fn)

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
