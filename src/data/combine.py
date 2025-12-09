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

import json  # noqa: I001
import logging
import os
import sys
import time
from datetime import UTC
from pathlib import Path
from socket import gethostname
from typing import NamedTuple

import cf_xarray  # Needed for the .cf accessor  # noqa: F401
import numpy as np
import pandas as pd
import xarray as xr
from utils import monotonic_increasing_time_indices, nudge_positions
from common_args import get_standard_lrauv_parser
from logs2netcdfs import AUV_NetCDF, TIME, TIME60HZ
from nc42netcdfs import BASE_LRAUV_PATH, GROUP
from utils import get_deployment_name

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
    variable_time_coord_mapping: dict = {}
    TIME_MATCH_TOLERANCE = 1e-6  # seconds tolerance for matching time values

    def __init__(
        self,
        log_file: str = None,
        verbose: int = 0,
        plot: str = None,
        commandline: str = "",
    ) -> None:
        """Initialize Combine_NetCDF with explicit parameters.

        Args:
            log_file: LRAUV log file path for processing (required for processing, optional for CLI)
            verbose: Verbosity level (0=WARN, 1=INFO, 2=DEBUG)
            plot: Optional plot specification
            commandline: Command line string for metadata
        """
        self.log_file = log_file
        self.verbose = verbose
        self.plot = plot
        self.commandline = commandline
        self.nudge_segment_count = None
        self.nudge_total_minutes = None
        if verbose:
            self.logger.setLevel(self._log_levels[verbose])

    def global_metadata(self):
        """Use instance variables to return a dictionary of
        metadata specific for the data that are written
        """
        from datetime import datetime

        # Try to get actual host name, fall back to container name
        actual_hostname = os.getenv("HOST_NAME", gethostname())

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
        metadata["variable_time_coord_mapping"] = json.dumps(self.variable_time_coord_mapping)
        log_file = self.log_file

        # Build title with optional deployment name
        title = f"Combined LRAUV data from {log_file}"
        deployment_name = get_deployment_name(log_file, BASE_LRAUV_PATH, self.logger)
        if deployment_name:
            title += f" - Deployment: {deployment_name}"
        metadata["title"] = title

        metadata["summary"] = (
            "Observational oceanographic data obtained from a Long Range Autonomous"
            " Underwater Vehicle mission with measurements at"
            " original sampling intervals. The data have been processed"
            " by MBARI's auv-python software."
        )
        if self.summary_fields:
            # Should be just one item in set, but just in case join them
            metadata["summary"] += " " + ". ".join(self.summary_fields)

        # Add nudging information to summary if available
        if self.nudge_segment_count is not None and self.nudge_total_minutes is not None:
            metadata["summary"] += (
                f" {self.nudge_segment_count} underwater segments over "
                f"{self.nudge_total_minutes:.1f} minutes nudged toward GPS fixes."
            )

        metadata["comment"] = (
            f"MBARI Long Range AUV data produced from original data"
            f" with execution of '{self.commandline}'' at {iso_now} on"
            f" host {actual_hostname}. Software available at"
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
                    .rename({f"{coord}": f"{coord}_qced"})
                    .rename(f"{var}_qced")
                )
            self.combined_nc = self.combined_nc.drop_vars(inst_vars)
            for var in inst_vars:
                self.logger.debug("Renaming %s_qced to %s", var, var)
                coord = next(iter(self.combined_nc[f"{var}_qced"].coords))
                self.combined_nc[var] = (
                    self.combined_nc[f"{var}_qced"]
                    .rename(
                        {f"{coord}": coord[:-5]},  # Remove '_qced' suffix from coord name
                    )
                    .rename(var)
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

    def _cons_group_time_coords(self, ds: xr.Dataset, group_name: str) -> dict:  # noqa: C901
        """Analyze and consolidate time coordinates for a group.

        Returns:
            dict: Contains consolidated time info with keys:
                - consolidated_time_name: name of consolidated coordinate (or None)
                - consolidated_time_data: the time coordinate data (or None)
                - time_coord_mapping: dict mapping original dims to consolidated dims
                - variable_time_coord_mapping: dict mapping variables to their time coords
        """
        if group_name.lower() == "universals":
            # Do not include the "time" record variable from universals group
            time_vars = {var: ds[var] for var in ds.variables if var.lower().endswith("_time")}
        else:
            time_vars = {var: ds[var] for var in ds.variables if var.lower().endswith("time")}

        if not time_vars:
            return {
                "consolidated_time_name": None,
                "consolidated_time_data": None,
                "time_coord_mapping": {},
                "variable_time_coord_mapping": {},
            }

        if len(time_vars) == 1:
            # Single time coordinate - use it as consolidated
            time_name = list(time_vars.keys())[0]
            consolidated_name = f"{group_name}_time"
            self.logger.info(
                "Group %s: Single time coordinate '%s' -> '%s'",
                group_name,
                time_name,
                consolidated_name,
            )
            time_coord_mapping = {time_name: consolidated_name}
            return {
                "consolidated_time_name": consolidated_name,
                "consolidated_time_data": ds[time_name],
                "time_coord_mapping": time_coord_mapping,
                "variable_time_coord_mapping": {
                    f"{group_name}_{k.split('_time')[0].lower()}": v
                    for k, v in time_coord_mapping.items()
                },
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
                self.logger.debug(
                    "Group %s: Time coordinate '%s' length %d differs from '%s' length %d",
                    group_name,
                    _name,
                    len(time_array),
                    first_time_name,
                    len(first_time),
                )
                break

            # Compare values with tolerance
            try:
                if not np.allclose(time_array.values, first_time.values, atol=1e-6):
                    all_identical = False
                    self.logger.debug(
                        "Group %s: Time coordinate '%s' values differ from '%s'",
                        group_name,
                        _name,
                        first_time_name,
                    )
                    break
            except TypeError:
                # Handle datetime arrays
                if not np.array_equal(time_array.values, first_time.values):
                    all_identical = False
                    self.logger.debug(
                        "Group %s: Time coordinate '%s' values differ from '%s'",
                        group_name,
                        _name,
                        first_time_name,
                    )
                    break

        if all_identical:
            # All time coordinates are identical - consolidate them
            consolidated_name = f"{group_name}_time"
            time_coord_mapping = dict.fromkeys(time_vars, consolidated_name)

            self.logger.info(
                "%-77s %s",
                f"Consoliding {len(time_vars)} coordinates to",
                consolidated_name,
            )

            return {
                "consolidated_time_name": consolidated_name,
                "consolidated_time_data": ds[first_time_name],
                "time_coord_mapping": time_coord_mapping,
                "variable_time_coord_mapping": {
                    f"{group_name}_{k.split('_time')[0].lower()}": consolidated_name
                    for k in time_vars
                },
            }

        # Time coordinates differ - keep them separate
        time_coord_mapping = {name: f"{group_name}_{name.lower()}" for name in time_vars}

        self.logger.info(
            "Group %s: Time coordinates differ - keeping separate: %s",
            group_name,
            list(time_vars.keys()),
        )

        return {
            "consolidated_time_name": None,
            "consolidated_time_data": None,
            "time_coord_mapping": time_coord_mapping,
            "variable_time_coord_mapping": {
                f"{group_name}_{k.split('_time')[0].lower()}": v
                for k, v in time_coord_mapping.items()
            },
        }

    def _add_time_coordinates_to_combined(self, time_info: dict, ds: xr.Dataset) -> None:
        """Add time coordinates to the combined dataset."""
        if time_info["consolidated_time_name"]:
            self._add_cons_time_coord(time_info)
        else:
            self._add_sep_time_coord(time_info, ds)

    def _add_cons_time_coord(self, time_info: dict) -> None:
        """Add a consolidated time coordinate to the combined dataset."""
        time_name = time_info["consolidated_time_name"]
        self.logger.info(
            "Adding consolidated time coordinate %-44s %s",
            f"{time_name} as",
            time_name,
        )
        self.combined_nc[time_name] = xr.DataArray(
            time_info["consolidated_time_data"].to_numpy(),
            dims=[time_name],
            coords={time_name: time_info["consolidated_time_data"].to_numpy()},
        )
        self.combined_nc[time_name].attrs = time_info["consolidated_time_data"].attrs.copy()

    def _add_sep_time_coord(self, time_info: dict, ds: xr.Dataset) -> None:
        """Add separate time coordinates to the combined dataset."""
        for orig_time_var, new_time_var in time_info["time_coord_mapping"].items():
            self.logger.info(
                "Adding time coordinate %-58s %s",
                f"{orig_time_var} as",
                new_time_var,
            )
            self.combined_nc[new_time_var] = xr.DataArray(
                ds[orig_time_var].to_numpy(),
                dims=[new_time_var],
                coords={new_time_var: ds[orig_time_var].to_numpy()},
            )
            self.combined_nc[new_time_var].attrs = ds[orig_time_var].attrs.copy()

    def _get_time_coordinate_data(self, time_info: dict, ds: xr.Dataset, orig_time_dim: str):
        """Get the appropriate time coordinate data for a variable."""
        if time_info["consolidated_time_name"]:
            return time_info["consolidated_time_data"].to_numpy()
        return ds[orig_time_dim].to_numpy()

    def _create_data_array_for_variable(
        self, ds: xr.Dataset, orig_var: str, dim_name: str, time_coord_data
    ) -> xr.DataArray:
        """Create a DataArray for a variable, handling unit conversions."""
        if orig_var in ("latitude", "longitude") and ds[orig_var].attrs.get("units") == "radians":
            data_array = xr.DataArray(
                ds[orig_var].to_numpy() * 180.0 / np.pi,
                dims=[dim_name],
                coords={dim_name: time_coord_data},
            )
            data_array.attrs = ds[orig_var].attrs.copy()
            data_array.attrs["units"] = "degrees"
            data_array.attrs["coordinates"] = f"{dim_name}"
        elif len(ds[orig_var].dims) == 2:  # noqa: PLR2004
            # Handle 2D arrays (time, array_index) - e.g. biolume_raw, digitized_raw_ad_counts_M
            second_dim_name = ds[orig_var].dims[1]
            second_dim_size = ds[orig_var].shape[1]
            self.logger.debug(
                "Reading 2 dimensional %s data arrays with shape %s",
                orig_var,
                ds[orig_var].shape,
            )
            data_array = xr.DataArray(
                ds[orig_var].to_numpy(),
                dims=[dim_name, second_dim_name],
                coords={
                    dim_name: time_coord_data,
                    second_dim_name: np.arange(second_dim_size),
                },
            )
            data_array.attrs = ds[orig_var].attrs.copy()
            data_array.attrs["comment"] = f"{orig_var} from group {ds.attrs.get('group_name', '')}"
            data_array.attrs["coordinates"] = f"{dim_name} {second_dim_name}"
        else:
            data_array = xr.DataArray(
                ds[orig_var].to_numpy(),
                dims=[dim_name],
                coords={dim_name: time_coord_data},
            )
            data_array.attrs = ds[orig_var].attrs.copy()
            data_array.attrs["comment"] = f"{orig_var} from group {ds.attrs.get('group_name', '')}"
            data_array.attrs["coordinates"] = f"{dim_name}"
        return data_array

    def _add_time_metadata_to_variable(self, var_name: str, dim_name: str) -> None:
        """Add required time metadata for cf_xarray decoding."""
        self.combined_nc[var_name].coords[dim_name].attrs["units"] = (
            "seconds since 1970-01-01T00:00:00Z"
        )
        self.combined_nc[var_name].coords[dim_name].attrs["standard_name"] = "time"

    def _process_group_vars(self, ds: xr.Dataset, group_name: str, time_info: dict) -> None:
        """Process all data variables in a group."""
        for orig_var in ds.variables:
            if orig_var.lower().endswith("time"):
                continue

            # Skip scalar variables (no dimensions)
            if len(ds[orig_var].dims) == 0:
                self.logger.debug("Skipping scalar variable: %s", orig_var)
                continue

            new_var = group_name + "_" + orig_var.lower()

            # Get the original time dimension for this variable
            orig_time_dim = ds[orig_var].dims[0]  # Assuming first dim is time

            # Check if this dimension has a mapping
            if orig_time_dim not in time_info["time_coord_mapping"]:
                self.logger.warning(
                    "No time mapping found for %s dimension %s", orig_var, orig_time_dim
                )
                continue

            dim_name = time_info["time_coord_mapping"][orig_time_dim]
            time_coord_data = self._get_time_coordinate_data(time_info, ds, orig_time_dim)

            self.logger.info("Adding variable %-65s %s", f"{orig_var} as", new_var)

            # Create the data array
            self.combined_nc[new_var] = self._create_data_array_for_variable(
                ds, orig_var, dim_name, time_coord_data
            )

            # Add time metadata
            self._add_time_metadata_to_variable(new_var, dim_name)

    def _add_consolidation_comment(self, time_info: dict) -> None:
        """Add a comment documenting time coordinate consolidation."""
        if time_info["consolidated_time_name"] in self.combined_nc.variables:
            mapping_info = ", ".join(
                [f"{orig} -> {new}" for orig, new in time_info["time_coord_mapping"].items()]
            )
            self.combined_nc[time_info["consolidated_time_name"]].attrs["comment"] = (
                f"Consolidated time coordinate from: {mapping_info}"
            )

    def _align_ubat_time_coordinates(self, ubat_2d, calib_coeff, time_dim, calib_time_dim):
        """Align UBAT and calibration coefficient time coordinates by finding common times."""
        ubat_time = self.combined_nc[time_dim].to_numpy()
        calib_time = self.combined_nc[calib_time_dim].to_numpy()

        # Find intersection of time values
        common_indices_ubat = []
        common_indices_calib = []

        for i, t_ubat in enumerate(ubat_time):
            # Find matching time in calib_time
            matches = np.where(np.abs(calib_time - t_ubat) < self.TIME_MATCH_TOLERANCE)[0]
            if len(matches) > 0:
                common_indices_ubat.append(i)
                common_indices_calib.append(matches[0])

        if len(common_indices_ubat) == 0:
            error_message = f"No common time values found between {time_dim} and {calib_time_dim}"
            raise ValueError(error_message)

        self.logger.info(
            "Found %d common time values out of %d in %s and %d in %s",
            len(common_indices_ubat),
            len(ubat_time),
            time_dim,
            len(calib_time),
            calib_time_dim,
        )

        # Subset both arrays to common times
        ubat_2d_aligned = ubat_2d.isel({time_dim: common_indices_ubat})
        calib_coeff_aligned = calib_coeff.isel({calib_time_dim: common_indices_calib})

        return ubat_2d_aligned, calib_coeff_aligned

    def _expand_ubat_to_60hz(self) -> None:
        """Expand UBAT digitized_raw_ad_counts 2D array into 60hz time series.

        Replaces the 2D array with a 1D 60Hz time series, analogous to how
        Dorado biolume_raw is stored with a time60hz coordinate.
        """
        ubat_var = "wetlabsubat_digitized_raw_ad_counts"

        if ubat_var not in self.combined_nc:
            self.logger.debug(
                "No UBAT digitized_raw_ad_counts variable found, skipping 60hz expansion"
            )
            return

        self.logger.info("Expanding UBAT %s to 60hz time series", ubat_var)

        # Get the 2D array (time, sample_index)
        ubat_2d = self.combined_nc[ubat_var]

        if len(ubat_2d.dims) != 2:  # noqa: PLR2004
            self.logger.warning("UBAT variable is not 2D, skipping 60hz expansion")
            return

        time_dim = ubat_2d.dims[0]
        n_samples = ubat_2d.shape[1]

        if "wetlabsubat_hv_step_calibration_coefficient" not in self.combined_nc:
            self.logger.warning("No UBAT calibration coefficient found, skipping 60hz expansion")
            return

        # Get calibration coefficient and verify dimensions match
        calib_coeff = self.combined_nc["wetlabsubat_hv_step_calibration_coefficient"]
        calib_time_dim = calib_coeff.dims[0]

        # Handle dimension mismatch by finding common time values
        ubat_time = self.combined_nc[time_dim].to_numpy()
        calib_time = self.combined_nc[calib_time_dim].to_numpy()

        if len(ubat_time) != len(calib_time):
            self.logger.warning(
                "Dimension mismatch: %s has %d elements but %s has %d elements - "
                "finding common time values",
                time_dim,
                len(ubat_time),
                calib_time_dim,
                len(calib_time),
            )
            ubat_2d, calib_coeff = self._align_ubat_time_coordinates(
                ubat_2d, calib_coeff, time_dim, calib_time_dim
            )
            ubat_time = ubat_2d.coords[time_dim].to_numpy()
            calib_time = calib_coeff.coords[calib_time_dim].to_numpy()

        # Verify the time coordinate values are now identical
        if not np.allclose(ubat_time, calib_time, rtol=1e-9):
            error_message = (
                f"Time coordinates {time_dim} and {calib_time_dim} have different values "
                "even after alignment"
            )
            raise ValueError(error_message)

        self.logger.info(
            "Verified dimensions match: %s and %s both have %d elements",
            time_dim,
            calib_time_dim,
            len(ubat_time),
        )

        # Multiply raw 60 hz values by the calibration coefficient
        # Broadcasting: calib_coeff is (m,) and ubat_2d is (m, 60)
        # This multiplies each row of ubat_2d by the corresponding coefficient
        ubat_2d_calibrated = ubat_2d * calib_coeff.to_numpy()[:, np.newaxis]

        # Get the time coordinate (use ubat_2d's time coordinate after alignment)
        time_coord = ubat_2d.coords[time_dim]
        n_times = len(time_coord)

        # Save original attributes before removing
        original_attrs = ubat_2d.attrs.copy()

        # Calculate 60hz time offsets (assuming samples span 1 second)
        # Each sample is 1/60th of a second apart
        # Subtract 0.5 seconds because 60Hz data are logged at the end of the 1-second period
        sample_offsets = np.arange(n_samples) / 60.0 - 0.5

        # Create 60hz time series by adding offsets to each 1Hz time
        time_60hz_list = []
        for i in range(n_times):
            base_time = time_coord.to_numpy()[i]
            # Add offsets to create 60 timestamps per second
            times_for_this_second = base_time + sample_offsets
            time_60hz_list.append(times_for_this_second)

        # Flatten the arrays
        time_60hz = np.concatenate(time_60hz_list)
        data_60hz = ubat_2d_calibrated.to_numpy().flatten()

        # Remove the old 2D variable
        del self.combined_nc[ubat_var]

        # Create new 60hz time coordinate with attributes
        time_60hz_name = f"{time_dim}_60hz"
        time_60hz_coord = xr.DataArray(
            time_60hz,
            dims=[time_60hz_name],
            name=time_60hz_name,
            attrs={
                "units": "seconds since 1970-01-01T00:00:00Z",
                "standard_name": "time",
                "long_name": "Time at 60Hz sampling rate",
            },
        )

        # Create replacement 1D variable with 60hz time coordinate
        self.combined_nc[ubat_var] = xr.DataArray(
            data_60hz,
            coords={time_60hz_name: time_60hz_coord},
            dims=[time_60hz_name],
            name=ubat_var,
        )

        # Restore and update attributes
        self.combined_nc[ubat_var].attrs = original_attrs
        self.combined_nc[ubat_var].attrs["long_name"] = "UBAT digitized raw AD counts at 60Hz"
        self.combined_nc[ubat_var].attrs["coordinates"] = time_60hz_name
        self.combined_nc[ubat_var].attrs["comment"] = (
            original_attrs.get("comment", "") + " Expanded from 2D to 1D 60Hz time series"
        )

        self.logger.info(
            "Replaced 2D %s with 1D 60hz time series: %d samples from %d 1Hz records",
            ubat_var,
            len(data_60hz),
            n_times,
        )

    def _initial_coordinate_qc(self) -> None:
        """Perform initial QC on core coordinate variables for specific log files."""
        if self.log_file in (
            "tethys/missionlogs/2012/20120908_20120920/20120909T010636/201209090106_201209091521.nc4",
            "brizo/missionlogs/2025/20250909_20250915/20250913T080940/202509130809_202509140809.nc4",
        ):
            self.logger.info("Performing initial coordinate QC for %s", self.log_file)
            self._range_qc_combined_nc(
                instrument="universals",
                variables=[
                    "universals_longitude",
                    "universals_latitude",
                ],
                ranges={
                    "universals_longitude": Range(-123.5, -121.5),
                    "universals_latitude": Range(35.0, 37.0),
                },
                set_to_nan=False,
            )
            self._range_qc_combined_nc(
                instrument="nal9602",
                variables=[
                    "nal9602_longitude_fix",
                    "nal9602_latitude_fix",
                ],
                ranges={
                    "nal9602_longitude_fix": Range(-123.5, -121.5),
                    "nal9602_latitude_fix": Range(35.0, 37.0),
                },
                set_to_nan=False,
            )

    def _add_nudged_coordinates(self, max_sec_diff_at_end: int = 10) -> None:
        """Add nudged longitude and latitude variables to the combined dataset."""
        self._initial_coordinate_qc()

        # Check if GPS fix variables exist
        if (
            "nal9602_longitude_fix" not in self.combined_nc
            or "nal9602_latitude_fix" not in self.combined_nc
        ):
            self.logger.warning(
                "No GPS fix variables found in combined dataset - "
                "skipping nudged coordinate creation"
            )
            return

        # Ensure GPS fixes have monotonically increasing timestamps
        gps_lon = self.combined_nc["nal9602_longitude_fix"]
        gps_lat = self.combined_nc["nal9602_latitude_fix"]
        gps_time_coord = gps_lon.coords[gps_lon.dims[0]]

        # Convert to pandas index which handles datetime comparisons properly
        gps_time_index = gps_time_coord.to_index()
        gps_monotonic = monotonic_increasing_time_indices(gps_time_index)
        if not np.all(gps_monotonic):
            monotonic_count = np.sum(gps_monotonic)
            self.logger.warning(
                "Filtered GPS fixes from %d to %d to ensure monotonically increasing timestamps",
                len(gps_lon),
                monotonic_count,
            )
            gps_lon = gps_lon.isel({gps_lon.dims[0]: gps_monotonic})
            gps_lat = gps_lat.isel({gps_lat.dims[0]: gps_monotonic})

        try:
            nudged_longitude, nudged_latitude, segment_count, segment_minsum = nudge_positions(
                nav_longitude=self.combined_nc["universals_longitude"],
                nav_latitude=self.combined_nc["universals_latitude"],
                gps_longitude=gps_lon,
                gps_latitude=gps_lat,
                logger=self.logger,
                auv_name="",
                mission="",
                log_file=self.log_file,
                max_sec_diff_at_end=max_sec_diff_at_end,
                create_plots=self.plot,
            )
        except ValueError as e:
            self.logger.error("Nudging positions failed: %s", e)  # noqa: TRY400
            return

        self.logger.info(
            "nudge_positions created %d segments with segment_minsum = %f",
            segment_count,
            segment_minsum,
        )

        # Calculate total underwater time and store for metadata
        time_coord = self.combined_nc[self.variable_time_coord_mapping["universals_longitude"]]
        time_diff = time_coord.to_numpy()[-1] - time_coord.to_numpy()[0]
        # Convert timedelta64 to seconds (handles nanosecond precision)
        total_seconds = float(time_diff / np.timedelta64(1, "s"))
        self.nudge_segment_count = segment_count
        self.nudge_total_minutes = total_seconds / 60.0

        self.combined_nc["nudged_longitude"] = xr.DataArray(
            nudged_longitude,
            coords=[
                self.combined_nc[
                    self.variable_time_coord_mapping["universals_longitude"]
                ].to_numpy()
            ],
            dims={f"nudged_{TIME}"},
            name="nudged_longitude",
        )
        self.combined_nc["nudged_longitude"].attrs = {
            "long_name": "Nudged Longitude",
            "standard_name": "longitude",
            "units": "degrees_east",
            "comment": (
                f"Dead reckoned positions from {segment_count} underwater segments "
                f"nudged to GPS positions"
            ),
        }
        self.combined_nc["nudged_latitude"] = xr.DataArray(
            nudged_latitude,
            coords=[
                self.combined_nc[self.variable_time_coord_mapping["universals_latitude"]].to_numpy()
            ],
            dims={f"nudged_{TIME}"},
            name="nudged_latitude",
        )
        self.combined_nc["nudged_latitude"].attrs = {
            "long_name": "Nudged Latitude",
            "standard_name": "latitude",
            "units": "degrees_north",
            "comment": (
                f"Dead reckoned positions from {segment_count} underwater segments "
                f"nudged to GPS positions"
            ),
        }

    def combine_groups(self) -> None:
        """Combine group files into a single NetCDF dataset with consolidated time coordinates."""
        src_dir = Path(BASE_LRAUV_PATH, Path(self.log_file).parent)
        group_files = sorted(src_dir.glob(f"{Path(self.log_file).stem}_{GROUP}_*.nc"))
        self.summary_fields = set()
        self.combined_nc = xr.Dataset()

        for group_file in group_files:
            self.logger.info("-" * 110)
            self.logger.info("Group file: %s", group_file.name)
            # Open group file without decoding to have np.allclose work properly
            with xr.open_dataset(group_file, decode_cf=False) as ds:
                # Group name to prepend variable names is lowercase with underscores removed
                group_name = group_file.stem.split(f"{GROUP}_")[1].replace("_", "").lower()
                time_info = self._cons_group_time_coords(ds, group_name)

                # Add time coordinate(s) to combined dataset
                self._add_time_coordinates_to_combined(time_info, ds)

                # Process all data variables in the group
                self._process_group_vars(ds, group_name, time_info)

                # Add consolidation comment if applicable
                self._add_consolidation_comment(time_info)

                # Collect variable coordinate mapping by group, which can be flattened
                self.variable_time_coord_mapping.update(time_info["variable_time_coord_mapping"])

        # Expand UBAT 2D arrays to 60hz time series
        self._expand_ubat_to_60hz()

        # Write intermediate file for cf_xarray decoding
        intermediate_file = self._intermediate_write_netcdf()
        with xr.open_dataset(intermediate_file, decode_cf=True) as ds:
            self.combined_nc = ds.load()

        # Add nudged coordinates
        self._add_nudged_coordinates()

        # Clean up intermediate file
        Path(intermediate_file).unlink()

    def _intermediate_write_netcdf(self) -> None:
        """Write out an intermediate combined netCDF file so that data can be
        read using decode_cf=True for nudge_positions() to work with cf accessors."""
        netcdfs_dir = Path(BASE_LRAUV_PATH, Path(self.log_file).parent)
        out_fn = Path(netcdfs_dir, f"{Path(self.log_file).stem}_combined_intermediate.nc")

        self.combined_nc.attrs = self.global_metadata()
        self.logger.info("Writing intermediate combined group data to %s", out_fn)
        if Path(out_fn).exists():
            Path(out_fn).unlink()
        self.combined_nc.to_netcdf(out_fn)
        self.logger.debug(
            "Data variables written: %s",
            ", ".join(sorted(self.combined_nc.variables)),
        )
        self.logger.info(
            "Wrote intermediate (_combined_intermediate.nc) netCDF file: %s",
            out_fn,
        )
        return out_fn

    def write_netcdf(self) -> None:
        """Write combined netCDF file using instance attributes."""
        netcdfs_dir = Path(BASE_LRAUV_PATH, Path(self.log_file).parent)
        out_fn = Path(netcdfs_dir, f"{Path(self.log_file).stem}_combined.nc4")

        self.combined_nc.attrs = self.global_metadata()
        self.logger.info("Writing combined group data to %s", out_fn)
        if Path(out_fn).exists():
            Path(out_fn).unlink()
        self.combined_nc.to_netcdf(out_fn)
        self.logger.debug(
            "Data variables written: %s",
            ", ".join(sorted(self.combined_nc.variables)),
        )
        self.logger.info("Wrote combined (_combined.nc4) netCDF file: %s", out_fn)

        return netcdfs_dir

    def process_command_line(self):
        """Process command line arguments using shared parser infrastructure."""
        examples = "Examples:" + "\n\n"
        examples += "  Combine original data from Group files for an LRAUV log file:\n"
        examples += (
            "    "
            + sys.argv[0]
            + " -v --log_file brizo/missionlogs/2025/"
            + "20250909_20250915/20250914T080941/"
            + "202509140809_202509150109.nc4\n"
        )

        # Use shared parser with combine-specific additions
        parser = get_standard_lrauv_parser(
            description=__doc__,
            epilog=examples,
        )

        # Add combine-specific arguments
        parser.add_argument(
            "--plot",
            action="store_true",
            help="Create intermediate plot(s) to help validate processing",
        )

        self.args = parser.parse_args()

        # Set instance attributes from parsed arguments
        self.log_file = self.args.log_file
        self.verbose = self.args.verbose
        self.plot = "--plot" if self.args.plot else None
        self.commandline = " ".join(sys.argv)
        self.logger.setLevel(self._log_levels[self.verbose])


if __name__ == "__main__":
    combine = Combine_NetCDF()
    combine.process_command_line()
    start = time.time()
    combine.combine_groups()
    combine.write_netcdf()
    combine.logger.info("Time to process: %.2f seconds", (time.time() - start))
