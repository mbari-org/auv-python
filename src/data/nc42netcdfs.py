#!/usr/bin/env python
"""
Extract instrument/group data from LRAUV .nc4 files into individual NetCDF files.

Makes the original data more accessible for analysis and visualization.
"""

__author__ = "Mike McCann"
__copyright__ = "Copyright 2025, Monterey Bay Aquarium Research Institute"

import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import git
import netCDF4
import numpy as np
import pooch
from common_args import get_standard_lrauv_parser
from utils import get_deployment_name

# Conditional imports for plotting (only when needed)
try:
    import matplotlib.pyplot as plt  # noqa: F401

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Local directory that serves as the work area for log_files and netcdf files
BASE_LRAUV_WEB = "https://dods.mbari.org/data/lrauv/"
BASE_LRAUV_PATH = Path(__file__).parent.joinpath("../../data/lrauv_data").resolve()
SUMMARY_SOURCE = "Original LRAUV data extracted from {}, group {}"
GROUPS = ["navigation", "ctd", "ecopuck"]  # Your actual group names
GROUP = "Group"  # A literal in the filename to use for identifying group .nc files

SCI_PARMS = {
    "/": [
        {"name": "longitude"},
        {"name": "latitude"},
        {"name": "depth"},
        {"name": "time"},
    ],
    "Aanderaa_O2": [{"name": "mass_concentration_of_oxygen_in_sea_water"}],
    "CTD_NeilBrown": [
        {"name": "sea_water_salinity"},
        {"name": "sea_water_temperature"},
    ],
    "CTD_Seabird": [
        {"name": "sea_water_salinity"},
        {"name": "sea_water_temperature"},
        {"name": "mass_concentration_of_oxygen_in_sea_water"},
    ],
    "ISUS": [{"name": "mole_concentration_of_nitrate_in_sea_water"}],
    "PAR_Licor": [{"name": "downwelling_photosynthetic_photon_flux_in_sea_water"}],
    "WetLabsBB2FL": [
        {"name": "mass_concentration_of_chlorophyll_in_sea_water"},
        {"name": "OutputChl"},
        {"name": "Output470"},
        {"name": "Output650"},
        {"name": "VolumeScatCoeff117deg470nm"},
        {"name": "VolumeScatCoeff117deg650nm"},
        {"name": "ParticulateBackscatteringCoeff470nm"},
        {"name": "ParticulateBackscatteringCoeff650nm"},
    ],
    "WetLabsSeaOWL_UV_A": [
        {"name": "concentration_of_chromophoric_dissolved_organic_matter_in_sea_water"},
        {"name": "mass_concentration_of_chlorophyll_in_sea_water"},
        {"name": "BackscatteringCoeff700nm"},
        {"name": "VolumeScatCoeff117deg700nm"},
        {"name": "mass_concentration_of_petroleum_hydrocarbons_in_sea_water"},
    ],
    "WetLabsUBAT": [
        {"name": "average_bioluminescence"},
        {"name": "flow_rate"},
        {"name": "digitized_raw_ad_counts"},
    ],
}

ENG_PARMS = {
    "BPC1": [
        {"name": "platform_battery_charge"},
        {"name": "platform_battery_voltage"},
    ],
    "BuoyancyServo": [{"name": "platform_buoyancy_position"}],
    "DeadReckonUsingMultipleVelocitySources": [
        {"name": "fix_residual_percent_distance_traveled"},
        {"name": "longitude"},
        {"name": "latitude"},
        {"name": "depth"},
    ],
    "DeadReckonUsingSpeedCalculator": [
        {"name": "fix_residual_percent_distance_traveled"},
        {"name": "longitude"},
        {"name": "latitude"},
        {"name": "depth"},
    ],
    "ElevatorServo": [{"name": "platform_elevator_angle"}],
    "MassServo": [{"name": "platform_mass_position"}],
    "NAL9602": [
        {"name": "time_fix"},
        {"name": "latitude_fix"},
        {"name": "longitude_fix"},
    ],
    "Onboard": [{"name": "platform_average_current"}],
    "RudderServo": [{"name": "platform_rudder_angle"}],
    "ThrusterServo": [{"name": "platform_propeller_rotation_rate"}],
    "CurrentEstimator": [
        {"name": "current_direction_navigation_frame"},
        {"name": "current_speed_navigation_frame"},
    ],
}

SCIENG_PARMS = {**SCI_PARMS, **ENG_PARMS}


class Extract:
    """Extract instrument/group data from LRAUV .nc4 files into individual NetCDF files."""

    logger = logging.getLogger(__name__)
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        "%(levelname)s %(asctime)s %(filename)s "
        "%(funcName)s():%(lineno)d [%(process)d] %(message)s",
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    _log_levels = (logging.WARN, logging.INFO, logging.DEBUG)

    def __init__(  # noqa: PLR0913
        self,
        log_file: str = None,
        plot_time: str = None,
        plot_universals: bool = False,  # noqa: FBT001, FBT002
        filter_monotonic_time: bool = True,  # noqa: FBT001, FBT002
        verbose: int = 0,
        commandline: str = "",
    ) -> None:
        """Initialize Extract with explicit parameters.

        Args:
            log_file: Log file path for processing
            plot_time: Optional plot time specification (e.g., /latitude_time)
            plot_universals: Plot longitude, latitude, depth time filtering for root group
            filter_monotonic_time: Filter out non-monotonic time values
            verbose: Verbosity level (0-2)
            commandline: Command line string for tracking
        """
        self.log_file = log_file
        self.plot_time = plot_time
        self.plot_universals = plot_universals
        self.filter_monotonic_time = filter_monotonic_time
        self.verbose = verbose
        self.commandline = commandline
        self.universals_plot_data = {}  # Store plot data for universals

    def download_with_pooch(self, url, local_dir, known_hash=None):
        """Download using pooch with caching and verification."""
        downloader = pooch.HTTPDownloader(timeout=(60, 300), progressbar=True)
        return pooch.retrieve(
            url=url,
            known_hash=known_hash,  # Optional but recommended for integrity
            fname=Path(url).name,
            path=local_dir,
            downloader=downloader,
        )

    def extract_groups_to_files_netcdf4(self, log_file: str) -> Path:
        """Extract each group from .nc4 file to a separate .nc file using netCDF4 library.

        Args:
            log_file: Relative path from BASE_LRAUV_WEB to .nc4 log_file

        Returns:
            netcdfs_dir: Local directory where NetCDF files were saved

        Note:
        The xarray library fails reading the WetLabsBB2FL group from this file:
        brizo/missionlogs/2025/20250909_20250915/20250914T080941/202509140809_202509150109.nc4
        with garbled data for the serial variable (using ncdump):
            serial = "<C0>$F<C4>!{<8D>\031@<AE>7\024[<FB><BF>P<C0><D4>]\001\030" ;
        but netCDF4 can skip over it and read the rest of the variables.
        """
        # Download over http so that we don't need to mount smb shares
        url = os.path.join(BASE_LRAUV_WEB, log_file)  # noqa: PTH118
        netcdfs_dir = Path(BASE_LRAUV_PATH, Path(log_file).parent)
        netcdfs_dir.mkdir(exist_ok=True, parents=True)

        self.logger.info("Downloading %s", url)
        input_file = self.download_with_pooch(url, netcdfs_dir)

        self.logger.info("Extracting data from %s", input_file)
        with netCDF4.Dataset(input_file, "r") as src_dataset:
            # Extract root group first
            self._extract_root_group(log_file, "/", src_dataset, netcdfs_dir)

            # Extract all other groups
            all_groups = list(src_dataset.groups.keys())
            for group_name in SCIENG_PARMS:
                if group_name == "/" or group_name not in all_groups:
                    if group_name != "/" and group_name not in all_groups:
                        self.logger.warning("Group %s not found in %s", group_name, input_file)
                    continue
                self._extract_single_group(log_file, group_name, src_dataset, netcdfs_dir)

        return netcdfs_dir

    def _extract_root_group(
        self, log_file: str, group_name: str, src_dataset: netCDF4.Dataset, output_dir: Path
    ):
        """Extract variables from the root group to <stem>_{GROUP}_Universals.nc."""
        root_parms = SCIENG_PARMS.get("/", [])
        if not root_parms:
            return

        self.logger.info("Extracting root group '/'")
        vars_to_extract, _ = self._get_available_variables(src_dataset, root_parms)

        # Add debugging output for root group processing
        self.logger.debug("=== ROOT GROUP DEBUG ===")
        self.logger.debug("Available variables: %s", sorted(vars_to_extract))
        self.logger.debug("Available dimensions: %s", sorted(src_dataset.dimensions.keys()))
        self.logger.debug(
            "Available coordinate variables: %s",
            [v for v in sorted(src_dataset.variables.keys()) if v in src_dataset.dimensions],
        )

        if vars_to_extract:
            output_file = output_dir / f"{Path(log_file).stem}_{GROUP}_Universals.nc"
            self._create_netcdf_file(
                log_file, group_name, src_dataset, vars_to_extract, output_file
            )
            self.logger.info("Extracted root group '/' to %s", output_file)
        else:
            self.logger.warning("No requested variables found in root group '/'")

    def _extract_single_group(
        self,
        log_file: str,
        group_name: str,
        src_dataset: netCDF4.Dataset,
        output_dir: Path,
    ):
        "Extract a single group to its own NetCDF file named like <stem>_{GROUP}_<group_name>.nc."
        group_parms = SCIENG_PARMS[group_name]

        try:
            self.logger.debug(" Group %s", group_name)
            src_group = src_dataset.groups[group_name]

            vars_to_extract, requested_vars = self._get_available_variables(src_group, group_parms)

            if vars_to_extract:
                output_file = output_dir / f"{Path(log_file).stem}_{GROUP}_{group_name}.nc"
                self._create_netcdf_file(
                    log_file, group_name, src_group, vars_to_extract, output_file
                )
                self.logger.info("Extracted %s to %s", group_name, output_file)
            else:
                self.logger.warning(
                    "No requested variables (%s) found in group %s", requested_vars, group_name
                )

        except KeyError:
            self.logger.warning("Group %s not found", group_name)

    def _get_available_variables(
        self, src_group: netCDF4.Group, group_parms: list[dict[str, Any]]
    ) -> list[str]:
        """Get the intersection of requested and available variables."""
        requested_vars = [p["name"] for p in group_parms if "name" in p]
        available_vars = list(src_group.variables.keys())
        vars_to_extract = [var for var in requested_vars if var in available_vars]

        self.logger.debug("  Variables to extract: %s", vars_to_extract)
        return vars_to_extract, requested_vars

    def _get_time_filters_for_variables(
        self, log_file: str, group_name: str, src_group: netCDF4.Group, vars_to_extract: list[str]
    ) -> dict[str, dict]:
        """Get time filtering information for time coordinates used by vars_to_extract.

        Returns:
            dict: Map of time_coord_name -> {"indices": list[int], "filtered": bool}
        """
        # Check if time filtering is enabled
        if not self.filter_monotonic_time:
            return {}

        self.logger.info("========================= Group %s =========================", group_name)
        # Find all time coordinates used by variables in extraction list
        time_coords_found = self._find_time_coordinates(group_name, src_group, vars_to_extract)

        # Add diagnostic check to compare original time coordinate values
        if len(time_coords_found) > 1:
            self._analyze_original_time_coordinates(src_group, time_coords_found, group_name)

        # Parse plot time settings once
        plot_group_name, plot_time_coord_name = self._parse_plot_time_argument()

        # Process each unique time coordinate found
        time_filters = {}
        for time_coord_name in sorted(time_coords_found):
            time_filter = self._process_single_time_coordinate(
                log_file,
                group_name,
                src_group,
                time_coord_name,
                plot_group_name,
                plot_time_coord_name,
            )
            time_filters[time_coord_name] = time_filter

        # For root group, apply intersection of valid indices before monotonic filtering
        if group_name == "/":
            time_filters = self._apply_universals_intersection(
                log_file, src_group, time_filters, vars_to_extract
            )
            time_filters = self._align_root_group_coordinates(time_filters, vars_to_extract)
            # Plot universals if requested
            if self.plot_universals:
                self._plot_universals_filtering()

        return time_filters

    def _analyze_original_time_coordinates(
        self, src_group: netCDF4.Group, time_coords_found: set[str], group_name: str
    ):
        """Quick diagnostic for Dead Reckoned timing issues in root group."""
        # Only analyze root group Dead Reckoned coordinates
        if group_name != "/":
            return

        if (
            "latitude_time" not in time_coords_found
            or "longitude_time" not in time_coords_found
            or "latitude_time" not in src_group.variables
            or "longitude_time" not in src_group.variables
        ):
            return

        lat_time = src_group.variables["latitude_time"][:]
        lon_time = src_group.variables["longitude_time"][:]

        # Quick check for Dead Reckoned timing synchronization
        min_len = min(len(lat_time), len(lon_time))
        if min_len == 0:
            return

        # Compare overlapping portion
        overlap_equal = np.array_equal(lat_time[:min_len], lon_time[:min_len])

        if overlap_equal and len(lat_time) == len(lon_time):
            self.logger.info(
                "Dead Reckoned timing: original latitude_time and longitude_time "
                "are properly synchronized"
            )
            return

        # Calculate timing differences for diagnosis
        time_diff = lon_time[:min_len] - lat_time[:min_len]
        non_zero_mask = time_diff != 0
        num_differences = np.sum(non_zero_mask)
        percent_different = 100.0 * num_differences / min_len

        if len(lat_time) != len(lon_time):
            self.logger.warning(
                "Dead Reckoned timing: Different array lengths - "
                "latitude_time: %d, longitude_time: %d",
                len(lat_time),
                len(lon_time),
            )

        if num_differences > 0:
            diff_values = time_diff[non_zero_mask]
            max_abs_diff = np.max(np.abs(diff_values))

            # Define thresholds for Dead Reckoned timing issues
            MAJOR_PERCENT_THRESHOLD = 50.0  # 50% different points
            MAJOR_TIME_THRESHOLD = 3600.0  # 1 hour difference
            MINOR_PERCENT_THRESHOLD = 5.0  # 5% different points
            MINOR_TIME_THRESHOLD = 60.0  # 1 minute difference

            if percent_different > MAJOR_PERCENT_THRESHOLD or max_abs_diff > MAJOR_TIME_THRESHOLD:
                self.logger.warning(
                    "Dead Reckoned timing: Significant synchronization issues detected - "
                    "%.1f%% of coordinates have timing differences (max: %.1f seconds)",
                    percent_different,
                    max_abs_diff,
                )
                self.logger.warning(
                    "Dead Reckoned timing: Differences begin at index %d",
                    np.where(non_zero_mask)[0][0],
                )
                lon_subset = lon_time[
                    max(0, np.where(non_zero_mask)[0][0] - 5) : np.where(non_zero_mask)[0][0] + 5
                ]
                lat_subset = lat_time[
                    max(0, np.where(non_zero_mask)[0][0] - 5) : np.where(non_zero_mask)[0][0] + 5
                ]
                self.logger.warning(
                    "Dead Reckoned timing: longitude_time around this index: %s",
                    " ".join(f"{val:14.2f}" for val in lon_subset),
                )
                self.logger.warning(
                    "Dead Reckoned timing: latitude_time around this index:  %s",
                    " ".join(f"{val:14.2f}" for val in lat_subset),
                )
            elif percent_different > MINOR_PERCENT_THRESHOLD or max_abs_diff > MINOR_TIME_THRESHOLD:
                self.logger.warning(
                    "Dead Reckoned timing: Minor synchronization issues detected - "
                    "%.1f%% of coordinates have timing differences (max: %.1f seconds)",
                    percent_different,
                    max_abs_diff,
                )
            else:
                self.logger.info(
                    "Dead Reckoned timing: Small timing differences detected - "
                    "%.1f%% of coordinates differ (max: %.1f seconds)",
                    percent_different,
                    max_abs_diff,
                )

    def _find_time_coordinates(
        self, group_name: str, src_group: netCDF4.Group, vars_to_extract: list[str]
    ) -> set[str]:
        """Find all time coordinates used by variables in extraction list."""
        time_coords_found = set()
        self.logger.debug(
            "=================================== Group: %s =======================================",
            group_name,
        )
        # Sort variables to make processing deterministic
        for var_name in sorted(vars_to_extract):
            if var_name in src_group.variables:
                var = src_group.variables[var_name]

                # Check each dimension to see if it's a time coordinate
                # Sort dimensions to make processing deterministic
                for dim_name in sorted(var.dimensions):
                    if dim_name in src_group.variables:
                        dim_var = src_group.variables[dim_name]

                        # Check if this dimension variable is a time coordinate
                        if self._is_time_variable(dim_name, dim_var):
                            time_coords_found.add(dim_name)

        return time_coords_found

    def _parse_plot_time_argument(self) -> tuple[str | None, str | None]:
        """Parse the --plot_time argument and return (group_name, time_coord_name)."""
        if not self.plot_time:
            return None, None

        plot_time = self.plot_time
        if not plot_time.startswith("/"):
            msg = "Invalid plot_time format, must be /<group_name>/<time_coord_name>"
            raise ValueError(msg)

        slash_count = plot_time.count("/")
        if slash_count == 1:
            return "/", plot_time[1:]
        if slash_count == 2:  # noqa: PLR2004
            parts = plot_time.split("/")[1:]
            return parts[0], parts[1]

        msg = "Invalid plot_time format, must be /<group_name>/<time_coord_name>"
        raise ValueError(msg)

    def _create_plot_data(
        self, log_file: str, group_name: str, time_coord_name: str, original_time_data
    ) -> dict:
        """Create plot data structure for time filtering visualization."""
        return {
            "original": original_time_data.copy(),
            "log_file": log_file,
            "group_name": group_name,
            "variable_name": time_coord_name,
        }

    def _create_time_filter_result(
        self, mono_indices: list[int], time_data_length: int, time_coord_name: str
    ) -> dict:
        """Create the result dictionary for a time filter."""
        filtered = len(mono_indices) < time_data_length
        comment = ""
        if filtered:
            removed_count = time_data_length - len(mono_indices)
            removed_percent = 100 * removed_count / time_data_length
            comment = (
                f"Filtered {removed_count} non-monotonic points "
                f"({time_data_length} -> {len(mono_indices)}), "
                f"{removed_percent:.2f}%"
            )
            self.logger.info("Time coordinate %s: %s", time_coord_name, comment)

        return {
            "indices": mono_indices,
            "filtered": filtered,
            "comment": comment,
        }

    def _process_single_time_coordinate(  # noqa: PLR0913
        self,
        log_file: str,
        group_name: str,
        src_group: netCDF4.Group,
        time_coord_name: str,
        plot_group_name: str | None,
        plot_time_coord_name: str | None,
    ) -> dict:
        """Process filtering for a single time coordinate."""
        time_var = src_group.variables[time_coord_name]
        original_time_data = time_var[:]
        self.logger.info("Time coordinate %s: %d points", time_coord_name, len(original_time_data))

        # Create plot data if this coordinate should be plotted
        plot_data = None
        should_plot = (
            plot_time_coord_name is not None
            and time_coord_name == plot_time_coord_name
            and group_name == plot_group_name
        )
        if should_plot:
            plot_data = self._create_plot_data(
                log_file, group_name, time_coord_name, original_time_data
            )

        # First filter out values that fall outside of reasonable bounds
        valid_indices = self._filter_valid_time_indices(original_time_data)

        # Get the valid time subset
        valid_time_data = original_time_data[valid_indices]

        # Apply monotonic filtering
        mono_indices_in_filtered = self._get_monotonic_indices(valid_time_data)

        # Convert monotonic indices back to original array indices
        final_indices = [valid_indices[i] for i in mono_indices_in_filtered]

        # For root group universals, store data for intersection step
        is_universal_coord = group_name == "/" and time_coord_name in [
            "longitude_time",
            "latitude_time",
            "depth_time",
            "time_time",
        ]

        if is_universal_coord:
            # Store intermediate data for intersection step
            if self.plot_universals:
                self.universals_plot_data[time_coord_name] = {
                    "original": original_time_data.copy(),
                    "valid_indices": valid_indices.copy(),
                    "valid_data": valid_time_data.copy(),
                    "mono_indices": final_indices.copy(),
                    "mono_data": valid_time_data[mono_indices_in_filtered].copy(),
                    "log_file": log_file,
                }
            # Return with monotonic-filtered indices - intersection happens later
            return {
                "indices": final_indices,
                "filtered": len(final_indices) < len(original_time_data),
                "comment": "Pre-intersection state",
                "mono_indices": final_indices,  # Store for intersection
                "original_time_data": original_time_data,  # Store for reprocessing
            }

        # Store data for plotting if requested
        if plot_data is not None:
            plot_data["valid_indices"] = valid_indices
            plot_data["valid_data"] = valid_time_data
            plot_data["final_indices"] = mono_indices_in_filtered
            plot_data["final_data"] = valid_time_data[mono_indices_in_filtered]
            self._plot_time_filtering(plot_data)

        return self._create_time_filter_result(
            final_indices, len(original_time_data), time_coord_name
        )

    def _apply_universals_intersection(
        self,
        log_file: str,
        src_group: netCDF4.Group,
        time_filters: dict,
        vars_to_extract: list[str],
    ) -> dict:
        """Apply intersection of monotonic-filtered indices for longitude, latitude, and depth.

        After outlier removal and monotonic filtering, we intersect the indices so that only
        time points that exist in all three coordinates are kept.
        """
        # Identify which universal coordinates are present
        universal_coords = ["longitude_time", "latitude_time", "depth_time", "time_time"]
        present_coords = [tc for tc in universal_coords if tc in time_filters]

        if len(present_coords) < 2:  # noqa: PLR2004
            # Need at least 2 coordinates to do intersection
            self.logger.info("Less than 2 universal coordinates found, skipping intersection")
            return time_filters

        # Get monotonic-filtered indices for each coordinate
        mono_indices_sets = {}
        for time_coord in present_coords:
            mono_indices_sets[time_coord] = set(time_filters[time_coord]["mono_indices"])

        # Compute intersection of all monotonic indices
        intersected_indices = set.intersection(*mono_indices_sets.values())
        intersected_indices_list = sorted(intersected_indices)

        self.logger.info(
            "Intersection of universal coordinates (after monotonic filtering): %s",
            ", ".join(f"{tc}: {len(mono_indices_sets[tc])} points" for tc in present_coords),
        )
        self.logger.info("After intersection: %d common points", len(intersected_indices_list))

        # Update each coordinate with intersected indices
        for time_coord in present_coords:
            original_time_data = time_filters[time_coord]["original_time_data"]

            # Update the filter with final indices
            time_filters[time_coord] = self._create_time_filter_result(
                intersected_indices_list, len(original_time_data), time_coord
            )

            # Update plot data if collecting for universals
            if self.plot_universals and time_coord in self.universals_plot_data:
                self.universals_plot_data[time_coord]["final_indices"] = (
                    intersected_indices_list.copy()
                )
                self.universals_plot_data[time_coord]["final_data"] = original_time_data[
                    intersected_indices_list
                ].copy()

        return time_filters

    def _is_time_variable(self, var_name: str, var) -> bool:
        """Check if a variable is a time coordinate variable."""
        # Check name pattern
        if var_name.lower().endswith("time"):
            return True

        # Check units
        if hasattr(var, "units"):
            units = getattr(var, "units", "").lower()
            time_patterns = ["seconds since", "days since", "hours since"]
            if any(pattern in units for pattern in time_patterns):
                return True

        return False

    def _filter_valid_time_indices(self, time_data) -> list[int]:
        """Filter out wildly invalid time values before monotonic filtering.

        Returns indices of time values that are reasonable Unix epoch timestamps.
        Uses time bounds from log file name.
        """
        # Parse filename like: 202509140809_202509150109.nc4
        # Format: YYYYMMDDHHMM_YYYYMMDDHHMM
        import re
        from datetime import timedelta
        from pathlib import Path

        filename = Path(self.log_file).stem
        match = re.search(r"(\d{12})_(\d{12})", filename)

        start_str = match.group(1)
        end_str = match.group(2)

        # Parse YYYYMMDDHHMM format
        start_time = datetime.strptime(start_str, "%Y%m%d%H%M").replace(tzinfo=UTC)
        end_time = datetime.strptime(end_str, "%Y%m%d%H%M").replace(tzinfo=UTC)

        # Add 10-minute buffer before and after
        MIN_UNIX_TIME = int((start_time - timedelta(minutes=10)).timestamp())
        MAX_UNIX_TIME = int((end_time + timedelta(minutes=10)).timestamp())

        self.logger.debug(
            "Using time bounds from log file: %s to %s",
            (start_time - timedelta(minutes=10)).isoformat(),
            (end_time + timedelta(minutes=10)).isoformat(),
        )

        # Convert to numpy array for efficient operations
        time_array = np.asarray(time_data)

        # Basic validity checks
        is_finite = np.isfinite(time_array)
        is_in_range = (time_array >= MIN_UNIX_TIME) & (time_array <= MAX_UNIX_TIME)
        valid_mask = is_finite & is_in_range

        # Get indices where all conditions are met
        valid_indices = np.where(valid_mask)[0].tolist()

        # Log filtering statistics
        total_count = len(time_array)
        outliers_found = total_count - len(valid_indices)

        if outliers_found > 0:
            non_finite = np.sum(~is_finite)
            out_of_range = np.sum(~is_in_range & is_finite)

            self.logger.info(
                "Pre-filtered %d invalid time values: %d non-finite, %d out-of-range",
                outliers_found,
                non_finite,
                out_of_range,
            )

        return valid_indices

    def _get_monotonic_indices(self, time_data) -> list[int]:
        """Get indices for monotonic time values from time data array."""
        mono_indices = []
        if len(time_data) > 0:
            # TODO: What if first point is not valid?  May need to a pre-filtering step.
            mono_indices.append(0)  # Always include first point

            for i in range(1, len(time_data)):
                if time_data[i] > time_data[mono_indices[-1]]:
                    mono_indices.append(i)
                else:
                    self.logger.debug(
                        "Non-monotonic time value at index %8d: %17.6f <= %17.6f",
                        i,
                        time_data[i],
                        time_data[mono_indices[-1]],
                    )

        return mono_indices

    def _plot_universals_filtering(self):
        """Plot longitude, latitude, and depth time coordinate filtering for root group."""
        if not MATPLOTLIB_AVAILABLE:
            self.logger.error("Matplotlib not available. Install with: uv add matplotlib")
            return

        if not self.universals_plot_data:
            self.logger.warning("No universals plot data collected")
            return

        import matplotlib.pyplot as plt

        # Determine which coordinates are available
        coords = []
        coord_names = []
        for coord in ["longitude", "latitude", "depth", "time"]:
            time_coord = f"{coord}_time"
            if time_coord in self.universals_plot_data:
                coords.append(coord)
                coord_names.append(time_coord)

        if not coords:
            self.logger.warning("No universal coordinates found for plotting")
            return

        n_coords = len(coords)
        fig, axes = plt.subplots(n_coords, 4, figsize=(18, 2.5 * n_coords), sharex="col")

        # Handle case of single coordinate
        if n_coords == 1:
            axes = axes.reshape(1, -1)

        log_file_name = self.universals_plot_data[coord_names[0]]["log_file"]
        fig.suptitle(
            f"Universal Coordinates Time Filtering\nFile: {log_file_name}",
            fontsize=14,
            fontweight="bold",
        )

        for i, (coord, time_coord) in enumerate(zip(coords, coord_names, strict=True)):
            data = self.universals_plot_data[time_coord]
            original = data["original"]
            valid_indices = data["valid_indices"]
            valid_data = data["valid_data"]
            mono_indices = data["mono_indices"]
            mono_data = data["mono_data"]
            final_indices = data["final_indices"]
            final_data = data["final_data"]

            self._plot_universals_row(
                axes[i],
                coord,
                original,
                valid_indices,
                valid_data,
                mono_indices,
                mono_data,
                final_indices,
                final_data,
                is_first_row=(i == 0),
            )

        # Set x-label only on bottom row
        for j in range(4):
            axes[-1, j].set_xlabel("Index")

        plt.tight_layout()
        plt.show()

        self.logger.info("Universals time filtering plot displayed")

    def _plot_universals_row(  # noqa: PLR0913
        self,
        axes_row,
        coord: str,
        original,
        valid_indices,
        valid_data,
        mono_indices,
        mono_data,
        final_indices,
        final_data,
        is_first_row: bool,  # noqa: FBT001
    ):
        """Plot a single row of the universals filtering plot."""
        # Column 1: Original data
        axes_row[0].plot(original, "b-", alpha=0.7)
        axes_row[0].set_ylabel(f"{coord}_time\n(Unix seconds)")
        if is_first_row:
            axes_row[0].set_title("Original")
        axes_row[0].grid(visible=True, alpha=0.3)
        axes_row[0].text(
            0.02,
            0.95,
            f"Points: {len(original)}",
            transform=axes_row[0].transAxes,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "lightblue", "alpha": 0.8},
        )

        # Column 2: After outlier removal
        axes_row[1].plot(valid_indices, valid_data, "m.-", alpha=0.7, markersize=2)
        if is_first_row:
            axes_row[1].set_title("After Outlier Removal")
        axes_row[1].grid(visible=True, alpha=0.3)
        removed_outliers = len(original) - len(valid_data)
        axes_row[1].text(
            0.02,
            0.95,
            f"Points: {len(valid_data)}\nRemoved: {removed_outliers}",
            transform=axes_row[1].transAxes,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "plum", "alpha": 0.8},
        )

        # Column 3: After monotonic removal
        axes_row[2].plot(mono_indices, mono_data, "g.-", alpha=0.7, markersize=2)
        if is_first_row:
            axes_row[2].set_title("After Monotonic Removal")
        axes_row[2].grid(visible=True, alpha=0.3)
        removed_monotonic = len(valid_data) - len(mono_data)
        axes_row[2].text(
            0.02,
            0.95,
            f"Points: {len(mono_data)}\nRemoved: {removed_monotonic}",
            transform=axes_row[2].transAxes,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "lightgreen", "alpha": 0.8},
        )

        # Column 4: After intersection
        axes_row[3].plot(final_indices, final_data, "r.-", alpha=0.7, markersize=2)
        if is_first_row:
            axes_row[3].set_title("After Intersection")
        axes_row[3].grid(visible=True, alpha=0.3)
        removed_intersection = len(mono_data) - len(final_data)
        total_removed = len(original) - len(final_data)
        axes_row[3].text(
            0.02,
            0.95,
            f"Points: {len(final_data)}\nRemoved: {removed_intersection}\n"
            f"Total removed: {total_removed} ({100 * total_removed / len(original):.1f}%)",
            transform=axes_row[3].transAxes,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "lightcoral", "alpha": 0.8},
        )

    def _plot_time_filtering(self, plot_data: dict):
        """Plot before and after time coordinate filtering."""
        if not MATPLOTLIB_AVAILABLE:
            self.logger.error("Matplotlib not available. Install with: uv add matplotlib")
            return

        # Import matplotlib here to avoid import errors when not needed
        import matplotlib.pyplot as plt  # noqa: F401

        original = plot_data["original"]
        valid_indices = plot_data["valid_indices"]
        valid_data = plot_data["valid_data"]
        final_indices = plot_data["final_indices"]
        final_data = plot_data["final_data"]

        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

        # Plot 1: Original data
        ax1.plot(original, "b-", label="Original", alpha=0.7)
        ax1.set_ylabel("Time Value")
        ax1.set_title(
            f"Time Coordinate Filtering: {plot_data['variable_name']}\n"
            f"File: {plot_data['log_file']}, Group: {plot_data['group_name']}"
        )
        ax1.legend()
        ax1.grid(visible=True, alpha=0.3)

        # Plot 2: After valid Values filtering
        ax2.plot(valid_indices, valid_data, "m.-", label="After Valid Values Filter", alpha=0.7)
        ax2.set_ylabel("Time Value")
        ax2.legend()
        ax2.grid(visible=True, alpha=0.3)
        ax2.text(
            0.02,
            0.60,
            f"Points removed: {len(original) - len(valid_data)}\n",
            transform=ax2.transAxes,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "wheat"},
        )

        # Plot 3: Final After Monotonic filtered data
        ax3.plot(final_indices, final_data, "r.-", label="After Monotonic Filter", alpha=0.7)
        ax3.set_xlabel("Index")
        ax3.set_ylabel("Time Value")
        ax3.legend()
        ax3.grid(visible=True, alpha=0.3)

        # Add statistics text
        stats_text = (
            f"Points removed: {len(valid_data) - len(final_data)}\n"
            f"Original points: {len(original)}\n"
            f"After final filter: {len(final_data)}\n"
            f"Total removed: {len(original) - len(final_data)} "
            f"({100 * (len(original) - len(final_data)) / len(original):.1f}%)"
        )
        ax3.text(
            0.02,
            0.90,
            stats_text,
            transform=ax3.transAxes,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "wheat"},
        )

        plt.tight_layout()
        plt.show()

        self.logger.info("Time filtering plot displayed for %s", plot_data["variable_name"])

    def _copy_variable_with_appropriate_time_filter(  # noqa: C901, PLR0912
        self,
        src_group: netCDF4.Group,
        dst_dataset: netCDF4.Dataset,
        var_name: str,
        time_filters: dict[str, dict],
    ):
        """Copy a variable with appropriate time filtering applied."""
        src_var = src_group.variables[var_name]

        # Skip variables that use time dimensions with 0 points
        for dim_name in src_var.dimensions:
            if (
                dim_name in time_filters
                and time_filters[dim_name]["filtered"]
                and len(time_filters[dim_name]["indices"]) == 0
            ):
                self.logger.debug(
                    "Skipping variable %s (uses dimension %s with 0 points)", var_name, dim_name
                )
                return

        # Create variable in destination
        try:
            dst_var = dst_dataset.createVariable(
                var_name,
                src_var.dtype,
                src_var.dimensions,
                zlib=True,
                complevel=4,
            )
        except ValueError as e:
            self.logger.warning(
                "Could not create variable %s in destination dataset: %s. ",
                var_name,
                str(e),
            )
            return

        # Check if this variable itself is a time coordinate that needs filtering
        if var_name in time_filters and time_filters[var_name]["filtered"]:
            # This is a time coordinate variable that needs filtering
            time_indices = time_filters[var_name]["indices"]
            dst_var[:] = src_var[:][time_indices]
            dst_var.setncattr("comment", time_filters[var_name]["comment"])
            self.logger.debug("Applied time filtering to time coordinate %s", var_name)

        # Check if this variable depends on any filtered time dimensions
        elif src_var.dimensions:
            # Find which (if any) of this variable's dimensions are filtered time coordinates
            filtered_dims = {}
            for dim_name in src_var.dimensions:
                if dim_name in time_filters and time_filters[dim_name]["filtered"]:
                    filtered_dims[dim_name] = time_filters[dim_name]["indices"]

            if filtered_dims:
                # Apply filtering for the appropriate dimensions
                self._apply_multidimensional_time_filter(src_var, dst_var, var_name, filtered_dims)
            else:
                # No time filtering needed
                dst_var[:] = src_var[:]
        else:
            # Scalar variable or no dimensions
            dst_var[:] = src_var[:]

        # Copy attributes
        for attr_name in src_var.ncattrs():
            dst_var.setncattr(attr_name, src_var.getncattr(attr_name))
        if var_name in time_filters and time_filters[var_name]["filtered"]:
            # Downstream process uses cf_xarray to recognize coordinates, add required attribute
            dst_var.setncattr("standard_name", "time")
        else:
            # Override any coordinates attribute in src with just the time coordinate
            dst_var.setncattr("coordinates", var_name + "_time")
            # Downstream process uses cf_xarray to recognize coordinates, add required attribute
            if src_group.name == "/" and var_name.startswith(("longitude", "latitude")):
                dst_var.setncattr("units", "radians")
            elif var_name.startswith("depth"):
                dst_var.setncattr("units", "meters")

        self.logger.debug("    Copied variable: %s", var_name)

    def _apply_multidimensional_time_filter(
        self, src_var, dst_var, var_name: str, filtered_dims: dict[str, list[int]]
    ):
        """Apply time filtering to a multi-dimensional variable."""
        # For now, handle the common case where time is the first dimension
        if len(filtered_dims) == 1:
            dim_name = list(filtered_dims.keys())[0]
            time_indices = filtered_dims[dim_name]

            if src_var.dimensions[0] == dim_name:
                # Time is first dimension
                if len(src_var.dimensions) == 1:
                    # 1D variable
                    dst_var[:] = src_var[:][time_indices]
                else:
                    # Multi-dimensional with time as first dimension
                    dst_var[:] = src_var[:][time_indices, ...]
                self.logger.debug(
                    "Applied time filtering to variable %s (dim: %s)", var_name, dim_name
                )
            else:
                # Time dimension is not first - more complex indexing needed
                self.logger.warning(
                    "Variable %s has filtered time dimension %s but not as first dimension - "
                    "copying all data",
                    var_name,
                    dim_name,
                )
                dst_var[:] = src_var[:]
        else:
            # Multiple time dimensions filtered - complex case
            self.logger.warning(
                "Variable %s has multiple filtered time dimensions - copying all data", var_name
            )
            dst_var[:] = src_var[:]

    def _create_dimensions_with_time_filters(
        self,
        src_group: netCDF4.Group,
        dst_dataset: netCDF4.Dataset,
        dims_needed: set[str],
        time_filters: dict[str, dict],
    ):
        """Create dimensions in the destination dataset, adjusting time dimensions if filtered."""
        # Use fixed dimensions for all - simpler and avoids NetCDF3 unlimited dimension issues
        for dim_name in dims_needed:
            if dim_name not in src_group.dimensions:
                continue

            src_dim = src_group.dimensions[dim_name]
            size = self._calculate_dimension_size(
                dim_name, src_dim, time_filters, should_be_unlimited=False
            )

            # Skip dimensions with 0 points to avoid NetCDF3 conflicts
            if size == 0:
                self.logger.debug("Skipping dimension %s with 0 points", dim_name)
                continue

            dst_dataset.createDimension(dim_name, size)

    def _calculate_dimension_size(
        self,
        dim_name: str,
        src_dim,
        time_filters: dict[str, dict],
        should_be_unlimited: bool,  # noqa: FBT001
    ) -> int:
        """Calculate the size for a dimension - always returns fixed size for simplicity."""
        is_filtered_time = dim_name in time_filters and time_filters[dim_name]["filtered"]

        if is_filtered_time:
            filtered_size = len(time_filters[dim_name]["indices"])
            self.logger.debug(
                "Created filtered fixed time dimension %s: %s -> %s",
                dim_name,
                len(src_dim),
                filtered_size,
            )
            return filtered_size

        # Non-filtered dimension - always fixed size
        size = len(src_dim)
        if src_dim.isunlimited():
            self.logger.debug(
                "Converting unlimited dimension %s to fixed size %s",
                dim_name,
                size,
            )
        else:
            self.logger.debug("Created fixed dimension %s: %s", dim_name, size)
        return size

    def _align_root_group_coordinates(
        self, time_filters: dict[str, dict], vars_to_extract: list[str]
    ) -> dict[str, dict]:
        """Align latitude and longitude indices in root group when they have different lengths.

        When time coordinate filtering removes different numbers of points from latitude_time
        and longitude_time, we need to use the union of both filtered indices to keep them
        aligned.

        Args:
            time_filters: Dictionary mapping time coordinate names to filter info
            vars_to_extract: List of variable names being extracted

        Returns:
            Modified time_filters with aligned indices for latitude and longitude
        """
        # Only apply to root group variables
        lat_vars = [v for v in vars_to_extract if v.startswith("latitude")]
        lon_vars = [v for v in vars_to_extract if v.startswith("longitude")]

        if not lat_vars or not lon_vars:
            return time_filters

        # Find the time coordinates for latitude and longitude
        lat_time_coords = [f"{v}_time" for v in lat_vars]
        lon_time_coords = [f"{v}_time" for v in lon_vars]

        # Get the filtered time coordinates that exist
        lat_filtered = [
            tc for tc in lat_time_coords if tc in time_filters and time_filters[tc]["filtered"]
        ]
        lon_filtered = [
            tc for tc in lon_time_coords if tc in time_filters and time_filters[tc]["filtered"]
        ]

        if not lat_filtered or not lon_filtered:
            return time_filters

        # For simplicity, handle the common case of single lat/lon time coordinates
        if len(lat_filtered) == 1 and len(lon_filtered) == 1:
            lat_tc = lat_filtered[0]
            lon_tc = lon_filtered[0]

            # Use numpy arrays for efficient intersection - indices are already lists
            lat_indices = np.array(time_filters[lat_tc]["indices"], dtype=np.int64)
            lon_indices = np.array(time_filters[lon_tc]["indices"], dtype=np.int64)

            # Quick check if they're already identical using numpy comparison
            if lat_indices.shape == lon_indices.shape and np.array_equal(lat_indices, lon_indices):
                return time_filters

            # Use numpy's intersect1d for efficient intersection of sorted arrays
            # assume_unique=True since indices come from filtered time coordinates
            aligned_indices = np.intersect1d(lat_indices, lon_indices, assume_unique=True)

            if len(aligned_indices) < len(lat_indices) or len(aligned_indices) < len(lon_indices):
                self.logger.info(
                    "Aligning root group coordinates: latitude has %d points, "
                    "longitude has %d points, using %d common indices",
                    len(lat_indices),
                    len(lon_indices),
                    len(aligned_indices),
                )

                # Convert back to list for consistency with the rest of the code
                aligned_list = aligned_indices.tolist()

                # Update both time filters with aligned indices
                time_filters[lat_tc]["indices"] = aligned_list
                time_filters[lon_tc]["indices"] = aligned_list

                # Update comments to reflect alignment
                alignment_note = " Aligned with longitude/latitude."
                if not time_filters[lat_tc]["comment"].endswith(alignment_note):
                    time_filters[lat_tc]["comment"] += alignment_note
                if not time_filters[lon_tc]["comment"].endswith(alignment_note):
                    time_filters[lon_tc]["comment"] += alignment_note

        return time_filters

    def _create_netcdf_file(  # noqa: PLR0913
        self,
        log_file: str,
        group_name: str,
        src_group: netCDF4.Group,
        vars_to_extract: list[str],
        output_file: Path,
    ):
        """Create a new NetCDF file with the specified variables and monotonic time."""
        # Get time filtering information for each time variable
        time_filters = self._get_time_filters_for_variables(
            log_file, group_name, src_group, vars_to_extract
        )

        with netCDF4.Dataset(output_file, "w", format="NETCDF3_CLASSIC") as dst_dataset:
            # Copy global attributes
            self._copy_global_attributes(src_group, dst_dataset)

            # Add standard global attributes
            log_file = self.log_file
            for attr_name, attr_value in self.global_metadata(log_file, group_name).items():
                dst_dataset.setncattr(attr_name, attr_value)

            # Add note about time filtering if applied
            if any(tf["filtered"] for tf in time_filters.values()):
                dst_dataset.setncattr(
                    "processing_note",
                    "Non-monotonic time values filtered from original, see variable comments",
                )

            # Create dimensions - may need to adjust time dimension sizes
            dims_needed = self._get_required_dimensions(src_group, vars_to_extract)
            self._create_dimensions_with_time_filters(
                src_group, dst_dataset, dims_needed, time_filters
            )

            # Copy coordinate variables with time filtering
            coord_vars = self._get_coordinate_variables(src_group, dims_needed, vars_to_extract)
            for var_name in coord_vars:
                self._copy_variable_with_appropriate_time_filter(
                    src_group, dst_dataset, var_name, time_filters
                )

            # Copy requested variables with time filtering
            for var_name in vars_to_extract:
                self._copy_variable_with_appropriate_time_filter(
                    src_group, dst_dataset, var_name, time_filters
                )

    def _copy_global_attributes(self, src_group: netCDF4.Group, dst_dataset: netCDF4.Dataset):
        """Copy global attributes from source to destination."""
        for attr_name in src_group.ncattrs():
            dst_dataset.setncattr(attr_name, src_group.getncattr(attr_name))

    def _get_required_dimensions(
        self, src_group: netCDF4.Group, vars_to_extract: list[str]
    ) -> set[str]:
        """Get all dimensions needed by the variables to extract."""
        dims_needed = set()
        for var_name in vars_to_extract:
            if var_name in src_group.variables:
                var = src_group.variables[var_name]
                dims_needed.update(var.dimensions)
        return dims_needed

    def _get_coordinate_variables(
        self, src_group: netCDF4.Group, dims_needed: set[str], vars_to_extract: list[str]
    ) -> list[str]:
        """Get coordinate variables that aren't already in vars_to_extract."""
        coord_vars = []
        for dim_name in dims_needed:
            if dim_name in src_group.variables and dim_name not in vars_to_extract:
                coord_vars.append(dim_name)  # noqa: PERF401
        return coord_vars

    def global_metadata(self, log_file: str, group_name: str):
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
        iso_now = datetime.now(UTC).isoformat() + "Z"

        metadata = {}
        metadata["netcdf_version"] = "4"
        metadata["Conventions"] = "CF-1.6"
        metadata["date_created"] = iso_now
        metadata["date_update"] = iso_now
        metadata["date_modified"] = iso_now

        metadata["distribution_statement"] = "Any use requires prior approval from MBARI"
        metadata["license"] = metadata["distribution_statement"]
        metadata["useconst"] = "Not intended for legal use. Data may contain inaccuracies."
        metadata["history"] = f"Created by {self.commandline} on {iso_now}"
        log_file = self.log_file

        # Build title with optional deployment name
        title = f"Extracted LRAUV data from {log_file}, Group: {group_name}"
        deployment_name = get_deployment_name(log_file, BASE_LRAUV_PATH, self.logger)
        if deployment_name:
            title += f" - Deployment: {deployment_name}"
        metadata["title"] = title

        metadata["source"] = (
            f"MBARI LRAUV data extracted from {log_file}"
            f" with execution of '{self.commandline}' at {iso_now}"
            f" using git commit {gitcommit} from"
            f" software at 'https://github.com/mbari-org/auv-python'"
        )
        metadata["group_name"] = group_name
        metadata["summary"] = (
            "Observational oceanographic data obtained from a Long Range Autonomous"
            " Underwater Vehicle mission with measurements at original sampling"
            f" intervals. The data in group {group_name} have been extracted from the"
            " original .nc4 log file with non-monotonic time values removed using"
            " MBARI's auv-python software"
        )
        return metadata

    def process_command_line(self):
        """Process command line arguments using shared parser infrastructure."""
        examples = "Examples:" + "\n\n"
        examples += "  Write to local missionnetcdfs direcory:\n"
        examples += "    " + sys.argv[0] + " --mission 2020.064.10\n"
        examples += "    " + sys.argv[0] + " --auv_name i2map --mission 2020.055.01\n\n"
        examples += "  Plot time coordinate filtering:\n"
        examples += (
            "    "
            + sys.argv[0]
            + " --log_file brizo/missionlogs/2025/20250909_20250915/20250914T080941/"
            + "202509140809_202509150109.nc4 --plot_time /latitude_time\n"
        )
        examples += (
            "    "
            + sys.argv[0]
            + " --log_file brizo/missionlogs/2025/20250909_20250915/20250914T080941/"
            + "202509140809_202509150109.nc4 --plot_universals\n"
        )

        # Use shared parser with nc42netcdfs-specific additions
        parser = get_standard_lrauv_parser(
            description=__doc__,
            epilog=examples,
        )

        # Add nc42netcdfs-specific arguments
        parser.add_argument(
            "--filter_monotonic_time",
            action="store_true",
            default=True,
            help="Filter out non-monotonic time values (default: True)",
        )
        parser.add_argument(
            "--no_filter_monotonic_time",
            dest="filter_monotonic_time",
            action="store_false",
            help="Keep all time values, including non-monotonic ones",
        )
        parser.add_argument(
            "--start",
            action="store",
            help="Convert a range of missions wth start time in YYYYMMDD format",
        )
        parser.add_argument(
            "--end",
            action="store",
            help="Convert a range of missions wth end time in YYYYMMDD format",
        )
        parser.add_argument(
            "--known_hash",
            action="store",
            help=(
                "Known hash for the file to be downloaded, e.g. "
                "d1235ead55023bea05e9841465d54a45dfab007a283320322e28b84438fb8a85"
            ),
        )
        parser.add_argument(
            "--plot_time",
            action="store",
            metavar="VARIABLE_NAME",
            help=(
                "Plot before and after time coordinate filtering for the specified variable. "
                "Shows the effect of outlier removal and monotonic filtering."
                "Format for <VARIABLE_NAME> is /Group/variable_name."
            ),
        )
        parser.add_argument(
            "--plot_universals",
            action="store_true",
            help=(
                "Plot time filtering for longitude, latitude, and depth coordinates "
                "in the root (/) group. Shows original, after outlier removal, after "
                "intersection, and after non-monotonic removal in 4-column layout."
            ),
        )

        self.args = parser.parse_args()

        # Set instance attributes from parsed arguments
        self.log_file = self.args.log_file
        self.plot_time = self.args.plot_time
        self.plot_universals = self.args.plot_universals
        self.filter_monotonic_time = self.args.filter_monotonic_time
        self.verbose = self.args.verbose
        self.commandline = " ".join(sys.argv)
        self.logger.setLevel(self._log_levels[self.verbose])


if __name__ == "__main__":
    extract = Extract()
    extract.process_command_line()
    extract.extract_groups_to_files_netcdf4(extract.args.log_file)
