#!/usr/bin/env python
"""
Extract instrument/group data from LRAUV .nc4 files into individual NetCDF files.

Makes the original data more accessible for analysis and visualization.
"""

__author__ = "Mike McCann"
__copyright__ = "Copyright 2025, Monterey Bay Aquarium Research Institute"

import argparse
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
        {
            "name": "concentration_of_colored_dissolved_organic_matter_in_sea_water",
            "rename": "colored_dissolved_organic_matter",
        },
        {"name": "longitude", "rename": "longitude"},
        {"name": "latitude", "rename": "latitude"},
        {"name": "depth", "rename": "depth"},
        {"name": "time", "rename": "time"},
    ],
    "Aanderaa_O2": [{"name": "mass_concentration_of_oxygen_in_sea_water", "rename": "oxygen"}],
    "CTD_NeilBrown": [
        {"name": "sea_water_salinity", "rename": "salinity"},
        {"name": "sea_water_temperature", "rename": "temperature"},
    ],
    "CTD_Seabird": [
        {"name": "sea_water_salinity", "rename": "salinity"},
        {"name": "sea_water_temperature", "rename": "temperature"},
        {
            "name": "mass_concentration_of_oxygen_in_sea_water",
            "rename": "mass_concentration_of_oxygen_in_sea_water",
        },
    ],
    "ISUS": [{"name": "mole_concentration_of_nitrate_in_sea_water", "rename": "nitrate"}],
    "PAR_Licor": [{"name": "downwelling_photosynthetic_photon_flux_in_sea_water", "rename": "PAR"}],
    "WetLabsBB2FL": [
        {"name": "mass_concentration_of_chlorophyll_in_sea_water", "rename": "chlorophyll"},
        {"name": "OutputChl", "rename": "chl"},
        {"name": "Output470", "rename": "bbp470"},
        {"name": "Output650", "rename": "bbp650"},
        {"name": "VolumeScatCoeff117deg470nm", "rename": "volumescatcoeff117deg470nm"},
        {"name": "VolumeScatCoeff117deg650nm", "rename": "volumescatcoeff117deg650nm"},
        {
            "name": "ParticulateBackscatteringCoeff470nm",
            "rename": "particulatebackscatteringcoeff470nm",
        },
        {
            "name": "ParticulateBackscatteringCoeff650nm",
            "rename": "particulatebackscatteringcoeff650nm",
        },
    ],
    "WetLabsSeaOWL_UV_A": [
        {
            "name": "concentration_of_chromophoric_dissolved_organic_matter_in_sea_water",
            "rename": "chromophoric_dissolved_organic_matter",
        },
        {"name": "mass_concentration_of_chlorophyll_in_sea_water", "rename": "chlorophyll"},
        {"name": "BackscatteringCoeff700nm", "rename": "BackscatteringCoeff700nm"},
        {"name": "VolumeScatCoeff117deg700nm", "rename": "VolumeScatCoeff117deg700nm"},
        {
            "name": "mass_concentration_of_petroleum_hydrocarbons_in_sea_water",
            "rename": "petroleum_hydrocarbons",
        },
    ],
    "WetLabsUBAT": [
        {"name": "average_bioluminescence", "rename": "average_bioluminescence"},
        {"name": "flow_rate", "rename": "ubat_flow_rate"},
        {"name": "digitized_raw_ad_counts", "rename": "digitized_raw_ad_counts"},
    ],
}

ENG_PARMS = {
    "BPC1": [
        {"name": "platform_battery_charge", "rename": "health_platform_battery_charge"},
        {"name": "platform_battery_voltage", "rename": "health_platform_average_voltage"},
    ],
    "BuoyancyServo": [
        {"name": "platform_buoyancy_position", "rename": "control_inputs_buoyancy_position"}
    ],
    "DeadReckonUsingMultipleVelocitySources": [
        {
            "name": "fix_residual_percent_distance_traveled",
            "rename": (
                "fix_residual_percent_distance_traveled_DeadReckonUsingMultipleVelocitySources"
            ),
        },
        {"name": "longitude", "rename": "pose_longitude_DeadReckonUsingMultipleVelocitySources"},
        {"name": "latitude", "rename": "pose_latitude_DeadReckonUsingMultipleVelocitySources"},
        {"name": "depth", "rename": "pose_depth_DeadReckonUsingMultipleVelocitySources"},
    ],
    "DeadReckonUsingSpeedCalculator": [
        {
            "name": "fix_residual_percent_distance_traveled",
            "rename": "fix_residual_percent_distance_traveled_DeadReckonUsingSpeedCalculator",
        },
        {"name": "longitude", "rename": "pose_longitude_DeadReckonUsingSpeedCalculator"},
        {"name": "latitude", "rename": "pose_latitude_DeadReckonUsingSpeedCalculator"},
        {"name": "depth", "rename": "pose_depth_DeadReckonUsingSpeedCalculator"},
    ],
    "ElevatorServo": [
        {"name": "platform_elevator_angle", "rename": "control_inputs_elevator_angle"}
    ],
    "MassServo": [{"name": "platform_mass_position", "rename": "control_inputs_mass_position"}],
    "NAL9602": [
        {"name": "time_fix", "rename": "fix_time"},
        {"name": "latitude_fix", "rename": "fix_latitude"},
        {"name": "longitude_fix", "rename": "fix_longitude"},
    ],
    "Onboard": [{"name": "platform_average_current", "rename": "health_platform_average_current"}],
    "RudderServo": [{"name": "platform_rudder_angle", "rename": "control_inputs_rudder_angle"}],
    "ThrusterServo": [
        {
            "name": "platform_propeller_rotation_rate",
            "rename": "control_inputs_propeller_rotation_rate",
        }
    ],
    "CurrentEstimator": [
        {
            "name": "current_direction_navigation_frame",
            "rename": "current_direction_navigation_frame",
        },
        {"name": "current_speed_navigation_frame", "rename": "current_speed_navigation_frame"},
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

    def show_variable_mapping(self):
        """Show the variable mapping."""
        for group, parms in sorted(SCIENG_PARMS.items()):
            print(f"Group: {group}")  # noqa: T201
            for parm in parms:
                name = parm.get("name", "N/A")
                rename = parm.get("rename", "N/A")
                print(f"  {name} -> {rename}")  # noqa: T201
            print()  # noqa: T201

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

    def get_groups_netcdf4(self, file_path):
        """Get list of groups using netCDF4 library."""
        with netCDF4.Dataset(file_path, "r") as dataset:
            return list(dataset.groups.keys())

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

        try:
            self.logger.info("Extracting root group '/'")
            vars_to_extract = self._get_available_variables(src_dataset, root_parms)

            if vars_to_extract:
                output_file = output_dir / f"{Path(log_file).stem}_{GROUP}_Universals.nc"
                self._create_netcdf_file(
                    log_file, group_name, src_dataset, vars_to_extract, output_file
                )
                self.logger.info("Extracted root group '/' to %s", output_file)
            else:
                self.logger.warning("No requested variables found in root group '/'")

        except Exception as e:  # noqa: BLE001
            self.logger.warning("Could not extract root group '/': %s", e)

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

            vars_to_extract = self._get_available_variables(src_group, group_parms)

            if vars_to_extract:
                output_file = output_dir / f"{Path(log_file).stem}_{GROUP}_{group_name}.nc"
                self._create_netcdf_file(
                    log_file, group_name, src_group, vars_to_extract, output_file
                )
                self.logger.info("Extracted %s to %s", group_name, output_file)
            else:
                self.logger.warning("No requested variables found in group %s", group_name)

        except KeyError:
            self.logger.warning("Group %s not found", group_name)
        # except Exception as e:  # noqa: BLE001
        #     self.logger.warning("Could not extract %s: %s", group_name, e)

    def _get_available_variables(
        self, src_group: netCDF4.Group, group_parms: list[dict[str, Any]]
    ) -> list[str]:
        """Get the intersection of requested and available variables."""
        requested_vars = [p["name"] for p in group_parms if "name" in p]
        available_vars = list(src_group.variables.keys())
        vars_to_extract = [var for var in requested_vars if var in available_vars]

        self.logger.debug("  Variables to extract: %s", vars_to_extract)
        return vars_to_extract

    def _find_time_coordinate(self, src_group: netCDF4.Group) -> str:
        """Find the time coordinate variable in a group using introspection.

        Returns:
            str: Name of the time coordinate variable, or empty string if not found
        """
        # Strategy 1: Look for variables with "time" in the name (most common)
        time_vars = [var_name for var_name in src_group.variables if "time" in var_name.lower()]
        if time_vars:
            # Prefer variables that start with 'time' (like time_NAL9602)
            time_vars.sort(key=lambda x: (not x.lower().startswith("time"), x))
            self.logger.debug("Found time coordinate %s via name pattern", time_vars[0])
            return time_vars[0]

        # Strategy 2: Look for variables with time-like units
        for var_name, var in src_group.variables.items():
            if hasattr(var, "units"):
                units = getattr(var, "units", "").lower()
                time_patterns = ["seconds since", "days since", "hours since"]
                if any(pattern in units for pattern in time_patterns):
                    self.logger.debug("Found time coordinate %s via units", var_name)
                    return var_name

        # Strategy 3: Look for unlimited dimension (backup)
        for dim_name, dim in src_group.dimensions.items():
            if dim.isunlimited() and dim_name in src_group.variables:
                self.logger.debug("Found time coordinate %s via unlimited dimension", dim_name)
                return dim_name

        self.logger.debug("No time coordinate found in group")
        return ""

    def _get_time_filters_for_variables(
        self, log_file: str, group_name: str, src_group: netCDF4.Group, vars_to_extract: list[str]
    ) -> dict[str, dict]:
        """Get time filtering information for time coordinates used by vars_to_extract.

        Returns:
            dict: Map of time_coord_name -> {"indices": list[int], "filtered": bool}
        """
        # Check if time filtering is enabled
        if not getattr(self.args, "filter_monotonic_time", True):
            return {}

        self.logger.info("========================= Group %s =========================", group_name)
        # Find all time coordinates used by variables in extraction list
        time_coords_found = self._find_time_coordinates(group_name, src_group, vars_to_extract)

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

        return time_filters

    def _find_time_coordinates(
        self, group_name: str, src_group: netCDF4.Group, vars_to_extract: list[str]
    ) -> set[str]:
        """Find all time coordinates used by variables in extraction list."""
        time_coords_found = set()
        self.logger.debug(
            "=================================== Group: %s =======================================",
            group_name,
        )
        for var_name in vars_to_extract:
            if var_name in src_group.variables:
                var = src_group.variables[var_name]

                # Check each dimension to see if it's a time coordinate
                for dim_name in var.dimensions:
                    if dim_name in src_group.variables:
                        dim_var = src_group.variables[dim_name]

                        # Check if this dimension variable is a time coordinate
                        if self._is_time_variable(dim_name, dim_var):
                            time_coords_found.add(dim_name)

        return time_coords_found

    def _parse_plot_time_argument(self) -> tuple[str | None, str | None]:
        """Parse the --plot_time argument and return (group_name, time_coord_name)."""
        if not getattr(self.args, "plot_time", None):
            return None, None

        plot_time = self.args.plot_time
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
        from scipy.signal import medfilt

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

        # Despike to remove single point outliers before getting monotonic indices
        time_data = medfilt(original_time_data[valid_indices], kernel_size=3)

        # Store valid indices and despiked data for plotting
        if plot_data is not None:
            plot_data["valid_indices"] = valid_indices
            plot_data["valid_data"] = original_time_data[valid_indices]
            plot_data["despiked"] = time_data.copy()

        # Now apply monotonic filtering to the valid subset
        mono_indices = self._get_monotonic_indices(time_data)

        # Generate plot if requested for this variable
        if plot_data is not None:
            plot_data["final_indices"] = mono_indices
            plot_data["final_data"] = time_data[mono_indices]
            self._plot_time_filtering(plot_data)

        return self._create_time_filter_result(mono_indices, len(time_data), time_coord_name)

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
        Uses numpy for efficient vectorized operations.
        """
        # LRAUV data bounds: September 2012 to current + 5 years buffer
        lrauv_start_date = datetime(2012, 9, 1, tzinfo=UTC)
        current_date = datetime.now(UTC)
        future_buffer_date = current_date.replace(year=current_date.year + 5)

        MIN_UNIX_TIME = int(lrauv_start_date.timestamp())  # September 1, 2012 UTC
        MAX_UNIX_TIME = int(future_buffer_date.timestamp())  # Current + 5 years buffer

        # Convert to numpy array for efficient operations
        time_array = np.asarray(time_data)

        # Create boolean masks for valid conditions
        is_finite = np.isfinite(time_array)
        is_in_range = (time_array >= MIN_UNIX_TIME) & (time_array <= MAX_UNIX_TIME)

        # Combine all conditions - all must be True for valid indices
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
        despiked = plot_data["despiked"]
        final_indices = plot_data["final_indices"]
        final_data = plot_data["final_data"]

        # Create figure with subplots
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 9), sharex=True)

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

        # Plot 3: After despiking
        ax3.plot(despiked, "g-", label="After Median Filter (3-point)", alpha=0.7)
        ax3.set_ylabel("Time Value")
        ax3.legend()
        ax3.grid(visible=True, alpha=0.3)
        ax3.text(
            0.02,
            0.60,
            f"Points removed: {len(valid_data) - len(despiked)}\n",
            transform=ax3.transAxes,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "wheat"},
        )

        # Plot 4: Final After Monotonic filtered data
        ax4.plot(final_indices, final_data, "r.-", label="After Monotonic Filter", alpha=0.7)
        ax4.set_xlabel("Index")
        ax4.set_ylabel("Time Value")
        ax4.legend()
        ax4.grid(visible=True, alpha=0.3)

        # Add statistics text
        stats_text = (
            f"Points removed: {len(despiked) - len(final_data)}\n"
            f"Original points: {len(original)}\n"
            f"After final filter: {len(final_data)}\n"
            f"Total removed: {len(original) - len(final_data)} "
            f"({100 * (len(original) - len(final_data)) / len(original):.1f}%)"
        )
        ax4.text(
            0.02,
            0.90,
            stats_text,
            transform=ax4.transAxes,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "wheat"},
        )

        plt.tight_layout()
        plt.show()

        self.logger.info("Time filtering plot displayed for %s", plot_data["variable_name"])

    def _copy_variable_with_appropriate_time_filter(
        self,
        src_group: netCDF4.Group,
        dst_dataset: netCDF4.Dataset,
        var_name: str,
        time_filters: dict[str, dict],
    ):
        """Copy a variable with appropriate time filtering applied."""
        try:
            src_var = src_group.variables[var_name]

            # Create variable in destination
            dst_var = dst_dataset.createVariable(
                var_name,
                src_var.dtype,
                src_var.dimensions,
                zlib=True,
                complevel=6,
                shuffle=True,
                fletcher32=True,
            )

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
                    self._apply_multidimensional_time_filter(
                        src_var, dst_var, var_name, filtered_dims
                    )
                else:
                    # No time filtering needed
                    dst_var[:] = src_var[:]
            else:
                # Scalar variable or no dimensions
                dst_var[:] = src_var[:]

            # Copy attributes
            for attr_name in src_var.ncattrs():
                dst_var.setncattr(attr_name, src_var.getncattr(attr_name))

            self.logger.debug("    Copied variable: %s", var_name)

        except Exception as e:  # noqa: BLE001
            self.logger.warning("Failed to copy variable %s: %s", var_name, e)

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
        for dim_name in dims_needed:
            if dim_name in src_group.dimensions:
                src_dim = src_group.dimensions[dim_name]

                # Check if this dimension corresponds to a filtered time variable
                if dim_name in time_filters and time_filters[dim_name]["filtered"]:
                    # Use the number of filtered time points
                    filtered_size = len(time_filters[dim_name]["indices"])
                    size = filtered_size if not src_dim.isunlimited() else None
                    self.logger.debug(
                        "Created filtered time dimension %s: %s -> %s",
                        dim_name,
                        len(src_dim),
                        size or filtered_size,
                    )
                else:
                    size = len(src_dim) if not src_dim.isunlimited() else None

                dst_dataset.createDimension(dim_name, size)

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

        with netCDF4.Dataset(output_file, "w", format="NETCDF4") as dst_dataset:
            # Copy global attributes
            self._copy_global_attributes(src_group, dst_dataset)

            # Add standard global attributes
            log_file = self.args.log_file
            for attr_name, attr_value in self.global_metadata(log_file, group_name).items():
                dst_dataset.setncattr(attr_name, attr_value)

            # Add note about time filtering if applied
            if any(tf["filtered"] for tf in time_filters.values()):
                dst_dataset.setncattr(
                    "processing_note",
                    "Non-monotonic time values filtered from original, see comment in variables",
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

    def _create_dimensions(
        self, src_group: netCDF4.Group, dst_dataset: netCDF4.Dataset, dims_needed: set[str]
    ):
        """Create dimensions in the destination dataset."""
        for dim_name in dims_needed:
            if dim_name in src_group.dimensions:
                src_dim = src_group.dimensions[dim_name]
                size = len(src_dim) if not src_dim.isunlimited() else None
                dst_dataset.createDimension(dim_name, size)

    def _get_coordinate_variables(
        self, src_group: netCDF4.Group, dims_needed: set[str], vars_to_extract: list[str]
    ) -> list[str]:
        """Get coordinate variables that aren't already in vars_to_extract."""
        coord_vars = []
        for dim_name in dims_needed:
            if dim_name in src_group.variables and dim_name not in vars_to_extract:
                coord_vars.append(dim_name)  # noqa: PERF401
        return coord_vars

    def _copy_variable(self, src_group: netCDF4.Group, dst_dataset: netCDF4.Dataset, var_name: str):
        """Helper method to copy a variable from source to destination."""
        try:
            src_var = src_group.variables[var_name]

            # Create variable in destination
            dst_var = dst_dataset.createVariable(
                var_name,
                src_var.dtype,
                src_var.dimensions,
                zlib=True,
                complevel=6,
                shuffle=True,
                fletcher32=True,
            )

            # Copy data and attributes
            dst_var[:] = src_var[:]
            for attr_name in src_var.ncattrs():
                dst_var.setncattr(attr_name, src_var.getncattr(attr_name))

            self.logger.debug("    Copied variable: %s", var_name)

        except Exception as e:  # noqa: BLE001
            self.logger.warning("Failed to copy variable %s: %s", var_name, e)

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
        log_file = self.args.log_file
        metadata["title"] = f"Extracted LRAUV data from {log_file}, Group: {group_name}"
        metadata["source"] = (
            f"MBARI LRAUV data extracted from {log_file}"
            f" with execution of '{self.commandline}' at {iso_now}"
            f" using git commit {gitcommit} from"
            f" software at 'https://github.com/mbari-org/auv-python'"
        )
        metadata["summary"] = (
            "Observational oceanographic data obtained from a Long Range Autonomous"
            " Underwater Vehicle mission with measurements at original sampling"
            f" intervals. The data in group {group_name} have been extracted from the"
            " original .nc4 log file with non-monotonic time values removed using"
            " MBARI's auv-python software"
        )
        return metadata

    def process_command_line(self):
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

        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter,
            description=__doc__,
            epilog=examples,
        )
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
            "--auv_name",
            action="store",
            help="Name of the AUV and the directory name for its data, e.g.: tethys, ahi, pontus",
        )
        parser.add_argument(
            "--log_file",
            action="store",
            help=(
                "Path to the log file for the mission, e.g.: "
                "brizo/missionlogs/2025/20250903_20250909/"
                "20250905T072042/202509050720_202509051653.nc4"
            ),
        )
        parser.add_argument(
            "--known_hash",
            action="store",
            help=(
                "Known hash for the file to be downloaded, e.g. "
                "d1235ead55023bea05e9841465d54a45dfab007a283320322e28b84438fb8a85"
            ),
        )
        (
            parser.add_argument(
                "--show_variable_mapping",
                action="store_true",
                help="Show the variable mapping: Group/variable_names -> their_renames",
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
    extract = Extract()
    extract.process_command_line()
    if extract.args.show_variable_mapping:
        extract.show_variable_mapping()
        sys.exit(0)
    else:
        extract.extract_groups_to_files_netcdf4(extract.args.log_file)
