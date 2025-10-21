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
from pathlib import Path
from typing import Any

import netCDF4
import pooch
import xarray as xr

# Local directory that serves as the work area for log_files and netcdf files
BASE_LRAUV_WEB = "https://dods.mbari.org/data/lrauv/"
BASE_LRAUV_PATH = Path(__file__).parent.joinpath("../../data/lrauv_data").resolve()
SUMMARY_SOURCE = "Original LRAUV data extracted from {}, group {}"
GROUPS = ["navigation", "ctd", "ecopuck"]  # Your actual group names

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
            "rename": "fix_residual_percent_distance_traveled_DeadReckonUsingMultipleVelocitySources",  # noqa: E501
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

        extract.logger.info("Downloading %s", url)
        input_file = extract.download_with_pooch(url, netcdfs_dir, self.args.known_hash)

        self.logger.info("Extracting data from %s", input_file)
        with netCDF4.Dataset(input_file, "r") as src_dataset:
            # Extract root group first
            self._extract_root_group(src_dataset, log_file, netcdfs_dir)

            # Extract all other groups
            all_groups = list(src_dataset.groups.keys())
            for group_name in SCIENG_PARMS:
                if group_name == "/" or group_name not in all_groups:
                    if group_name != "/" and group_name not in all_groups:
                        self.logger.warning("Group %s not found in %s", group_name, input_file)
                    continue
                self._extract_single_group(src_dataset, group_name, log_file, netcdfs_dir)

        return netcdfs_dir

    def _extract_root_group(self, src_dataset: netCDF4.Dataset, log_file: str, output_dir: Path):
        """Extract variables from the root group to <stem>_Group_Universals.nc."""
        root_parms = SCIENG_PARMS.get("/", [])
        if not root_parms:
            return

        try:
            self.logger.info("Extracting root group '/'")
            vars_to_extract = self._get_available_variables(src_dataset, root_parms)

            if vars_to_extract:
                output_file = output_dir / f"{Path(log_file).stem}_Group_Universals.nc"
                self._create_netcdf_file(src_dataset, vars_to_extract, output_file)
                self.logger.info("Extracted root group '/' to %s", output_file)
            else:
                self.logger.warning("No requested variables found in root group '/'")

        except Exception as e:  # noqa: BLE001
            self.logger.warning("Could not extract root group '/': %s", e)

    def _extract_single_group(
        self, src_dataset: netCDF4.Dataset, group_name: str, log_file: str, output_dir: Path
    ):
        """Extract a single group to its own NetCDF file named like <stem>_Group_<group_name>.nc."""
        group_parms = SCIENG_PARMS[group_name]

        try:
            self.logger.info(" Group %s", group_name)
            src_group = src_dataset.groups[group_name]

            vars_to_extract = self._get_available_variables(src_group, group_parms)

            if vars_to_extract:
                output_file = output_dir / f"{Path(log_file).stem}_Group_{group_name}.nc"
                self._create_netcdf_file(src_group, vars_to_extract, output_file)
                self.logger.info("Extracted %s to %s", group_name, output_file)
            else:
                self.logger.warning("No requested variables found in group %s", group_name)

        except KeyError:
            self.logger.warning("Group %s not found", group_name)
        except Exception as e:  # noqa: BLE001
            self.logger.warning("Could not extract %s: %s", group_name, e)

    def _get_available_variables(
        self, src_group: netCDF4.Group, group_parms: list[dict[str, Any]]
    ) -> list[str]:
        """Get the intersection of requested and available variables."""
        requested_vars = [p["name"] for p in group_parms if "name" in p]
        available_vars = list(src_group.variables.keys())
        vars_to_extract = [var for var in requested_vars if var in available_vars]

        self.logger.debug("  Variables to extract: %s", vars_to_extract)
        return vars_to_extract

    def _create_netcdf_file(
        self, src_group: netCDF4.Group, vars_to_extract: list[str], output_file: Path
    ):
        """Create a new NetCDF file with the specified variables."""
        with netCDF4.Dataset(output_file, "w", format="NETCDF4") as dst_dataset:
            # Copy global attributes
            self._copy_global_attributes(src_group, dst_dataset)

            # Create dimensions
            dims_needed = self._get_required_dimensions(src_group, vars_to_extract)
            self._create_dimensions(src_group, dst_dataset, dims_needed)

            # Copy coordinate variables
            coord_vars = self._get_coordinate_variables(src_group, dims_needed, vars_to_extract)
            for var_name in coord_vars:
                self._copy_variable(src_group, dst_dataset, var_name)

            # Copy requested variables
            for var_name in vars_to_extract:
                self._copy_variable(src_group, dst_dataset, var_name)

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

    def extract_groups_to_files(self, input_file, output_dir):
        """Extract each group to a separate NetCDF file."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        all_groups = self.get_groups_netcdf4(input_file)

        self.logger.info("Extracting data from %s", input_file)
        for group_name, group_parms in SCIENG_PARMS.items():
            if group_name not in all_groups:
                self.logger.warning("Group %s not found in %s", group_name, input_file)
                continue
            try:
                self.logger.info(" Group %s", group_name)
                ds = xr.open_dataset(input_file, group=group_name)
                output_file = output_dir / f"{group_name}.nc"
                # Output only the variables of interest
                parms = [p["name"] for p in group_parms if "name" in p]
                self.logger.debug("  Variables to extract: %s", parms)
                ds = ds[parms]
                ds.to_netcdf(path=str(output_file), format="NETCDF4")
                ds.close()
                self.logger.info("Extracted %s to %s", group_name, output_file)
            except (FileNotFoundError, OSError, ValueError):
                self.logger.warning("Could not extract %s", group_name)
            except KeyError:
                self.logger.warning("Variable %s not found in group %s", parms, group_name)
            except TypeError:
                self.logger.warning(
                    "Type error processing group %s: %s", group_name, sys.exc_info()
                )

    def process_command_line(self):
        examples = "Examples:" + "\n\n"
        examples += "  Write to local missionnetcdfs direcory:\n"
        examples += "    " + sys.argv[0] + " --mission 2020.064.10\n"
        examples += "    " + sys.argv[0] + " --auv_name i2map --mission 2020.055.01\n"

        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter,
            description=__doc__,
            epilog=examples,
        )

        parser.add_argument(
            "--base_path",
            action="store",
            default=BASE_LRAUV_PATH,
            help=(
                "Base directory for missionlogs and missionnetcdfs, "
                "default: auv_data in repo data directory"
            ),
        )
        parser.add_argument(
            "--title",
            action="store",
            help="A short description of the dataset",
        )
        parser.add_argument(
            "--summary",
            action="store",
            help="Additional information about the dataset",
        )

        parser.add_argument(
            "--noinput",
            action="store_true",
            help="Execute without asking for a response, e.g.  to not ask to re-download file",
        )
        parser.add_argument(
            "--clobber",
            action="store_true",
            help="Use with --noinput to overwrite existing downloaded log files",
        )
        parser.add_argument(
            "--noreprocess",
            action="store_true",
            help="Use with --noinput to not re-process existing downloaded log files",
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
