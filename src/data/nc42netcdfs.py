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

import netCDF4
import pooch
import xarray as xr

# Local directory that serves as the work area for log_files and netcdf files
BASE_LRAUV_WEB = "https://dods.mbari.org/data/lrauv/"
BASE_PATH = Path(__file__).parent.joinpath("../../data/lrauv_data").resolve()
SUMMARY_SOURCE = "Original LRAUV data extracted from {}, group {}"
GROUPS = ["navigation", "ctd", "ecopuck"]  # Your actual group names

SCI_PARMS = {
    "/": [
        {
            "name": "concentration_of_colored_dissolved_organic_matter_in_sea_water",
            "rename": "colored_dissolved_organic_matter",
        }
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

    def extract_groups_to_files(self, input_file, output_dir):
        """Extract each group to a separate NetCDF file."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        all_groups = self.get_groups_netcdf4(input_file)

        for group_name, group_parms in SCIENG_PARMS.items():
            if group_name not in all_groups:
                self.logger.warning("Group %s not found in %s", group_name, input_file)
                continue
            try:
                ds = xr.open_dataset(input_file, group=group_name)
                output_file = output_dir / f"{group_name}.nc"
                # Output only the variables of interest
                parms = [p["name"] for p in group_parms if "name" in p]
                ds = ds[parms]
                ds.to_netcdf(path=str(output_file), format="NETCDF4")
                ds.close()
                self.logger.info("Extracted %s to %s", group_name, output_file)
            except (FileNotFoundError, OSError, ValueError):
                self.logger.warning("Could not extract %s", group_name)
            except KeyError:
                self.logger.warning("Variable %s not found in group %s", parms, group_name)

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
            default=BASE_PATH,
            help="Base directory for missionlogs and missionnetcdfs, default: auv_data",
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
    url = os.path.join(BASE_LRAUV_WEB, extract.args.log_file)  # noqa: PTH118
    output_dir = Path(BASE_PATH, Path(extract.args.log_file).parent)
    extract.logger.info("Downloading %s", url)
    input_file = extract.download_with_pooch(url, output_dir, extract.args.known_hash)
    extract.extract_groups_to_files(input_file, output_dir)
