#!/usr/bin/env python
"""
Resample variables from mission netCDF file to common time axis

Read all the record variables stored at original instrument sampling rate
from netCDF file and resample them to common time axis.
"""

__author__ = "Mike McCann"
__copyright__ = "Copyright 2021, Monterey Bay Aquarium Research Institute"

import logging  # noqa: I001
import re
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from socket import gethostname

import cf_xarray  # Needed for the .cf accessor  # noqa: F401
import git
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from pysolar.solar import get_altitude
from scipy import signal

from common_args import get_standard_lrauv_parser
from dorado_info import dorado_info
from logs2netcdfs import AUV_NetCDF, BASE_PATH, MISSIONNETCDFS, SUMMARY_SOURCE, TIME
from nc42netcdfs import BASE_LRAUV_PATH, BASE_LRAUV_WEB

MF_WIDTH = 3
FREQ = "1S"
PLOT_SECONDS = 300
AUVCTD_OPENDAP_BASE = "http://dods.mbari.org/opendap/data/auvctd"
FLASH_THRESHOLD = 1.0e11
DEPTH_THRESHOLD = 10.0  # meters


class InvalidAlignFile(Exception):
    pass


class Resampler:
    logger = logging.getLogger(__name__)
    _handler = logging.StreamHandler()
    _handler.setFormatter(AUV_NetCDF._formatter)
    logger.addHandler(_handler)
    _log_levels = (logging.WARN, logging.INFO, logging.DEBUG)

    def __init__(  # noqa: PLR0913
        self,
        auv_name: str = None,
        mission: str = None,
        log_file: str = None,
        freq: str = FREQ,
        mf_width: int = MF_WIDTH,
        flash_threshold: float = None,
        verbose: int = 0,
        plot: bool = None,  # noqa: FBT001
        commandline: str = "",
    ) -> None:
        """Initialize Resampler with explicit parameters.

        Args:
            auv_name: Name of the AUV vehicle
            mission: Mission identifier
            log_file: Log file path (for LRAUV processing)
            freq: Resampling frequency (default: '1S')
            mf_width: Median filter width (default: 3)
            flash_threshold: Flash detection threshold
            verbose: Verbosity level (0-2)
            plot: Enable plotting
            commandline: Command line string for tracking
        """
        self.auv_name = auv_name
        self.mission = mission
        self.log_file = log_file
        self.freq = freq
        self.mf_width = mf_width
        self.flash_threshold = flash_threshold
        self.verbose = verbose
        self.plot = plot
        self.commandline = commandline

        plt.rcParams["figure.figsize"] = (15, 5)
        self.resampled_nc = xr.Dataset()
        iso_now = datetime.now(tz=UTC).isoformat().split(".")[0] + "Z"
        # Common static attributes for all auv platforms
        self.metadata = {}
        self.metadata["netcdf_version"] = "4"
        self.metadata["Conventions"] = "CF-1.6"
        self.metadata["date_created"] = iso_now
        self.metadata["date_update"] = iso_now
        self.metadata["date_modified"] = iso_now
        self.metadata["featureType"] = "trajectory"

    def _build_global_metadata(self) -> None:
        """
        Call following saving of coordinates and variables from resample_mission()
        """
        # Skip dynamic metadata during testing to ensure reproducible results
        if "pytest" in sys.modules:
            return {}
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
        iso_now = datetime.now(tz=UTC).isoformat().split(".")[0] + "Z"

        # Ensure that only the latitude, longitude, and depth variables have
        # standard_name attributes equal to "latitude", "longitude", and "depth" so that
        # the .cf[] accessor works correctly
        for var in self.resampled_nc.data_vars:
            standard_name = self.resampled_nc[var].attrs.get("standard_name")
            if standard_name in ["latitude", "longitude", "depth"]:
                if var in {"latitude", "longitude", "depth"}:
                    continue
                self.logger.info("Removing standard_name attribute from variable %s", var)
                del self.resampled_nc[var].attrs["standard_name"]

        # Common dynamic attributes for all auv platforms
        self.metadata["time_coverage_start"] = str(min(self.resampled_nc.time.values))
        self.metadata["time_coverage_end"] = str(max(self.resampled_nc.time.values))
        self.metadata["time_coverage_duration"] = str(
            pd.to_datetime(max(self.resampled_nc.time.values))
            - pd.to_datetime(min(self.resampled_nc.time.values)),
        )
        self.metadata["geospatial_vertical_min"] = min(
            self.resampled_nc.cf["depth"].values,
        )
        self.metadata["geospatial_vertical_max"] = max(
            self.resampled_nc.cf["depth"].values,
        )
        self.metadata["geospatial_lat_min"] = min(
            self.resampled_nc.cf["latitude"].values,
        )
        self.metadata["geospatial_lat_max"] = max(
            self.resampled_nc.cf["latitude"].values,
        )
        self.metadata["geospatial_lon_min"] = min(
            self.resampled_nc.cf["longitude"].values,
        )
        self.metadata["geospatial_lon_max"] = max(
            self.resampled_nc.cf["longitude"].values,
        )
        self.metadata["license"] = "Any use requires prior approval from MBARI"
        self.metadata["history"] = f"Created by {self.commandline} on {iso_now}"

        if "process_" in self.commandline:  # noqa: SIM108
            # Either process_i2map.py or process_dorado.py was used which
            # means the starting point was the original log files
            from_data = "original log files"
        else:
            # Otherwise if resample.py was used, the starting point is _align.nc
            from_data = "aligned data"
        self.metadata["source"] = (
            f"MBARI Dorado-class AUV data produced from {from_data}"
            f" with execution of '{self.commandline}' at {iso_now} on"
            f" host {gethostname()} using git commit {gitcommit} from"
            f" software at 'https://github.com/mbari-org/auv-python'"
        )
        self.metadata["summary"] = (
            f"Observational oceanographic data obtained from an Autonomous"
            f" Underwater Vehicle mission with measurements sampled at"
            f" {self.freq} intervals."
            f" Data processed at {iso_now} using MBARI's auv-python software."
        )
        return None

    def dorado_global_metadata(self) -> dict:  # noqa: PLR0912
        """Use instance variables to return a dictionary of
        metadata specific for the data that are written.
        Calls _build_global_metadata() first to populate common metadata.
        """
        # Skip dynamic metadata during testing to ensure reproducible results
        if "pytest" in sys.modules:
            return {}

        # First populate common metadata (git commit, host, geospatial bounds, etc.)
        self._build_global_metadata()

        # Then add dorado-specific metadata
        self.metadata["title"] = "Calibrated, "
        try:
            if dorado_info[self.mission].get("program"):
                self.metadata["title"] = (
                    f"{dorado_info[self.mission]['program']} program - calibrated, "
                )
        except KeyError:
            self.logger.warning(
                "No entry for for mission %s program in dorado_info.py",
                self.mission,
            )
        self.metadata["title"] += (
            f"aligned, and resampled AUV sensor data from {self.auv_name} mission {self.mission}"
        )
        if "summary" in self.ds.attrs:
            self.metadata["summary"] = self.ds.attrs["summary"]
        try:
            self.metadata["summary"] += (
                f" Processing log file: {AUVCTD_OPENDAP_BASE}/surveys/"
                f"{self.mission.split('.')[0]}/netcdf/"
                f"{self.auv_name}_{self.mission}_processing.log"
            )
        except KeyError:
            # Likely no _1S.nc file was created, hence no summary to append to
            self.logger.warning(
                "Could not add processing log file to summary matadata for mission %s",
                self.mission,
            )

        try:
            if dorado_info[self.mission].get("program"):
                self.metadata["program"] = dorado_info[self.mission].get("program")
            if dorado_info[self.mission].get("comment"):
                self.metadata["comment"] = dorado_info[self.mission].get("comment")
        except KeyError:
            self.logger.warning(
                "No entry for for mission %s program or comment in dorado_info.py",
                self.mission,
            )
        try:
            # Parse from ctd1_depth comment: "using SensorOffset(x=1.003, y=0.0001)"
            self.metadata["comment"] += (
                f". Variable depth pitch corrected using"
                f" {self.ds['ctd1_depth'].attrs['comment'].split('using ')[1]}"
            )
        except KeyError:
            self.logger.warning(
                "No comment for pitch correction in ctd1_depth for mission %s",
                self.mission,
            )

        return self.metadata

    def i2map_global_metadata(self) -> dict:
        """Use instance variables to return a dictionary of
        metadata specific for the data that are written.
        Calls _build_global_metadata() first to populate common metadata.
        """
        # Skip dynamic metadata during testing to ensure reproducible results
        if "pytest" in sys.modules:
            return {}

        # First populate common metadata (git commit, host, geospatial bounds, etc.)
        self._build_global_metadata()

        # Then add i2map-specific metadata
        self.metadata["title"] = (
            f"Calibrated, aligned, and resampled AUV sensor data from"
            f" {self.auv_name} mission {self.mission}"
        )
        # Append location of original data files to summary
        self.metadata["summary"] = self.ds.attrs.get
        matches = re.search(
            "(" + SUMMARY_SOURCE.replace("{}", r".+$") + ")",
            self.ds.attrs["summary"],
        )
        if matches:
            self.metadata["summary"] += (
                " "
                + matches.group(1)
                + f".  Processing log file: {AUVCTD_OPENDAP_BASE}/surveys/"
                + f"{self.mission.split('.')[0]}/netcdf/"
                + f"{self.auv_name}_{self.mission}_processing.log"
            )
            # Append shortened location of original data files to title
            # Useful for I2Map data as it's in a YYYY/MM directory structure
            self.metadata["title"] += (
                ", original data in /mbari/M3/master/i2MAP/: "
                + matches.group(1)
                .replace("Original log files copied from ", "")
                .replace("/Volumes/M3/master/i2MAP/", "")
            )
        try:
            # Parse from seabird25p_depth comment: "using SensorOffset(x=1.003, y=0.0001)"
            self.metadata["comment"] = self.metadata.get("comment", "")
            if self.metadata["comment"]:
                self.metadata["comment"] = ". "
            self.metadata["comment"] += (
                f"Variable depth pitch corrected using"
                f" {self.ds['seabird25p_depth'].attrs['comment'].split('using ')[1]}"
            )
        except KeyError:
            self.logger.warning(
                "No entry for for mission %s comment in dorado_info.py",
                self.mission,
            )

        return self.metadata

    def lrauv_global_metadata(self) -> dict:
        """Use instance variables to return a dictionary of
        metadata specific for LRAUV data that are written.
        Calls _build_global_metadata() first to populate common metadata.
        """
        # Skip dynamic metadata during testing to ensure reproducible results
        if "pytest" in sys.modules:
            return {}

        # First populate common metadata (git commit, host, geospatial bounds, etc.)
        self._build_global_metadata()

        # Then add LRAUV-specific metadata
        # Preserve title and summary from align.nc if available
        if "title" in self.ds.attrs:
            self.metadata["title"] = self.ds.attrs["title"].replace(
                "Combined and aligned LRAUV", "Combined, Aligned, and Resampled LRAUV"
            )
        else:
            self.metadata["title"] = (
                f"Resampled LRAUV data from {self.log_file} at {self.freq} intervals"
            )

        if "summary" in self.ds.attrs:
            self.metadata["summary"] = self.ds.attrs["summary"]
        # Add resampling information and processing log file link to the summary
        self.metadata["summary"] += (
            f" Data resampled to {self.freq} intervals following {self.mf_width} "
            f"point median filter."
        )
        self.metadata["summary"] += (
            f". Processing log file: {BASE_LRAUV_WEB}/"
            f"{self.log_file.replace('.nc4', '_processing.log')}"
        )

        # Preserve comment from align.nc if available, otherwise use default
        if "comment" in self.ds.attrs:
            self.metadata["comment"] = self.ds.attrs["comment"]

        return self.metadata

    def instruments_variables(self, nc_file: str) -> dict:
        """
        Return a dictionary of all the variables in the mission netCDF file,
        keyed by instrument name
        """
        self.logger.info("Reading variables from %s mission netCDF file", nc_file)
        last_instr = None
        instr_vars = defaultdict(list)
        for variable in self.ds:
            instr, *_ = variable.split("_")
            if instr == "navigation":
                freq = "0.1S"  # noqa: F841
            elif instr in {"gps", "depth"}:
                continue
            if instr != last_instr:
                instr_vars[instr].append(variable)
        return instr_vars

    def resample_coordinates(self, instr: str, mf_width: int, freq: str) -> None:
        self.logger.info(
            "Resampling coordinates depth, latitude and longitude with"
            " frequency %s following %d point median filter ",
            freq,
            mf_width,
        )
        # Original
        try:
            self.df_o[f"{instr}_depth"] = self.ds[f"{instr}_depth"].to_pandas()
        except KeyError:
            self.logger.warning(
                "Variable %s_depth not found in %s align.nc file",
                instr,
                self.mission,
            )
            self.logger.info(
                "Cannot continue without a pitch corrected depth coordinate",
            )
            if self.log_file:
                msg = f"A CTD depth was not found in {self.ds.encoding['source']}"
            else:
                msg = f"{instr}_depth not found in {self.auv_name}_{self.mission}_align.nc"
            raise InvalidAlignFile(msg) from None
        try:
            self.df_o[f"{instr}_latitude"] = self.ds[f"{instr}_latitude"].to_pandas()
            self.df_o[f"{instr}_longitude"] = self.ds[f"{instr}_longitude"].to_pandas()
        except KeyError:
            msg = (
                f"Variable {instr}_latitude or {instr}_longitude not found in "
                f"{self.mission} align.nc file"
            )
            self.logger.warning(msg)
            raise InvalidAlignFile(msg) from None
        # Median Filtered - back & forward filling nan values at ends
        self.df_o[f"{instr}_depth_mf"] = (
            self.ds[f"{instr}_depth"]
            .rolling(**{instr + "_time": mf_width}, center=True)
            .median()
            .to_pandas()
        )
        self.df_o[f"{instr}_depth_mf"] = self.df_o[f"{instr}_depth_mf"].bfill()
        self.df_o[f"{instr}_depth_mf"] = self.df_o[f"{instr}_depth_mf"].ffill()
        self.df_o[f"{instr}_latitude_mf"] = (
            self.ds[f"{instr}_latitude"]
            .rolling(**{instr + "_time": mf_width}, center=True)
            .median()
            .to_pandas()
        )
        self.df_o[f"{instr}_latitude_mf"] = self.df_o[f"{instr}_latitude_mf"].bfill()
        self.df_o[f"{instr}_latitude_mf"] = self.df_o[f"{instr}_latitude_mf"].ffill()
        self.df_o[f"{instr}_longitude_mf"] = (
            self.ds[f"{instr}_longitude"]
            .rolling(**{instr + "_time": mf_width}, center=True)
            .median()
            .to_pandas()
        )
        self.df_o[f"{instr}_longitude_mf"] = self.df_o[f"{instr}_longitude_mf"].bfill()
        self.df_o[f"{instr}_longitude_mf"] = self.df_o[f"{instr}_longitude_mf"].ffill()
        # Resample to center of freq https://stackoverflow.com/a/69945592/1281657
        aggregator = ".mean() aggregator"
        # This is the common depth for all the instruments - the instruments that
        # matter (ctds, hs2, biolume, lopc) are all in the nose of the vehicle
        # (at least in November 2020)
        # and we want to use the same pitch corrected depth for all of them.
        self.df_r["depth"] = (
            self.df_o[f"{instr}_depth_mf"]
            .shift(0.5, freq=freq.lower())
            .resample(freq.lower())
            .mean()
        )
        self.df_r["latitude"] = (
            self.df_o[f"{instr}_latitude_mf"]
            .shift(0.5, freq=freq.lower())
            .resample(freq.lower())
            .mean()
        )
        self.df_r["longitude"] = (
            self.df_o[f"{instr}_longitude_mf"]
            .shift(0.5, freq=freq.lower())
            .resample(freq.lower())
            .mean()
        )
        return aggregator

    def save_coordinates(
        self,
        instr: str,
        mf_width: int,
        freq: str,
        aggregator: str,
    ) -> None:
        in_fn = self.ds.encoding["source"].split("/")[-1]
        self.df_r["depth"].index.rename("time", inplace=True)  # noqa: PD002
        self.resampled_nc["depth"] = self.df_r["depth"].to_xarray()
        self.df_r["latitude"].index.rename("time", inplace=True)  # noqa: PD002
        self.resampled_nc["latitude"] = self.df_r["latitude"].to_xarray()
        self.df_r["longitude"].index.rename("time", inplace=True)  # noqa: PD002
        self.resampled_nc["longitude"] = self.df_r["longitude"].to_xarray()
        self.resampled_nc["depth"].attrs = self.ds[f"{instr}_depth"].attrs
        self.resampled_nc["depth"].attrs["standard_name"] = "depth"
        self.resampled_nc["depth"].attrs["comment"] += (
            f". {self.ds[f'{instr}_depth'].attrs['comment']}"
            f" mean sampled at {self.freq} intervals following"
            f" {self.mf_width} point median filter."
        )
        self.resampled_nc["latitude"].attrs = self.ds[f"{instr}_latitude"].attrs
        self.resampled_nc["latitude"].attrs["comment"] += (
            f" Variable {instr}_latitude from {in_fn}"
            f" median filtered with {mf_width} samples"
            f" and resampled with {aggregator} to {freq} intervals."
        )
        self.resampled_nc["longitude"].attrs = self.ds[f"{instr}_longitude"].attrs
        self.resampled_nc["longitude"].attrs["comment"] += (
            f" Variable {instr}_longitude from {in_fn}"
            f" median filtered with {mf_width} samples"
            f" and resampled with {aggregator} to {freq} intervals."
        )

    def select_nighttime_bl_raw(
        self,
        stride: int = 3000,
    ) -> tuple[pd.Series, list[datetime], list[datetime]]:
        """
        Select nighttime biolume_raw data for multiple nights in a mission.
        Default stride of 3000 gives 10-minute resolution from 5 Hz navigation data.

        Returns:
            nighttime_bl_raw: A pandas Series containing nighttime biolume_raw data.
            sunsets: A list of sunset times for each night.
            sunrises: A list of sunrise times for each night.
        """
        lat_var, lon_var = self._find_lat_lon_variables()
        lat = float(self.ds[lat_var].median())
        lon = float(self.ds[lon_var].median())
        self.logger.debug("Getting sun altitudes for nighttime selection")
        sun_alts = []
        # Get the time coordinate for the latitude variable
        time_coord = self.ds[lat_var].dims[0]
        for ts in self.ds[time_coord].to_numpy()[::stride]:
            # About 10-minute resolution from 5 Hz navigation data
            sun_alts.append(  # noqa: PERF401
                get_altitude(
                    lat,
                    lon,
                    datetime.fromtimestamp(ts.astype(int) / 1.0e9, tz=UTC),
                ),
            )

        # Find sunset and sunrise - where sun altitude changes sign
        sign_changes = np.where(np.diff(np.sign(sun_alts)))[0]
        ss_sr_times = self.ds[time_coord].isel({time_coord: sign_changes * stride}).to_numpy()
        self.logger.debug("Sunset and sunrise times: %s", ss_sr_times)

        sunsets = []
        sunrises = []
        nighttime_bl_raw = pd.Series(dtype="float64")

        # Iterate over sunset and sunrise pairs
        for i in range(0, len(ss_sr_times) - 1, 2):
            sunset = ss_sr_times[i] + pd.to_timedelta(1, "h")  # 1 hour past sunset
            sunrise = ss_sr_times[i + 1] - pd.to_timedelta(1, "h")  # 1 hour before sunrise
            sunsets.append(sunset)
            sunrises.append(sunrise)

            self.logger.info(
                "Extracting biolume_raw data between sunset %s and sunrise %s",
                sunset,
                sunrise,
            )
            nighttime_data = (
                self.ds["biolume_raw"]
                .where(
                    (self.ds["biolume_time60hz"] > sunset)
                    & (self.ds["biolume_time60hz"] < sunrise),
                )
                .to_pandas()
                .dropna()
            )
            # This complication is needed because concat will not like an empty DataFrame
            nighttime_bl_raw = (
                nighttime_bl_raw.copy()
                if nighttime_data.empty
                else nighttime_data.copy()
                if nighttime_bl_raw.empty
                else pd.concat([nighttime_bl_raw, nighttime_data])  # if both DataFrames non empty
            )

        if not sunsets or not sunrises:
            self.logger.info("No sunset or sunrise found during this mission.")
        return nighttime_bl_raw, sunsets, sunrises

    def _find_lat_lon_variables(self) -> tuple[str, str]:
        """Find latitude and longitude variables in the dataset.

        Searches for variables ending in _latitude and _longitude.
        Prefers navigation_, nudged_, or onboard_ prefixes in that order.

        Returns:
            tuple: (lat_var_name, lon_var_name)

        Raises:
            KeyError: If no latitude/longitude variables are found
        """
        lat_vars = [v for v in self.ds.variables if v.endswith("_latitude")]
        lon_vars = [v for v in self.ds.variables if v.endswith("_longitude")]

        if not lat_vars or not lon_vars:
            msg = (
                f"No latitude/longitude variables found. "
                f"Available variables: {list(self.ds.variables.keys())}"
            )
            raise KeyError(msg)

        # Prefer navigation_, then nudged_, then onboard_, then any other
        for prefix in ["navigation_", "nudged_", "onboard_"]:
            for lat_var in lat_vars:
                if lat_var.startswith(prefix):
                    lon_var = prefix + "longitude"
                    if lon_var in lon_vars:
                        self.logger.debug("Using %s and %s for coordinates", lat_var, lon_var)
                        return lat_var, lon_var

        # Fall back to first available pair
        lat_var = lat_vars[0]
        lon_var = lon_vars[0]
        self.logger.info("Using first available coordinates: %s and %s", lat_var, lon_var)
        return lat_var, lon_var

    def add_profile(self, depth_threshold: float) -> None:
        # Find depth vertices value using scipy's find_peaks algorithm
        options = {"prominence": 10, "width": 30}
        peaks_pos, _ = signal.find_peaks(self.resampled_nc["depth"], **options)
        peaks_neg, _ = signal.find_peaks(-self.resampled_nc["depth"], **options)
        # Need to add the first and last time values to the list of peaks
        peaks = np.concatenate(
            (peaks_pos, peaks_neg, [0], [len(self.resampled_nc["depth"]) - 1]),
        )
        peaks.sort(kind="mergesort")
        s_peaks = self.resampled_nc["depth"][peaks].to_pandas()

        # Assign a profile number to each time value
        profiles = []
        count = 1
        k = 0
        for tv in self.resampled_nc["time"].to_numpy():
            if tv > s_peaks.index[k + 1]:
                # Encountered a new simple_depth point
                k += 1
                if abs(s_peaks.iloc[k + 1] - s_peaks.iloc[k]) > depth_threshold:
                    # Completed downcast or upcast
                    count += 1
            profiles.append(count)
            if k > len(s_peaks) - 2:
                break

        self.resampled_nc["profile_number"] = xr.DataArray(
            profiles,
            dims="time",
            coords=[self.resampled_nc["time"].to_numpy()],
            name="profile_number",
        )
        self.resampled_nc["profile_number"].attrs["coordinates"] = "time depth latitude longitude"
        self.resampled_nc["profile_number"].attrs = {
            "long_name": "Profile number",
        }

    def set_proxy_parameters(self, mission_start: datetime) -> tuple[float, float]:
        # The parameters used to calculate bioluminescence proxies should be changed depending
        # on the time period considered, as described below.

        # Changes in HS2/biolum configurations:

        # beginning of UBAT surveys
        # [period1]
        # 2007.344.00 first survey with bathyphotometer mounted in the nose instead of side-mounted
        # [period2]
        # Jan 1st 2009 changing HS2 sensor (new bbp channels), first mission is 2009.055.05 I believe  # noqa: E501
        # [period3]
        # new UBAT installed in 2010 (first mission 2010.277.01)
        # [period4]
        # UBAT serviced early 2025, may require a new parameterization in the future but we can keep
        # period4 for now until present.

        # Parameters should be:

        # period1: calibration=0.0016691, ratioAdinos=5.0119E13
        # period2: calibration=0.0016691, ratioAdinos=2.5119E13
        # period3: calibration=0.0047101, ratioAdinos=1.0000E14
        # period4: calibration=0.0049859, ratioAdinos=3.8019E13 to test (previously based on
        # 2010-2020: calibration=0.0047118, ratioAdinos=3.9811E13)

        # Set start datetime from year and year-day
        period1_start = datetime(2003, 1, 1) + timedelta(days=225)  # noqa: DTZ001
        period2_start = datetime(2007, 1, 1) + timedelta(days=343)  # noqa: DTZ001
        period3_start = datetime(2009, 1, 1) + timedelta(days=54)  # noqa: DTZ001
        period4_start = datetime(2010, 1, 1) + timedelta(days=276)  # noqa: DTZ001
        if mission_start >= period1_start and mission_start < period2_start:
            # period1: 2003.225 to 2007.343
            self.logger.info("Setting biolume proxy parameters for period1")
            proxy_cal_factor = 0.0016691
            proxy_ratio_adinos = 5.0119e13
        elif mission_start >= period2_start and mission_start < period3_start:
            # period2: 2007.343 to 2009.054
            self.logger.info("Setting biolume proxy parameters for period2")
            proxy_cal_factor = 0.0016691
            proxy_ratio_adinos = 2.5119e13
        elif mission_start >= period3_start and mission_start < period4_start:
            # period3: 2009.054 to 2010.275
            self.logger.info("Setting biolume proxy parameters for period3")
            proxy_cal_factor = 0.0047101
            proxy_ratio_adinos = 1.0000e14
        elif mission_start >= period4_start:
            # period4: 2010.275 to present
            self.logger.info("Setting biolume proxy parameters for period4")
            proxy_cal_factor = 0.0049859
            proxy_ratio_adinos = 3.8019e13
        else:
            # Should not happen, but if it does, use the values used in Notebook 5.2
            self.logger.warning(
                "Mission start %s is before period1_start %s - Setting original parameters",
                mission_start,
                period1_start,
            )
            proxy_cal_factor = 0.0047118
            proxy_ratio_adinos = 3.9811e13
        return proxy_cal_factor, proxy_ratio_adinos

    def add_biolume_proxies(  # noqa: PLR0913, PLR0915
        self,
        freq,
        window_size_secs: int = 5,
        envelope_mini: float = 1.5e10,
        flash_threshold: float = FLASH_THRESHOLD,
        proxy_ratio_adinos: float = 3.9811e13,  # 4-Oct-2010 to 2-Dec-2020 value
        proxy_cal_factor: float = 0.00470,  # Same as used in 5.2-mpm-bg_biolume-PiO-paper.ipynb
    ) -> tuple[pd.Series, list[datetime], list[datetime]]:
        # Add variables via the calculations according to Appendix B in
        # "Using fluorescence and bioluminescence sensors to characterize
        # auto- and heterotrophic plankton communities" by Messie et al."
        # https://www.sciencedirect.com/science/article/pii/S0079661118300478
        # Translation to Python demonstrated in notebooks/5.2-mpm-bg_biolume-PiO-paper.ipynb

        self.logger.info("Adding biolume proxy variables computed from biolume_raw")
        sample_rate = 60  # Assume all biolume_raw data is sampled at 60 Hz
        window_size = window_size_secs * sample_rate

        # s_biolume_raw includes daytime data - see below for nighttime_bl_raw
        s_biolume_raw = self.ds["biolume_raw"].to_pandas().dropna()

        # Compute background biolumenesence envelope
        self.logger.debug("Applying rolling min filter")
        min_bg_unsmoothed = s_biolume_raw.rolling(
            window_size,
            min_periods=0,
            center=True,
        ).min()
        min_bg = (
            min_bg_unsmoothed.rolling(window_size, min_periods=0, center=True).mean().to_numpy()
        )

        self.logger.debug("Applying rolling median filter")
        med_bg_unsmoothed = s_biolume_raw.rolling(
            window_size,
            min_periods=0,
            center=True,
        ).median()
        s_med_bg = med_bg_unsmoothed.rolling(
            window_size,
            min_periods=0,
            center=True,
        ).mean()
        med_bg = s_med_bg.to_numpy()
        max_bg = med_bg * 2.0 - min_bg
        # envelope_mini: minimum value for the envelope (max_bgrd - med_bgrd)
        # to avoid very dim flashes when the background is low
        # (default 1.5E10 ph/s)
        max_bg[max_bg - med_bg < envelope_mini] = (
            med_bg[max_bg - med_bg < envelope_mini] + envelope_mini
        )

        # Find the high and low peaks
        self.logger.debug("Finding peaks")
        peaks, _ = signal.find_peaks(s_biolume_raw, height=max_bg)
        s_peaks = pd.Series(s_biolume_raw.iloc[peaks], index=s_biolume_raw.index[peaks])
        s_med_bg_peaks = pd.Series(s_med_bg.iloc[peaks], index=s_biolume_raw.index[peaks])
        if self.flash_threshold:
            flash_threshold = self.flash_threshold
        flash_threshold_note = f"Computed with flash_threshold = {flash_threshold:.0e}"
        self.logger.info("Using flash_threshold = %.4e", flash_threshold)
        nbflash_high = s_peaks[s_peaks > (s_med_bg_peaks + flash_threshold)]
        nbflash_low = s_peaks[s_peaks <= (s_med_bg_peaks + flash_threshold)]

        # Construct full time series of flashes with NaNs for non-flash values
        s_nbflash_high = pd.Series(np.nan, index=s_biolume_raw.index)
        s_nbflash_high.loc[nbflash_high.index] = nbflash_high
        s_nbflash_low = pd.Series(np.nan, index=s_biolume_raw.index)
        s_nbflash_low.loc[nbflash_low.index] = nbflash_low

        # Count the number of flashes per second - use 15 second window stepping every second
        flash_count_seconds = 15
        flash_window = flash_count_seconds * sample_rate
        self.logger.debug("Counting flashes using %d second window", flash_count_seconds)
        nbflash_high_counts = (
            s_nbflash_high.rolling(flash_window, step=1, min_periods=0, center=True)
            .count()
            .resample(freq.lower())
            .mean()
            / flash_count_seconds
        )
        nbflash_low_counts = (
            s_nbflash_low.rolling(flash_window, step=1, min_periods=0, center=True)
            .count()
            .resample(freq.lower())
            .mean()
            / flash_count_seconds
        )

        flow = self.ds[["biolume_flow"]]["biolume_flow"].to_pandas().resample("1s").mean().ffill()

        # Flow sensor is not always on, so fill in 0.0 values with 350 ml/s
        zero_note = ""
        num_zero_flow = len(np.where(flow == 0)[0])
        if num_zero_flow > 0:
            zero_note = (
                f"Zero flow values found: {num_zero_flow} of {len(flow)} - replaced with 350 ml/s"
            )
            self.logger.info(zero_note)
            flow = flow.replace(0.0, 350.0)

        # Compute flashes per liter - pandas.Series.divide() will match indexes
        # Units: flashes per liter = (flashes per second / mL/s) * 1000 mL/L
        self.logger.info("Computing flashes per liter: nbflash_high, nbflash_low")
        self.df_r["biolume_nbflash_high"] = nbflash_high_counts.divide(flow) * 1000
        self.df_r["biolume_nbflash_high"].attrs["long_name"] = (
            "High intensity flashes (copepods proxy)"
        )
        self.df_r["biolume_nbflash_high"].attrs["units"] = "flashes/liter"
        self.df_r["biolume_nbflash_high"].attrs["comment"] = f"{zero_note} - {flash_threshold_note}"

        self.df_r["biolume_nbflash_low"] = nbflash_low_counts.divide(flow) * 1000
        self.df_r["biolume_nbflash_low"].attrs["long_name"] = (
            "Low intensity flashes (Larvacean proxy)"
        )
        self.df_r["biolume_nbflash_low"].attrs["units"] = "flashes/liter"
        self.df_r["biolume_nbflash_low"].attrs["comment"] = f"{zero_note} - {flash_threshold_note}"

        # Flash intensity in ph/s - proxy for small jellies - for entire mission, not just nightime
        all_raw = self.ds[["biolume_raw"]]["biolume_raw"].to_pandas()
        med_bg_60 = pd.Series(
            np.interp(all_raw.index, s_med_bg.index, med_bg),
            index=all_raw.index,
        )
        intflash = (
            (all_raw - med_bg_60)
            .rolling(flash_window, min_periods=0, center=True)
            .max()
            .resample("1s")
            .mean()
        )
        self.logger.info(
            "Saving flash intensity: biolume_intflash - the upper bound of the background envelope",
        )
        self.df_r["biolume_intflash"] = intflash
        self.df_r["biolume_intflash"].attrs["long_name"] = "Flashes intensity (small jellies proxy)"
        self.df_r["biolume_intflash"].attrs["units"] = "photons/s"
        self.df_r["biolume_intflash"].attrs["comment"] = (
            f" intensity of flashes from {sample_rate} Hz biolume_raw variable in {freq} intervals."
        )

        # Make min_bg a 1S pd.Series so that we can divide by flow, matching indexes
        s_min_bg = min_bg_unsmoothed.rolling(
            window_size,
            min_periods=0,
            center=True,
        ).mean()
        bg_biolume = pd.Series(s_min_bg, index=s_biolume_raw.index).resample("1s").mean()
        self.logger.info("Saving Background bioluminescence (dinoflagellates proxy)")
        self.df_r["biolume_bg_biolume"] = bg_biolume.divide(flow) * 1000
        self.df_r["biolume_bg_biolume"].attrs["long_name"] = (
            "Background bioluminescence (dinoflagellates proxy)"
        )
        self.df_r["biolume_bg_biolume"].attrs["units"] = "photons/liter"
        self.df_r["biolume_bg_biolume"].attrs["comment"] = zero_note

        fluo = None
        nighttime_bl_raw, sunsets, sunrises = self.select_nighttime_bl_raw()
        if nighttime_bl_raw.empty:
            self.logger.info(
                "No nighttime_bl_raw data to compute adinos, diatoms, hdinos proxies",
            )
        else:
            # (2) Phytoplankton proxies - use median filtered hs2_fl700 1S data
            if "hs2_fl700" not in self.ds:
                self.logger.info(
                    "No hs2_fl700 data. Not computing adinos, diatoms, and hdinos",
                )
                return fluo, sunsets, sunrises
            fluo = (
                self.resampled_nc["hs2_fl700"]
                .where(
                    (self.resampled_nc["time"] > min(sunsets))
                    & (self.resampled_nc["time"] < max(sunrises)),
                )
                .to_pandas()
                .resample(freq.lower())
                .mean()
            )
            # Set negative values from hs2_fl700 to NaN
            fluo[fluo < 0] = np.nan
            self.logger.info("Using proxy_ratio_adinos = %.4e", proxy_ratio_adinos)
            self.logger.info("Using proxy_cal_factor = %.6f", proxy_cal_factor)

            nighttime_bg_biolume = (
                pd.Series(s_min_bg, index=nighttime_bl_raw.index).resample("1s").mean()
            )
            nighttime_bg_biolume_perliter = nighttime_bg_biolume.divide(flow) * 1000
            pseudo_fluorescence = nighttime_bg_biolume_perliter / proxy_ratio_adinos
            self.df_r["biolume_proxy_adinos"] = (
                np.minimum(fluo, pseudo_fluorescence) / proxy_cal_factor
            )
            self.df_r["biolume_proxy_adinos"].attrs["comment"] = (
                f"Autotrophic dinoflagellate proxy using proxy_ratio_adinos"
                f" = {proxy_ratio_adinos:.4e} and proxy_cal_factor = {proxy_cal_factor:.6f}"
            )
            self.df_r["biolume_proxy_hdinos"] = (
                pseudo_fluorescence - np.minimum(fluo, pseudo_fluorescence)
            ) / proxy_cal_factor
            self.df_r["biolume_proxy_hdinos"].attrs["comment"] = (
                f"Heterotrophic dinoflagellate proxy using proxy_ratio_adinos"
                f" = {proxy_ratio_adinos:.4e} and proxy_cal_factor = {proxy_cal_factor:.6f}"
            )
            biolume_proxy_diatoms = (fluo - pseudo_fluorescence) / proxy_cal_factor
            biolume_proxy_diatoms[biolume_proxy_diatoms < 0] = 0
            self.df_r["biolume_proxy_diatoms"] = biolume_proxy_diatoms
            self.df_r["biolume_proxy_diatoms"].attrs["comment"] = (
                f"Diatom proxy using proxy_ratio_adinos"
                f" = {proxy_ratio_adinos:.4e} and proxy_cal_factor = {proxy_cal_factor:.6f}"
            )

        return fluo, sunsets, sunrises

    def add_wetlabsubat_proxies(  # noqa: PLR0913, PLR0915, C901, PLR0912
        self,
        freq,
        window_size_secs: int = 5,
        envelope_mini: float = 1.5e10,
        flash_threshold: float = FLASH_THRESHOLD,
        proxy_ratio_adinos: float = 3.9811e13,  # Default value for LRAUV
        proxy_cal_factor: float = 0.00470,  # Default value for LRAUV
    ) -> tuple[pd.Series, list[datetime], list[datetime]]:
        """Add biolume proxy variables computed from wetlabsubat_digitized_raw_ad_counts.

        This is parallel to add_biolume_proxies() but for LRAUV wetlabsubat data.
        Computations follow Appendix B in Messie et al. 2019.
        https://www.sciencedirect.com/science/article/pii/S0079661118300478
        """
        self.logger.info(
            "Adding wetlabsubat proxy variables computed from wetlabsubat_digitized_raw_ad_counts"
        )
        sample_rate = 60  # Assume all digitized_raw_ad_counts data is sampled at 60 Hz
        window_size = window_size_secs * sample_rate

        # s_ubat_raw includes daytime data - see below for nighttime data
        s_ubat_raw = self.ds["wetlabsubat_digitized_raw_ad_counts"].to_pandas().dropna()

        # Compute background biolumenesence envelope
        self.logger.debug("Applying rolling min filter")
        min_bg_unsmoothed = s_ubat_raw.rolling(
            window_size,
            min_periods=0,
            center=True,
        ).min()
        min_bg = (
            min_bg_unsmoothed.rolling(window_size, min_periods=0, center=True).mean().to_numpy()
        )

        self.logger.debug("Applying rolling median filter")
        med_bg_unsmoothed = s_ubat_raw.rolling(
            window_size,
            min_periods=0,
            center=True,
        ).median()
        s_med_bg = med_bg_unsmoothed.rolling(
            window_size,
            min_periods=0,
            center=True,
        ).mean()
        med_bg = s_med_bg.to_numpy()
        max_bg = med_bg * 2.0 - min_bg
        # envelope_mini: minimum value for the envelope (max_bgrd - med_bgrd)
        # to avoid very dim flashes when the background is low
        max_bg[max_bg - med_bg < envelope_mini] = (
            med_bg[max_bg - med_bg < envelope_mini] + envelope_mini
        )

        # Find the high and low peaks
        self.logger.debug("Finding peaks")
        peaks, _ = signal.find_peaks(s_ubat_raw, height=max_bg)
        s_peaks = pd.Series(s_ubat_raw.iloc[peaks], index=s_ubat_raw.index[peaks])
        s_med_bg_peaks = pd.Series(s_med_bg.iloc[peaks], index=s_ubat_raw.index[peaks])
        if self.flash_threshold:
            flash_threshold = self.flash_threshold
        flash_threshold_note = f"Computed with flash_threshold = {flash_threshold:.0e}"
        self.logger.info("Using flash_threshold = %.4e", flash_threshold)
        nbflash_high = s_peaks[s_peaks > (s_med_bg_peaks + flash_threshold)]
        nbflash_low = s_peaks[s_peaks <= (s_med_bg_peaks + flash_threshold)]

        # Construct full time series of flashes with NaNs for non-flash values
        s_nbflash_high = pd.Series(np.nan, index=s_ubat_raw.index)
        s_nbflash_high.loc[nbflash_high.index] = nbflash_high
        s_nbflash_low = pd.Series(np.nan, index=s_ubat_raw.index)
        s_nbflash_low.loc[nbflash_low.index] = nbflash_low

        # Count the number of flashes per second - use 15 second window stepping every second
        flash_count_seconds = 15
        flash_window = flash_count_seconds * sample_rate
        self.logger.debug("Counting flashes using %d second window", flash_count_seconds)
        nbflash_high_counts = (
            s_nbflash_high.rolling(flash_window, step=1, min_periods=0, center=True)
            .count()
            .resample(freq.lower())
            .mean()
            / flash_count_seconds
        )
        nbflash_low_counts = (
            s_nbflash_low.rolling(flash_window, step=1, min_periods=0, center=True)
            .count()
            .resample(freq.lower())
            .mean()
            / flash_count_seconds
        )

        # Get flow data - try both flow_rate and flow variable names
        flow = None
        if "wetlabsubat_flow_rate" in self.ds:
            flow = (
                self.ds[["wetlabsubat_flow_rate"]]["wetlabsubat_flow_rate"]
                .to_pandas()
                .resample("1s")
                .mean()
                .ffill()
            )
            self.logger.info("Using wetlabsubat_flow_rate for flow calculations")
        elif "wetlabsubat_flow" in self.ds:
            flow = (
                self.ds[["wetlabsubat_flow"]]["wetlabsubat_flow"]
                .to_pandas()
                .resample("1s")
                .mean()
                .ffill()
            )
            self.logger.info("Using wetlabsubat_flow for flow calculations")

        # Flow sensor is not always on or may not be present, fill in 0.0 values with 350 ml/s
        zero_note = ""
        if flow is None:
            self.logger.info("No flow data found - using constant 350 ml/s")
            # Create flow series with same index as resampled data
            flow = pd.Series(350.0, index=nbflash_high_counts.index)
            zero_note = "No flow data available - used constant 350 ml/s"
        else:
            num_zero_flow = len(np.where(flow == 0)[0])
            if num_zero_flow > 0:
                zero_note = (
                    f"Zero flow values found: {num_zero_flow} of {len(flow)} "
                    f"- replaced with 350 ml/s"
                )
                self.logger.info(zero_note)
                flow = flow.replace(0.0, 350.0)

        # Compute flashes per liter - pandas.Series.divide() will match indexes
        # Units: flashes per liter = (flashes per second / mL/s) * 1000 mL/L
        self.logger.info(
            "Computing flashes per liter: wetlabsubat_nbflash_high, wetlabsubat_nbflash_low"
        )
        self.df_r["wetlabsubat_nbflash_high"] = nbflash_high_counts.divide(flow) * 1000
        self.df_r["wetlabsubat_nbflash_high"].attrs["long_name"] = (
            "High intensity flashes (copepods proxy)"
        )
        self.df_r["wetlabsubat_nbflash_high"].attrs["units"] = "flashes/liter"
        self.df_r["wetlabsubat_nbflash_high"].attrs["comment"] = (
            f"{zero_note} - {flash_threshold_note}"
        )

        self.df_r["wetlabsubat_nbflash_low"] = nbflash_low_counts.divide(flow) * 1000
        self.df_r["wetlabsubat_nbflash_low"].attrs["long_name"] = (
            "Low intensity flashes (Larvacean proxy)"
        )
        self.df_r["wetlabsubat_nbflash_low"].attrs["units"] = "flashes/liter"
        self.df_r["wetlabsubat_nbflash_low"].attrs["comment"] = (
            f"{zero_note} - {flash_threshold_note}"
        )

        # Flash intensity - proxy for small jellies - for entire mission, not just nightime
        all_raw = self.ds[["wetlabsubat_digitized_raw_ad_counts"]][
            "wetlabsubat_digitized_raw_ad_counts"
        ].to_pandas()
        med_bg_60 = pd.Series(
            np.interp(all_raw.index, s_med_bg.index, med_bg),
            index=all_raw.index,
        )
        intflash = (
            (all_raw - med_bg_60)
            .rolling(flash_window, min_periods=0, center=True)
            .max()
            .resample("1s")
            .mean()
        )
        self.logger.info(
            "Saving flash intensity: wetlabsubat_intflash - "
            "the upper bound of the background envelope"
        )
        self.df_r["wetlabsubat_intflash"] = intflash
        self.df_r["wetlabsubat_intflash"].attrs["long_name"] = (
            "Flashes intensity (small jellies proxy)"
        )
        self.df_r["wetlabsubat_intflash"].attrs["units"] = "counts"
        self.df_r["wetlabsubat_intflash"].attrs["comment"] = (
            f"intensity of flashes from {sample_rate} Hz "
            f"wetlabsubat_digitized_raw_ad_counts variable in {freq} intervals."
        )

        # Make min_bg a 1S pd.Series so that we can divide by flow, matching indexes
        s_min_bg = min_bg_unsmoothed.rolling(
            window_size,
            min_periods=0,
            center=True,
        ).mean()
        bg_biolume = pd.Series(s_min_bg, index=s_ubat_raw.index).resample("1s").mean()
        self.logger.info("Saving Background bioluminescence (dinoflagellates proxy)")
        self.df_r["wetlabsubat_bg_biolume"] = bg_biolume.divide(flow) * 1000
        self.df_r["wetlabsubat_bg_biolume"].attrs["long_name"] = (
            "Background bioluminescence (dinoflagellates proxy)"
        )
        self.df_r["wetlabsubat_bg_biolume"].attrs["units"] = "counts/liter"
        self.df_r["wetlabsubat_bg_biolume"].attrs["comment"] = zero_note

        fluo = None
        nighttime_ubat_raw, sunsets, sunrises = self.select_nighttime_ubat_raw()
        if nighttime_ubat_raw.empty:
            self.logger.info(
                "No nighttime wetlabsubat data to compute adinos, diatoms, hdinos proxies",
            )
        else:
            # (2) Phytoplankton proxies - look for wetlabsbb2fl fluorescence/chlorophyll data
            fluo_var = None
            for var in self.resampled_nc.variables:
                if "wetlabsbb2fl" in var.lower() and (
                    "fl" in var.lower() or "chlorophyll" in var.lower()
                ):
                    fluo_var = var
                    break

            if fluo_var is None:
                self.logger.info(
                    "No wetlabsbb2fl fluorescence data found. "
                    "Not computing adinos, diatoms, and hdinos"
                )
                return fluo, sunsets, sunrises

            self.logger.info("Using %s for phytoplankton proxy calculations", fluo_var)
            fluo = (
                self.resampled_nc[fluo_var]
                .where(
                    (self.resampled_nc["time"] > min(sunsets))
                    & (self.resampled_nc["time"] < max(sunrises)),
                )
                .to_pandas()
                .resample(freq.lower())
                .mean()
            )
            # Set negative values from fluorescence to NaN
            fluo[fluo < 0] = np.nan
            self.logger.info("Using proxy_ratio_adinos = %.4e", proxy_ratio_adinos)
            self.logger.info("Using proxy_cal_factor = %.6f", proxy_cal_factor)

            nighttime_bg_biolume = (
                pd.Series(s_min_bg, index=nighttime_ubat_raw.index).resample("1s").mean()
            )
            nighttime_bg_biolume_perliter = nighttime_bg_biolume.divide(flow) * 1000
            pseudo_fluorescence = nighttime_bg_biolume_perliter / proxy_ratio_adinos
            self.df_r["wetlabsubat_proxy_adinos"] = (
                np.minimum(fluo, pseudo_fluorescence) / proxy_cal_factor
            )
            self.df_r["wetlabsubat_proxy_adinos"].attrs["comment"] = (
                f"Autotrophic dinoflagellate proxy using proxy_ratio_adinos"
                f" = {proxy_ratio_adinos:.4e} and proxy_cal_factor = {proxy_cal_factor:.6f}"
            )
            self.df_r["wetlabsubat_proxy_hdinos"] = (
                pseudo_fluorescence - np.minimum(fluo, pseudo_fluorescence)
            ) / proxy_cal_factor
            self.df_r["wetlabsubat_proxy_hdinos"].attrs["comment"] = (
                f"Heterotrophic dinoflagellate proxy using proxy_ratio_adinos"
                f" = {proxy_ratio_adinos:.4e} and proxy_cal_factor = {proxy_cal_factor:.6f}"
            )
            wetlabsubat_proxy_diatoms = (fluo - pseudo_fluorescence) / proxy_cal_factor
            wetlabsubat_proxy_diatoms[wetlabsubat_proxy_diatoms < 0] = 0
            self.df_r["wetlabsubat_proxy_diatoms"] = wetlabsubat_proxy_diatoms
            self.df_r["wetlabsubat_proxy_diatoms"].attrs["comment"] = (
                f"Diatom proxy using proxy_ratio_adinos"
                f" = {proxy_ratio_adinos:.4e} and proxy_cal_factor = {proxy_cal_factor:.6f}"
            )

        return fluo, sunsets, sunrises

    def select_nighttime_ubat_raw(
        self,
        stride: int = 3000,
    ) -> tuple[pd.Series, list[datetime], list[datetime]]:
        """
        Select nighttime wetlabsubat_digitized_raw_ad_counts data for multiple nights in a mission.
        Parallel to select_nighttime_bl_raw() but for LRAUV wetlabsubat data.
        Default stride of 3000 gives 10-minute resolution from 5 Hz navigation data.

        Returns:
            nighttime_ubat_raw: A pandas Series containing nighttime ubat data.
            sunsets: A list of sunset times for each night.
            sunrises: A list of sunrise times for each night.
        """
        lat_var, lon_var = self._find_lat_lon_variables()
        lat = float(self.ds[lat_var].median())
        lon = float(self.ds[lon_var].median())
        self.logger.debug("Getting sun altitudes for nighttime selection")
        sun_alts = []
        # Get the time coordinate for the latitude variable
        time_coord = self.ds[lat_var].dims[0]
        for ts in self.ds[time_coord].to_numpy()[::stride]:
            # About 10-minute resolution from 5 Hz navigation data
            sun_alts.append(  # noqa: PERF401
                get_altitude(
                    lat,
                    lon,
                    datetime.fromtimestamp(ts.astype(int) / 1.0e9, tz=UTC),
                ),
            )

        # Find sunset and sunrise - where sun altitude changes sign
        sign_changes = np.where(np.diff(np.sign(sun_alts)))[0]
        ss_sr_times = self.ds[time_coord].isel({time_coord: sign_changes * stride}).to_numpy()
        self.logger.debug("Sunset and sunrise times: %s", ss_sr_times)

        sunsets = []
        sunrises = []
        nighttime_ubat_raw = pd.Series(dtype="float64")

        # Iterate over sunset and sunrise pairs
        for i in range(0, len(ss_sr_times) - 1, 2):
            sunset = ss_sr_times[i] + pd.to_timedelta(1, "h")  # 1 hour past sunset
            sunrise = ss_sr_times[i + 1] - pd.to_timedelta(1, "h")  # 1 hour before sunrise
            sunsets.append(sunset)
            sunrises.append(sunrise)

            self.logger.info(
                "Extracting wetlabsubat_digitized_raw_ad_counts data "
                "between sunset %s and sunrise %s",
                sunset,
                sunrise,
            )
            nighttime_data = (
                self.ds["wetlabsubat_digitized_raw_ad_counts"]
                .where(
                    (self.ds["wetlabsubat_time_60hz"] > sunset)
                    & (self.ds["wetlabsubat_time_60hz"] < sunrise),
                )
                .to_pandas()
                .dropna()
            )
            # This complication is needed because concat will not like an empty DataFrame
            nighttime_ubat_raw = (
                nighttime_ubat_raw.copy()
                if nighttime_data.empty
                else nighttime_data.copy()
                if nighttime_ubat_raw.empty
                else pd.concat([nighttime_ubat_raw, nighttime_data])  # if both DataFrames non empty
            )

        if not sunsets or not sunrises:
            self.logger.info("No sunset or sunrise found during this mission.")
        return nighttime_ubat_raw, sunsets, sunrises

    def correct_biolume_proxies(  # noqa: C901, PLR0912, PLR0913, PLR0915
        self,
        biolume_fluo: pd.Series,  # from add_biolume_proxies
        biolume_sunsets: list[datetime],  # from add_biolume_proxies
        biolume_sunrises: list[datetime],  # from add_biolume_proxies
        depth_threshold: float,
        adinos_threshold: float = 0.1,
        correction_threshold: int = 3,
        fluo_bl_threshold: float = 0.4,
        corr_type: str = "pearson",  # "spearman" or "pearson"
        minutes_from_surface_threshold: int = 5,
    ) -> None:
        variables = [
            "biolume_proxy_diatoms",
            "biolume_proxy_adinos",
            "biolume_proxy_hdinos",
            "biolume_bg_biolume",
        ]
        try:
            df_p = self.df_r[variables].copy(deep=True)
        except KeyError:
            # We didn't add biolum proxies this round...
            return

        # Save the attrs for later as the correction process will drop them
        saved_attrs = defaultdict(list)
        for var in variables:
            saved_attrs[var] = self.df_r[var].attrs

        df_p["biolume_fluo"] = biolume_fluo
        df_p["fluoBL_corr"] = np.full_like(df_p.biolume_fluo, np.nan)

        depth_series = self.resampled_nc["depth"].to_series()
        # df_p["depth"] = depth_series.reindex(df_p.index, method="ffill")
        df_p = pd.merge_asof(
            df_p,
            depth_series.to_frame(),
            left_index=True,
            right_index=True,
            direction="nearest",
        )

        self.logger.info(
            "correct proxies: df_p depth count=%d nans=%d, resampled_nc depth count=%d nans=%d",
            df_p.depth.count(),
            df_p.depth.isna().sum(),
            self.resampled_nc.depth.count(),
            self.resampled_nc.depth.isnull().sum(),  # noqa: PD003
        )

        profile_series = self.resampled_nc["profile_number"].to_series()
        # df_p["profile_number"] = profile_series.reindex(df_p.index, method="ffill")
        df_p = pd.merge_asof(
            df_p,
            profile_series.to_frame(),
            left_index=True,
            right_index=True,
            direction="nearest",
        )

        self.logger.info(
            "correct proxies: df_p max profile=%d (nans=%d), resampled_nc max profile=%d (nans=%d)",
            df_p.profile_number.max(),
            df_p.profile_number.isna().sum(),
            self.resampled_nc.profile_number.max(),
            self.resampled_nc.profile_number.isnull().sum(),  # noqa: PD003
        )

        # new proxies are the "N" fields
        for new, old in zip(
            ["diatomsN", "adinosN", "hdinosN"],
            ["biolume_proxy_diatoms", "biolume_proxy_adinos", "biolume_proxy_hdinos"],
            strict=False,
        ):
            df_p[new] = np.full_like(df_p[old], np.nan)

        def _interval_contains_sunevent(
            start: pd.Timestamp, end: pd.Timestamp, events: pd.DatetimeIndex
        ) -> bool:
            mask = (events >= start) & (events <= end)
            return mask.any()

        biolume_sunsets = pd.DatetimeIndex(biolume_sunsets).sort_values()
        biolume_sunrises = pd.DatetimeIndex(biolume_sunrises).sort_values()
        profile_intervals = (
            df_p.groupby("profile_number")
            .apply(lambda g: (g.index.min(), g.index.max()), include_groups=False)
            .rename("interval")
            .apply(pd.Series)
            .rename(columns={0: "start", 1: "end"})
        )
        profile_intervals["has_sunset"] = profile_intervals.apply(
            lambda row: _interval_contains_sunevent(row["start"], row["end"], biolume_sunsets),
            axis=1,
        )
        profile_intervals["has_sunrise"] = profile_intervals.apply(
            lambda row: _interval_contains_sunevent(row["start"], row["end"], biolume_sunrises),
            axis=1,
        )
        profile_intervals["has_sunevent"] = (
            profile_intervals["has_sunrise"] | profile_intervals["has_sunset"]
        )
        df_p["has_sunset"] = df_p["profile_number"].map(profile_intervals["has_sunset"])
        df_p["has_sunrise"] = df_p["profile_number"].map(profile_intervals["has_sunrise"])
        df_p["has_sunevent"] = df_p["profile_number"].map(profile_intervals["has_sunevent"])

        # Set all the proxies to nan and then add in the valid values in the loop below
        self.df_r["biolume_proxy_adinos"] = np.nan
        self.df_r["biolume_proxy_diatoms"] = np.nan
        self.df_r["biolume_proxy_hdinos"] = np.nan

        # compute correlation per profil and then correct proxies
        profil = df_p.profile_number
        dt_5mins = np.timedelta64(timedelta(minutes=minutes_from_surface_threshold))
        for iprofil_ in range(1, int(np.max(profil)) + 1):
            iprofil = profil == iprofil_
            has_sunevent = df_p.loc[iprofil, "has_sunevent"].any()
            if has_sunevent:  # set proxies for this profile to NaN
                self.logger.info(
                    "Processing profile=%d for proxy correction: found sun event -- set NaN",
                    iprofil_,
                )
                target_indices = df_p.index[iprofil]
                self.df_r.loc[target_indices, "biolume_proxy_adinos"] = np.nan
                self.df_r.loc[target_indices, "biolume_proxy_diatoms"] = np.nan
                self.df_r.loc[target_indices, "biolume_proxy_hdinos"] = np.nan
                continue
            # excludes surface, must be within 5 min of it
            ideep = iprofil & (df_p.depth > depth_threshold)
            itime = (df_p.index > (df_p.index[ideep].min() - dt_5mins)) & (
                df_p.index < (df_p.index[ideep].max() + dt_5mins)
            )
            iprofil = iprofil & itime
            if not np.any(iprofil):
                # print(f'no corrections possible for {iprofil_=}')
                continue
            auv_profil = df_p.loc[iprofil]
            self.logger.info(
                "Processing profile=%d for proxy correction: total_points=%d > thresh=%d ?",
                iprofil_,
                auv_profil.shape[0],
                correction_threshold,
            )
            if auv_profil.shape[0] > correction_threshold:
                if (
                    np.sum(auv_profil.biolume_proxy_adinos > adinos_threshold)
                    < correction_threshold
                ):
                    if auv_profil.biolume_proxy_adinos.count() == 0:  # all proxies are NaN so skip
                        self.logger.info(
                            "Correcting proxies: valid adinos=%d < thresh=%d -- all NaN so skip",
                            np.sum(auv_profil.biolume_proxy_adinos > adinos_threshold),
                            correction_threshold,
                        )
                        continue
                    # no correction for low fluo & biolum values
                    fluoBL_corr = 1.0
                    self.logger.info(
                        "Correcting proxies: valid adinos=%d < thresh=%d"
                        " -- using fluoBL_corr=%.4f, total_size_adinos=%d, nans=%d",
                        np.sum(auv_profil.biolume_proxy_adinos > adinos_threshold),
                        correction_threshold,
                        fluoBL_corr,
                        auv_profil.biolume_proxy_adinos.shape[0],
                        auv_profil.biolume_proxy_adinos.isna().sum(),
                    )
                else:
                    # correlation between fluo and bg_biolum computed on high
                    # adino values for each profile
                    idepth = (
                        auv_profil.depth
                        <= auv_profil.depth[
                            auv_profil.biolume_proxy_adinos > adinos_threshold
                        ].max()
                    )
                    auv_profil_idepth = auv_profil[
                        ["biolume_fluo", "biolume_bg_biolume", "depth"]
                    ].loc[idepth]
                    # pandas' corr ignores NaN
                    fluoBL_corr = auv_profil_idepth.biolume_fluo.corr(
                        auv_profil_idepth.biolume_bg_biolume, method=corr_type
                    )
                    self.logger.info(
                        "Correcting proxies: valid adinos=%d > thresh=%d"
                        " -- using fluoBL_corr=%.4f, total_size_idepth=%d, nans=%d,"
                        " min_depth=%.4f, max_depth=%.4f",
                        np.sum(auv_profil.biolume_proxy_adinos > adinos_threshold),
                        correction_threshold,
                        fluoBL_corr,
                        auv_profil_idepth.shape[0],
                        auv_profil.biolume_proxy_adinos.isna().sum(),
                        auv_profil_idepth.depth.min(),
                        auv_profil_idepth.depth.max(),
                    )

                # save correlation
                df_p.loc[iprofil, "fluoBL_corr"] = fluoBL_corr
                # self.logger.info(
                #    "Correcting proxies for profile=%d using fluoBL_corr=%.4f",
                #    iprofil_,
                #    fluoBL_corr,
                # )

                # scale between 0 and 1 first
                fluoBL_correctionfactor = (fluoBL_corr + 1.0) / 2.0

                # then scale between fluo_bl_threshold and 1
                fluoBL_correctionfactor = (
                    fluoBL_correctionfactor * (1.0 - fluo_bl_threshold) + fluo_bl_threshold
                )

                # can happen if fluo_bl_threshold is negative
                fluoBL_correctionfactor = max(fluoBL_correctionfactor, 0.0)

                df_p.loc[iprofil, "adinosN"] = (
                    df_p.biolume_proxy_adinos[iprofil] * fluoBL_correctionfactor
                )

                # preserving adinos+diatoms
                df_p.loc[iprofil, "diatomsN"] = (
                    df_p.biolume_proxy_adinos[iprofil]
                    + df_p.biolume_proxy_diatoms[iprofil]
                    - df_p.adinosN[iprofil]
                )

                # preserving adinos+hdinos
                df_p.loc[iprofil, "hdinosN"] = (
                    df_p.biolume_proxy_adinos[iprofil]
                    + df_p.biolume_proxy_hdinos[iprofil]
                    - df_p.adinosN[iprofil]
                )

                target_indices = df_p.index[iprofil]
                self.df_r.loc[target_indices, "biolume_proxy_adinos"] = df_p.adinosN.loc[iprofil]
                self.df_r.loc[target_indices, "biolume_proxy_diatoms"] = df_p.diatomsN.loc[iprofil]
                self.df_r.loc[target_indices, "biolume_proxy_hdinos"] = df_p.hdinosN.loc[iprofil]
            else:
                self.logger.info(
                    "profile=%d skipped for proxy correction",
                    iprofil_,
                )
        # Copy the attrs back to self.df_r[] as they were lost in the processing
        # Also add the fluo_bl_threshold value to the comment attribute
        for var in saved_attrs:
            self.df_r[var].attrs = saved_attrs[var]
            if var in ["biolume_proxy_diatoms", "biolume_proxy_adinos", "biolume_proxy_hdinos"]:
                self.df_r[var].attrs["comment"] += (
                    f"; corrected with fluo_bl_threshold={fluo_bl_threshold}"
                )

    def resample_variable(  # noqa: PLR0913
        self,
        instr: str,
        variable: str,
        mf_width: int,
        freq: str,
        mission_start: pd.Timestamp,
        mission_end: pd.Timestamp,
        instrs_to_pad: dict[str, timedelta],
        depth_threshold: float,
    ) -> None:
        # Get the time variable name from the dimension of the variable
        timevar = self.ds[variable].dims[0]
        if instr == "biolume" and variable == "biolume_raw":
            # Only biolume_avg_biolume and biolume_flow treated like other data
            # All other biolume variables in self.df_r[] are computed from biolume_raw
            proxy_cal_factor, proxy_ratio_adinos = self.set_proxy_parameters(mission_start)
            biolume_fluo, biolume_sunsets, biolume_sunrises = self.add_biolume_proxies(
                freq=freq,
                proxy_cal_factor=proxy_cal_factor,
                proxy_ratio_adinos=proxy_ratio_adinos,
            )
            self.correct_biolume_proxies(
                biolume_fluo,
                biolume_sunsets,
                biolume_sunrises,
                depth_threshold,
            )
        elif instr == "wetlabsubat" and variable == "wetlabsubat_digitized_raw_ad_counts":
            # All wetlabsubat proxy variables are computed from wetlabsubat_digitized_raw_ad_counts
            # Use default parameters for LRAUV - these may need adjustment in the future
            proxy_cal_factor = 0.00470
            proxy_ratio_adinos = 3.9811e13
            self.add_wetlabsubat_proxies(
                freq=freq,
                proxy_cal_factor=proxy_cal_factor,
                proxy_ratio_adinos=proxy_ratio_adinos,
            )
        else:
            self.df_o[variable] = self.ds[variable].to_pandas()
            self.df_o[f"{variable}_mf"] = (
                self.ds[variable].rolling(**{timevar: mf_width}, center=True).median().to_pandas()
            )
            # Resample to center of freq https://stackoverflow.com/a/69945592/1281657
            self.logger.info(
                "Resampling %s with frequency %s following %d point median filter",
                variable,
                freq,
                mf_width,
            )
            if instr in instrs_to_pad:
                self.logger.info(
                    "Padding %s with %s of NaNs to the end of mission",
                    variable,
                    instrs_to_pad[instr],
                )
                dt_index = pd.date_range(mission_start, mission_end, freq=freq.lower())
                self.df_r[variable] = pd.Series(np.nan, index=dt_index)
                instr_data = (
                    self.df_o[f"{variable}_mf"]
                    .shift(0.5, freq=freq.lower())
                    .resample(freq.lower())
                    .mean()
                )
                self.df_r.loc[instr_data.index, variable] = instr_data
            else:
                self.df_r[variable] = (
                    self.df_o[f"{variable}_mf"]
                    .shift(0.5, freq=freq.lower())
                    .resample(freq.lower())
                    .mean()
                )
        return ".mean() aggregator"

    def plot_coordinates(self, instr: str, freq: str, plot_seconds: float) -> None:
        self.logger.info("Plotting resampled data")
        # Use sample rate to get indices for plotting plot_seconds of data
        o_end = int(
            self.ds[f"{instr}_depth"].attrs["instrument_sample_rate_hz"] * plot_seconds,
        )
        r_end = int(plot_seconds / float(freq[:-1]))
        df_op = self.df_o.iloc[:o_end]
        df_rp = self.df_r.iloc[:r_end]

        _, ax = plt.subplots(nrows=3, figsize=(18, 10))
        df_op[f"{instr}_depth"].plot(ax=ax[0])
        df_op[f"{instr}_depth_mf"].plot(ax=ax[0])
        df_rp["depth"].plot(linestyle="--", ax=ax[0], marker="o", markersize=2)
        df_op[f"{instr}_latitude"].plot(ax=ax[1])
        df_op[f"{instr}_latitude_mf"].plot(ax=ax[1])
        df_rp["latitude"].plot(linestyle="--", ax=ax[1], marker="o", markersize=2)
        df_op[f"{instr}_longitude"].plot(ax=ax[2])
        df_op[f"{instr}_longitude_mf"].plot(ax=ax[2])
        df_rp["longitude"].plot(linestyle="--", ax=ax[2], marker="o", markersize=2)
        ax[0].set_ylabel("Depth")
        ax[0].legend(["Original", "Median Filtered", "Resampled"])
        ax[1].set_ylabel("Latitude")
        ax[1].legend(["Original", "Median Filtered", "Resampled"])
        ax[2].set_ylabel("Longitude")
        ax[2].legend(["Original", "Median Filtered", "Resampled"])
        ax[2].set_xlabel("Time")
        ax[0].set_title(f"{instr} coordinates")
        plt.show()

    def plot_variable(
        self,
        instr: str,
        variable: str,
        freq: str,
        plot_seconds: float,
    ) -> None:
        self.logger.info("Plotting resampled data")
        # Use sample rate to get indices for plotting plot_seconds of data
        o_end = int(self.ds[variable].attrs["instrument_sample_rate_hz"] * plot_seconds)
        r_end = int(plot_seconds / float(freq[:-1]))
        df_op = self.df_o.iloc[:o_end]
        df_rp = self.df_r.iloc[:r_end]

        # Different freqs on same axes - https://stackoverflow.com/a/13873014/1281657
        ax = df_op.plot.line(y=[variable, f"{variable}_mf"])
        df_rp.plot.line(
            y=[variable],
            ax=ax,
            marker="o",
            linestyle="dotted",
            markersize=2,
        )
        try:
            units = self.ds[variable].attrs["units"]
        except KeyError:
            units = ""
        ax.set_ylabel(f"{self.ds[variable].attrs['long_name']} ({units})")
        ax.legend(["Original", "Median Filtered", "Resampled"])
        ax.set_xlabel("Time")
        ax.set_title(f"{instr} {variable}")
        plt.show()

    def get_mission_start_end(
        self,
        nc_file: str,
        min_crit: int = 5,
    ) -> tuple[datetime, datetime, str]:
        """Some i2map missions have the seabird25p instrument failing
        to record shortly after the start of the mission.  Examine the
        start and end times of each instrument and collect instruments
        that end more than `min_crit` minutes earlier than the overall
        mission_end time.
        """
        mission_start = datetime.max  # noqa: DTZ901
        mission_end = datetime.min  # noqa: DTZ901
        instrs_to_pad = {}
        self.logger.info("Determining mission start and end times")
        time_coords = []
        for instr in self.instruments_variables(nc_file):
            time_coord = f"{instr}_{TIME}"
            try:
                mission_start = min(pd.to_datetime(self.ds[time_coord].min().values), mission_start)
                mission_end = max(pd.to_datetime(self.ds[time_coord].max().values), mission_end)
                time_coords.append(time_coord)
            except KeyError:
                # Likely an LRAUV _combined.nc file with multiple different dimensions in a Group
                self.logger.info(
                    "Ignoring expected time_coord that could not be found: %s", time_coord
                )
        for time_coord in time_coords:
            duration = mission_end - pd.to_datetime(self.ds[time_coord].max().values)
            self.logger.info(
                "%-10s: %s to %s (%s before mission_end)",
                time_coord.split("_")[0],
                self.ds[time_coord].min().values,
                self.ds[time_coord].max().values,
                duration,
            )
            if mission_end - pd.to_datetime(
                self.ds[time_coord].max().values,
            ) > timedelta(minutes=min_crit):
                instrs_to_pad[time_coord.split("_")[0]] = duration
                self.logger.warning(
                    "Instrument %s has a gap > %d minutes at the end of the mission: %s",
                    time_coord.split("_")[0],
                    min_crit,
                    mission_end - pd.to_datetime(self.ds[time_coord].max().values),
                )
        # Convert to datetime with no fractional seconds
        # Naive datetime objects are used for compatibility
        # with Pandas Series in resample_coordinates()
        mission_start = datetime.strptime(  # noqa: DTZ007
            mission_start.strftime("%Y-%m-%dT%H:%M:%S"),
            "%Y-%m-%dT%H:%M:%S",
        )
        mission_end = datetime.strptime(  # noqa: DTZ007
            mission_end.strftime("%Y-%m-%dT%H:%M:%S"),
            "%Y-%m-%dT%H:%M:%S",
        )
        return mission_start, mission_end, instrs_to_pad

    def resample_mission(  # noqa: C901, PLR0912, PLR0915, PLR0913
        self,
        nc_file: str,  # align.nc file
        mf_width: int = MF_WIDTH,
        freq: str = FREQ,
        plot_seconds: float = PLOT_SECONDS,
        depth_threshold: float = DEPTH_THRESHOLD,
    ) -> None:
        # Change depth_threshold here should a particular mission require it, e.g.:
        # if "2023.192.01" in nc_file: ...
        self.logger.info("Using depth_threshold = %.2f m", depth_threshold)

        pd.options.plotting.backend = "matplotlib"
        self.ds = xr.open_dataset(nc_file)
        mission_start, mission_end, instrs_to_pad = self.get_mission_start_end(nc_file)
        last_instr = ""
        for icount, (instr, variables) in enumerate(
            self.instruments_variables(nc_file).items(),
        ):
            if icount == 0:
                self.df_o = pd.DataFrame()  # original dataframe
                self.df_r = pd.DataFrame()  # resampled dataframe
                # Choose an instrument to use for the resampled coordinates
                # All the instruments we care about are in the nose of the vehicle
                # Use the pitch corrected depth coordinate for 'ctd1' for dorado,
                # 'seabird25p' for i2map.  The depth coordinate for pitch_corrected_instr
                # must be as complete as possible as it's used for all the other
                # nosecone instruments.  If we are processing LRAUV data then
                # use 'ctddseabird', otherwise start with 'ctd1' and fall back to
                # 'seabird25p' if needed for i2map missions. Early LRAUV missions
                # had only CTD_NeilBrown instruments, later ones had CTD_Seabird.
                pitch_corrected_instr = "ctdseabird" if self.log_file else "ctd1"
                if f"{pitch_corrected_instr}_depth" not in self.ds:
                    pitch_corrected_instr = "ctdneilbrown"
                if f"{pitch_corrected_instr}_depth" not in self.ds:
                    pitch_corrected_instr = "seabird25p"
                    if pitch_corrected_instr in instrs_to_pad:
                        # Use navigation if seabird25p failed to record
                        # data all the way to the end of the mission
                        pitch_corrected_instr = "navigation"
                aggregator = self.resample_coordinates(
                    pitch_corrected_instr,
                    mf_width,
                    freq,
                )
                self.save_coordinates(instr, mf_width, freq, aggregator)
                if self.plot:
                    self.plot_coordinates(instr, freq, plot_seconds)
                self.add_profile(depth_threshold=depth_threshold)
            if instr != last_instr:
                # Start with new dataframes for each instrument
                self.df_o = pd.DataFrame()
                self.df_r = pd.DataFrame()
            for variable in variables:
                if instr == "biolume" and variable == "biolume_raw":
                    # resample_variable() creates new proxy variables
                    # not in the original align.nc file
                    self.resample_variable(
                        instr,
                        variable,
                        mf_width,
                        freq,
                        mission_start,
                        mission_end,
                        instrs_to_pad,
                        depth_threshold,
                    )
                    for var in self.df_r:
                        if var not in variables:
                            # save new proxy variable
                            self.df_r[var].index.rename("time", inplace=True)  # noqa: PD002
                            self.resampled_nc[var] = self.df_r[var].to_xarray()
                            self.resampled_nc[var].attrs = self.df_r[var].attrs
                            self.resampled_nc[var].attrs["coordinates"] = (
                                "time depth latitude longitude"
                            )
                elif instr == "wetlabsubat" and variable == "wetlabsubat_digitized_raw_ad_counts":
                    # resample_variable() creates new proxy variables for LRAUV
                    # not in the original align.nc file
                    self.resample_variable(
                        instr,
                        variable,
                        mf_width,
                        freq,
                        mission_start,
                        mission_end,
                        instrs_to_pad,
                        depth_threshold,
                    )
                    for var in self.df_r:
                        if var not in variables:
                            # save new proxy variable
                            self.df_r[var].index.rename("time", inplace=True)  # noqa: PD002
                            self.resampled_nc[var] = self.df_r[var].to_xarray()
                            self.resampled_nc[var].attrs = self.df_r[var].attrs
                            self.resampled_nc[var].attrs["coordinates"] = (
                                "time depth latitude longitude"
                            )
                elif variable in {"biolume_latitude", "biolume_longitude"}:
                    self.logger.info(
                        "Not saving instrument coordinate variable %s to resampled file",
                        variable,
                    )
                else:
                    aggregator = self.resample_variable(
                        instr,
                        variable,
                        mf_width,
                        freq,
                        mission_start,
                        mission_end,
                        instrs_to_pad,
                        depth_threshold,
                    )
                    self.df_r[variable].index.rename("time", inplace=True)  # noqa: PD002
                    self.resampled_nc[variable] = self.df_r[variable].to_xarray()
                    self.resampled_nc[variable].attrs = self.ds[variable].attrs
                    self.resampled_nc[variable].attrs["coordinates"] = (
                        "time depth latitude longitude"
                    )
                    self.resampled_nc[variable].attrs["comment"] += (
                        f" median filtered with {mf_width} samples"
                        f" and resampled with {aggregator} to {freq} intervals."
                    )
                    if self.plot:
                        self.plot_variable(instr, variable, freq, plot_seconds)

        # Call vehicle-specific metadata method which will call _build_global_metadata()
        if self.auv_name.lower() == "dorado":
            self.resampled_nc.attrs = self.dorado_global_metadata()
        elif self.auv_name.lower() == "i2map":
            self.resampled_nc.attrs = self.i2map_global_metadata()
        else:
            # Assume LRAUV for any other vehicle
            self.resampled_nc.attrs = self.lrauv_global_metadata()
        self.resampled_nc["time"].attrs = {
            "standard_name": "time",
            "long_name": "Time (UTC)",
        }
        out_fn = str(nc_file).replace("_align.nc", f"_{freq}.nc")
        if self.flash_threshold and self.flash_threshold != FLASH_THRESHOLD:
            # Append flash_threshold to output filename
            ft_ending = f"_ft{self.flash_threshold:.0E}.nc".replace("E+", "E")
            out_fn = out_fn.replace(".nc", ft_ending)
        self.resampled_nc.to_netcdf(path=out_fn, format="NETCDF4_CLASSIC")
        self.logger.info("Saved resampled mission to %s", out_fn)

    def process_command_line(self):
        """Process command line arguments using shared parser infrastructure."""
        # Use shared parser with resample-specific additions
        parser = get_standard_lrauv_parser(
            description=__doc__,
        )

        # Add resampling arguments (freq and mf_width)
        parser.add_argument(
            "--freq",
            type=str,
            default=FREQ,
            help=f"Resampling frequency, default: {FREQ}",
        )
        parser.add_argument(
            "--mf_width",
            type=int,
            default=MF_WIDTH,
            help=f"Median filter width for smoothing, default: {MF_WIDTH}",
        )

        # Add resample-specific arguments
        parser.add_argument("--plot", action="store_true", help="Plot data")
        parser.add_argument(
            "--plot_seconds",
            action="store",
            default=PLOT_SECONDS,
            type=float,
            help="Plot seconds of data",
        )
        parser.add_argument(
            "--flash_threshold",
            action="store",
            type=float,
            help=(
                f"Override the default flash_threshold value of {FLASH_THRESHOLD:.0E} "
                "and append to output filename"
            ),
        )

        self.args = parser.parse_args()

        # Set instance attributes from parsed arguments
        self.auv_name = self.args.auv_name
        self.mission = self.args.mission
        self.log_file = self.args.log_file
        self.freq = self.args.freq
        self.mf_width = self.args.mf_width
        self.flash_threshold = self.args.flash_threshold
        self.verbose = self.args.verbose
        self.plot = self.args.plot
        self.commandline = " ".join(sys.argv)
        self.logger.setLevel(self._log_levels[self.verbose])


if __name__ == "__main__":
    resamp = Resampler()
    resamp.process_command_line()
    if resamp.args.log_file:
        netcdfs_dir = Path(BASE_LRAUV_PATH, f"{Path(resamp.args.log_file).parent}")
        nc_file = Path(netcdfs_dir, f"{Path(resamp.args.log_file).stem}_align.nc")
    else:
        file_name = f"{resamp.args.auv_name}_{resamp.args.mission}_align.nc"
        nc_file = Path(
            BASE_PATH,
            resamp.args.auv_name,
            MISSIONNETCDFS,
            resamp.args.mission,
            file_name,
        )
    p_start = time.time()
    # Everything that Resampler needs should be in the self described nc_file
    # whether it is Dorado/i2MAP or LRAUV
    resamp.resample_mission(
        nc_file,
        mf_width=resamp.args.mf_width,
        freq=resamp.args.freq,
        plot_seconds=resamp.args.plot_seconds,
    )
    resamp.logger.info("Time to process: %.2f seconds", (time.time() - p_start))
