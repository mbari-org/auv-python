#!/usr/bin/env python
"""
Resample variables from mission netCDF file to common time axis

Read all the record variables stored at original instrument sampling rate
from netCDF file and resample them to common time axis.
"""

__author__ = "Mike McCann"
__copyright__ = "Copyright 2021, Monterey Bay Aquarium Research Institute"

import argparse
import logging
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from socket import gethostname
from typing import Dict, Tuple

import cf_xarray  # Needed for the .cf accessor
import git
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from dorado_info import dorado_info
from logs2netcdfs import BASE_PATH, MISSIONNETCDFS, SUMMARY_SOURCE, TIME, AUV_NetCDF
from pysolar.solar import get_altitude
from scipy import signal
from utils import simplify_points

MF_WIDTH = 3
FREQ = "1S"
PLOT_SECONDS = 300


class InvalidAlignFile(Exception):
    pass


class Resampler:
    logger = logging.getLogger(__name__)
    _handler = logging.StreamHandler()
    _handler.setFormatter(AUV_NetCDF._formatter)
    logger.addHandler(_handler)
    _log_levels = (logging.WARN, logging.INFO, logging.DEBUG)

    def __init__(self) -> None:
        plt.rcParams["figure.figsize"] = (15, 5)
        self.resampled_nc = xr.Dataset()
        iso_now = datetime.utcnow().isoformat().split(".")[0] + "Z"
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
        repo = git.Repo(search_parent_directories=True)
        try:
            gitcommit = repo.head.object.hexsha
        except (ValueError, BrokenPipeError) as e:
            self.logger.warning(
                "could not get head commit sha for %s: %s", repo.remotes.origin.url, e
            )
            gitcommit = "<failed to get git commit>"
        iso_now = datetime.utcnow().isoformat().split(".")[0] + "Z"
        # Common dynamic attributes for all auv platforms
        self.metadata["time_coverage_start"] = str(min(self.resampled_nc.time.values))
        self.metadata["time_coverage_end"] = str(max(self.resampled_nc.time.values))
        self.metadata["time_coverage_duration"] = str(
            datetime.utcfromtimestamp(
                max(self.resampled_nc.time.values).astype("float64") / 1.0e9
            )
            - datetime.utcfromtimestamp(
                min(self.resampled_nc.time.values).astype("float64") / 1.0e9
            )
        )
        self.metadata["geospatial_vertical_min"] = min(
            self.resampled_nc.cf["depth"].values
        )
        self.metadata["geospatial_vertical_max"] = max(
            self.resampled_nc.cf["depth"].values
        )
        self.metadata["geospatial_lat_min"] = min(
            self.resampled_nc.cf["latitude"].values
        )
        self.metadata["geospatial_lat_max"] = max(
            self.resampled_nc.cf["latitude"].values
        )
        self.metadata["geospatial_lon_min"] = min(
            self.resampled_nc.cf["longitude"].values
        )
        self.metadata["geospatial_lon_max"] = max(
            self.resampled_nc.cf["longitude"].values
        )
        self.metadata["license"] = "Any use requires prior approval from MBARI"
        self.metadata["history"] = f"Created by {self.commandline} on {iso_now}"

        if "process_" in self.commandline:
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
            f" {self.args.freq} intervals."
            f" Data processed at {iso_now} using MBARI's auv-python software."
        )

    def dorado_global_metadata(self) -> dict:
        """Use instance variables to return a dictionary of
        metadata specific for the data that are written
        """
        self.metadata["title"] = f"Calibrated, "
        try:
            if dorado_info[self.args.mission].get("program"):
                self.metadata[
                    "title"
                ] = f"{dorado_info[self.args.mission]['program']} program - calibrated, "
        except KeyError:
            self.logger.warning(
                f"No entry for for mission {self.args.mission} program in dorado_info.py"
            )
        self.metadata["title"] += (
            f"aligned, and resampled AUV sensor data from"
            f" {self.args.auv_name} mission {self.args.mission}"
        )
        try:
            self.metadata["summary"] += (
                " Processing log file: http://dods.mbari.org/opendap/data/auvctd/surveys/"
                f"{self.args.mission.split('.')[0]}/netcdf/"
                f"{self.args.auv_name}_{self.args.mission}_processing.log"
            )
        except KeyError as e:
            # Likely no _1S.nc file was created, hence no summary to append to
            self.logger.warning(
                f"Could not add processing log file to summary matadata for mission {self.args.mission}"
            )

        try:
            if dorado_info[self.args.mission].get("program"):
                self.metadata["program"] = dorado_info[self.args.mission].get("program")
            if dorado_info[self.args.mission].get("comment"):
                self.metadata["comment"] = dorado_info[self.args.mission].get("comment")
        except KeyError:
            self.logger.warning(
                f"No entry for for mission {self.args.mission} program or comment in dorado_info.py"
            )
        try:
            # Parse from ctd1_depth comment: "using SensorOffset(x=1.003, y=0.0001)"
            self.metadata["comment"] += (
                f". Variable depth pitch corrected using"
                f" {self.ds['ctd1_depth'].attrs['comment'].split('using ')[1]}"
            )
        except KeyError:
            self.logger.warning(
                f"No entry for for mission {self.args.mission} comment in dorado_info.py"
            )

        return self.metadata

    def i2map_global_metadata(self) -> dict:
        """Use instance variables to return a dictionary of
        metadata specific for the data that are written
        """
        self.metadata["title"] = (
            f"Calibrated, aligned, and resampled AUV sensor data from"
            f" {self.args.auv_name} mission {self.args.mission}"
        )
        # Append location of original data files to summary
        matches = re.search(
            "(" + SUMMARY_SOURCE.replace("{}", r".+$") + ")",
            self.ds.attrs["summary"],
        )
        if matches:
            self.metadata["summary"] += (
                " "
                + matches.group(1)
                + ".  Processing log file: http://dods.mbari.org/opendap/data/auvctd/surveys/"
                + f"{self.args.mission.split('.')[0]}/netcdf/"
                + f"{self.args.auv_name}_{self.args.mission}_processing.log"
            )
            # Append shortened location of original data files to title
            # Useful for I2Map data as it's in a YYYY/MM directory structure
            self.metadata["title"] += (
                ", "
                + "original data in /mbari/M3/master/i2MAP/: "
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
                f"No entry for for mission {self.args.mission} comment in dorado_info.py"
            )

        return self.metadata

    def instruments_variables(self, nc_file: str) -> dict:
        """
        Return a dictionary of all the variables in the mission netCDF file,
        keyed by instrument name
        """
        self.logger.info(f"Reading variables from {nc_file} mission netCDF file")
        last_instr = None
        instr_vars = defaultdict(list)
        for variable in self.ds.keys():
            instr, *_ = variable.split("_")
            if instr == "navigation":
                freq = "0.1S"
            elif instr == "gps" or instr == "depth":
                continue
            if instr != last_instr:
                instr_vars[instr].append(variable)
        return instr_vars

    def resample_coordinates(self, instr: str, mf_width: int, freq: str) -> None:
        self.logger.info(
            f"Resampling coordinates depth, latitude and longitude with"
            f" frequency {freq} following {mf_width} point median filter "
        )
        # Original
        try:
            self.df_o[f"{instr}_depth"] = self.ds[f"{instr}_depth"].to_pandas()
        except KeyError:
            self.logger.warning(
                f"Variable {instr}_depth not found in {self.args.mission} align.nc file"
            )
            self.logger.info(
                "Cannot continue without a pitch corrected depth coordinate"
            )
            raise InvalidAlignFile(
                f"{instr}_depth not found in {self.args.auv_name}_{self.args.mission}_align.nc"
            )
        self.df_o[f"{instr}_latitude"] = self.ds[f"{instr}_latitude"].to_pandas()
        self.df_o[f"{instr}_longitude"] = self.ds[f"{instr}_longitude"].to_pandas()
        # Median Filtered - back & forward filling nan values at ends
        self.df_o[f"{instr}_depth_mf"] = (
            self.ds[f"{instr}_depth"]
            .rolling(**{instr + "_time": mf_width}, center=True)
            .median()
            .to_pandas()
        )
        self.df_o[f"{instr}_depth_mf"] = self.df_o[f"{instr}_depth_mf"].fillna(
            method="bfill"
        )
        self.df_o[f"{instr}_depth_mf"] = self.df_o[f"{instr}_depth_mf"].fillna(
            method="ffill"
        )
        self.df_o[f"{instr}_latitude_mf"] = (
            self.ds[f"{instr}_latitude"]
            .rolling(**{instr + "_time": mf_width}, center=True)
            .median()
            .to_pandas()
        )
        self.df_o[f"{instr}_latitude_mf"] = self.df_o[f"{instr}_latitude_mf"].fillna(
            method="bfill"
        )
        self.df_o[f"{instr}_latitude_mf"] = self.df_o[f"{instr}_latitude_mf"].fillna(
            method="ffill"
        )
        self.df_o[f"{instr}_longitude_mf"] = (
            self.ds[f"{instr}_longitude"]
            .rolling(**{instr + "_time": mf_width}, center=True)
            .median()
            .to_pandas()
        )
        self.df_o[f"{instr}_longitude_mf"] = self.df_o[f"{instr}_longitude_mf"].fillna(
            method="bfill"
        )
        self.df_o[f"{instr}_longitude_mf"] = self.df_o[f"{instr}_longitude_mf"].fillna(
            method="ffill"
        )
        # Resample to center of freq https://stackoverflow.com/a/69945592/1281657
        aggregator = ".mean() aggregator"
        # This is the common depth for all the instruments - the instruments that
        # matter (ctds, hs2, biolume, lopc) are all in the nose of the vehicle
        # (at least in November 2020)
        # and we want to use the same pitch corrected depth for all of them.
        self.df_r["depth"] = (
            self.df_o[f"{instr}_depth_mf"].shift(0.5, freq=freq).resample(freq).mean()
        )
        self.df_r["latitude"] = (
            self.df_o[f"{instr}_latitude_mf"]
            .shift(0.5, freq=freq)
            .resample(freq)
            .mean()
        )
        self.df_r["longitude"] = (
            self.df_o[f"{instr}_longitude_mf"]
            .shift(0.5, freq=freq)
            .resample(freq)
            .mean()
        )
        return aggregator

    def save_coordinates(
        self, instr: str, mf_width: int, freq: str, aggregator: str
    ) -> None:
        in_fn = self.ds.encoding["source"].split("/")[-1]
        self.df_r["depth"].index.rename("time", inplace=True)
        self.resampled_nc["depth"] = self.df_r["depth"].to_xarray()
        self.df_r["latitude"].index.rename("time", inplace=True)
        self.resampled_nc["latitude"] = self.df_r["latitude"].to_xarray()
        self.df_r["longitude"].index.rename("time", inplace=True)
        self.resampled_nc["longitude"] = self.df_r["longitude"].to_xarray()
        self.resampled_nc["depth"].attrs = self.ds[f"{instr}_depth"].attrs
        self.resampled_nc["depth"].attrs["comment"] += (
            f". {self.ds[f'{instr}_depth'].attrs['comment']}"
            f" mean sampled at {self.args.freq} intervals following"
            f" {self.args.mf_width} point median filter."
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
        self, stride: int = 3000
    ) -> Tuple[pd.Series, datetime, datetime]:
        # Select only the nighttime biolume_raw data,
        # one hour past sunset to one before sunrise
        # Default stride of 3000 give 10 minute resolution
        # from 5 hz navigation data
        lat = float(self.ds["navigation_latitude"].median())
        lon = float(self.ds["navigation_longitude"].median())
        self.logger.debug("Getting sun altitudes for nighttime selection")
        sun_alts = []
        for ts in self.ds["navigation_time"].values[::stride]:
            # About 10 minute resolution from 5 hz navigation data
            sun_alts.append(
                get_altitude(
                    lat,
                    lon,
                    datetime.utcfromtimestamp(ts.astype(int) / 1.0e9).replace(
                        tzinfo=timezone.utc
                    ),
                )
            )
        # Find sunset and sunrise - where sun altitude changes sign
        sign_changes = np.where(np.diff(np.sign(sun_alts)))[0]
        ss_sr_times = (
            self.ds["navigation_time"]
            .isel({"navigation_time": sign_changes * stride})
            .values
        )
        self.logger.debug(f"Sunset sunrise times {ss_sr_times}")
        sunset = None
        sunrise = None
        if np.sign(sun_alts[0]) == 1:
            # Sun is up at start of mission
            if len(ss_sr_times) == 2:
                sunset, sunrise = ss_sr_times
                sunset += pd.to_timedelta(1, "h")
                sunrise -= pd.to_timedelta(1, "h")
            elif len(ss_sr_times) == 1:
                sunset = ss_sr_times[0]
                sunset += pd.to_timedelta(1, "h")
                sunrise = self.ds["biolume_time60hz"].values[-1]
                self.logger.warning(
                    f"Could not find sunrise time, using last time in dataset: {sunrise}"
                )
            else:
                self.logger.info("Sun is up at start, but no sunset in this mission")
        if np.sign(sun_alts[0]) == -1:
            try:
                self.logger.warning(
                    f"Sun is not up at start of mission: {ss_sr_times[0]}: alt = {sun_alts[0]}"
                )
            except IndexError:
                # Likely no values in ss_sr_times[]
                self.logger.warning("Sun is not up at start of mission")

        if sunset is None and sunrise is None:
            self.logger.info(
                f"No sunset during this mission. No biolume_raw data will be extracted."
            )
            nighttime_bl_raw = pd.Series(dtype="float64")
        else:
            self.logger.info(
                f"Extracting biolume_raw data between sunset {sunset} and sunrise {sunrise}"
            )
            nighttime_bl_raw = (
                (
                    self.ds["biolume_raw"].where(
                        (self.ds["biolume_time60hz"] > sunset)
                        & (self.ds["biolume_time60hz"] < sunrise)
                    )
                )
                .to_pandas()
                .dropna()
            )

        return nighttime_bl_raw, sunset, sunrise

    def add_profile(self, depth_threshold: float = 15) -> None:
        # Find depth vertices value using scipy's find_peaks algorithm
        options = dict(prominence=10, width=30)
        peaks_pos, _ = signal.find_peaks(self.resampled_nc["depth"], **options)
        peaks_neg, _ = signal.find_peaks(-self.resampled_nc["depth"], **options)
        # Need to add the first and last time values to the list of peaks
        peaks = np.concatenate(
            (peaks_pos, peaks_neg, [0], [len(self.resampled_nc["depth"]) - 1])
        )
        peaks.sort(kind="mergesort")
        s_peaks = self.resampled_nc["depth"][peaks].to_pandas()

        # Assign a profile number to each time value
        profiles = []
        count = 1
        k = 0
        for tv in self.resampled_nc["time"].values:
            if tv > s_peaks.index[k + 1]:
                # Encountered a new simple_depth point
                k += 1
                if abs(s_peaks[k + 1] - s_peaks[k]) > depth_threshold:
                    # Completed downcast or upcast
                    count += 1
            profiles.append(count)
            if k > len(s_peaks) - 2:
                break

        self.resampled_nc["profile_number"] = profiles
        self.resampled_nc["profile_number"].attrs = {
            "long_name": "Profile number",
        }

    def add_biolume_proxies(
        self,
        freq,
        window_size_secs: int = 5,
        envelope_mini: float = 1.5e10,
        flash_threshold: float = 1.5e11,
        proxy_ratio_adinos: float = 3.9811e13,  # 4-Oct-2010 to 2-Dec-2020 value
        proxy_cal_factor=0.00470,  # Same as used in 5.2-mpm-bg_biolume-PiO-paper.ipynb
    ) -> None:
        # Add variables via the calculations according to Appendix B in
        # "Using fluorescence and bioluminescence sensors to characterize
        # auto- and heterotrophic plankton communities" by Messie et al."
        # https://www.sciencedirect.com/science/article/pii/S0079661118300478
        # Translation to Python demonstrated in notebooks/5.2-mpm-bg_biolume-PiO-paper.ipynb

        # (1) Dinoflagellate and zooplankton proxies

        self.logger.info("Adding biolume proxy variables computed from biolume_raw")
        sample_rate = 60  # Assume all biolume_raw data is sampled at 60 Hz
        window_size = window_size_secs * sample_rate
        s_biolume_raw, sunset, sunrise = self.select_nighttime_bl_raw()
        if s_biolume_raw.empty:
            self.logger.info("No biolume_raw data to compute proxies")
            return

        # Compute background biolumenesence envelope
        self.logger.debug("Applying rolling min filter")
        min_bg_unsmoothed = s_biolume_raw.rolling(
            window_size, min_periods=0, center=True
        ).min()
        min_bg = (
            min_bg_unsmoothed.rolling(window_size, min_periods=0, center=True)
            .mean()
            .values
        )

        self.logger.debug("Applying rolling median filter")
        med_bg_unsmoothed = s_biolume_raw.rolling(
            window_size, min_periods=0, center=True
        ).median()
        s_med_bg = med_bg_unsmoothed.rolling(
            window_size, min_periods=0, center=True
        ).mean()
        med_bg = s_med_bg.values
        max_bg = med_bg * 2.0 - min_bg
        # envelope_mini: minimum value for the envelope (max_bgrd - med_bgrd) to avoid very dim flashes when the background is low (default 1.5E10 ph/s)
        max_bg[max_bg - med_bg < envelope_mini] = (
            med_bg[max_bg - med_bg < envelope_mini] + envelope_mini
        )

        # Find the high and low peaks
        self.logger.debug("Finding peaks")
        peaks, _ = signal.find_peaks(s_biolume_raw, height=max_bg)
        s_peaks = pd.Series(s_biolume_raw[peaks], index=s_biolume_raw.index[peaks])
        s_med_bg_peaks = pd.Series(s_med_bg[peaks], index=s_biolume_raw.index[peaks])
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
        self.logger.debug(f"Counting flashes using {flash_count_seconds} second window")
        nbflash_high_counts = (
            s_nbflash_high.rolling(flash_window, step=1, min_periods=0, center=True)
            .count()
            .resample(freq)
            .mean()
            / flash_count_seconds
        )
        nbflash_low_counts = (
            s_nbflash_low.rolling(flash_window, step=1, min_periods=0, center=True)
            .count()
            .resample(freq)
            .mean()
            / flash_count_seconds
        )

        flow = (
            self.ds[["biolume_flow"]]["biolume_flow"]
            .to_pandas()
            .resample("1S")
            .mean()
            .fillna(method="ffill")
        )

        # Flow sensor is not always on, so fill in 0.0 values with 350 ml/s
        zero_note = ""
        num_zero_flow = len(np.where(flow == 0)[0])
        if num_zero_flow > 0:
            zero_note = f"Zero flow values found: {num_zero_flow} of {len(flow)} - replaced with 350 ml/s"
            self.logger.info(zero_note)
            flow = flow.replace(0.0, 350.0)

        # Compute flashes per liter - pandas.Series.divide() will match indexes
        # Units: flashes per liter = (flashes per second / mL/s) * 1000 mL/L
        self.logger.info("Computing flashes per liter: nbflash_high, nbflash_low")
        self.df_r["biolume_nbflash_high"] = nbflash_high_counts.divide(flow) * 1000
        self.df_r["biolume_nbflash_high"].attrs[
            "long_name"
        ] = "High intensity flashes (copepods proxy)"
        self.df_r["biolume_nbflash_high"].attrs["units"] = "flashes/liter"
        self.df_r["biolume_nbflash_high"].attrs["comment"] = zero_note

        self.df_r["biolume_nbflash_low"] = nbflash_low_counts.divide(flow) * 1000
        self.df_r["biolume_nbflash_low"].attrs[
            "long_name"
        ] = "Low intensity flashes (Larvacean proxy)"
        self.df_r["biolume_nbflash_low"].attrs["units"] = "flashes/liter"
        self.df_r["biolume_nbflash_low"].attrs["comment"] = zero_note

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
            .resample("1S")
            .mean()
        )
        self.logger.info(
            "Saving flash intensity: biolume_intflash - the upper bound of the background envelope"
        )
        self.df_r["biolume_intflash"] = intflash
        self.df_r["biolume_intflash"].attrs[
            "long_name"
        ] = "Flashes intensity (small jellies proxy)"
        self.df_r["biolume_intflash"].attrs["units"] = "photons/s"
        self.df_r["biolume_intflash"].attrs["comment"] = (
            f" intensity of flashes from {sample_rate} Hz biolume_raw variable"
            f" in {freq} intervals."
        )

        # Make min_bg a 1S pd.Series so that we can divide by flow, matching indexes
        bg_biolume = pd.Series(min_bg, index=s_biolume_raw.index).resample("1S").mean()
        self.logger.info("Saving Background bioluminescence (dinoflagellates proxy)")
        self.df_r["biolume_bg_biolume"] = bg_biolume.divide(flow) * 1000
        self.df_r["biolume_bg_biolume"].attrs[
            "long_name"
        ] = "Background bioluminescence (dinoflagellates proxy)"
        self.df_r["biolume_bg_biolume"].attrs["units"] = "photons/liter"
        self.df_r["biolume_bg_biolume"].attrs["comment"] = zero_note

        # (2) Phytoplankton proxies - use median filtered hs2_fl700 1S data
        if "hs2_fl700" not in self.ds:
            self.logger.info(
                "No hs2_fl700 data. Not computing biolume_proxy_adinos and biolume_proxy_hdinos"
            )
            return
        fluo = (
            self.resampled_nc["hs2_fl700"]
            .where(
                (self.resampled_nc["time"] > sunset)
                & (self.resampled_nc["time"] < sunrise)
            )
            .to_pandas()
            .resample(freq)
            .mean()
        )
        # Set negative values from hs2_fl700 to NaN
        fluo[fluo < 0] = np.nan
        self.logger.info(f"Using proxy_ratio_adinos = {proxy_ratio_adinos:.4e}")
        self.logger.info(f"Using proxy_cal_factor = {proxy_cal_factor:.6f}")
        pseudo_fluorescence = self.df_r["biolume_bg_biolume"] / proxy_ratio_adinos
        self.df_r["biolume_proxy_adinos"] = (
            np.minimum(fluo, pseudo_fluorescence) / proxy_cal_factor
        )
        self.df_r["biolume_proxy_adinos"].attrs[
            "comment"
        ] = f"Autotrophic dinoflagellate proxy using proxy_ratio_adinos = {proxy_ratio_adinos:.4e} and proxy_cal_factor = {proxy_cal_factor:.6f}"
        self.df_r["biolume_proxy_hdinos"] = (
            pseudo_fluorescence - np.minimum(fluo, pseudo_fluorescence)
        ) / proxy_cal_factor
        self.df_r["biolume_proxy_hdinos"].attrs[
            "comment"
        ] = f"Heterotrophic dinoflagellate proxy using proxy_ratio_adinos = {proxy_ratio_adinos:.4e} and proxy_cal_factor = {proxy_cal_factor:.6f}"
        biolume_proxy_diatoms = (fluo - pseudo_fluorescence) / proxy_cal_factor
        biolume_proxy_diatoms[biolume_proxy_diatoms < 0] = 0
        self.df_r["biolume_proxy_diatoms"] = biolume_proxy_diatoms
        self.df_r["biolume_proxy_diatoms"].attrs[
            "comment"
        ] = f"Diatom proxy using proxy_ratio_adinos = {proxy_ratio_adinos:.4e} and proxy_cal_factor = {proxy_cal_factor:.6f}"

    def resample_variable(
        self,
        instr: str,
        variable: str,
        mf_width: int,
        freq: str,
        mission_start: pd.Timestamp,
        mission_end: pd.Timestamp,
        instrs_to_pad: Dict[str, timedelta],
    ) -> None:
        timevar = f"{instr}_{TIME}"
        if instr == "biolume" and variable == "biolume_raw":
            # Only biolume_avg_biolume and biolume_flow treated like other data
            # All other biolume variables in self.df_r[] are computed from biolume_raw
            self.add_biolume_proxies(freq)
        else:
            self.df_o[variable] = self.ds[variable].to_pandas()
            self.df_o[f"{variable}_mf"] = (
                self.ds[variable]
                .rolling(**{timevar: mf_width}, center=True)
                .median()
                .to_pandas()
            )
            # Resample to center of freq https://stackoverflow.com/a/69945592/1281657
            self.logger.info(
                f"Resampling {variable} with frequency {freq} following {mf_width} point median filter "
            )
            if instr in instrs_to_pad.keys():
                self.logger.info(
                    f"Padding {variable} with {instrs_to_pad[instr]} of NaNs to the end of mission"
                )
                dt_index = pd.date_range(mission_start, mission_end, freq=freq)
                self.df_r[variable] = pd.Series(np.NaN, index=dt_index)
                instr_data = (
                    self.df_o[f"{variable}_mf"]
                    .shift(0.5, freq=freq)
                    .resample(freq)
                    .mean()
                )
                self.df_r[variable].loc[instr_data.index] = instr_data
            else:
                self.df_r[variable] = (
                    self.df_o[f"{variable}_mf"]
                    .shift(0.5, freq=freq)
                    .resample(freq)
                    .mean()
                )
        return ".mean() aggregator"

    def plot_coordinates(self, instr: str, freq: str, plot_seconds: float) -> None:
        self.logger.info("Plotting resampled data")
        # Use sample rate to get indices for plotting plot_seconds of data
        o_end = int(
            self.ds[f"{instr}_depth"].attrs["instrument_sample_rate_hz"] * plot_seconds
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
        self, instr: str, variable: str, freq: str, plot_seconds: float
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
        self, nc_file: str, min_crit: int = 5
    ) -> Tuple[datetime, datetime, str]:
        """Some i2map missions have the seabird25p instrument failing
        to record shortly after the start of the mission.  Examine the
        start and end times of each instrument and collect instruments
        that end more than `min_crit` minutes earlier than the overall
        mission_end time.
        """
        mission_start = datetime.max
        mission_end = datetime.min
        instrs_to_pad = {}
        for instr, _ in self.instruments_variables(nc_file).items():
            time_coord = f"{instr}_{TIME}"
            if pd.to_datetime(self.ds[time_coord].min().values) < mission_start:
                mission_start = pd.to_datetime(self.ds[time_coord].min().values)
            if pd.to_datetime(self.ds[time_coord].max().values) > mission_end:
                mission_end = pd.to_datetime(self.ds[time_coord].max().values)
        for instr, _ in self.instruments_variables(nc_file).items():
            time_coord = f"{instr}_{TIME}"
            duration = mission_end - pd.to_datetime(self.ds[time_coord].max().values)
            self.logger.info(
                "%-10s: %s to %s (%s before mission_end)",
                instr,
                self.ds[time_coord].min().values,
                self.ds[time_coord].max().values,
                duration,
            )
            if mission_end - pd.to_datetime(
                self.ds[time_coord].max().values
            ) > timedelta(minutes=min_crit):
                instrs_to_pad[instr] = duration
                self.logger.warning(
                    "Instrument %s has a gap > %d minutes at the end of the mission: %s",
                    instr,
                    min_crit,
                    mission_end - pd.to_datetime(self.ds[time_coord].max().values),
                )
        # Convert to datetime with no fractional seconds
        mission_start = datetime.strptime(
            mission_start.strftime("%Y-%m-%dT%H:%M:%S"), "%Y-%m-%dT%H:%M:%S"
        )
        mission_end = datetime.strptime(
            mission_end.strftime("%Y-%m-%dT%H:%M:%S"), "%Y-%m-%dT%H:%M:%S"
        )
        return mission_start, mission_end, instrs_to_pad

    def resample_mission(
        self,
        nc_file: str,
        mf_width: int = MF_WIDTH,
        freq: str = FREQ,
        plot_seconds: float = PLOT_SECONDS,
    ) -> None:
        pd.options.plotting.backend = "matplotlib"
        self.ds = xr.open_dataset(nc_file)
        mission_start, mission_end, instrs_to_pad = self.get_mission_start_end(nc_file)
        last_instr = ""
        for icount, (instr, variables) in enumerate(
            self.instruments_variables(nc_file).items()
        ):
            if icount == 0:
                self.df_o = pd.DataFrame()  # original dataframe
                self.df_r = pd.DataFrame()  # resampled dataframe
                # Choose an instrument to use for the resampled coordinates
                # All the instruments we care about are in the nose of the vehicle
                # Use the pitch corrected depth coordinate for 'ctd1' for dorado,
                # 'seabird25p' for i2map.
                pitch_corrected_instr = "ctd1"
                if f"{pitch_corrected_instr}_depth" not in self.ds:
                    pitch_corrected_instr = "seabird25p"
                    if pitch_corrected_instr in instrs_to_pad.keys():
                        # Use navigation if seabird25p failed to record
                        # data all the way to the end of the mission
                        pitch_corrected_instr = "navigation"
                aggregator = self.resample_coordinates(
                    pitch_corrected_instr, mf_width, freq
                )
                self.save_coordinates(instr, mf_width, freq, aggregator)
                if self.args.plot:
                    self.plot_coordinates(instr, freq, plot_seconds)
            if instr != last_instr:
                # Start with new dataframes for each instrument
                self.df_o = pd.DataFrame()
                self.df_r = pd.DataFrame()
            for variable in variables:
                if instr == "biolume" and variable == "biolume_raw":
                    # resample_variable() creates new proxy variables not in the original align.nc file
                    self.resample_variable(
                        instr,
                        variable,
                        mf_width,
                        freq,
                        mission_start,
                        mission_end,
                        instrs_to_pad,
                    )
                    for var in self.df_r.keys():
                        if var not in variables:
                            # save new proxy variable
                            self.df_r[var].index.rename("time", inplace=True)
                            self.resampled_nc[var] = self.df_r[var].to_xarray()
                            self.resampled_nc[var].attrs = self.df_r[var].attrs
                            self.resampled_nc[var].attrs[
                                "coordinates"
                            ] = "time depth latitude longitude"
                else:
                    aggregator = self.resample_variable(
                        instr,
                        variable,
                        mf_width,
                        freq,
                        mission_start,
                        mission_end,
                        instrs_to_pad,
                    )
                    self.df_r[variable].index.rename("time", inplace=True)
                    self.resampled_nc[variable] = self.df_r[variable].to_xarray()
                    self.resampled_nc[variable].attrs = self.ds[variable].attrs
                    self.resampled_nc[variable].attrs[
                        "coordinates"
                    ] = "time depth latitude longitude"
                    self.resampled_nc[variable].attrs["comment"] += (
                        f" median filtered with {mf_width} samples"
                        f" and resampled with {aggregator} to {freq} intervals."
                    )
                    if self.args.plot:
                        self.plot_variable(instr, variable, freq, plot_seconds)
        self.add_profile()
        try:
            self._build_global_metadata()
        except KeyError as e:
            self.logger.error(
                f"Missing global attribute {e} in {nc_file}. "
                "Cannot add global metadata to resampled mission."
            )
        if self.args.auv_name.lower() == "dorado":
            self.resampled_nc.attrs = self.dorado_global_metadata()
        elif self.args.auv_name.lower() == "i2map":
            self.resampled_nc.attrs = self.i2map_global_metadata()
        self.resampled_nc["time"].attrs = {
            "standard_name": "time",
            "long_name": "Time (UTC)",
        }
        out_fn = nc_file.replace("_align.nc", f"_{freq}.nc")
        self.resampled_nc.to_netcdf(path=out_fn, format="NETCDF4_CLASSIC")
        self.logger.info(f"Saved resampled mission to {out_fn}")

    def process_command_line(self):
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter,
            description=__doc__,
        )
        parser.add_argument(
            "--base_path",
            action="store",
            default=BASE_PATH,
            help="Base directory for missionlogs and"
            " missionnetcdfs, default: auv_data",
        ),
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
        ),
        parser.add_argument("--plot", action="store_true", help="Plot data")
        parser.add_argument(
            "--plot_seconds",
            action="store",
            default=PLOT_SECONDS,
            type=float,
            help="Plot seconds of data",
        )
        parser.add_argument(
            "--mf_width",
            action="store",
            default=MF_WIDTH,
            type=int,
            help="Median filter width",
        )
        parser.add_argument(
            "--freq",
            action="store",
            default=FREQ,
            help="Resample freq",
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
                [f"{i}: {v}" for i, v, in enumerate(("WARN", "INFO", "DEBUG"))]
            ),
        )
        self.args = parser.parse_args()
        self.logger.setLevel(self._log_levels[self.args.verbose])
        self.commandline = " ".join(sys.argv)


if __name__ == "__main__":
    resamp = Resampler()
    resamp.process_command_line()
    file_name = f"{resamp.args.auv_name}_{resamp.args.mission}_align.nc"
    nc_file = os.path.join(
        BASE_PATH,
        resamp.args.auv_name,
        MISSIONNETCDFS,
        resamp.args.mission,
        file_name,
    )
    p_start = time.time()
    resamp.resample_mission(
        nc_file,
        mf_width=resamp.args.mf_width,
        freq=resamp.args.freq,
        plot_seconds=resamp.args.plot_seconds,
    )
    resamp.logger.info(f"Time to process: {(time.time() - p_start):.2f} seconds")
