#!/usr/bin/env python
"""
Resample variables from mission netCDF file to common time axis

Read all the record variables stored at original instrument sampling rate
from netCDF file and resample them to common time axis.
"""

__author__ = "Mike McCann"
__copyright__ = "Copyright 2021, Monterey Bay Aquarium Research Institute"

import argparse
import git
import logging
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from socket import gethostname

import cf_xarray  # Needed for the .cf accessor
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

from logs2netcdfs import BASE_PATH, MISSIONNETCDFS, SUMMARY_SOURCE

MF_WIDTH = 3
FREQ = "1S"
PLOT_SECONDS = 300


class Resampler:
    logger = logging.getLogger(__name__)
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        "%(levelname)s %(asctime)s %(filename)s "
        "%(funcName)s():%(lineno)d %(message)s"
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    _log_levels = (logging.WARN, logging.INFO, logging.DEBUG)

    def __init__(self) -> None:
        plt.rcParams["figure.figsize"] = (15, 5)
        self.resampled_nc = xr.Dataset()

    def global_metadata(self):
        """Use instance variables to return a dictionary of
        metadata specific for the data that are written
        """
        repo = git.Repo(search_parent_directories=True)
        gitcommit = repo.head.object.hexsha
        iso_now = datetime.utcnow().isoformat() + "Z"

        metadata = {}
        metadata["netcdf_version"] = "4"
        metadata["Conventions"] = "CF-1.6"
        metadata["date_created"] = iso_now
        metadata["date_update"] = iso_now
        metadata["date_modified"] = iso_now
        metadata["featureType"] = "trajectory"

        metadata["time_coverage_start"] = str(min(self.resampled_nc.time.values))
        metadata["time_coverage_end"] = str(max(self.resampled_nc.time.values))
        metadata["time_coverage_duration"] = str(
            datetime.utcfromtimestamp(
                max(self.resampled_nc.time.values).astype("float64") / 1.0e9
            )
            - datetime.utcfromtimestamp(
                min(self.resampled_nc.time.values).astype("float64") / 1.0e9
            )
        )
        metadata["geospatial_vertical_min"] = min(self.resampled_nc.cf["depth"].values)
        metadata["geospatial_vertical_max"] = max(self.resampled_nc.cf["depth"].values)
        metadata["geospatial_lat_min"] = min(self.resampled_nc.cf["latitude"].values)
        metadata["geospatial_lat_max"] = max(self.resampled_nc.cf["latitude"].values)
        metadata["geospatial_lon_min"] = min(self.resampled_nc.cf["longitude"].values)
        metadata["geospatial_lon_max"] = max(self.resampled_nc.cf["longitude"].values)
        metadata[
            "distribution_statement"
        ] = "Any use requires prior approval from MBARI"
        metadata["license"] = metadata["distribution_statement"]
        metadata[
            "useconst"
        ] = "Not intended for legal use. Data may contain inaccuracies."
        metadata["history"] = f"Created by {self.commandline} on {iso_now}"

        metadata["title"] = (
            f"Calibrated, aligned, and resampled AUV sensor data from"
            f" {self.args.auv_name} mission {self.args.mission}"
        )
        metadata["summary"] = (
            f"Observational oceanographic data obtained from an Autonomous"
            f" Underwater Vehicle mission with measurements sampled at"
            f" {self.args.freq} intervals."
            f" Data processed using MBARI's auv-python software."
        )
        # Append location of original data files to summary
        matches = re.search(
            "(" + SUMMARY_SOURCE.replace("{}", r".+$") + ")",
            self.ds.attrs["summary"],
        )
        if matches:
            metadata["summary"] += (
                " "
                + matches.group(1)
                + ".  Processing log file: http://dods.mbari.org/opendap/data/auvctd/surveys/"
                + f"{self.args.mission.split('.')[0]}/netcdf/"
                + f"{self.args.auv_name}_{self.args.mission}_processing.log"
            )
            if self.args.auv_name.lower() == "i2map":
                # Append shortened location of original data files to title
                # Useful for I2Map data as it's in a YYYY/MM directory structure
                metadata["title"] += (
                    ", "
                    + "original data in /mbari/M3/master/i2MAP/: "
                    + matches.group(1)
                    .replace("Original log files copied from ", "")
                    .replace("/Volumes/M3/master/i2MAP/", "")
                )
        metadata["source"] = (
            f"MBARI Dorado-class AUV data produced from aligned data"
            f" with execution of '{self.commandline}' at {iso_now} on"
            f" host {gethostname()} using git commit {gitcommit} from"
            f" software at 'https://github.com/mbari-org/auv-python'"
        )

        return metadata

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
        self.df_o[f"{instr}_depth"] = self.ds[f"{instr}_depth"].to_pandas()
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
            f" Variable {instr}_depth from {in_fn}"
            f" median filtered with {mf_width} samples"
            f" and resampled with {aggregator} to {freq} intervals."
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

    def resample_variable(
        self, instr: str, variable: str, mf_width: int, freq: str
    ) -> None:
        self.df_o[variable] = self.ds[variable].to_pandas()
        self.df_o[f"{variable}_mf"] = (
            self.ds[variable]
            .rolling(**{instr + "_time": mf_width}, center=True)
            .median()
            .to_pandas()
        )
        # Resample to center of freq https://stackoverflow.com/a/69945592/1281657
        aggregator = ".mean() aggregator"
        self.logger.info(
            f"Resampling {variable} with frequency {freq} following {mf_width} point median filter "
        )
        self.df_r[variable] = (
            self.df_o[f"{variable}_mf"].shift(0.5, freq=freq).resample(freq).mean()
        )
        return aggregator

    def save_variable(
        self, variable: str, mf_width: int, freq: str, aggregator: str
    ) -> None:
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

    def resample_mission(
        self,
        nc_file: str,
        mf_width: int = MF_WIDTH,
        freq: str = FREQ,
        plot_seconds: float = PLOT_SECONDS,
    ) -> None:
        pd.options.plotting.backend = "matplotlib"
        self.ds = xr.open_dataset(nc_file)
        last_instr = ""
        for icount, (instr, variables) in enumerate(
            self.instruments_variables(nc_file).items()
        ):
            if icount == 0:
                self.df_o = pd.DataFrame()  # original dataframe
                self.df_r = pd.DataFrame()  # resampled dataframe
                aggregator = self.resample_coordinates(instr, mf_width, freq)
                self.save_coordinates(instr, mf_width, freq, aggregator)
                if self.args.plot:
                    self.plot_coordinates(instr, freq, plot_seconds)
            if instr != last_instr:
                # Start with new dataframes for each instrument
                self.df_o = pd.DataFrame()
                self.df_r = pd.DataFrame()
            for variable in variables:
                aggregator = self.resample_variable(instr, variable, mf_width, freq)
                self.save_variable(variable, mf_width, freq, aggregator)
                if self.args.plot:
                    self.plot_variable(instr, variable, freq, plot_seconds)
        self.resampled_nc.attrs = self.global_metadata()
        self.resampled_nc["time"].attrs = {
            "standard_name": "time",
            "long_name": "Time (UTC)",
        }
        out_fn = nc_file.replace("_align.nc", f"_{freq}.nc")
        self.resampled_nc.to_netcdf(path=out_fn, format="NETCDF3_CLASSIC")
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
