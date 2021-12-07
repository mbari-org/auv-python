#!/usr/bin/env python
"""
Resample variables from mission netCDF file to common time axis

Read all the record variables stored at original instrument sampling rate
from netCDF file and resample them to common time axis.
"""

__author__ = "Mike McCann"
__copyright__ = "Copyright 2021, Monterey Bay Aquarium Research Institute"

import argparse
from collections import defaultdict
import logging
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

from logs2netcdfs import BASE_PATH, MISSIONNETCDFS


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

    def instruments_variables(self) -> dict:
        """
        Return a dictionary of all the variables in the mission netCDF file,
        keyed by instrument name
        """
        self.logger.info(f"Reading variables from {nc_file} mission netCDF file")
        last_instr = None
        instr_vars = defaultdict(list)
        for variable in self.ds.keys():
            instr, _ = variable.split("_")
            if instr == "navigation":
                freq = "0.1S"
            elif instr == "gps" or instr == "depth":
                continue
            if instr != last_instr:
                instr_vars[instr].append(variable)
        return instr_vars

    def resample_coordinates(self, instr: str, mf_width: int, freq: str) -> None:
        # Original
        self.df_o[f"{instr}_depth"] = self.ds[f"{instr}_depth"].to_pandas()
        self.df_o[f"{instr}_latitude"] = self.ds[f"{instr}_latitude"].to_pandas()
        self.df_o[f"{instr}_longitude"] = self.ds[f"{instr}_longitude"].to_pandas()
        # Median Filtered
        self.df_o[f"{instr}_depth_mf"] = (
            self.ds[f"{instr}_depth"]
            .rolling(**{instr + "_time": mf_width}, center=True)
            .median()
            .to_pandas()
        )
        self.df_o[f"{instr}_latitude_mf"] = (
            self.ds[f"{instr}_latitude"]
            .rolling(**{instr + "_time": mf_width}, center=True)
            .median()
            .to_pandas()
        )
        self.df_o[f"{instr}_longitude_mf"] = (
            self.ds[f"{instr}_longitude"]
            .rolling(**{instr + "_time": mf_width}, center=True)
            .median()
            .to_pandas()
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
        self.resampled_nc["depth"] = self.df_r["depth"].to_xarray()
        self.resampled_nc["latitude"] = self.df_r["latitude"].to_xarray()
        self.resampled_nc["longitude"] = self.df_r["longitude"].to_xarray()
        self.resampled_nc["depth"].attrs = self.ds[f"{instr}_depth"].attrs
        self.resampled_nc["depth"].attrs["comment"] += (
            " Instrument"
            f" aligned _depth median filtered with {mf_width} samples"
            f" and resampled with {aggregator} to {freq} intervals."
        )
        self.resampled_nc["latitude"].attrs = self.ds[f"{instr}_latitude"].attrs
        self.resampled_nc["latitude"].attrs["comment"] += (
            " Instrument"
            f" aligned _latitude median filtered with {mf_width} samples"
            f" and resampled with {aggregator} to {freq} intervals."
        )
        self.resampled_nc["longitude"].attrs = self.ds[f"{instr}_longitude"].attrs
        self.resampled_nc["longitude"].attrs["comment"] += (
            " Instrument"
            f" aligned _longitude median filtered with {mf_width} samples"
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
        self.logger.info(f"Resampling {variable} with frequency {freq} into df_r")
        self.df_r[variable] = (
            self.df_o[f"{variable}_mf"].shift(0.5, freq=freq).resample(freq).mean()
        )
        return aggregator

    def save_variable(
        self, variable: str, mf_width: int, freq: str, aggregator: str
    ) -> None:
        self.resampled_nc[variable] = self.df_r[variable].to_xarray()
        self.resampled_nc[variable].attrs = self.ds[variable].attrs
        self.resampled_nc[variable].attrs["comment"] += (
            f" Calibrated {variable} median filtered with {mf_width} samples"
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
        ax.set_ylabel(
            f"{self.ds[variable].attrs['long_name']} ({self.ds[variable].attrs['units']})"
        )
        ax.legend(["Original", "Median Filtered", "Resampled"])
        ax.set_xlabel("Time")
        ax.set_title(f"{instr} {variable}")
        plt.show()

    def resample_mission(
        self,
        nc_file: str,
        mf_width: int = 3,
        freq: str = "1S",
        plot_seconds: float = 300,
    ) -> None:
        pd.options.plotting.backend = "matplotlib"
        self.ds = xr.open_dataset(nc_file)
        for icount, (instr, variables) in enumerate(
            self.instruments_variables().items()
        ):
            if icount == 0:
                self.df_o = pd.DataFrame()  # original dataframe
                self.df_r = pd.DataFrame()  # resampled dataframe
                aggregator = self.resample_coordinates(instr, mf_width, freq)
                self.save_coordinates(instr, mf_width, freq, aggregator)
                if self.args.plot:
                    self.plot_coordinates(instr, freq, plot_seconds)
            for variable in variables:
                aggregator = self.resample_variable(instr, variable, mf_width, freq)
                self.save_variable(variable, mf_width, freq, aggregator)
                if self.args.plot:
                    self.plot_variable(instr, variable, freq, plot_seconds)
        self.resampled_nc.to_netcdf(nc_file.replace("_align.nc", f"_{freq}.nc"))

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
            default=300,
            type=float,
            help="Plot sesonds of data",
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
    resamp.resample_mission(nc_file, plot_seconds=resamp.args.plot_seconds)
    print("Done")
