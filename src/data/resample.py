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

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

from datetime import timedelta
from logs2netcdfs import BASE_PATH, MISSIONNETCDFS
from process_i2map import TEST_LIST


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
        self.df_o[f"{instr}_latitude"] = self.ds[f"{instr}_depth"].to_pandas()
        self.df_o[f"{instr}_longitude"] = self.ds[f"{instr}_depth"].to_pandas()
        # Median Filetered
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
        self.df_r[f"{instr}_latitude"] = (
            self.df_o[f"{instr}_latitude_mf"]
            .shift(0.5, freq=freq)
            .resample(freq)
            .mean()
        )
        self.df_r[f"{instr}_longitude"] = (
            self.df_o[f"{instr}_longitude_mf"]
            .shift(0.5, freq=freq)
            .resample(freq)
            .mean()
        )
        self.df_r[f"{instr}_depth"] = (
            self.df_o[f"{instr}_depth_mf"].shift(0.5, freq=freq).resample(freq).mean()
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
        self.logger.info(f"Resampling {variable} with frequency {freq} into df_r")
        # Resample to center if freq https://stackoverflow.com/a/69945592/1281657
        self.df_r[variable] = (
            self.df_o[f"{variable}_mf"].shift(0.5, freq=freq).resample(freq).mean()
        )

    def plot_sampled(
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
        ax = df_op.plot.line(y=[f"{instr}_depth", f"{instr}_depth_mf"])
        df_rp.plot.line(
            y=[f"{instr}_depth"],
            ax=ax,
            marker="o",
            linestyle="dotted",
            markersize=2,
        )
        plt.show()

    def resample_mission(
        self,
        nc_file: str,
        mf_width: int = 3,
        freq: str = "1.0S",
        plot_seconds: float = 180,
    ) -> None:
        pd.options.plotting.backend = "matplotlib"
        self.ds = xr.open_dataset(nc_file)
        for instr, variables in self.instruments_variables().items():
            self.df_o = pd.DataFrame()  # original dataframe
            self.df_r = pd.DataFrame()  # resampled dataframe
            self.resample_coordinates(instr, mf_width, freq)
            for variable in variables:
                self.resample_variable(instr, variable, mf_width, freq)
                self.resampled_nc[variable] = self.df_r[variable].to_xarray()
                if self.args.plot:
                    self.plot_sampled(instr, variable, freq, plot_seconds)

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
    ##for mission in TEST_LIST:
    file_name = f"{resamp.args.auv_name}_{resamp.args.mission}_align.nc"
    nc_file = os.path.join(
        BASE_PATH,
        resamp.args.auv_name,
        MISSIONNETCDFS,
        resamp.args.mission,
        file_name,
    )
    resamp.resample_mission(nc_file)
    print("Done")
