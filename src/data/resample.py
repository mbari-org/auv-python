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

    def resample_variables(
        self,
        nc_file,
        mf_width=3,
        freq="1.0S",
        plot_seconds=60,
    ):
        pd.options.plotting.backend = "matplotlib"
        ds = xr.open_dataset(nc_file)
        last_instr = None
        for variable in ds.keys():
            instr, _ = variable.split("_")
            if instr == "navigation":
                freq = "0.1S"
            elif instr == "gps" or instr == "depth":
                continue
            if instr != last_instr:
                # New instrument, so create a new dataframes
                df_o = pd.DataFrame()  # original dataframe
                df_r = pd.DataFrame()  # resampled dataframe
            print(f"Median filtering {variable} with width {mf_width}")
            df_o[variable] = ds[variable].to_pandas()
            df_o[f"{variable}_mf"] = (
                ds[variable]
                .rolling(**{instr + "_time": mf_width}, center=True)
                .median()
                .to_pandas()
            )
            # Convert all coordinate variables to Pandas Series and resample
            print(f"Resampling {variable} with frequency {freq}")
            df_r[f"{instr}_latitude"] = (
                ds[instr + "_latitude"].to_pandas().resample(freq).mean()
            )
            df_r[f"{instr}_longitude"] = (
                ds[instr + "_longitude"].to_pandas().resample(freq).mean()
            )
            df_r[f"{instr}_depth"] = (
                ds[instr + "_depth"].to_pandas().resample(freq).mean()
            )
            df_r[f"{variable}_rs"] = ds[variable].to_pandas().resample(freq).mean()

            # Use sample rate to get indices for plotting plot_seconds of data
            o_end = int(ds[variable].attrs["instrument_sample_rate_hz"] * plot_seconds)
            r_end = int(plot_seconds / float(freq[:-1]))
            df_op = df_o.iloc[:o_end]
            df_rp = df_r.iloc[:r_end]
            # Different freqs on same axes - https://stackoverflow.com/a/13873014/1281657
            ax = df_op.plot.line(y=[variable, f"{variable}_mf"])
            df_rp.plot.line(
                y=[f"{variable}_rs"],
                ax=ax,
                marker="o",
                linestyle="dotted",
                markersize=2,
            )

            plt.show()
            last_instr = instr

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
    resamp.resample_variables(nc_file)
    print("Done")
