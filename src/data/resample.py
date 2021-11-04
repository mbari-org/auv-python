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

import xarray as xr
from scipy.signal import medfilt

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

    def resample_variables(self, nc_file):
        ds = xr.open_dataset(nc_file)
        for variable in ds.variables:
            print(f"Resampling {variable}")
            ds["smooth_lat"] = (
                ds["navigation_latitude"]
                .rolling(navigation_time=5, center=True)
                .median()
            )
            lat_mf = xr.apply_ufunc(
                medfilt,
                ds.navigation_latitude,
                kwargs={"kernel_size": 3},
                input_core_dims=[["navigation_time"]],
                vectorize=True,
            )
            pass

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
    file_name = f"{resamp.args.auv_name}_{resamp.args.mission}.nc"
    nc_file = os.path.join(
        BASE_PATH,
        resamp.args.auv_name,
        MISSIONNETCDFS,
        resamp.args.mission,
        file_name,
    )
    resamp.resample_variables(nc_file)
    print("Done")
