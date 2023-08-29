#!/usr/bin/env python
"""
Create "quick look" plots and other products from processed data.

"""

__author__ = "Mike McCann"
__copyright__ = "Copyright 2023, Monterey Bay Aquarium Research Institute"

import argparse
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import xarray as xr
from gulper import Gulper
from logs2netcdfs import BASE_PATH, MISSIONNETCDFS, AUV_NetCDF
from resample import AUVCTD_OPENDAP_BASE, FREQ

MISSIONODVS = "missionodvs"


class CreateProducts:
    logger = logging.getLogger(__name__)
    _handler = logging.StreamHandler()
    _handler.setFormatter(AUV_NetCDF._formatter)
    logger.addHandler(_handler)
    _log_levels = (logging.WARN, logging.INFO, logging.DEBUG)

    # Column name format required by ODV - will be tab delimited
    odv_column_names = [
        "Cruise",
        "Station",
        "Type",
        "mon/day/yr",
        "hh:mm",
        "Lon (degrees_east)",
        "Lat (degrees_north)",
        "Bot. Depth [m]",
        "Bottle Number [count]",
        "QF",
        "DEPTH [m]",
        "QF",
        "TEMPERATURE [°C]",
        "QF",
        "SALINITY [PSS78]",
        "QF",
        "Oxygen [ml/l]",
        "QF",
        "NITRATE [µmol/kg]",
        "QF",
        "ChlFluor [raw]",
        "QF",
        "bbp420 [m^{-1}]",
        "QF",
        "bbp700 [m^{-1}]",
        "QF",
        "PAR [V]",
        "QF",
        "YearDay [day]",
        "QF",
    ]

    def _open_ds(self):
        if self.args.local:
            self.ds = xr.open_dataset(
                os.path.join(
                    BASE_PATH,
                    self.args.auv_name,
                    MISSIONNETCDFS,
                    self.args.mission,
                    f"{self.args.auv_name}_{self.args.mission}_{FREQ}.nc",
                )
            )
        else:
            # Requires mission to have been processed and archived to AUVCTD
            self.ds = xr.open_dataset(
                os.path.join(
                    AUVCTD_OPENDAP_BASE,
                    "surveys",
                    self.args.mission.split(".")[0],
                    "netcdf",
                    f"{self.args.auv_name}_{self.args.mission}_{FREQ}.nc",
                )
            )

    def plot_biolume(self) -> str:
        "Create biolume plot"
        pass

    def plot_2column(self) -> str:
        "Create 2column plot"
        pass

    def _get_best_ctd(self) -> str:
        """Determine best CTD to use for ODV lookup table based on metadata"""
        best_ctd = "ctd1"  # default to ctd1 if no metadata
        if "comment" not in self.ds.attrs:
            self.logger.warning("No comment attribute in dataset")
            return best_ctd
        matches = re.search(r"Best CTD is ([\S]+)", self.ds.attrs["comment"])
        if matches:
            self.logger.info(f"Best CTD is {matches.group(1)}")
            best_ctd = matches.group(1)
        else:
            matches = re.search(r"ctdToUse = ([\S]+)", self.ds.attrs["comment"])
            if matches:
                self.logger.info(f"ctdToUse = {matches.group(1)}")
                best_ctd = matches.group(1)

        return best_ctd

    def gulper_odv(self, sec_bnds: int = 1) -> str:
        "Create gulper bottle numbers and data at sample collection (ODV tab-delimited) file"

        gulper = Gulper()
        gulper.args = argparse.Namespace()
        gulper.args.auv_name = "dorado"
        gulper.args.mission = self.args.mission
        gulper.args.local = self.args.local
        gulper.args.verbose = self.args.verbose
        gulper.args.start_esecs = self.args.start_esecs
        gulper.logger.setLevel(self._log_levels[self.args.verbose])
        gulper.logger.addHandler(self._handler)

        gulper_times = gulper.parse_gulpers()
        if not gulper_times:
            self.logger.info(f"No gulper times found for {self.args.mission}")
            return
        odv_dir = os.path.join(
            BASE_PATH, self.args.auv_name, MISSIONODVS, self.args.mission
        )
        Path(odv_dir).mkdir(parents=True, exist_ok=True)
        gulper_odv_filename = os.path.join(
            odv_dir,
            f"{self.args.auv_name}_{self.args.mission}_{FREQ}_Gulper.txt",
        )
        self._open_ds()

        # Replace red and blue backscatter variables with names used before 2012
        if "hs2_bb470" in self.ds:
            self.odv_column_names[22] = "bbp470 [m^{-1}]"
        if "hs2_bb676" in self.ds:
            self.odv_column_names[24] = "bbp676 [m^{-1}]"

        best_ctd = self._get_best_ctd()
        with open(gulper_odv_filename, "w") as f:
            f.write("\t".join(self.odv_column_names) + "\n")
            for bottle, esec in gulper_times.items():
                self.logger.debug(f"bottle: {bottle} of {len(gulper_times)}")
                gulper_data = self.ds.sel(
                    time=slice(
                        datetime.utcfromtimestamp(esec - sec_bnds),
                        datetime.utcfromtimestamp(esec + sec_bnds),
                    )
                )
                for count, name in enumerate(self.odv_column_names):
                    if name == "Cruise":
                        f.write(f"{self.args.auv_name}_{self.args.mission}_{FREQ}")
                    elif name == "Station":
                        f.write(f'{int(gulper_data["profile_number"].values.mean()):d}')
                    elif name == "Type":
                        f.write("B")
                    elif name == "mon/day/yr":
                        f.write(
                            f'{gulper_data.cf["T"][0].dt.month.values:02d}/'
                            f'{gulper_data.cf["T"][0].dt.day.values:02d}/'
                            f'{gulper_data.cf["T"][0].dt.year.values}'
                        )
                    elif name == "hh:mm":
                        f.write(
                            f'{gulper_data.cf["T"][0].dt.hour.values:02d}:'
                            f'{gulper_data.cf["T"][0].dt.minute.values:02d}'
                        )
                    elif name == "Lon (degrees_east)":
                        f.write(
                            f'{gulper_data.cf["longitude"].values.mean() + 360.0:9.5f}'
                        )
                    elif name == "Lat (degrees_north)":
                        f.write(f'{gulper_data.cf["latitude"].values.mean():8.5f}')
                    elif name == "Bot. Depth [m]":
                        f.write(
                            f"{float(1000):8.1f}"
                        )  # TODO: add proper bottom depth values
                    elif name == "Bottle Number [count]":
                        f.write(f"{bottle}")
                    elif name == "QF":
                        f.write("0")  # TODO: add proper quality flag values
                    elif name == "DEPTH [m]":
                        f.write(f'{gulper_data.cf["depth"].values.mean():6.2f}')
                    elif name == "TEMPERATURE [°C]":
                        temp = gulper_data[f"{best_ctd}_temperature"].values.mean()
                        f.write(f"{temp:5.2f}")
                    elif name == "SALINITY [PSS78]":
                        sal = gulper_data[f"{best_ctd}_salinity"].values.mean()
                        f.write(f"{sal:6.3f}")
                    elif name == "Oxygen [ml/l]":
                        f.write(f'{gulper_data["ctd1_oxygen_mll"].values.mean():5.3f}')
                    elif name == "NITRATE [µmol/kg]":
                        if "isus_nitrate" in gulper_data:
                            no3 = gulper_data["isus_nitrate"].dropna(dim="time").values
                            if no3.any():
                                f.write(f"{no3.mean():6.3f}")
                            else:
                                f.write("NaN")
                        else:
                            f.write("NaN")
                    elif name == "ChlFluor [raw]":
                        if "hs2_fl700" in gulper_data:
                            f.write(f'{gulper_data["hs2_fl700"].values.mean():11.8f}')
                        elif "hs2_fl676" in gulper_data:
                            f.write(f'{gulper_data["hs2_fl676"].values.mean():.8f}')
                    elif name == "bbp420 [m^{-1}]":
                        f.write(f'{gulper_data["hs2_bb420"].values.mean():8.7f}')
                    elif name == "bbp470 [m^{-1}]":
                        f.write(f'{gulper_data["hs2_bb470"].values.mean():8.7f}')
                    elif name == "bbp700 [m^{-1}]":
                        f.write(f'{gulper_data["hs2_bb700"].values.mean():8.7f}')
                    elif name == "bbp676 [m^{-1}]":
                        f.write(f'{gulper_data["hs2_bb676"].values.mean():8.7f}')
                    elif name == "PAR [V]":
                        if "ctd2_par" in gulper_data:
                            f.write(f'{gulper_data["ctd2_par"].values.mean():6.3f}')
                        else:
                            f.write("NaN")
                    elif name == "YearDay [day]":
                        fractional_ns = gulper_data.cf["T"][0] - gulper_data.cf["T"][
                            0
                        ].dt.floor("D")
                        fractional_day = float(fractional_ns) / 86400000000000.0
                        f.write(
                            f'{gulper_data.cf["T"][0].dt.dayofyear.values + fractional_day:9.5f}'
                        )
                    if count < len(self.odv_column_names) - 1:
                        f.write("\t")
                f.write("\n")
        self.logger.info(
            f"Wrote {len(gulper_times)} Gulper data lines to {gulper_odv_filename}"
        )

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
            f" missionnetcdfs, default: {BASE_PATH}",
        ),
        parser.add_argument(
            "--auv_name",
            action="store",
            default="dorado",
            help="dorado (default), i2map",
        )
        parser.add_argument(
            "--mission",
            action="store",
            help="Mission directory, e.g.: 2020.064.10",
        ),
        parser.add_argument(
            "--start_esecs",
            help="Start time of mission in epoch seconds, optional for gulper time lookup",
            type=float,
        )
        parser.add_argument("--local", help="Read local files", action="store_true")
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
    cp = CreateProducts()
    cp.process_command_line()
    p_start = time.time()
    cp.gulper_odv()
    cp.logger.info(f"Time to process: {(time.time() - p_start):.2f} seconds")
