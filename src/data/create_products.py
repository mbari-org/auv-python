#!/usr/bin/env python
"""
Create "quick look" plots and other products from processed data.

"""

__author__ = "Mike McCann"
__copyright__ = "Copyright 2023, Monterey Bay Aquarium Research Institute"

import argparse
import contextlib
import logging
import os
import re
import sys
import time
from pathlib import Path

import cmocean
import matplotlib  # noqa: ICN001
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import xarray as xr
from gulper import Gulper
from logs2netcdfs import BASE_PATH, MISSIONNETCDFS, AUV_NetCDF
from resample import AUVCTD_OPENDAP_BASE, FREQ
from scipy.interpolate import griddata

MISSIONODVS = "missionodvs"
MISSIONIMAGES = "missionimages"


class CreateProducts:
    logger = logging.getLogger(__name__)
    _handler = logging.StreamHandler()
    _handler.setFormatter(AUV_NetCDF._formatter)
    logger.addHandler(_handler)
    _log_levels = (logging.WARN, logging.INFO, logging.DEBUG)

    # Column name format required by ODV - will be tab delimited
    ODV_COLUMN_NAMES = [  # noqa: RUF012
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
    cmocean_lookup = {  # noqa: RUF012
        "sea_water_temperature": "thermal",
        "sea_water_salinity": "haline",
        "sea_water_sigma_t": "dense",
        "mass_concentration_of_chlorophyll_in_sea_water": "algae",
        "mass_concentration_of_oxygen_in_sea_water": "oxy",
        "downwelling_photosynthetic_photon_flux_in_sea_water": "solar",
        "surface_downwelling_shortwave_flux_in_air": "solar",
        "platform_pitch_angle": "balance",
        "platform_roll_angle": "balance",
        "northward_sea_water_velocity": "balance",
        "eastward_sea_water_velocity": "balance",
        "northward_wind": "balance",
        "eastward_wind": "balance",
    }

    def _open_ds(self):
        local_nc = Path(
            BASE_PATH,
            self.args.auv_name,
            MISSIONNETCDFS,
            self.args.mission,
            f"{self.args.auv_name}_{self.args.mission}_{FREQ}.nc",
        )
        # Requires mission to have been processed and archived to AUVCTD
        dap_url = os.path.join(  # noqa: PTH118
            AUVCTD_OPENDAP_BASE,
            "surveys",
            self.args.mission.split(".")[0],
            "netcdf",
            f"{self.args.auv_name}_{self.args.mission}_{FREQ}.nc",
        )
        try:
            self.ds = xr.open_dataset(dap_url)
        except OSError:
            self.logger.debug("%s not available yet", dap_url)
            self.ds = xr.open_dataset(local_nc)

    def _grid_dims(self) -> tuple:
        # From Matlab code in plot_sections.m:
        # auvnav positions are too fine for distance calculations, they resolve
        # spiral ascents and circling while on station
        # subsample, interpolate back, use subsampled - interpolated positions
        # for distance calculation
        # npos = length(auvnav.fixLat);
        # if npos > 400,
        # 	nSubSample = 200;			% Test plots show that about 200 removes the spirals
        # else
        #   nSubSample = 1;
        # end
        # fixLonSubSamp = auvnav.fixLon(1:nSubSample:npos);
        # fixLatSubSamp = auvnav.fixLat(1:nSubSample:npos);
        # fixLonSubIntrp = interp1(auvnav.fixTime(1:nSubSample:npos), fixLonSubSamp, auvnav.fixTime, 'linear', 'extrap');  # noqa: E501
        # fixLatSubIntrp = interp1(auvnav.fixTime(1:nSubSample:npos), fixLatSubSamp, auvnav.fixTime, 'linear', 'extrap');  # noqa: E501
        # [xFix yFix] = geo2utm(fixLonSubIntrp, fixLatSubIntrp);
        # dxFix = [0; diff(xFix - xFix(1))];
        # dyFix = [0; diff(yFix - yFix(1))];
        # distnav = cumsum(sqrt(dxFix.^2 + dyFix.^2));	% in m
        # dists = distnav / 1000; 	% in km

        utm_zone = int(31 + (self.ds.cf["longitude"].to_numpy().mean() // 6))
        MAX_LONGITUDE_VALUES = 400
        n_subsample = 200 if len(self.ds.cf["longitude"].to_numpy()) > MAX_LONGITUDE_VALUES else 1
        lon_sub_intrp = np.interp(
            self.ds.cf["time"].to_numpy().astype(np.int64),
            self.ds.cf["time"].to_numpy()[::n_subsample].astype(np.int64),
            self.ds.cf["longitude"].to_numpy()[::n_subsample],
        )
        lat_sub_intrp = np.interp(
            self.ds.cf["time"].to_numpy().astype(np.int64),
            self.ds.cf["time"].to_numpy()[::n_subsample].astype(np.int64),
            self.ds.cf["latitude"].to_numpy()[::n_subsample],
        )
        x, y = pyproj.Proj(proj="utm", zone=utm_zone, ellps="WGS84")(
            lon_sub_intrp,
            lat_sub_intrp,
        )
        dx = np.insert(np.diff(x - x[0]), 0, 0)
        dy = np.insert(np.diff(y - y[0]), 0, 0)
        distnav = xr.DataArray(
            np.cumsum(np.sqrt(dx**2 + dy**2)),
            dims=("time",),
            coords={"time": self.ds.cf["time"].to_numpy()},
            attrs={
                "long_name": "distance along track",
                "units": "m",
            },
        )

        # Horizontal gridded to 3x the number of profiles
        idist = np.linspace(
            distnav.to_numpy().min(),
            distnav.to_numpy().max(),
            3 * self.ds["profile_number"].to_numpy()[-1],
        )
        # Vertical gridded to .5 m
        iz = np.arange(2.0, self.ds.cf["depth"].max(), 0.5)
        if not iz.any():
            self.logger.warning(
                "Gridding vertical for a surface only mission: {self.ds.cf['depth'].max() =}",
            )
            iz = np.arange(0, self.ds.cf["depth"].max(), 0.05)

        return idist, iz, distnav

    def _profile_bottoms(
        self,
        distnav: xr.DataArray,
        window_frac: float = 0.01,
    ) -> np.array:
        """Return array of distance and depth points defining the bottom of the profiles
        where there is no data"""

        # Create a DataArray of depths indexed by distance
        depth_dist = xr.DataArray(
            self.ds.cf["depth"].to_numpy(),
            dims=("dist",),
            coords={"dist": distnav.to_numpy() / 1000.0},
        )
        # Set rolling window to fraction of the total distance of the mission
        window = int(len(distnav) * window_frac)
        return depth_dist.rolling(dist=window).max()

    def _plot_var(  # noqa: PLR0913
        self,
        var: str,
        idist: np.array,
        iz: np.array,
        distnav: np.array,
        fig: matplotlib.figure.Figure,
        ax: matplotlib.axes.Axes,
        row: int,
        col: int,
        profile_bottoms: xr.DataArray,
        scale: str = "linear",
        num_colors: int = 256,
    ):
        var_to_plot = (
            np.log10(self.ds[var].to_numpy()) if scale == "log" else self.ds[var].to_numpy()
        )
        scafac = max(idist) / max(iz)
        gridded_var = griddata(
            (distnav.to_numpy() / 1000.0 / scafac, self.ds.cf["depth"].to_numpy()),
            var_to_plot,
            ((idist / scafac / 1000.0)[None, :], iz[:, None]),
            method="linear",
            rescale=True,
        )
        color_map_name = "cividis"
        with contextlib.suppress(KeyError):
            color_map_name = self.cmocean_lookup.get(
                self.ds[var].attrs["standard_name"],
                "cividis",
            )
        try:
            cmap = plt.get_cmap(color_map_name)
        except ValueError:
            # Likely a cmocean colormap
            cmap = getattr(cmocean.cm, color_map_name)

        v2_5 = np.percentile(var_to_plot[~np.isnan(var_to_plot)], 2.5)
        v97_5 = np.percentile(var_to_plot[~np.isnan(var_to_plot)], 97.5)
        norm = matplotlib.colors.BoundaryNorm(
            np.linspace(v2_5, v97_5, num_colors),
            num_colors,
        )

        self.logger.info(
            "%s using %s cmap with ranges %.1f %.1f",
            var,
            color_map_name,
            v2_5,
            v97_5,
        )
        ax[row, col].set_ylim(max(iz), min(iz))
        cntrf = ax[row, col].contourf(
            idist / 1000.0,
            iz,
            gridded_var,
            cmap=cmap,
            norm=norm,
            extend="both",
            levels=np.linspace(v2_5, v97_5, num_colors),
        )
        # Blank out the countoured data below the bottom of the profiles
        xb = np.append(
            profile_bottoms.get_index("dist").to_numpy(),
            [
                ax[row, col].get_xlim()[1],
                ax[row, col].get_xlim()[1],
                ax[row, col].get_xlim()[0],
                ax[row, col].get_xlim()[0],
            ],
        )
        yb = np.append(
            profile_bottoms.to_numpy(),
            [
                profile_bottoms.to_numpy()[-1],
                ax[row][col].get_ylim()[0],
                ax[row, col].get_ylim()[0],
                profile_bottoms.to_numpy()[0],
            ],
        )
        ax[row][col].fill(list(reversed(xb)), list(reversed(yb)), "w")

        ax[row, col].set_ylabel("Depth (m)")
        cb = fig.colorbar(cntrf, ax=ax[row, col])
        cb.locator = matplotlib.ticker.LinearLocator(numticks=3)
        cb.minorticks_off()
        cb.update_ticks()
        cb.ax.set_yticklabels([f"{x:.1f}" for x in cb.get_ticks()])
        if scale == "log":
            cb.set_label(
                f"{self.ds[var].attrs['long_name']}\n[log10({self.ds[var].attrs['units']})]",
                fontsize=7,
            )
        else:
            cb.set_label(
                f"{self.ds[var].attrs['long_name']} [{self.ds[var].attrs['units']}]",
                fontsize=9,
            )

    def plot_2column(self) -> str:
        """Create 2column plot similar to plot_sections.m and stoqs/utils/Viz/plotting.py
        Construct a 2D grid of distance and depth and for each parameter grid the data
        to create a shaded plot in each subplot.
        """
        self._open_ds()

        idist, iz, distnav = self._grid_dims()
        scfac = max(idist) / max(iz)  # noqa: F841

        fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(18, 10))
        fig.tight_layout()

        best_ctd = self._get_best_ctd()
        profile_bottoms = self._profile_bottoms(distnav)
        row = 0
        col = 1
        for var, scale in (
            ("ctd1_oxygen_mll", "linear"),
            ("density", "linear"),
            ("hs2_bb420", "linear"),
            (f"{best_ctd}_temperature", "linear"),
            ("hs2_bb700", "linear"),
            (f"{best_ctd}_salinity", "linear"),
            ("hs2_fl700", "linear"),
            ("isus_nitrate", "linear"),
            ("biolume_avg_biolume", "log"),
        ):
            self.logger.info("Plotting %s...", var)
            if var not in self.ds:
                self.logger.warning("%s not in dataset", var)
                ax[row, col].get_xaxis().set_visible(False)
                ax[row, col].get_yaxis().set_visible(False)
            else:
                self._plot_var(
                    var,
                    idist,
                    iz,
                    distnav,
                    fig,
                    ax,
                    row,
                    col,
                    profile_bottoms,
                    scale=scale,
                )
            if row != 4:  # noqa: PLR2004
                ax[row, col].get_xaxis().set_visible(False)

            if col == 1:
                row += 1
                col = 0
            else:
                col = 1

        # Save plot to file
        images_dir = Path(BASE_PATH, self.args.auv_name, MISSIONIMAGES)
        Path(images_dir).mkdir(parents=True, exist_ok=True)

        plt.savefig(
            Path(
                images_dir,
                f"{self.args.auv_name}_{self.args.mission}_{FREQ}_2column.png",
            ),
        )

    def plot_biolume(self) -> str:
        "Create biolume plot"

    def _get_best_ctd(self) -> str:
        """Determine best CTD to use for ODV lookup table based on metadata"""
        best_ctd = "ctd1"  # default to ctd1 if no metadata
        if "comment" not in self.ds.attrs:
            self.logger.warning("No comment attribute in dataset")
            return best_ctd
        matches = re.search(r"Best CTD is (ctd1|ctd2)", self.ds.attrs["comment"])
        if matches:
            self.logger.info("Best CTD is %s", matches.group(1))
            best_ctd = matches.group(1)
        else:
            matches = re.search(r"ctdToUse = ([\S]+)", self.ds.attrs["comment"])
            if matches:
                self.logger.info("ctdToUse = %s", matches.group(1))
                best_ctd = matches.group(1)

        return best_ctd

    def gulper_odv(self, sec_bnds: int = 1) -> str:  # noqa: C901, PLR0912, PLR0915
        "Create gulper bottle numbers and data at sample collection (ODV tab-delimited) file"

        gulper = Gulper()
        gulper.args = argparse.Namespace()
        gulper.args.base_path = self.args.base_path
        gulper.args.auv_name = self.args.auv_name
        gulper.args.mission = self.args.mission
        gulper.args.local = self.args.local
        gulper.args.verbose = self.args.verbose
        gulper.args.start_esecs = self.args.start_esecs
        gulper.logger.setLevel(self._log_levels[self.args.verbose])
        gulper.logger.addHandler(self._handler)

        gulper_times = gulper.parse_gulpers()
        if not gulper_times:
            self.logger.info("No gulper times found for %s", self.args.mission)
            return
        odv_dir = Path(
            BASE_PATH,
            self.args.auv_name,
            MISSIONODVS,
            self.args.mission,
        )
        Path(odv_dir).mkdir(parents=True, exist_ok=True)
        gulper_odv_filename = Path(
            odv_dir,
            f"{self.args.auv_name}_{self.args.mission}_{FREQ}_Gulper.txt",
        )
        self._open_ds()

        # Replace red and blue backscatter variables with names used before 2012
        odv_column_names = self.ODV_COLUMN_NAMES.copy()
        if "hs2_bb470" in self.ds:
            odv_column_names[22] = "bbp470 [m^{-1}]"
        if "hs2_bb676" in self.ds:
            odv_column_names[24] = "bbp676 [m^{-1}]"

        best_ctd = self._get_best_ctd()
        with gulper_odv_filename.open("w") as f:
            f.write("\t".join(odv_column_names) + "\n")
            for bottle, esec in gulper_times.items():
                self.logger.debug("bottle: %d of %d", bottle, len(gulper_times))
                gulper_data = self.ds.sel(
                    time=slice(
                        np.datetime64(int((esec - sec_bnds) * 1e9), "ns"),
                        np.datetime64(int((esec + sec_bnds) * 1e9), "ns"),
                    ),
                )
                for count, name in enumerate(odv_column_names):
                    if name == "Cruise":
                        f.write(f"{self.args.auv_name}_{self.args.mission}_{FREQ}")
                    elif name == "Station":
                        f.write(f"{int(gulper_data['profile_number'].to_numpy().mean()):d}")
                    elif name == "Type":
                        f.write("B")
                    elif name == "mon/day/yr":
                        f.write(
                            f"{gulper_data.cf['T'][0].dt.month.to_numpy():02d}/"
                            f"{gulper_data.cf['T'][0].dt.day.to_numpy():02d}/"
                            f"{gulper_data.cf['T'][0].dt.year.to_numpy()}",
                        )
                    elif name == "hh:mm":
                        f.write(
                            f"{gulper_data.cf['T'][0].dt.hour.to_numpy():02d}:"
                            f"{gulper_data.cf['T'][0].dt.minute.to_numpy():02d}",
                        )
                    elif name == "Lon (degrees_east)":
                        f.write(
                            f"{gulper_data.cf['longitude'].to_numpy().mean() + 360.0:9.5f}",
                        )
                    elif name == "Lat (degrees_north)":
                        f.write(f"{gulper_data.cf['latitude'].to_numpy().mean():8.5f}")
                    elif name == "Bot. Depth [m]":
                        f.write(
                            f"{float(1000):8.1f}",
                        )  # TODO: add proper bottom depth values
                    elif name == "Bottle Number [count]":
                        f.write(f"{bottle}")
                    elif name == "QF":
                        f.write("0")  # TODO: add proper quality flag values
                    elif name == "DEPTH [m]":
                        f.write(f"{gulper_data.cf['depth'].to_numpy().mean():6.2f}")
                    elif name == "TEMPERATURE [°C]":
                        temp = gulper_data[f"{best_ctd}_temperature"].to_numpy().mean()
                        f.write(f"{temp:5.2f}")
                    elif name == "SALINITY [PSS78]":
                        sal = gulper_data[f"{best_ctd}_salinity"].to_numpy().mean()
                        f.write(f"{sal:6.3f}")
                    elif name == "Oxygen [ml/l]":
                        f.write(f"{gulper_data['ctd1_oxygen_mll'].to_numpy().mean():5.3f}")
                    elif name == "NITRATE [µmol/kg]":
                        if "isus_nitrate" in gulper_data:
                            no3 = gulper_data["isus_nitrate"].dropna(dim="time").to_numpy()
                            if no3.any():
                                f.write(f"{no3.mean():6.3f}")
                            else:
                                f.write("NaN")
                        else:
                            f.write("NaN")
                    elif name == "ChlFluor [raw]":
                        if "hs2_fl700" in gulper_data:
                            f.write(f"{gulper_data['hs2_fl700'].to_numpy().mean():11.8f}")
                        elif "hs2_fl676" in gulper_data:
                            f.write(f"{gulper_data['hs2_fl676'].to_numpy().mean():.8f}")
                        else:
                            f.write("NaN")
                    elif name == "bbp420 [m^{-1}]":
                        if "hs2_bb420" in gulper_data:
                            f.write(f"{gulper_data['hs2_bb420'].to_numpy().mean():8.7f}")
                        else:
                            f.write("NaN")
                    elif name == "bbp470 [m^{-1}]":
                        f.write(f"{gulper_data['hs2_bb470'].to_numpy().mean():8.7f}")
                    elif name == "bbp700 [m^{-1}]":
                        if "hs2_bb700" in gulper_data:
                            f.write(f"{gulper_data['hs2_bb700'].to_numpy().mean():8.7f}")
                        else:
                            f.write("NaN")
                    elif name == "bbp676 [m^{-1}]":
                        f.write(f"{gulper_data['hs2_bb676'].to_numpy().mean():8.7f}")
                    elif name == "PAR [V]":
                        if "ctd2_par" in gulper_data:
                            f.write(f"{gulper_data['ctd2_par'].to_numpy().mean():6.3f}")
                        else:
                            f.write("NaN")
                    elif name == "YearDay [day]":
                        fractional_ns = gulper_data.cf["T"][0] - gulper_data.cf["T"][0].dt.floor(
                            "D"
                        )
                        fractional_day = float(fractional_ns) / 86400000000000.0
                        f.write(
                            f"{gulper_data.cf['T'][0].dt.dayofyear.to_numpy() + fractional_day:9.5f}",  # noqa: E501
                        )
                    if count < len(odv_column_names) - 1:
                        f.write("\t")
                f.write("\n")
        self.logger.info(
            "Wrote %d Gulper data lines to %s",
            len(gulper_times),
            gulper_odv_filename,
        )

    def process_command_line(self):
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter,
            description=__doc__,
        )
        (
            parser.add_argument(
                "--base_path",
                action="store",
                default=BASE_PATH,
                help=f"Base directory for missionlogs and missionnetcdfs, default: {BASE_PATH}",
            ),
        )
        parser.add_argument(
            "--auv_name",
            action="store",
            default="dorado",
            help="dorado (default), i2map",
        )
        (
            parser.add_argument(
                "--mission",
                action="store",
                help="Mission directory, e.g.: 2020.064.10",
            ),
        )
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
                [f"{i}: {v}" for i, v in enumerate(("WARN", "INFO", "DEBUG"))],
            ),
        )
        self.args = parser.parse_args()
        self.logger.setLevel(self._log_levels[self.args.verbose])
        self.commandline = " ".join(sys.argv)


if __name__ == "__main__":
    cp = CreateProducts()
    cp.process_command_line()
    p_start = time.time()
    cp.plot_2column()
    cp.plot_biolume()
    # cp.gulper_odv()
    cp.logger.info("Time to process: %.2f seconds", (time.time() - p_start))
