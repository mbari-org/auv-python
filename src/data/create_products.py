#!/usr/bin/env python
"""
Create "quick look" plots and other products from processed data.

"""

__author__ = "Mike McCann"
__copyright__ = "Copyright 2023, Monterey Bay Aquarium Research Institute"

import argparse  # noqa: I001
import contextlib
import logging
import os
import re
import sys
import time
from pathlib import Path

import cmocean
import gsw
import matplotlib  # noqa: ICN001
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import xarray as xr

from common_args import DEFAULT_BASE_PATH, get_standard_dorado_parser
from gulper import Gulper
from logs2netcdfs import AUV_NetCDF, MISSIONNETCDFS
from resample import AUVCTD_OPENDAP_BASE, FREQ
from scipy.interpolate import griddata

# Define BASE_PATH for backward compatibility
BASE_PATH = DEFAULT_BASE_PATH

MISSIONODVS = "missionodvs"
MISSIONIMAGES = "missionimages"


class CreateProducts:
    logger = logging.getLogger(__name__)
    _handler = logging.StreamHandler()
    _handler.setFormatter(AUV_NetCDF._formatter)
    logger.addHandler(_handler)
    _log_levels = (logging.WARN, logging.INFO, logging.DEBUG)

    def __init__(  # noqa: PLR0913
        self,
        auv_name: str = None,
        mission: str = None,
        base_path: str = str(BASE_PATH),
        start_esecs: float = None,
        local: bool = False,  # noqa: FBT001, FBT002
        verbose: int = 0,
        commandline: str = "",
    ):
        """Initialize CreateProducts with explicit parameters.

        Args:
            auv_name: Name of the AUV vehicle
            mission: Mission identifier
            base_path: Base path for output files
            start_esecs: Start epoch seconds for processing
            local: Local processing flag
            verbose: Verbosity level (0-2)
            commandline: Command line string for tracking
        """
        self.auv_name = auv_name
        self.mission = mission
        self.base_path = base_path
        self.start_esecs = start_esecs
        self.local = local
        self.verbose = verbose
        self.commandline = commandline

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
        "sea_water_density": "dense",
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
            self.auv_name,
            MISSIONNETCDFS,
            self.mission,
            f"{self.auv_name}_{self.mission}_{FREQ}.nc",
        )
        # Requires mission to have been processed and archived to AUVCTD
        dap_url = os.path.join(  # noqa: PTH118
            AUVCTD_OPENDAP_BASE,
            "surveys",
            self.mission.split(".")[0],
            "netcdf",
            f"{self.auv_name}_{self.mission}_{FREQ}.nc",
        )
        try:
            self.ds = xr.open_dataset(dap_url)
        except OSError:
            self.logger.debug("%s not available yet", dap_url)
            self.ds = xr.open_dataset(local_nc)

    def _compute_density(self, best_ctd: str = "ctd1") -> None:
        """Compute sigma-t density from temperature and salinity using EOS-80.

        Args:
            best_ctd: The CTD instrument to use for temperature and salinity
        """
        if "density" in self.ds:
            self.logger.debug("density already exists in dataset")
            return

        temp_var = f"{best_ctd}_temperature"
        sal_var = f"{best_ctd}_salinity"

        if temp_var not in self.ds or sal_var not in self.ds:
            self.logger.warning(
                "Cannot compute density: %s or %s not in dataset",
                temp_var,
                sal_var,
            )
            return

        # Get temperature, salinity, and pressure (approximated from depth)
        temp = self.ds[temp_var].to_numpy()
        sal = self.ds[sal_var].to_numpy()

        # Compute sigma-t using gsw (uses EOS-80 formulation)
        # sigma-t is density - 1000 kg/m³
        density = gsw.density.sigma0(sal, temp)

        # Add to dataset
        self.ds["density"] = xr.DataArray(
            density,
            dims=self.ds[temp_var].dims,
            coords=self.ds[temp_var].coords,
            attrs={
                "long_name": "Sigma-t",
                "standard_name": "sea_water_density",
                "units": "kg/m^3",
                "comment": f"Computed from {temp_var} and {sal_var} using gsw.density.sigma0",
            },
        )
        self.logger.info("Computed density (sigma-t) from %s and %s", temp_var, sal_var)

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
        # Vertical gridded to .5 m, rounded down to nearest 50m
        max_depth = np.floor(self.ds.cf["depth"].max() / 50) * 50
        iz = np.arange(2.0, max_depth, 0.5)
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

    def _plot_track_map(
        self, map_ax: matplotlib.axes.Axes, reference_ax: matplotlib.axes.Axes
    ) -> None:
        """Plot AUV track map on left side with title and times on right.

        Args:
            map_ax: The axes object to plot the map on
            reference_ax: The axes below to align with (for left edge)
        """
        # Get lat/lon data
        lons = self.ds.cf["longitude"].to_numpy()
        lats = self.ds.cf["latitude"].to_numpy()

        # Get time data for start/end
        times = self.ds.cf["time"].to_numpy()
        start_time = pd.to_datetime(times[0]).strftime("%Y-%m-%d %H:%M:%S")
        end_time = pd.to_datetime(times[-1]).strftime("%Y-%m-%d %H:%M:%S")

        # Get title from netCDF attributes
        title = self.ds.attrs.get("title", f"{self.auv_name} {self.mission}")

        # Get the position of the reference axes below to align with
        ref_pos = reference_ax.get_position()

        # Store original position
        pos = map_ax.get_position()

        # Make the plot square by using equal aspect (do this first)
        map_ax.set_aspect("equal", adjustable="datalim")

        # Plot the track with depth coloring
        depths = self.ds.cf["depth"].to_numpy()
        map_ax.scatter(
            lons,
            lats,
            c=depths,
            cmap="viridis_r",  # Reversed so shallow is yellow, deep is dark
            s=1,
            alpha=0.6,
        )

        # Add start and end markers
        map_ax.plot(lons[0], lats[0], "go", markersize=8, label="Start", zorder=5)
        map_ax.plot(lons[-1], lats[-1], "r^", markersize=8, label="End", zorder=5)

        # Set fixed axis limits for Monterey Bay area
        map_ax.set_xlim([-122.41, -121.77])
        map_ax.set_ylim([36.5, 37.0])

        # Now position map aligned with left edge of reference, 50% width
        # Use a square aspect ratio based on the y-dimension
        map_height = pos.height
        aspect_ratio = (37.0 - 36.5) / (122.41 - 121.77)  # data aspect ratio
        map_width = map_height / aspect_ratio * 0.7  # scale to fit nicely

        map_ax.set_position([ref_pos.x0, pos.y0, map_width, map_height])

        # Remove axes, labels, and ticks but keep the border
        map_ax.set_xticks([])
        map_ax.set_yticks([])
        map_ax.set_xlabel("")
        map_ax.set_ylabel("")

        # Add a border around the map
        for spine in map_ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1)

        # Add legend
        map_ax.legend(loc="upper right", fontsize=7, framealpha=0.9)

        # Add text on the right side with title and times
        # Wrap title for better formatting
        import textwrap

        wrapped_title = textwrap.fill(title, width=40)

        text_content = f"{wrapped_title}\n\nStart: {start_time}\nEnd:   {end_time}"

        # Get updated position after aspect adjustment
        updated_pos = map_ax.get_position()

        # Position text in figure coordinates, to the right of the map
        text_x = updated_pos.x0 + updated_pos.width + 0.01
        text_y = updated_pos.y0 + updated_pos.height * 0.5

        map_ax.figure.text(
            text_x,
            text_y,
            text_content,
            fontsize=11,
            fontweight="bold",
            family="sans-serif",
            verticalalignment="center",
            horizontalalignment="left",
        )

    def _plot_var(  # noqa: C901, PLR0912, PLR0913, PLR0915
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
        best_ctd: str = "ctd1",
    ):
        # Handle both 1D and 2D axis arrays
        curr_ax = ax[row] if col == 0 and hasattr(ax, "ndim") and ax.ndim == 1 else ax[row, col]

        # Check if variable exists and has valid data
        no_data = False
        if var not in self.ds:
            self.logger.warning("%s not in dataset", var)
            no_data = True
        else:
            var_to_plot = (
                np.log10(self.ds[var].to_numpy()) if scale == "log" else self.ds[var].to_numpy()
            )
            valid_data = var_to_plot[~np.isnan(var_to_plot)]
            if len(valid_data) == 0:
                self.logger.warning("%s has no valid data", var)
                no_data = True

        # If no data, set up minimal axes and return early
        if no_data:
            curr_ax.set_xlim([min(idist) / 1000.0, max(idist) / 1000.0])
            curr_ax.set_ylim([max(iz), min(iz)])

            # Only show y-label on left column or top of right column
            if col == 0 or (col == 1 and row == 0):
                curr_ax.set_ylabel("Depth (m)")
            else:
                curr_ax.set_ylabel("")

            # Set y-axis ticks at 0, 50, 100, 150, etc.
            y_min = max(iz)
            y_ticks = np.arange(0, int(y_min) + 50, 50)
            curr_ax.set_yticks(y_ticks)

            # Add "No Data" text
            curr_ax.text(
                0.5,
                0.5,
                "No Data",
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                transform=curr_ax.transAxes,
            )
            return

        # Normal plotting path - we have valid data
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

        v2_5 = np.percentile(valid_data, 2.5)
        v97_5 = np.percentile(valid_data, 97.5)

        # Check for invalid percentiles
        if np.isnan(v2_5) or np.isnan(v97_5) or v2_5 == v97_5:
            self.logger.warning(
                "%s has invalid range (v2.5=%.2f, v97.5=%.2f)",
                var,
                v2_5,
                v97_5,
            )
            # Set up minimal axes and return early
            curr_ax.set_xlim([min(idist) / 1000.0, max(idist) / 1000.0])
            curr_ax.set_ylim([max(iz), min(iz)])

            # Only show y-label on left column or top of right column
            if col == 0 or (col == 1 and row == 0):
                curr_ax.set_ylabel("Depth (m)")
            else:
                curr_ax.set_ylabel("")

            # Set y-axis ticks at 0, 50, 100, 150, etc.
            y_min = max(iz)
            y_ticks = np.arange(0, int(y_min) + 50, 50)
            curr_ax.set_yticks(y_ticks)

            # Add "No Data" text
            curr_ax.text(
                0.5,
                0.5,
                "No Data",
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                transform=curr_ax.transAxes,
            )
            return

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
        curr_ax.set_ylim(max(iz), min(iz))
        cntrf = curr_ax.contourf(
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
                curr_ax.get_xlim()[1],
                curr_ax.get_xlim()[1],
                curr_ax.get_xlim()[0],
                curr_ax.get_xlim()[0],
            ],
        )
        yb = np.append(
            profile_bottoms.to_numpy(),
            [
                profile_bottoms.to_numpy()[-1],
                curr_ax.get_ylim()[0],
                curr_ax.get_ylim()[0],
                profile_bottoms.to_numpy()[0],
            ],
        )
        curr_ax.fill(list(reversed(xb)), list(reversed(yb)), "w")

        # Only show y-label on left column or top of right column
        if col == 0 or (col == 1 and row == 0):
            curr_ax.set_ylabel("Depth (m)")
        else:
            curr_ax.set_ylabel("")

        # Set y-axis ticks at 0, 50, 100, 150, etc.
        y_min, y_max = curr_ax.get_ylim()
        # Since y-axis is inverted (max at bottom), y_min is the deeper value
        y_ticks = np.arange(0, int(y_min) + 50, 50)
        curr_ax.set_yticks(y_ticks)

        cb = fig.colorbar(cntrf, ax=curr_ax)
        cb.locator = matplotlib.ticker.LinearLocator(numticks=3)
        cb.minorticks_off()
        cb.update_ticks()
        cb.ax.set_yticklabels([f"{x:.1f}" for x in cb.get_ticks()])

        # Get long_name and units with fallbacks
        long_name = self.ds[var].attrs.get("long_name", var)
        units = self.ds[var].attrs.get("units", "")

        if scale == "log" and units:
            cb.set_label(f"{long_name}\n[log10({units})]", fontsize=7)
        elif scale == "log":
            cb.set_label(f"{long_name}\n[log10]", fontsize=7)
        elif units:
            cb.set_label(f"{long_name} [{units}]", fontsize=9)
        else:
            cb.set_label(long_name, fontsize=9)

    def plot_2column(self) -> str:
        """Create 2column plot similar to plot_sections.m and stoqs/utils/Viz/plotting.py
        Construct a 2D grid of distance and depth and for each parameter grid the data
        to create a shaded plot in each subplot.
        """
        self._open_ds()

        idist, iz, distnav = self._grid_dims()
        scfac = max(idist) / max(iz)  # noqa: F841

        fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(18, 10))
        plt.subplots_adjust(hspace=0.15, wspace=0.1, left=0.05, right=0.98, top=0.96, bottom=0.03)

        best_ctd = self._get_best_ctd()

        # Compute density (sigma-t) if not already present
        self._compute_density(best_ctd)

        # Create map in top-left subplot (row=0, col=0), aligned with ax[1,0] below
        self._plot_track_map(ax[0, 0], ax[1, 0])

        profile_bottoms = self._profile_bottoms(distnav)
        row = 1  # Start at row 1, col 0 (below the map)
        col = 0
        for var, scale in (
            ("density", "linear"),
            (f"{best_ctd}_temperature", "linear"),
            (f"{best_ctd}_salinity", "linear"),
            ("isus_nitrate", "linear"),
            ("ctd1_oxygen_mll", "linear"),
            ("hs2_bbp420", "linear"),
            ("hs2_bbp700", "linear"),
            ("hs2_fl700", "linear"),
            ("biolume_avg_biolume", "linear"),
        ):
            self.logger.info("Plotting %s...", var)
            if var not in self.ds:
                self.logger.warning("%s not in dataset, plotting with no data", var)

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
                best_ctd=best_ctd,
            )
            if row != 4:  # noqa: PLR2004
                ax[row, col].get_xaxis().set_visible(False)
            else:
                # Add x-axis label only to bottom row
                ax[row, col].set_xlabel("Distance (km)")

            # Column-major order: fill down first column, then second column
            if row == 4 and col == 0:  # noqa: PLR2004
                # Finished first column, move to top of second column
                row = 0
                col = 1
            else:
                # Move down in current column
                row += 1

        # Save plot to file
        images_dir = Path(BASE_PATH, self.auv_name, MISSIONIMAGES, self.mission)
        Path(images_dir).mkdir(parents=True, exist_ok=True)

        output_file = Path(
            images_dir,
            f"{self.auv_name}_{self.mission}_{FREQ}_2column.png",
        )
        plt.savefig(output_file, dpi=100, bbox_inches="tight")
        plt.show()
        plt.close(fig)

        self.logger.info("Saved 2column plot to %s", output_file)
        return str(output_file)

    def plot_biolume(self) -> str:
        """Create bioluminescence plot showing raw signal and proxy variables"""
        self._open_ds()

        # Check if biolume variables exist
        biolume_vars = [v for v in self.ds.variables if v.startswith("biolume_")]
        if not biolume_vars:
            self.logger.warning("No biolume variables found in dataset")
            return None

        idist, iz, distnav = self._grid_dims()
        profile_bottoms = self._profile_bottoms(distnav)

        # Create figure with subplots for biolume variables
        num_plots = min(len(biolume_vars), 6)  # Limit to 6 most important variables
        fig, ax = plt.subplots(nrows=num_plots, ncols=1, figsize=(18, 12))
        if num_plots == 1:
            ax = [ax]
        fig.tight_layout(rect=[0, 0.03, 1, 0.96])

        # Priority order for biolume variables to plot
        priority_vars = [
            "biolume_avg_biolume",
            "biolume_bg_biolume",
            "biolume_nbflash_high",
            "biolume_nbflash_low",
            "biolume_proxy_diatoms",
            "biolume_proxy_adinos",
        ]

        vars_to_plot = []
        for pvar in priority_vars:
            if pvar in self.ds:
                scale = "log" if "avg_biolume" in pvar or "bg_biolume" in pvar else "linear"
                vars_to_plot.append((pvar, scale))
            if len(vars_to_plot) >= num_plots:
                break

        for i, (var, scale) in enumerate(vars_to_plot):
            self.logger.info("Plotting %s...", var)
            self._plot_var(
                var,
                idist,
                iz,
                distnav,
                fig,
                ax,
                i,
                0,
                profile_bottoms,
                scale=scale,
            )
            if i != num_plots - 1:
                ax[i].get_xaxis().set_visible(False)
            else:
                ax[i].set_xlabel("Distance (km)")

        # Add title to the figure
        title = f"{self.auv_name} {self.mission} - Bioluminescence"
        if "title" in self.ds.attrs:
            title = f"{self.ds.attrs['title']} - Bioluminescence"
        fig.suptitle(title, fontsize=12, fontweight="bold")

        # Save plot to file
        images_dir = Path(BASE_PATH, self.auv_name, MISSIONIMAGES, self.mission)
        Path(images_dir).mkdir(parents=True, exist_ok=True)

        output_file = Path(
            images_dir,
            f"{self.auv_name}_{self.mission}_{FREQ}_biolume.png",
        )
        plt.savefig(output_file, dpi=100, bbox_inches="tight")
        plt.close(fig)

        self.logger.info("Saved biolume plot to %s", output_file)
        return str(output_file)

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
        gulper.args.base_path = self.base_path
        gulper.args.auv_name = self.auv_name
        gulper.args.mission = self.mission
        gulper.args.local = self.local
        gulper.args.verbose = self.verbose
        gulper.args.start_esecs = self.start_esecs
        gulper.logger.setLevel(self._log_levels[self.verbose])
        gulper.logger.addHandler(self._handler)

        gulper_times = gulper.parse_gulpers()
        if not gulper_times:
            self.logger.info("No gulper times found for %s", self.mission)
            return
        odv_dir = Path(
            BASE_PATH,
            self.auv_name,
            MISSIONODVS,
            self.mission,
        )
        Path(odv_dir).mkdir(parents=True, exist_ok=True)
        gulper_odv_filename = Path(
            odv_dir,
            f"{self.auv_name}_{self.mission}_{FREQ}_Gulper.txt",
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
                        f.write(f"{self.auv_name}_{self.mission}_{FREQ}")
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
        """Process command line arguments using shared parser infrastructure."""
        # Use shared parser with create_products-specific additions
        parser = get_standard_dorado_parser(
            description=__doc__,
        )

        # Add create_products-specific arguments
        parser.add_argument(
            "--start_esecs",
            help="Start time of mission in epoch seconds, optional for gulper time lookup",
            type=float,
        )

        self.args = parser.parse_args()
        self.commandline = " ".join(sys.argv)

        # Update instance attributes from parsed arguments
        self.auv_name = self.args.auv_name
        self.mission = self.args.mission
        self.base_path = self.args.base_path
        self.start_esecs = self.args.start_esecs
        self.local = self.args.local
        self.verbose = self.args.verbose

        self.logger.setLevel(self._log_levels[self.args.verbose])


if __name__ == "__main__":
    cp = CreateProducts()
    cp.process_command_line()
    p_start = time.time()
    cp.plot_2column()
    cp.plot_biolume()
    # cp.gulper_odv()
    cp.logger.info("Time to process: %.2f seconds", (time.time() - p_start))
