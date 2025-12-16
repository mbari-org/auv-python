#!/usr/bin/env python
"""
Create "quick look" plots and other products from processed data.

"""

__author__ = "Mike McCann"
__copyright__ = "Copyright 2023, Monterey Bay Aquarium Research Institute"

import argparse  # noqa: I001
import logging
import os
import re
import sys
import time
from pathlib import Path

import cmocean
import contextily as ctx
import gsw
import matplotlib  # noqa: ICN001
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pooch
import pyproj
import xarray as xr

from common_args import DEFAULT_BASE_PATH, get_standard_dorado_parser
from gulper import Gulper
from logs2netcdfs import AUV_NetCDF, MISSIONNETCDFS
from resample import AUVCTD_OPENDAP_BASE, FREQ
from scipy.interpolate import griddata

# Optional import for bathymetry data
# Set GMT library path for macOS systems with MacPorts or Homebrew installations
if sys.platform == "darwin":
    # Try common installation paths, including MacPorts GMT6 subdirectory structure
    gmt_search_paths = [
        "/opt/local/lib/gmt6/lib",  # MacPorts GMT6
        "/opt/local/lib",  # MacPorts older versions
        "/opt/homebrew/lib",  # Homebrew ARM
        "/usr/local/lib",  # Homebrew Intel
    ]
    for gmt_path in gmt_search_paths:
        gmt_path_obj = Path(gmt_path)
        if gmt_path_obj.exists() and any(gmt_path_obj.glob("libgmt.*")):
            os.environ.setdefault("GMT_LIBRARY_PATH", gmt_path)
            break

try:
    import pygmt

    PYGMT_AVAILABLE = True

    # Download Monterey Bay bathymetry grid using pooch for local caching
    MONTEREY_BAY_GRID_URL = "https://stoqs.mbari.org/terrain/Monterey25.grd"
    MONTEREY_BAY_GRID = pooch.retrieve(
        url=MONTEREY_BAY_GRID_URL,
        known_hash=None,  # Could add SHA256 hash for verification
        fname="Monterey25.grd",
        path=pooch.os_cache("auv-python"),
        progressbar=False,
    )
except Exception:  # noqa: BLE001
    # Any failure (ImportError, GMTCLibNotFoundError, etc.) - just continue without pygmt
    PYGMT_AVAILABLE = False
    MONTEREY_BAY_GRID = None

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

    # Maximum length for long_name before using variable name instead
    MAX_LONG_NAME_LENGTH = 40

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
        "volume_fraction_of_oxygen_in_sea_water": "oxy",
        "moles_of_oxygen_per_unit_mass_in_sea_water": "oxy",
    }
    # Fallback colormap lookup by variable name
    variable_colormap_lookup = {  # noqa: RUF012
        "nitrate": "matter",
        "hs2_bbp420": "BuPu",
        "hs2_bbp700": "Reds",
        "hs2_bbp470": "BuPu",
        "hs2_bbp676": "Reds",
        "hs2_fl700": "algae",
        "hs2_fl676": "algae",
        "ecopuck_chla": "algae",
        "biolume_avg_biolume": "cividis",
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

    def _get_bathymetry(self, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
        """Get bathymetry depths at given lon/lat positions using pygmt.

        Args:
            lons: Array of longitude values
            lats: Array of latitude values

        Returns:
            Array of bathymetry depths (positive down) in meters, or None if pygmt unavailable
        """
        if not PYGMT_AVAILABLE:
            self.logger.warning("pygmt not available, will not be plotting bottom depths")
            return None

        # Use local Monterey Bay grid if available and coordinates are in range
        # Otherwise fall back to global grids
        points = pd.DataFrame({"lon": lons, "lat": lats})

        # Check if coordinates are within Monterey Bay region
        MB_LON_RANGE = (-122.5, -121.5)
        MB_LAT_RANGE = (36.0, 37.5)
        in_mb_region = (
            (lons >= MB_LON_RANGE[0]).all()
            and (lons <= MB_LON_RANGE[1]).all()
            and (lats >= MB_LAT_RANGE[0]).all()
            and (lats <= MB_LAT_RANGE[1]).all()
        )

        if in_mb_region and MONTEREY_BAY_GRID:
            self.logger.info("Using local Monterey Bay bathymetry grid")
            result = pygmt.grdtrack(
                points=points,
                grid=MONTEREY_BAY_GRID,
                newcolname="depth",
            )
            # Convert to positive depths (meters below sea surface)
            bathymetry = -result["depth"].to_numpy()
            self.logger.info(
                "Retrieved bathymetry data from Monterey Bay grid (min: %.1f m, max: %.1f m)",
                bathymetry.min(),
                bathymetry.max(),
            )
            return bathymetry

        # Fall back to global grids
        # Try GEBCO first (higher resolution), fall back to ETOPO1
        result = pygmt.grdtrack(
            points=points,
            grid="@earth_relief_15s",  # 15 arc-second resolution (~450m)
            newcolname="depth",
        )

        # Convert to positive depths (meters below sea surface)
        bathymetry = -result["depth"].to_numpy()
        self.logger.info(
            "Retrieved bathymetry data using pygmt (min: %.1f m, max: %.1f m)",
            bathymetry.min(),
            bathymetry.max(),
        )
        return bathymetry

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

    def _get_gulper_locations(self, distnav: xr.DataArray) -> dict:
        """Get gulper bottle locations in distance/depth space.

        Returns:
            Dictionary mapping bottle number to (distance_km, depth_m) tuple
        """
        gulper = Gulper()
        gulper.args = argparse.Namespace()
        gulper.args.base_path = self.base_path
        gulper.args.auv_name = self.auv_name
        gulper.args.mission = self.mission
        gulper.args.local = self.local
        gulper.args.verbose = 0  # Suppress gulper logging
        gulper.args.start_esecs = self.start_esecs
        gulper.logger.setLevel(logging.WARNING)

        gulper_times = gulper.parse_gulpers()
        if not gulper_times:
            return {}

        locations = {}
        for bottle, esec in gulper_times.items():
            # Find closest time index
            time_ns = np.datetime64(int(esec * 1e9), "ns")
            time_idx = np.abs(self.ds.cf["time"].to_numpy() - time_ns).argmin()

            # Get distance and depth at that time
            dist_km = distnav.to_numpy()[time_idx] / 1000.0
            depth_m = self.ds.cf["depth"].to_numpy()[time_idx]

            locations[bottle] = (dist_km, depth_m)

        return locations

    def _get_colormap_name(self, var: str) -> str:
        """Get colormap name for a variable.

        Tries in order:
        1. Lookup by standard_name attribute
        2. Lookup by matching variable name parts
        3. Default to 'cividis'

        Args:
            var: Variable name

        Returns:
            Colormap name
        """
        # Try standard_name first
        if var in self.ds and "standard_name" in self.ds[var].attrs:
            standard_name = self.ds[var].attrs["standard_name"]
            if standard_name in self.cmocean_lookup:
                return self.cmocean_lookup[standard_name]

        # Fallback: try matching variable name parts
        var_lower = var.lower()
        for key, colormap in self.variable_colormap_lookup.items():
            if key in var_lower:
                return colormap

        # Default
        return "cividis"

    def _plot_track_map(  # noqa: PLR0915
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

        # Convert to Web Mercator for contextily
        import pyproj

        transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        x_merc, y_merc = transformer.transform(lons, lats)

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

        # Plot the track with profile_number coloring in Web Mercator coordinates
        profile_numbers = self.ds["profile_number"].to_numpy()
        scatter = map_ax.scatter(
            x_merc,
            y_merc,
            c=profile_numbers,
            cmap="jet",
            s=1,
            alpha=0.6,
        )

        # Add start and end markers
        map_ax.plot(x_merc[0], y_merc[0], "go", markersize=8, label="Start", zorder=5)
        map_ax.plot(x_merc[-1], y_merc[-1], "r^", markersize=8, label="End", zorder=5)

        # Set fixed axis limits for Monterey Bay area (in Web Mercator)
        lon_bounds = [-122.41, -121.77]
        lat_bounds = [36.5, 37.0]
        x_bounds, y_bounds = transformer.transform(lon_bounds, lat_bounds)
        map_ax.set_xlim(x_bounds)
        map_ax.set_ylim(y_bounds)

        # Add basemap
        ctx.add_basemap(
            map_ax,
            crs="EPSG:3857",
            source=ctx.providers.OpenStreetMap.Mapnik,
            alpha=0.6,
            zorder=0,
        )

        # Now position map aligned with left edge of reference, 50% width
        # Use a square aspect ratio based on the y-dimension
        map_height = pos.height
        aspect_ratio = (37.0 - 36.5) / (122.41 - 121.77)  # data aspect ratio
        map_width = map_height / aspect_ratio * 0.7  # scale to fit nicely

        map_ax.set_position([ref_pos.x0, pos.y0, map_width, map_height])

        # Add colorbar for profile numbers - create manually positioned axes
        # to avoid affecting map position
        # Position colorbar to the right of the map
        cbar_width = 0.01
        cbar_pad = 0.005
        cbar_ax = map_ax.figure.add_axes(
            [ref_pos.x0 + map_width + cbar_pad, pos.y0, cbar_width, map_height]
        )
        cbar = map_ax.figure.colorbar(
            scatter,
            cax=cbar_ax,
            orientation="vertical",
        )
        cbar.set_label("Profile Number", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

        # Remove axes, labels, and ticks but keep the border
        map_ax.set_xticks([])
        map_ax.set_yticks([])
        map_ax.set_xlabel("")
        map_ax.set_ylabel("")

        # Add a border around the map
        for spine in map_ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1)

        # Add text on the right side with title and times
        # Wrap title for better formatting
        import textwrap

        wrapped_title = textwrap.fill(title, width=40)

        # Get updated position after aspect adjustment
        updated_pos = map_ax.get_position()

        # Position text in figure coordinates, to the right of the colorbar
        # Account for colorbar width, padding, and space for colorbar label
        text_x = ref_pos.x0 + map_width + cbar_pad + cbar_width + 0.03
        text_y = updated_pos.y0 + updated_pos.height  # Align with top of map

        # Add title
        map_ax.figure.text(
            text_x,
            text_y,
            wrapped_title,
            fontsize=11,
            fontweight="bold",
            family="sans-serif",
            verticalalignment="top",
            horizontalalignment="left",
            color="black",
        )

        # Calculate approximate title height based on number of lines
        # Each line is roughly 0.015 in figure coordinates at fontsize 11
        num_title_lines = wrapped_title.count("\n") + 1
        title_height_approx = num_title_lines * 0.02

        # Position start/end text below the title with some spacing
        start_end_y = text_y - title_height_approx - 0.03

        # Add start marker and text
        map_ax.figure.text(
            text_x,
            start_end_y,
            "● ",
            fontsize=14,
            fontweight="bold",
            family="sans-serif",
            verticalalignment="center",
            horizontalalignment="left",
            color="green",
        )
        map_ax.figure.text(
            text_x + 0.015,
            start_end_y,
            f"Start: {start_time}",
            fontsize=11,
            fontweight="bold",
            family="sans-serif",
            verticalalignment="center",
            horizontalalignment="left",
            color="black",
        )

        # Add end marker and text
        map_ax.figure.text(
            text_x,
            start_end_y - 0.025,
            "▲ ",
            fontsize=14,
            fontweight="bold",
            family="sans-serif",
            verticalalignment="center",
            horizontalalignment="left",
            color="red",
        )
        map_ax.figure.text(
            text_x + 0.015,
            start_end_y - 0.025,
            f"End  : {end_time}",
            fontsize=11,
            fontweight="bold",
            family="sans-serif",
            verticalalignment="center",
            horizontalalignment="left",
            color="black",
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
        gulper_locations: dict = None,
        bottom_depths: np.array = None,
        best_ctd: str = None,
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
            # Filter out both NaN and infinite values (e.g., log10(0) = -inf)
            valid_data = var_to_plot[~np.isnan(var_to_plot) & ~np.isinf(var_to_plot)]
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

            # Add a fake colorbar to maintain layout consistency
            # Create a dummy mappable with a simple gray colormap
            dummy_data = np.array([[0, 1]])
            dummy_im = curr_ax.imshow(
                dummy_data,
                cmap="gray",
                aspect="auto",
                visible=False,
            )
            cb = fig.colorbar(dummy_im, ax=curr_ax)

            # Hide the colorbar ticks and tick labels but keep the label visible
            cb.set_ticks([])
            cb.ax.set_facecolor("white")
            cb.outline.set_visible(False)

            # Get variable name for the label
            if var in self.ds:
                long_name = self.ds[var].attrs.get("long_name", var)
                units = self.ds[var].attrs.get("units", "")
            else:
                # Extract readable name from variable string (e.g., "isus_nitrate" -> "Nitrate")
                long_name = var.split("_")[-1].capitalize()
                units = ""

            # Use variable name if long_name is too long
            if len(long_name) > self.MAX_LONG_NAME_LENGTH:
                long_name = var

            # Add label to the colorbar area
            if units:
                cb.set_label(f"{long_name} [{units}]", fontsize=9)
            else:
                cb.set_label(long_name, fontsize=9)

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

        color_map_name = self._get_colormap_name(var)
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

        # Add bathymetry as grey filled contour if available
        if bottom_depths is not None:
            # Pair bottom_depths with distance along track
            dist_km = distnav.to_numpy() / 1000.0
            # Create polygon for seafloor: from left to right along bottom,
            # then back along axis bottom
            xb_bathy = np.append(
                dist_km,
                [curr_ax.get_xlim()[1], curr_ax.get_xlim()[1], curr_ax.get_xlim()[0], dist_km[0]],
            )
            yb_bathy = np.append(
                bottom_depths,
                [bottom_depths[-1], curr_ax.get_ylim()[0], curr_ax.get_ylim()[0], bottom_depths[0]],
            )
            curr_ax.fill(xb_bathy, yb_bathy, color="#CCCCCC", zorder=1, alpha=0.8)

        # Add measurement location dots for density plot
        if var == "density":
            curr_ax.scatter(
                distnav.to_numpy() / 1000.0,
                self.ds.cf["depth"].to_numpy(),
                s=0.1,
                c="white",
                alpha=0.1,
                zorder=3,
            )

        # Add gulper bottle locations
        if gulper_locations:
            for bottle, (dist, depth) in gulper_locations.items():
                curr_ax.text(
                    dist,
                    depth - 5,
                    str(bottle),
                    fontsize=7,
                    ha="center",
                    va="top",
                    color="black",
                    fontweight="bold",
                    zorder=5,
                )

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

        # Use scientific notation with offset for hs2 variables
        if var.startswith("hs2_"):
            # Calculate the order of magnitude
            tick_values = cb.get_ticks()
            max_abs = max(abs(tick_values.min()), abs(tick_values.max()))
            if max_abs > 0:
                order = int(np.floor(np.log10(max_abs)))
                scale = 10**order
                # Set clean tick labels
                cb.ax.set_yticklabels([f"{x / scale:.2f}" for x in tick_values])
                # Add offset text
                cb.ax.text(
                    1.5,
                    1.05,
                    f"10$^{{{order}}}$",
                    transform=cb.ax.transAxes,
                    fontsize=9,
                    verticalalignment="bottom",
                )
        else:
            cb.ax.set_yticklabels([f"{x:.1f}" for x in cb.get_ticks()])

        # Get long_name and units with fallbacks
        long_name = self.ds[var].attrs.get("long_name", var)
        units = self.ds[var].attrs.get("units", "")

        # Use variable name if long_name is too long
        if len(long_name) > self.MAX_LONG_NAME_LENGTH:
            long_name = var

        if scale == "log" and units:
            cb.set_label(f"{long_name}\n[log10({units})]", fontsize=7)
        elif scale == "log":
            cb.set_label(f"{long_name}\n[log10]", fontsize=7)
        elif units:
            cb.set_label(f"{long_name} [{units}]", fontsize=9)
        else:
            cb.set_label(long_name, fontsize=9)

        # Add CTD label for density, temperature, and salinity plots
        if best_ctd and (var == "density" or "_temperature" in var or "_salinity" in var):
            # Position above the plot area (outside y-axis limits)
            # Since y-axis is inverted (depth), min(iz) is at top, so go even less (shallower)
            y_pos = (
                min(iz) - (max(iz) - min(iz)) * 0.025
            )  # 2.5% above the top (half the whitespace)
            x_pos = curr_ax.get_xlim()[0]  # Left edge of plot
            curr_ax.text(
                x_pos,
                y_pos,
                best_ctd,
                fontsize=8,
                fontweight="bold",
                verticalalignment="bottom",
                horizontalalignment="left",
                clip_on=False,
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
        plt.subplots_adjust(hspace=0.15, wspace=0.01, left=0.05, right=1.01, top=0.96, bottom=0.06)

        best_ctd = self._get_best_ctd()

        # Compute density (sigma-t) if not already present
        self._compute_density(best_ctd)

        # Create map in top-left subplot (row=0, col=0), aligned with ax[1,0] below
        self._plot_track_map(ax[0, 0], ax[1, 0])

        # Parse gulper locations
        gulper_locations = self._get_gulper_locations(distnav)

        profile_bottoms = self._profile_bottoms(distnav)

        bottom_depths = self._get_bathymetry(
            self.ds.cf["longitude"].to_numpy(),
            self.ds.cf["latitude"].to_numpy(),
        )

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
            ("biolume_avg_biolume", "log"),
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
                gulper_locations=gulper_locations,
                bottom_depths=bottom_depths,
                best_ctd=best_ctd,
            )
            if row != 4:  # noqa: PLR2004
                ax[row, col].get_xaxis().set_visible(False)
            else:
                # Add x-axis label only to bottom row
                ax[row, col].set_xlabel("Distance along track (km)")

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
        fig.tight_layout(rect=[0, 0.06, 0.99, 0.96])

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
                ax[i].set_xlabel("Distance along track (km)")

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

        # Get bathymetry data for all gulper locations if available
        bathymetry_dict = {}
        if PYGMT_AVAILABLE:
            gulper_lons = []
            gulper_lats = []
            for esec in gulper_times.values():
                gulper_data = self.ds.sel(
                    time=slice(
                        np.datetime64(int((esec - sec_bnds) * 1e9), "ns"),
                        np.datetime64(int((esec + sec_bnds) * 1e9), "ns"),
                    ),
                )
                gulper_lons.append(gulper_data.cf["longitude"].to_numpy().mean())
                gulper_lats.append(gulper_data.cf["latitude"].to_numpy().mean())

            bathymetry = self._get_bathymetry(np.array(gulper_lons), np.array(gulper_lats))
            if bathymetry is not None:
                for i, bottle in enumerate(gulper_times.keys()):
                    bathymetry_dict[bottle] = bathymetry[i]

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
                        # Use pygmt bathymetry if available, otherwise default to 1000m
                        if bottle in bathymetry_dict:
                            f.write(f"{bathymetry_dict[bottle]:8.1f}")
                        else:
                            f.write(f"{float(1000):8.1f}")
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
    cp.gulper_odv()
    cp.logger.info("Time to process: %.2f seconds", (time.time() - p_start))
