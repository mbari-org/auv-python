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
import contextily as ctx
import gsw
import matplotlib  # noqa: ICN001
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pooch
import pyproj
import scipy
import xarray as xr

from common_args import DEFAULT_BASE_PATH, get_standard_dorado_parser
from gulper import Gulper
from logs2netcdfs import AUV_NetCDF, MISSIONNETCDFS
from nc42netcdfs import BASE_LRAUV_PATH, BASE_LRAUV_WEB
from resample import AUVCTD_OPENDAP_BASE, FREQ, LRAUV_OPENDAP_BASE
from scipy.interpolate import griddata
from sipper import Sipper

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
        known_hash="cef6a575ed9a311230201aa0c3e3075535a31e87fd8282c3dda94823bfd08484",
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
        log_file: str = None,
        freq: str = FREQ,
        use_scatter: bool = True,  # noqa: FBT001, FBT002
        ds: xr.Dataset = None,
        output_dir: Path = None,
        plot_name_stem: str = None,
        nc_files: list[str] | None = None,
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
            log_file: Path to LRAUV log file (alternative to auv_name/mission)
            freq: Resampling frequency (default: '1S')
            use_scatter: Use scatter plots instead of contour plots (default: True)
            nc_files: Per-log OPeNDAP URL list (deployment plots only). When set,
                log-file boundary lines are drawn on the deployment plot.
        """
        self.auv_name = auv_name
        self.mission = mission
        self.base_path = base_path
        self.start_esecs = start_esecs
        self.local = local
        self.verbose = verbose
        self.commandline = commandline
        self.log_file = log_file
        self.freq = freq
        self.use_scatter = use_scatter
        self.ds = ds
        self.output_dir = output_dir
        self.plot_name_stem = plot_name_stem
        self.nc_files = nc_files

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
        "proxy_adinos": "algae",
        "proxy_diatoms": "Purples",
        "proxy_hdinos": "Blues",
        "avgrois": "YlOrRd",
        "casetemp": "thermal",
        "casehumidity": "PuBu",
        "casepress": "deep",
        "pitch_angle": "balance",
        "average_current": "jet",
        "battery_charge": "jet",
    }

    centered_vars = {"pitch_angle"}  # noqa: RUF012
    scatter_marker_size = {"average_current": 5, "battery_charge": 10}  # noqa: RUF012

    # Override labels for variables whose name is not a good y-axis label.
    variable_display_names: dict[str, str] = {"density": "Sigma-t"}  # noqa: RUF012

    # Fallback display metadata for variables that may be absent from the dataset
    # or whose NetCDF attributes are missing.  Each entry is (units, colormap).
    variable_fallback_metadata: dict[str, tuple[str, str]] = {  # noqa: RUF012
        # ── core oceanography ─────────────────────────────────────────────────
        "density": ("kg/m^3", "dense"),
        "ctd1_temperature": ("°C", "thermal"),
        "ctd2_temperature": ("°C", "thermal"),
        "ctd1_salinity": ("1", "haline"),
        "ctd2_salinity": ("1", "haline"),
        "isus_nitrate": ("umol/L", "matter"),
        "ctd1_oxygen_mll": ("ml/L", "oxy"),
        "ctd2_oxygen_mll": ("ml/L", "oxy"),
        # ── HS2 backscatter / fluorescence ────────────────────────────────────
        "hs2_bbp420": ("m^-1 sr^-1", "BuPu"),
        "hs2_bbp470": ("m^-1 sr^-1", "BuPu"),
        "hs2_bbp676": ("m^-1 sr^-1", "Reds"),
        "hs2_bbp700": ("m^-1 sr^-1", "Reds"),
        "hs2_fl676": ("counts", "algae"),
        "hs2_fl700": ("counts", "algae"),
        "ecopuck_chla": ("ug/L", "algae"),
        # ── bioluminescence (Dorado) ───────────────────────────────────────────
        "biolume_flow": ("ml/s", "cividis"),
        "biolume_avg_biolume": ("photons/s/cm^2", "cividis"),
        "biolume_intflash": ("photons/flash", "cividis"),
        "biolume_bg_biolume": ("photons/s/cm^2", "cividis"),
        "biolume_nbflash_high": ("counts", "cividis"),
        "biolume_nbflash_low": ("counts", "cividis"),
        "biolume_proxy_diatoms": ("", "Purples"),
        "biolume_proxy_adinos": ("", "algae"),
        "biolume_proxy_hdinos": ("", "Blues"),
        # ── LRAUV CTD Seabird ─────────────────────────────────────────────────
        "ctdseabird_sea_water_temperature": ("degC", "thermal"),
        "ctdseabird_sea_water_salinity": ("psu", "haline"),
        "ctdseabird_mass_concentration_of_oxygen_in_sea_water": ("ug/l", "oxy"),
        # ── LRAUV other ───────────────────────────────────────────────────────
        "onboard_platform_average_current": ("mA", "jet"),
        "bpc1_platform_battery_charge": ("Ah", "jet"),
        "universals_platform_pitch_angle": ("degrees", "balance"),
        # ── LRAUV WetLabs BB2FL ───────────────────────────────────────────────
        "wetlabsbb2fl_particulatebackscatteringcoeff470nm": ("1/m", "BuPu"),
        "wetlabsbb2fl_particulatebackscatteringcoeff650nm": ("1/m", "Reds"),
        "wetlabsbb2fl_mass_concentration_of_chlorophyll_in_sea_water": ("ug/l", "algae"),
        # ── LRAUV UBAT bioluminescence ────────────────────────────────────────
        "wetlabsubat_flow_rate": ("ml/s", "cividis"),
        "wetlabsubat_flow": ("ml/s", "cividis"),
        "wetlabsubat_average_bioluminescence": ("photons/s/cm^2", "cividis"),
        "wetlabsubat_intflash": ("photons/flash", "cividis"),
        "wetlabsubat_bg_biolume": ("photons/s/cm^2", "cividis"),
        "wetlabsubat_nbflash_high": ("counts", "cividis"),
        "wetlabsubat_nbflash_low": ("counts", "cividis"),
        "wetlabsubat_proxy_diatoms": ("", "Purples"),
        "wetlabsubat_proxy_adinos": ("", "algae"),
        "wetlabsubat_proxy_hdinos": ("", "Blues"),
        # ── Planktivore ───────────────────────────────────────────────────────
        "backseat_planktivore_hm_avgrois": ("counts", "YlOrRd"),
        "backseat_planktivore_lm_avgrois": ("counts", "YlOrRd"),
        "backseat_planktivore_casetemp": ("°C", "thermal"),
        "backseat_planktivore_casehumidity": ("%", "PuBu"),
        "backseat_planktivore_casepress": ("dbar", "deep"),
    }

    def _open_ds(self):
        if self.ds is not None:
            return
        if self._is_lrauv():
            # Open LRAUV resampled file - transform log_file to point to _1S.nc file
            # Convert from original .nc4 to resampled _1S.nc format
            resampled_file = re.sub(r"\.nc4?$", f"_{self.freq}.nc", str(self.log_file))
            log_path = Path(BASE_LRAUV_PATH, resampled_file)
            dap_url = os.path.join(LRAUV_OPENDAP_BASE, resampled_file)  # noqa: PTH118
            try:
                self.logger.info("Opening local LRAUV resampled file: %s", log_path)
                self.ds = xr.open_dataset(log_path)
            except (OSError, FileNotFoundError):
                self.logger.info("Local file not available, trying OPENDAP: %s", dap_url)
                self.ds = xr.open_dataset(dap_url)
        else:
            # Open Dorado mission file - try local first, then OPENDAP
            local_nc = Path(
                BASE_PATH,
                self.auv_name,
                MISSIONNETCDFS,
                self.mission,
                f"{self.auv_name}_{self.mission}_{self.freq}.nc",
            )
            dap_url = os.path.join(  # noqa: PTH118
                AUVCTD_OPENDAP_BASE,
                "surveys",
                self.mission.split(".")[0],
                "netcdf",
                f"{self.auv_name}_{self.mission}_{self.freq}.nc",
            )
            try:
                self.logger.info("Opening local Dorado file: %s", local_nc)
                self.ds = xr.open_dataset(local_nc)
            except (OSError, FileNotFoundError):
                self.logger.info("Local file not available, trying OPENDAP: %s", dap_url)
                self.ds = xr.open_dataset(dap_url)

    def _compute_density(self, best_ctd: str = "ctd1") -> None:
        """Compute sigma-t density from temperature and salinity using EOS-80.

        Args:
            best_ctd: The CTD instrument to use for temperature and salinity
        """
        if "density" in self.ds:
            self.logger.debug("density already exists in dataset")
            if not self.ds["density"].attrs.get("long_name"):
                self.ds["density"].attrs.update({"long_name": "Sigma-t", "units": "kg/m^3"})
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

    def _compute_density_lrauv(self) -> None:
        """Compute sigma-t density from temperature and salinity using EOS-80 for LRAUV data.

        LRAUV uses variable names without instrument prefix: 'temperature' and 'salinity'.
        """
        if "density" in self.ds:
            self.logger.debug("density already exists in dataset")
            if not self.ds["density"].attrs.get("long_name"):
                self.ds["density"].attrs.update({"long_name": "Sigma-t", "units": "kg/m^3"})
            return

        temp_var = "ctdseabird_sea_water_temperature"
        sal_var = "ctdseabird_sea_water_salinity"

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

    def _is_lrauv(self) -> bool:
        """Detect if processing LRAUV data based on parameters."""
        return self.log_file is not None

    def _get_plot_variables(self, best_ctd: str) -> list:
        """Get vehicle-specific list of variables to plot.

        Args:
            best_ctd: The CTD instrument identifier

        Returns:
            List of (variable_name, scale) tuples
        """
        if self._is_lrauv():
            return self._get_lrauv_plot_variables()
        return self._get_dorado_plot_variables(best_ctd)

    def _get_dorado_plot_variables(self, best_ctd: str) -> list:
        """Get Dorado-specific plot variables."""
        return [
            ("density", "linear"),
            (f"{best_ctd}_temperature", "linear"),
            (f"{best_ctd}_salinity", "linear"),
            ("isus_nitrate", "linear"),
            ("ctd1_oxygen_mll", "linear"),
            ("hs2_bbp420", "linear"),
            ("hs2_bbp700", "linear"),
            ("hs2_fl700", "linear"),
            ("biolume_avg_biolume", "log"),
        ]

    def _get_lrauv_plot_variables(self) -> list:
        """Get LRAUV-specific plot variables.

        Returns variables commonly available in LRAUV log files.
        """
        return [
            ("density", "linear"),
            ("ctdseabird_sea_water_temperature", "linear"),
            ("ctdseabird_sea_water_salinity", "linear"),
            ("bpc1_platform_battery_charge", "linear"),
            ("ctdseabird_mass_concentration_of_oxygen_in_sea_water", "linear"),
            ("wetlabsbb2fl_particulatebackscatteringcoeff470nm", "linear"),
            ("wetlabsbb2fl_particulatebackscatteringcoeff650nm", "linear"),
            ("wetlabsbb2fl_mass_concentration_of_chlorophyll_in_sea_water", "linear"),
            ("universals_platform_pitch_angle", "linear"),
        ]

    def _get_dorado_biolume_variables(self) -> list:
        """Get Dorado-specific bioluminescence plot variables for plot_biolume_2column()."""
        return [
            ("biolume_flow", "linear"),
            ("biolume_avg_biolume", "log"),
            ("biolume_intflash", "linear"),
            ("biolume_bg_biolume", "log"),
            ("biolume_nbflash_high", "linear"),
            ("biolume_nbflash_low", "linear"),
            ("biolume_proxy_diatoms", "linear"),
            ("biolume_proxy_adinos", "linear"),
            ("biolume_proxy_hdinos", "linear"),
        ]

    def _get_lrauv_ubat_flow_variable(self) -> str:
        """Get the preferred LRAUV UBAT flow variable name.

        Newer datasets use wetlabsubat_flow_rate; some older datasets use wetlabsubat_flow.
        """
        if hasattr(self, "ds"):
            if "wetlabsubat_flow_rate" in self.ds:
                return "wetlabsubat_flow_rate"
            if "wetlabsubat_flow" in self.ds:
                return "wetlabsubat_flow"
        return "wetlabsubat_flow_rate"

    def _get_lrauv_biolume_variables(self) -> list:
        """Get LRAUV-specific bioluminescence plot variables for plot_biolume_2column()."""
        return [
            (self._get_lrauv_ubat_flow_variable(), "linear"),
            ("wetlabsubat_average_bioluminescence", "log"),
            ("wetlabsubat_intflash", "linear"),
            ("wetlabsubat_bg_biolume", "log"),
            ("wetlabsubat_nbflash_high", "linear"),
            ("wetlabsubat_nbflash_low", "linear"),
            ("wetlabsubat_proxy_diatoms", "linear"),
            ("wetlabsubat_proxy_adinos", "linear"),
            ("wetlabsubat_proxy_hdinos", "linear"),
        ]

    def _get_biolume_plot_variables(self) -> list:
        """Get vehicle-specific bioluminescence variables for plot_biolume_2column()."""
        if self._is_lrauv():
            return self._get_lrauv_biolume_variables()
        return self._get_dorado_biolume_variables()

    def _get_planktivore_plot_variables(self) -> list:
        """Get planktivore + context variables for plot_planktivore_2column()."""
        return [
            ("backseat_planktivore_hm_avgrois", "linear"),
            ("backseat_planktivore_lm_avgrois", "linear"),
            ("backseat_planktivore_casetemp", "linear"),
            ("backseat_planktivore_casehumidity", "linear"),
            ("density", "linear"),
            ("wetlabsbb2fl_particulatebackscatteringcoeff470nm", "linear"),
            ("wetlabsbb2fl_particulatebackscatteringcoeff650nm", "linear"),
            ("wetlabsbb2fl_mass_concentration_of_chlorophyll_in_sea_water", "linear"),
            ("backseat_planktivore_casepress", "linear"),
        ]

    def _log_file_distance_ranges(self, distnav: xr.DataArray) -> list[tuple[str, float, float]]:
        """Return (label, start_km, end_km) for each nc_file in self.nc_files.

        The label is the log-directory timestamp (e.g. ``20250414T205440``),
        taken from the second-to-last URL component.  Start and end times are
        parsed from the filename (second-to-last and last URL components differ):
        filename format ``<start>_<end>_<freq>.nc``
        (e.g. ``202504142054_202504150400_1S.nc``).
        Entries that cannot be parsed, and duplicate log directories, are
        silently skipped.
        """
        if not self.nc_files:
            return []

        seen_starts: set[str] = set()
        times_np = distnav.coords["time"].to_numpy()
        times_idx = pd.DatetimeIndex(times_np)
        if times_idx.tz is None:
            times_idx = times_idx.tz_localize("UTC")
        dist_km = distnav.to_numpy() / 1000.0

        result: list[tuple[str, float, float]] = []
        for url in self.nc_files:
            # label from parent dir: e.g. "20250414T205440"
            # times from filename:   e.g. "202504142054_202504150400_1S.nc"
            label = url.rstrip("/").split("/")[-2]
            filename = url.rstrip("/").split("/")[-1]
            parts = filename.split("_")
            try:
                t_start = pd.Timestamp(parts[0]).tz_localize("UTC")
                t_end = pd.Timestamp(parts[1]).tz_localize("UTC")
            except Exception:  # noqa: BLE001
                self.logger.debug("Could not parse start/end times from filename: %s", filename)
                continue

            if label in seen_starts:
                continue
            seen_starts.add(label)

            mask = (times_idx >= t_start) & (times_idx < t_end)
            if not mask.any():
                self.logger.debug("No data in distnav for %s, skipping", label)
                continue
            result.append((label, float(dist_km[mask][0]), float(dist_km[mask][-1])))

        return result

    def _plot_log_file_boundaries(
        self,
        fig: matplotlib.figure.Figure,
        map_ax: matplotlib.axes.Axes,
        ref_data_ax: matplotlib.axes.Axes,
        distnav: xr.DataArray,
    ) -> None:
        """Draw horizontal log-file boundary segments between the map and first data row.

        Creates a thin axes in the vertical gap between *map_ax* (the map/title
        block at row 0) and *ref_data_ax* (the first data subplot at row 1, left
        column).  Each log file gets one horizontal line segment spanning its
        distance range, labelled with the log-directory timestamp.  Only drawn
        for deployment plots (``self.nc_files`` must be set).
        """
        if not self.nc_files:
            return

        ranges = self._log_file_distance_ranges(distnav)
        if not ranges:
            return

        map_pos = map_ax.get_position()
        data_pos = ref_data_ax.get_position()

        # Vertical span: from the top of the first data row to the bottom of the map
        y0 = data_pos.y1  # top of first data subplot
        y1 = map_pos.y0  # bottom of map (after _plot_track_map repositioning)
        height = y1 - y0
        if height <= 0:
            self.logger.debug("No vertical gap for log-boundary axes (height=%.4f)", height)
            return

        # Derive x range from distnav — ref_data_ax xlim is not yet set at this
        # point in plot construction (the _plot_var calls come after this)
        x_km_min = float(distnav.to_numpy()[0]) / 1000.0
        x_km_max = float(distnav.to_numpy()[-1]) / 1000.0

        bar_ax = fig.add_axes([data_pos.x0, y0, data_pos.width, height])
        bar_ax.axis("off")

        cmap = plt.get_cmap("tab10")
        for idx, (label, d_start, d_end) in enumerate(ranges):
            color = cmap(idx % 10)
            y_line = idx * 0.25 + 0.5
            bar_ax.plot(
                [d_start, d_end],
                [y_line, y_line],
                color=color,
                linewidth=2,
                solid_capstyle="butt",
            )
            bar_ax.plot(d_start, y_line, "|", color=color, markersize=6, markeredgewidth=1.5)
            bar_ax.plot(d_end, y_line, "|", color=color, markersize=6, markeredgewidth=1.5)
            # Label centred on the segment, clipped to the axes
            x_mid = (d_start + d_end) / 2.0
            bar_ax.text(
                x_mid,
                y_line + 0.25,
                label,
                fontsize=6,
                ha="center",
                va="bottom",
                color=color,
                clip_on=False,
            )

        # Set limits after plotting so they override matplotlib's autoscaling
        bar_ax.set_xlim(x_km_min, x_km_max)
        bar_ax.set_ylim(0, len(ranges))

    _PER_LOG_CSS = """
    body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
         background:#f4f6f9;color:#1a1a2e;margin:0}
    header{background:#0d1b2a;color:#fff;padding:1.5rem 2rem;
           border-bottom:4px solid #00b4d8}
    header h1{font-size:1.2rem;font-weight:600}
    main{max-width:1400px;margin:0 auto;padding:1.5rem 2rem}
    h2{font-size:1rem;font-weight:600;color:#0d1b2a;
       border-left:4px solid #00b4d8;padding-left:.6rem;margin:1.2rem 0 .75rem}
    .plots{display:flex;flex-wrap:wrap;gap:1rem}
    .plot-card{background:#fff;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,.1);
               overflow:hidden;max-width:700px}
    .plot-card img{width:100%;height:auto;display:block}
    .plot-card figcaption{padding:.3rem .7rem;font-size:.78rem;color:#6c757d}
    .links a{display:inline-block;margin:.3rem .4rem 0 0;padding:.4rem 1rem;
             background:#00b4d8;color:#fff;text-decoration:none;
             border-radius:6px;font-size:.9rem}
    .links a:hover{background:#90e0ef;color:#0d1b2a}
"""

    def write_per_log_html(self) -> str | None:
        """Write a styled HTML page alongside the per-log PNGs.

        Only meaningful for single-log processing (``nc_files`` is None).
        Looks for any ``{stem}_{freq}_2column_*.png`` files that exist and embeds them.
        Returns the path of the written HTML, or None if no PNGs found.
        """
        if not self._is_lrauv() or not self.log_file:
            return None
        out_dir = Path(BASE_LRAUV_PATH, Path(self.log_file).parent)
        stem = Path(self.log_file).stem
        html_path = out_dir / f"{stem}_{self.freq}.html"

        png_suffixes = ("_2column_cmocean.png", "_2column_biolume.png", "_2column_planktivore.png")
        cards = ""
        for suffix in png_suffixes:
            png_path = out_dir / f"{stem}_{self.freq}{suffix}"
            if png_path.exists():
                name = png_path.name
                cards += (
                    f'      <figure class="plot-card">\n'
                    f'        <a href="{name}"><img src="{name}" alt="{name}" loading="lazy"></a>\n'
                    f"        <figcaption>{name}</figcaption>\n"
                    f"      </figure>\n"
                )

        if not cards:
            self.logger.debug("No per-log PNGs found; skipping write_per_log_html")
            return None

        # Build OPeNDAP link from the log file path
        nc4_name = f"{stem}.nc4"
        nc4_url = (
            BASE_LRAUV_WEB.rstrip("/") + "/" + str(Path(self.log_file).parent) + f"/{nc4_name}"
        )

        title = f"{stem} — {self.freq} resampled"
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{title}</title>
  <style>{self._PER_LOG_CSS}  </style>
</head>
<body>
  <header><h1>{title}</h1></header>
  <main>
    <section>
      <h2>Plots</h2>
      <div class="plots">
{cards}      </div>
    </section>
    <section>
      <h2>Data</h2>
      <div class="links">
        <a href="{nc4_url}">&#128190;&nbsp;{nc4_name} (DODS)</a>
      </div>
    </section>
  </main>
</body>
</html>
"""
        html_path.write_text(html, encoding="utf-8")
        self.logger.info("Wrote per-log HTML to %s", html_path)
        return str(html_path)

    def _plot_nighttime_indicator(  # noqa: PLR0915
        self,
        fig: matplotlib.figure.Figure,
        ref_ax: matplotlib.axes.Axes,
        distnav: xr.DataArray,
    ) -> None:
        """Draw a thin nighttime indicator strip just above ref_ax.

        Fills black bars over the distance axis wherever the sun is below the horizon.
        Uses the figure's existing white space without adjusting other subplot dimensions.
        """
        from datetime import UTC, timedelta, timezone  # noqa: PLC0415

        try:
            from pysolar import solar  # noqa: PLC0415
        except ImportError:
            self.logger.warning("pysolar not available; skipping nighttime indicator")
            return

        times = pd.to_datetime(self.ds.cf["time"].to_numpy())
        lats = self.ds.cf["latitude"].to_numpy()
        lons = self.ds.cf["longitude"].to_numpy()
        dist_km = distnav.to_numpy() / 1000.0

        # Subsample for speed (pysolar is slow)
        n = len(times)
        step = max(1, n // 500)
        idx = np.arange(0, n, step)

        is_night = np.zeros(len(idx), dtype=bool)
        for k, i in enumerate(idx):
            try:
                alt = solar.get_altitude(
                    float(lats[i]),
                    float(lons[i]),
                    times[i].to_pydatetime().replace(tzinfo=UTC),
                )
                is_night[k] = alt < 0
            except Exception:  # noqa: BLE001
                self.logger.debug("pysolar altitude failed at index %d", i)

        sub_dist = dist_km[idx]

        # Create a thin axes above ref_ax using figure-normalized coordinates
        bbox = ref_ax.get_position()
        indicator_height = 0.004  # ~4 px at 100 dpi on a 10-inch-tall figure
        gap = 0.022
        night_ax = fig.add_axes([bbox.x0, bbox.y1 + gap, bbox.width, indicator_height])
        night_ax.set_xlim(sub_dist[0], ref_ax.get_xlim()[1])
        night_ax.set_ylim(0, 1)
        night_ax.axis("off")

        # Fill contiguous nighttime spans as black bars
        in_night = False
        night_start = None
        for k in range(len(sub_dist)):
            if is_night[k] and not in_night:
                night_start = sub_dist[k]
                in_night = True
            elif not is_night[k] and in_night:
                night_ax.axvspan(night_start, sub_dist[k - 1], color="black", lw=0)
                in_night = False
        if in_night:
            night_ax.axvspan(night_start, sub_dist[-1], color="black", lw=0)

        # Draw a short tick and local date label at each local midnight
        utc_offset_h = int(round(float(np.median(lons)) / 15))
        local_tz = timezone(timedelta(hours=utc_offset_h))
        times_s = np.array([t.timestamp() for t in times])
        day = times[0].to_pydatetime().astimezone(local_tz).date()
        end_day = times[-1].to_pydatetime().astimezone(local_tz).date()
        while day <= end_day:
            midnight_local = pd.Timestamp(year=day.year, month=day.month, day=day.day, tz=local_tz)
            ts = midnight_local.timestamp()
            if times_s[0] <= ts <= times_s[-1]:
                dist_mid = float(np.interp(ts, times_s, dist_km))
                night_ax.plot(
                    [dist_mid, dist_mid],
                    [0.0, -1.0],
                    color="black",
                    linewidth=0.8,
                    clip_on=False,
                )
                night_ax.text(
                    dist_mid,
                    -1.2,
                    "Local:",
                    fontsize=6,
                    ha="right",
                    va="top",
                    color="black",
                    clip_on=False,
                )
                night_ax.text(
                    dist_mid,
                    -1.2,
                    f"{day.day} {day.strftime('%b %Y')}",
                    fontsize=6,
                    ha="left",
                    va="top",
                    color="black",
                    clip_on=False,
                )
            day += timedelta(days=1)

    def _grid_dims(self, plot_vars: list[str] | None = None) -> tuple:
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

        try:
            utm_zone = int(31 + (self.ds.cf["longitude"].mean() // 6))
        except (KeyError, ValueError):
            self.logger.warning(
                "Cannot compute mean longitude for UTM zone calculation, "
                "longitude data may be empty or contain NaNs",
            )
            return np.array([]), np.array([]), xr.DataArray()
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

        # Check for and remove NaN values in distnav
        if np.isnan(distnav.to_numpy()).any():
            nan_count = np.isnan(distnav.to_numpy()).sum()
            self.logger.warning(
                "distnav contains %d NaN values, %.1f%% of %d, removing them",
                nan_count,
                100.0 * nan_count / len(distnav),
                len(distnav),
            )
            # Filter out NaN values and corresponding time coordinates
            valid_mask = ~np.isnan(distnav.to_numpy())
            distnav = xr.DataArray(
                distnav.to_numpy()[valid_mask],
                dims=("time",),
                coords={"time": distnav.coords["time"].to_numpy()[valid_mask]},
                attrs=distnav.attrs,
            )

        # Horizontal gridded to 3x the number of profiles
        idist = np.linspace(
            distnav.to_numpy()[0],
            distnav.to_numpy()[-1],
            int(3 * self.ds["profile_number"].to_numpy()[-1]),
        )
        # Vertical gridded to .5 m, rounded down to nearest 10m (minimum 10m)
        # Use only depths where at least one sensor variable has valid data to
        # exclude bogus depth values recorded when no valid sensor data was logged
        # (e.g. from memory corruption events)
        depth_values = self.ds.cf["depth"].to_numpy()
        time_dim = self.ds.cf["depth"].dims[0]
        nav_vars = {"depth", "latitude", "longitude", "profile_number"}
        has_valid_sensor_data = np.zeros(len(depth_values), dtype=bool)
        vars_to_check = [
            v for v in (plot_vars or self.ds.data_vars) if v in self.ds and "pitch" not in v
        ]
        for var in vars_to_check:
            if var not in nav_vars and time_dim in self.ds[var].dims and self.ds[var].ndim == 1:
                has_valid_sensor_data |= ~np.isnan(self.ds[var].to_numpy())
        depths_with_data = depth_values[has_valid_sensor_data]
        if len(depths_with_data) > 0 and not np.all(np.isnan(depths_with_data)):
            max_depth = max(np.floor(np.nanmax(depths_with_data) / 10) * 10, 10)
        else:
            max_depth = max(np.floor(np.nanmax(depth_values) / 10) * 10, 10)
        iz = np.arange(0, max_depth, 0.5)

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
        points = pd.DataFrame({"lon": lons, "lat": lats}).dropna()
        if len(pd.DataFrame({"lon": lons, "lat": lats})) != len(points):
            self.logger.warning(
                "Some lon/lat points have NaNs, these will be skipped for bathymetry retrieval"
            )

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
            try:
                result = pygmt.grdtrack(
                    points=points,
                    grid=MONTEREY_BAY_GRID,
                    newcolname="depth",
                )
            except Exception as e:  # noqa: BLE001
                self.logger.warning(
                    "Failed to retrieve bathymetry from Monterey Bay grid: %s. "
                    "Continuing without bathymetry.",
                    e,
                )
                return None
            else:
                # Convert to positive depths (meters below sea surface)
                bathymetry = -result["depth"].to_numpy()
                self.logger.info(
                    "Retrieved bathymetry data from Monterey Bay grid (min: %.1f m, max: %.1f m)",
                    bathymetry.min(),
                    bathymetry.max(),
                )
                return bathymetry

        return None

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

    def _get_sipper_locations(self, distnav: xr.DataArray) -> dict:
        """Get sipper sample locations in distance/depth space.

        For deployment plots (self.nc_files is set), scans the syslog of every
        log directory so samples from all logs are captured.

        Returns:
            Dictionary mapping sample number to (distance_km, depth_m) tuple
        """
        if not self.log_file:
            return {}

        sipper = Sipper()
        sipper.args = argparse.Namespace()
        sipper.args.local = self.local
        sipper.args.verbose = 0  # Suppress sipper logging
        sipper.logger.setLevel(logging.WARNING)

        if self.nc_files:
            # Deployment mode: derive a log_file path for each nc_file so we
            # can read the syslog from each individual log directory.
            log_files = [
                re.sub(
                    rf"_{re.escape(self.freq)}\.nc$",
                    ".nc4",
                    nc.replace(LRAUV_OPENDAP_BASE.rstrip("/") + "/", ""),
                )
                for nc in self.nc_files
            ]
            sipper_times: dict = {}
            for lf in log_files:
                sipper.args.log_file = lf
                with contextlib.suppress(FileNotFoundError):
                    sipper_times.update(sipper.parse_sippers())
        else:
            sipper.args.log_file = self.log_file
            sipper_times = sipper.parse_sippers()

        if not sipper_times:
            return {}

        locations = {}
        for sample_num, esec in sipper_times.items():
            # Find closest time index
            time_ns = np.datetime64(int(esec * 1e9), "ns")
            time_idx = np.abs(self.ds.cf["time"].to_numpy() - time_ns).argmin()

            # Get distance and depth at that time
            dist_km = distnav.to_numpy()[time_idx] / 1000.0
            depth_m = self.ds.cf["depth"].to_numpy()[time_idx]

            locations[sample_num] = (dist_km, depth_m)

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
        self,
        map_ax: matplotlib.axes.Axes,
        reference_ax: matplotlib.axes.Axes,
        night_ref_ax: matplotlib.axes.Axes | None = None,
    ) -> None:
        """Plot AUV track map on left side with title and times on right.

        Args:
            map_ax: The axes object to plot the map on
            reference_ax: The axes below to align with (for left edge)
            night_ref_ax: The axes used by _plot_nighttime_indicator (ax[0,1]); when
                provided the map top is shifted to align with the indicator top.
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
        start_time = pd.to_datetime(times[0]).strftime("%Y-%m-%d %H:%M:%S UTC")
        end_time = pd.to_datetime(times[-1]).strftime("%Y-%m-%d %H:%M:%S UTC")

        # Get title from netCDF attributes
        if self._is_lrauv() and self.plot_name_stem:
            # Derive dlist path (vehicle/missionlogs/year/dlist_dir) from log_file
            lf_parts = Path(self.log_file).parts
            dlist_path = "/".join(lf_parts[:4]) if len(lf_parts) >= 4 else self.log_file  # noqa: PLR2004
            deployment_name = self.plot_name_stem.replace("_", " ")
            title = (
                "Combined, Aligned, and Resampled LRAUV instrument data from "
                f"Deployment:\n{deployment_name}\n{dlist_path}"
            )
        else:
            title = self.ds.attrs.get("title", f"{self.auv_name} {self.mission}")

        # Get the position of the reference axes below to align with
        ref_pos = reference_ax.get_position()

        # Store original position
        pos = map_ax.get_position()

        # Use fixed Monterey Bay bounds when the track fits within them; otherwise
        # compute bounds dynamically from the data with a 10% margin.
        fixed_lon_bounds = [-122.41, -121.77]
        fixed_lat_bounds = [36.5, 37.0]
        valid_lons = lons[~np.isnan(lons)]
        valid_lats = lats[~np.isnan(lats)]
        if (
            valid_lons.size > 0
            and valid_lats.size > 0
            and valid_lons.min() >= fixed_lon_bounds[0]
            and valid_lons.max() <= fixed_lon_bounds[1]
            and valid_lats.min() >= fixed_lat_bounds[0]
            and valid_lats.max() <= fixed_lat_bounds[1]
        ):
            lon_bounds = fixed_lon_bounds
            lat_bounds = fixed_lat_bounds
        else:
            lon_margin = (valid_lons.max() - valid_lons.min()) * 0.1 or 0.05
            lat_margin = (valid_lats.max() - valid_lats.min()) * 0.1 or 0.05
            lon_bounds = [valid_lons.min() - lon_margin, valid_lons.max() + lon_margin]
            lat_bounds = [valid_lats.min() - lat_margin, valid_lats.max() + lat_margin]
        x_bounds, y_bounds = transformer.transform(lon_bounds, lat_bounds)
        map_ax.set_xlim(x_bounds)
        map_ax.set_ylim(y_bounds)

        # Make the plot square by using equal aspect with explicit box adjustment
        map_ax.set_aspect("equal", adjustable="box")

        # Plot the track colored by a cumulative profile number that keeps
        # incrementing across concatenated log files (profile_number resets to
        # zero at the start of each log file).
        profile_numbers = self.ds["profile_number"].to_numpy()
        cumulative_profile = profile_numbers.copy().astype(float)
        offset = 0
        for i in range(1, len(profile_numbers)):
            if profile_numbers[i] < profile_numbers[i - 1]:
                offset += profile_numbers[i - 1]
            cumulative_profile[i] = profile_numbers[i] + offset
        scatter = map_ax.scatter(
            x_merc,
            y_merc,
            c=cumulative_profile,
            cmap="jet",
            s=1,
            alpha=0.6,
        )

        # Add start and end markers
        map_ax.plot(x_merc[0], y_merc[0], "go", markersize=8, label="Start", zorder=5)
        map_ax.plot(x_merc[-1], y_merc[-1], "r^", markersize=8, label="End", zorder=5)

        # Add basemap with explicit zoom to ensure consistent rendering across platforms
        ctx.add_basemap(
            map_ax,
            crs="EPSG:3857",
            source=ctx.providers.OpenStreetMap.Mapnik,
            alpha=0.6,
            zorder=0,
            zoom=11,  # Explicit zoom for consistent rendering
        )

        # Re-apply axis limits after basemap to ensure they're respected
        map_ax.set_xlim(x_bounds)
        map_ax.set_ylim(y_bounds)

        # Size the map to 20% of the left column's horizontal span.
        map_height = pos.height
        cbar_width = 0.01
        cbar_pad = 0.005
        right_boundary = night_ref_ax.get_position().x0 if night_ref_ax is not None else pos.x1
        map_width = (right_boundary - ref_pos.x0) * 0.20

        # Align map top with the nighttime indicator top when ref axes is available.
        # Constants must match _plot_nighttime_indicator: gap=0.022, height=0.004.
        if night_ref_ax is not None:
            night_top = night_ref_ax.get_position().y1 + 0.022 + 0.004
            map_y0 = night_top - map_height
        else:
            map_y0 = pos.y0

        map_ax.set_position([ref_pos.x0, map_y0, map_width, map_height])

        # Force aspect ratio again after positioning for consistency across platforms
        map_ax.set_aspect("equal", adjustable="box")

        # Add colorbar for profile numbers - create manually positioned axes
        # to avoid affecting map position
        # Position colorbar to the right of the map
        cbar_ax = map_ax.figure.add_axes(
            [ref_pos.x0 + map_width + cbar_pad, map_y0, cbar_width, map_height]
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
        # Wrap each pre-formatted line independently to preserve intentional newlines
        import textwrap

        wrapped_title = "\n".join(textwrap.fill(line, width=40) for line in title.split("\n"))

        # Compute text block height so the whole block can be centered on the map.
        # line_height is approximate figure-coordinate height per line at fontsize 11.
        line_height = 0.02
        gap_after_title = 0.03
        time_row_spacing = 0.025
        num_title_lines = wrapped_title.count("\n") + 1
        title_height_approx = num_title_lines * line_height
        total_text_height = title_height_approx + gap_after_title + time_row_spacing

        # Position text in figure coordinates, to the right of the colorbar.
        # Center the full block vertically on the map.
        text_x = ref_pos.x0 + map_width + cbar_pad + cbar_width + 0.03
        text_y = map_y0 + map_height / 2 + total_text_height / 2

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

        # Position start/end text below the title with some spacing
        start_end_y = text_y - title_height_approx - gap_after_title

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
            start_end_y - time_row_spacing,
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
            start_end_y - time_row_spacing,
            f"End  : {end_time}",
            fontsize=11,
            fontweight="bold",
            family="sans-serif",
            verticalalignment="center",
            horizontalalignment="left",
            color="black",
        )

    def _wrap_label_text(self, text: str, max_chars: int = 20) -> str:
        """Wrap long text into multiple lines at underscore boundaries.

        Args:
            text: Text to wrap
            max_chars: Maximum characters per line

        Returns:
            Text with newlines inserted at appropriate positions
        """
        if len(text) <= max_chars:
            return text

        # Special case: insert underscore before digit suffix in long compound words
        # This allows breaking variables like "volumescatcoeff117deg650nm" and
        # "particulatebackscatteringcoeff470nm" at natural word boundaries
        import re

        # Pattern 1: Break after 'deg' before 3-digit number (volumescatcoeff117deg650nm)
        text = re.sub(r"(deg)(\d{3})", r"\1_\2", text)

        # Pattern 2: Break after 'coeff' before 3-digit number (particulatebackscatteringcoeff470nm)
        text = re.sub(r"(coeff)(\d{3})", r"\1_\2", text)

        # Pattern 3: Break after 'High intensity' or 'Low intensity' in flash labels
        text = re.sub(r"\b((?:High|Low) intensity)\s+", r"\1\n", text)

        # Split on underscores to find natural break points
        parts = text.split("_")
        lines = []
        current_line = ""

        for i, part in enumerate(parts):
            # Don't add underscore back for parts that are purely numeric suffixes (like "470nm")
            # These were split by our regex insertions above and should appear without underscore
            modified_part = "_" + part if i > 0 and not re.match(r"^\d{3}", part) else part

            # Check if adding this part would exceed max_chars
            if current_line and len(current_line + modified_part) > max_chars:
                lines.append(current_line)
                current_line = modified_part
            else:
                current_line += modified_part

        # Add the last line
        if current_line:
            lines.append(current_line)

        return "\n".join(lines)

    def _resolve_label(self, var: str) -> tuple[str, str, str]:
        """Return (display_name, units, colormap) for a variable.

        Priority: dataset attrs > variable_display_names > variable name.
        """
        if var in self.ds and self.ds[var].attrs.get("long_name"):
            long_name = self.ds[var].attrs["long_name"]
            units = self.ds[var].attrs.get("units", "")
            color_map_name = self._get_colormap_name(var)
        else:
            long_name = self.variable_display_names.get(var, var)
            if var in self.variable_fallback_metadata:
                units, color_map_name = self.variable_fallback_metadata[var]
            else:
                units = ""
                color_map_name = self._get_colormap_name(var)

        if len(long_name) > self.MAX_LONG_NAME_LENGTH:
            long_name = var
        long_name = self._wrap_label_text(long_name)
        return long_name, units, color_map_name

    def _setup_no_data_axes(  # noqa: PLR0913
        self,
        curr_ax: matplotlib.axes.Axes,
        idist: np.array,
        iz: np.array,
        fig: matplotlib.figure.Figure,
        var: str,
        row: int,
        col: int,
        text: str = "No Data",
        scale: str = "linear",
    ):
        """Set up minimal axes for plots with no data.

        Args:
            curr_ax: The axes object to configure
            idist: Distance array
            iz: Depth array
            fig: Figure object for colorbar
            var: Variable name for colorbar label
            row: Row index in subplot grid
            col: Column index in subplot grid
            text: Text to display in the center of the plot
            scale: Scale type ("linear" or "log") used to format the colorbar label
        """
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

        # Add text
        curr_ax.text(
            0.5,
            0.5,
            text,
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            transform=curr_ax.transAxes,
        )

        long_name, units, color_map_name = self._resolve_label(var)

        # Show a colorbar (no ticks) using the resolved colormap
        try:
            cmap = plt.get_cmap(color_map_name)
        except ValueError:
            cmap = getattr(cmocean.cm, color_map_name)
        dummy_data = np.array([[0, 1]])
        dummy_im = curr_ax.imshow(
            dummy_data,
            cmap=cmap,
            aspect="auto",
            visible=False,
        )
        cb = fig.colorbar(dummy_im, ax=curr_ax, pad=0.01)
        cb.set_ticks([])

        if scale == "log" and units:
            cb.set_label(f"{long_name}\n[log10({units})]", fontsize=7)
        elif scale == "log":
            cb.set_label(f"{long_name}\n[log10]", fontsize=7)
        elif units:
            cb.set_label(f"{long_name}\n[{units}]", fontsize=8)
        else:
            cb.set_label(long_name, fontsize=8)

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
        # Route to scatter or contour plot based on flag
        if self.use_scatter:
            return self._plot_var_scatter(
                var,
                idist,
                iz,
                distnav,
                fig,
                ax,
                row,
                col,
                profile_bottoms,
                scale,
                num_colors,
                gulper_locations,
                bottom_depths,
                best_ctd,
            )
        return self._plot_var_contour(
            var,
            idist,
            iz,
            distnav,
            fig,
            ax,
            row,
            col,
            profile_bottoms,
            scale,
            num_colors,
            gulper_locations,
            bottom_depths,
            best_ctd,
        )

    def _plot_var_scatter(  # noqa: C901, PLR0912, PLR0913, PLR0915
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
        """Plot variable as scatter plot (no gridding/interpolation)."""
        # Handle both 1D and 2D axis arrays
        curr_ax = ax[row] if col == 0 and hasattr(ax, "ndim") and ax.ndim == 1 else ax[row, col]

        # Check if variable exists and has valid data
        no_data = False
        if var not in self.ds:
            self.logger.warning("%s not in dataset", var)
            no_data = True
        else:
            if scale == "log":
                # Filter out zeros and negative values before log10 to avoid warnings
                data = self.ds[var].to_numpy()
                var_to_plot = np.where(data > 0, np.log10(data), np.nan)
            else:
                var_to_plot = self.ds[var].to_numpy()
            # Filter out both NaN and infinite values
            valid_data = var_to_plot[~np.isnan(var_to_plot) & ~np.isinf(var_to_plot)]
            if len(valid_data) == 0:
                self.logger.warning("%s has no valid data", var)
                no_data = True

        # If no data, set up minimal axes and return early
        if no_data:
            self._setup_no_data_axes(curr_ax, idist, iz, fig, var, row, col, scale=scale)
            return

        # Get color map
        color_map_name = self._get_colormap_name(var)
        try:
            cmap = plt.get_cmap(color_map_name)
        except ValueError:
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
            self._setup_no_data_axes(
                curr_ax, idist, iz, fig, var, row, col, text="Invalid Data Range", scale=scale
            )
            return

        if any(key in var.lower() for key in self.centered_vars):
            abs_max = max(abs(v2_5), abs(v97_5))
            norm = matplotlib.colors.Normalize(vmin=-abs_max, vmax=abs_max)
        else:
            norm = matplotlib.colors.Normalize(vmin=v2_5, vmax=v97_5)

        self.logger.info(
            "%s using %s cmap with ranges %.1f %.1f",
            var,
            color_map_name,
            v2_5,
            v97_5,
        )

        # Set axis limits
        curr_ax.set_xlim([min(distnav.to_numpy()) / 1000.0, max(distnav.to_numpy()) / 1000.0])
        curr_ax.set_ylim(max(iz), min(iz))

        # Create scatter plot with actual data points
        marker_size = next(
            (size for key, size in self.scatter_marker_size.items() if key in var.lower()), 1
        )
        scatter = curr_ax.scatter(
            distnav.to_numpy() / 1000.0,
            self.ds.cf["depth"].to_numpy(),
            c=var_to_plot,
            s=marker_size,
            cmap=cmap,
            norm=norm,
            alpha=0.8,
            rasterized=True,
        )

        # Add bathymetry as grey filled contour if available
        if bottom_depths is not None:
            dist_km = distnav.to_numpy() / 1000.0
            xb_bathy = np.append(
                dist_km,
                [curr_ax.get_xlim()[1], curr_ax.get_xlim()[1], curr_ax.get_xlim()[0], dist_km[0]],
            )
            yb_bathy = np.append(
                bottom_depths,
                [bottom_depths[-1], curr_ax.get_ylim()[0], curr_ax.get_ylim()[0], bottom_depths[0]],
            )
            try:
                curr_ax.fill(xb_bathy, yb_bathy, color="#CCCCCC", zorder=1, alpha=0.8)
            except ValueError as e:
                self.logger.warning("Could not fill bathymetry area: %s", e)  # noqa: TRY400

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

        # Set y-axis ticks adaptively based on depth range
        y_min, y_max = curr_ax.get_ylim()
        tick_step = 10 if y_min <= 50 else 50  # noqa: PLR2004
        y_ticks = np.arange(0, int(y_min) + tick_step, tick_step)
        curr_ax.set_yticks(y_ticks)

        cb = fig.colorbar(scatter, ax=curr_ax, pad=0.01)
        cb.locator = matplotlib.ticker.LinearLocator(numticks=3)
        cb.minorticks_off()
        cb.update_ticks()

        # Format tick labels intelligently based on value range
        tick_values = cb.get_ticks()
        if len(tick_values) > 0:
            value_range = abs(tick_values.max() - tick_values.min())
            max_val = abs(tick_values).max()

            # Threshold constants for tick label formatting
            VERY_LARGE_VALUE_THRESHOLD = 9_999
            LARGE_VALUE_THRESHOLD = 100
            LARGE_RANGE_THRESHOLD = 10
            MEDIUM_VALUE_THRESHOLD = 10

            # Choose format based on magnitude and range
            if max_val > VERY_LARGE_VALUE_THRESHOLD:
                # Very large values (e.g. flash intensity): use scientific notation
                labels = [f"{x:.3g}" for x in tick_values]
            elif max_val >= LARGE_VALUE_THRESHOLD or value_range >= LARGE_RANGE_THRESHOLD:
                # Large values or large range: use integers
                labels = [f"{int(round(x))}" for x in tick_values]
            elif max_val >= MEDIUM_VALUE_THRESHOLD:
                # Medium values: 1 decimal place
                labels = [f"{x:.1f}" for x in tick_values]
            elif max_val >= 1:
                # Values around 1-10: 2 decimal places
                labels = [f"{x:.2f}" for x in tick_values]
            else:
                # Small values: use scientific notation
                labels = [f"{x:.2g}" for x in tick_values]

            cb.ax.set_yticks(tick_values)
            cb.ax.set_yticklabels(labels)

        long_name, units, _ = self._resolve_label(var)

        if scale == "log" and units:
            cb.set_label(f"{long_name}\n[log10({units})]", fontsize=7)
        elif scale == "log":
            cb.set_label(f"{long_name}\n[log10]", fontsize=7)
        elif units:
            cb.set_label(f"{long_name}\n[{units}]", fontsize=8)
        else:
            cb.set_label(long_name, fontsize=8)

        # Add CTD label for density, temperature, and salinity plots
        if best_ctd and (var == "density" or "_temperature" in var or "_salinity" in var):
            y_pos = min(iz) - (max(iz) - min(iz)) * 0.025
            x_pos = curr_ax.get_xlim()[0]
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

    def _plot_var_contour(  # noqa: C901, PLR0912, PLR0913, PLR0915
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
        """Plot variable as contour plot (with gridding/interpolation)."""
        # Handle both 1D and 2D axis arrays
        curr_ax = ax[row] if col == 0 and hasattr(ax, "ndim") and ax.ndim == 1 else ax[row, col]

        # Check if variable exists and has valid data
        no_data = False
        if var not in self.ds:
            self.logger.warning("%s not in dataset", var)
            no_data = True
        else:
            if scale == "log":
                # Filter out zeros and negative values before log10 to avoid warnings
                data = self.ds[var].to_numpy()
                var_to_plot = np.where(data > 0, np.log10(data), np.nan)
            else:
                var_to_plot = self.ds[var].to_numpy()
            # Filter out both NaN and infinite values (e.g., log10(0) = -inf)
            valid_data = var_to_plot[~np.isnan(var_to_plot) & ~np.isinf(var_to_plot)]
            if len(valid_data) == 0:
                self.logger.warning("%s has no valid data", var)
                no_data = True

        # If no data, set up minimal axes and return early
        if no_data:
            self._setup_no_data_axes(curr_ax, idist, iz, fig, var, row, col, scale=scale)
            return

        # Normal plotting path - we have valid data
        scafac = max(idist) / max(iz)
        try:
            gridded_var = griddata(
                (distnav.to_numpy() / 1000.0 / scafac, self.ds.cf["depth"].to_numpy()),
                var_to_plot,
                ((idist / scafac / 1000.0)[None, :], iz[:, None]),
                method="linear",
                rescale=True,
            )
        except (ValueError, scipy.spatial._qhull.QhullError) as e:
            self.logger.error("Error in griddata for %s: %s", var, e)  # noqa: TRY400
            self._setup_no_data_axes(
                curr_ax, idist, iz, fig, var, row, col, text="Failed to Grid Data", scale=scale
            )
            return
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
            self._setup_no_data_axes(
                curr_ax, idist, iz, fig, var, row, col, text="Invalid Data Range", scale=scale
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

        if profile_bottoms is not None:
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

        # Set y-axis ticks adaptively based on depth range
        y_min, y_max = curr_ax.get_ylim()
        # Since y-axis is inverted (max at bottom), y_min is the deeper value
        tick_step = 10 if y_min <= 50 else 50  # noqa: PLR2004
        y_ticks = np.arange(0, int(y_min) + tick_step, tick_step)
        curr_ax.set_yticks(y_ticks)

        cb = fig.colorbar(cntrf, ax=curr_ax, pad=0.01)
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
                # Set ticks explicitly before setting labels to avoid warning
                cb.ax.set_yticks(tick_values)
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
            tick_values = cb.get_ticks()
            if len(tick_values) > 0:
                value_range = abs(tick_values.max() - tick_values.min())
                max_val = abs(tick_values).max()

                # Threshold constants for tick label formatting
                VERY_LARGE_VALUE_THRESHOLD = 9_999
                LARGE_VALUE_THRESHOLD = 100
                LARGE_RANGE_THRESHOLD = 10
                MEDIUM_VALUE_THRESHOLD = 10

                # Choose format based on magnitude and range
                if max_val > VERY_LARGE_VALUE_THRESHOLD:
                    # Very large values (e.g. flash intensity): use scientific notation
                    labels = [f"{x:.3g}" for x in tick_values]
                elif max_val >= LARGE_VALUE_THRESHOLD or value_range >= LARGE_RANGE_THRESHOLD:
                    # Large values or large range: use integers
                    labels = [f"{int(round(x))}" for x in tick_values]
                elif max_val >= MEDIUM_VALUE_THRESHOLD:
                    # Medium values: 1 decimal place
                    labels = [f"{x:.1f}" for x in tick_values]
                elif max_val >= 1:
                    # Values around 1-10: 2 decimal places
                    labels = [f"{x:.2f}" for x in tick_values]
                else:
                    # Small values: use scientific notation
                    labels = [f"{x:.2g}" for x in tick_values]

                cb.ax.set_yticks(tick_values)
                cb.ax.set_yticklabels(labels)

        long_name, units, _ = self._resolve_label(var)

        if scale == "log" and units:
            cb.set_label(f"{long_name}\n[log10({units})]", fontsize=7)
        elif scale == "log":
            cb.set_label(f"{long_name}\n[log10]", fontsize=7)
        elif units:
            cb.set_label(f"{long_name}\n[{units}]", fontsize=8)
        else:
            cb.set_label(long_name, fontsize=8)

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

    def plot_2column(self) -> str:  # noqa: C901, PLR0912, PLR0915
        """Create 2column plot similar to plot_sections.m and stoqs/utils/Viz/plotting.py
        Construct a 2D grid of distance and depth and for each parameter grid the data
        to create a shaded plot in each subplot.
        """
        # Skip plotting in pytest environment - too many prerequisites for CI
        if "pytest" in sys.modules:
            self.logger.info("Skipping plot_2column in pytest environment")
            return None

        self._open_ds()

        # Early return if no plot variables present in dataset
        # Use a quick pre-check with LRAUV or Dorado variables (excluding computed 'density')
        plot_variables = self._get_plot_variables(None if self._is_lrauv() else "ctd1")
        if not any(var in self.ds for var, _ in plot_variables if var != "density"):
            self.logger.warning(
                "No plot variables found in dataset, skipping plot_2column",
            )
            return None

        idist, iz, distnav = self._grid_dims([var for var, _ in plot_variables])
        if idist.size == 0 or iz.size == 0 or distnav.size == 0:
            self.logger.warning("Skipping plot_2column due to missing gridding dimensions")
            return None

        fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(18, 10))
        plt.subplots_adjust(hspace=0.15, wspace=0.04, left=0.05, right=0.97, top=0.96, bottom=0.06)

        # Compute density (sigma-t) if not already present
        best_ctd = None
        if self._is_lrauv():
            self.logger.info("LRAUV mission detected for 2column plot")
            self._compute_density_lrauv()
        else:
            self.logger.info("Dorado mission detected for 2column plot")
            best_ctd = self._get_best_ctd()
            self._compute_density(best_ctd)

        # Create map in top-left subplot (row=0, col=0), aligned with ax[1,0] below
        self._plot_track_map(ax[0, 0], ax[1, 0], ax[0, 1])

        # Parse sample locations - vehicle specific
        if self.auv_name and self.mission:
            try:
                # Dorado missions use gulper
                gulper_locations = self._get_gulper_locations(distnav)
            except FileNotFoundError as e:
                self.logger.warning("Error retrieving gulper locations: %s", e)  # noqa: TRY400
                gulper_locations = {}
        else:
            # LRAUV missions may use sipper or ESP
            try:
                gulper_locations = self._get_sipper_locations(distnav)
            except FileNotFoundError as e:
                self.logger.warning("Error retrieving sipper locations: %s", e)  # noqa: TRY400
                gulper_locations = {}
            # TODO: Implement _get_esp_locations(distnav)

        try:
            profile_bottoms = self._profile_bottoms(distnav)
        except (TypeError, ValueError) as e:
            self.logger.warning("Error computing profile bottoms: %s", e)  # noqa: TRY400
            profile_bottoms = None

        try:
            bottom_depths = self._get_bathymetry(
                self.ds.cf["longitude"].to_numpy(),
                self.ds.cf["latitude"].to_numpy(),
            )
        except ValueError as e:  # noqa: BLE001
            self.logger.warning("Error retrieving bathymetry: %s", e)  # noqa: TRY400
            bottom_depths = None

        row = 1  # Start at row 1, col 0 (below the map)
        col = 0

        # Get vehicle-specific plot variables
        plot_variables = self._get_plot_variables(best_ctd)

        for var, scale in plot_variables:
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

        self._plot_nighttime_indicator(fig, ax[0, 1], distnav)
        self._plot_log_file_boundaries(fig, ax[0, 0], ax[1, 0], distnav)
        # Save plot to file
        if self._is_lrauv():
            out_dir = (
                self.output_dir
                if self.output_dir is not None
                else Path(BASE_LRAUV_PATH, f"{Path(self.log_file).parent}")
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = (
                self.plot_name_stem if self.plot_name_stem is not None else Path(self.log_file).stem
            )
            output_file = Path(out_dir, f"{stem}_{self.freq}_2column_cmocean.png")
        else:
            images_dir = Path(BASE_PATH, self.auv_name, MISSIONIMAGES, self.mission)
            Path(images_dir).mkdir(parents=True, exist_ok=True)
            output_file = Path(
                images_dir, f"{self.auv_name}_{self.mission}_{self.freq}_2column_cmocean.png"
            )
        plt.savefig(output_file, dpi=100, bbox_inches="tight")
        plt.show()
        plt.close(fig)

        self.logger.info("Saved 2column plot to %s", output_file)
        return str(output_file)

    def plot_biolume_2column(self) -> str:  # noqa: C901, PLR0912, PLR0915
        """Create 2-column bioluminescence plot with map, flow, and biolume proxies.

        Layout (5 rows x 2 columns, column-major order):
          (0,0) track map   (0,1) flow
          (1,0) avg_biolume (1,1) intflash
          (2,0) bg_biolume  (2,1) nbflash_high
          (3,0) nbflash_low (3,1) proxy_diatoms
          (4,0) proxy_adinos(4,1) proxy_hdinos
        """
        # Skip plotting in pytest environment - too many prerequisites for CI
        if "pytest" in sys.modules:
            self.logger.info("Skipping plot_biolume_2column in pytest environment")
            return None

        self._open_ds()

        # Early return if no biolume plot variables present in dataset
        plot_variables = self._get_biolume_plot_variables()
        if not any(var in self.ds for var, _ in plot_variables):
            self.logger.warning(
                "No biolume plot variables found in dataset, skipping plot_biolume_2column",
            )
            return None

        idist, iz, distnav = self._grid_dims([var for var, _ in plot_variables])
        if idist.size == 0 or iz.size == 0 or distnav.size == 0:
            self.logger.warning("Skipping plot_biolume_2column due to missing gridding dimensions")
            return None

        fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(18, 10))
        plt.subplots_adjust(hspace=0.15, wspace=0.04, left=0.05, right=0.97, top=0.96, bottom=0.06)

        # Compute density (sigma-t) if not already present
        best_ctd = None
        if self._is_lrauv():
            self.logger.info("LRAUV mission detected for biolume 2column plot")
            self._compute_density_lrauv()
        else:
            self.logger.info("Dorado mission detected for biolume 2column plot")
            best_ctd = self._get_best_ctd()
            self._compute_density(best_ctd)

        # Create map in top-left subplot (row=0, col=0), aligned with ax[1,0] below
        self._plot_track_map(ax[0, 0], ax[1, 0], ax[0, 1])

        # Sample locations (Dorado: Gulper, LRAUV: Sipper)
        if self.auv_name and self.mission:
            try:
                gulper_locations = self._get_gulper_locations(distnav)
            except FileNotFoundError as e:
                self.logger.warning("Error retrieving gulper locations: %s", e)  # noqa: TRY400
                gulper_locations = {}
        else:
            try:
                gulper_locations = self._get_sipper_locations(distnav)
            except FileNotFoundError as e:
                self.logger.warning("Error retrieving sipper locations: %s", e)  # noqa: TRY400
                gulper_locations = {}

        try:
            profile_bottoms = self._profile_bottoms(distnav)
        except (TypeError, ValueError) as e:
            self.logger.warning("Error computing profile bottoms: %s", e)  # noqa: TRY400
            profile_bottoms = None

        try:
            bottom_depths = self._get_bathymetry(
                self.ds.cf["longitude"].to_numpy(),
                self.ds.cf["latitude"].to_numpy(),
            )
        except ValueError as e:  # noqa: BLE001
            self.logger.warning("Error retrieving bathymetry: %s", e)  # noqa: TRY400
            bottom_depths = None

        row = 1  # Start at row 1, col 0 (below the map)
        col = 0

        plot_variables = self._get_biolume_plot_variables()

        for var, scale in plot_variables:
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
                ax[row, col].set_xlabel("Distance along track (km)")

            # Column-major order: fill down first column, then second column
            if row == 4 and col == 0:  # noqa: PLR2004
                row = 0
                col = 1
            else:
                row += 1

        # Draw nighttime indicator strip just above ax[0,1] now that its x-limits are final
        self._plot_nighttime_indicator(fig, ax[0, 1], distnav)
        self._plot_log_file_boundaries(fig, ax[0, 0], ax[1, 0], distnav)

        # Save plot to file
        if self._is_lrauv():
            out_dir = (
                self.output_dir
                if self.output_dir is not None
                else Path(BASE_LRAUV_PATH, f"{Path(self.log_file).parent}")
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = (
                self.plot_name_stem if self.plot_name_stem is not None else Path(self.log_file).stem
            )
            output_file = Path(out_dir, f"{stem}_{self.freq}_2column_biolume.png")
        else:
            images_dir = Path(BASE_PATH, self.auv_name, MISSIONIMAGES, self.mission)
            Path(images_dir).mkdir(parents=True, exist_ok=True)
            output_file = Path(
                images_dir, f"{self.auv_name}_{self.mission}_{self.freq}_2column_biolume.png"
            )
        plt.savefig(output_file, dpi=100, bbox_inches="tight")
        plt.show()
        plt.close(fig)

        self.logger.info("Saved biolume 2column plot to %s", output_file)
        return str(output_file)

    def plot_planktivore_2column(self) -> str:  # noqa: C901, PLR0912, PLR0915
        """Create 2-column planktivore plot with map, ROIs, engineering, and context.

        Layout (5 rows x 2 columns, column-major order):
          (0,0) track map       (0,1) density
          (1,0) hm_avgrois      (1,1) backscatter470
          (2,0) lm_avgrois      (2,1) backscatter650
          (3,0) casetemp        (3,1) chlorophyll
          (4,0) casehumidity    (4,1) casepress
        """
        # Skip plotting in pytest environment - too many prerequisites for CI
        if "pytest" in sys.modules:
            self.logger.info("Skipping plot_planktivore_2column in pytest environment")
            return None

        self._open_ds()

        # Early return if no planktivore plot variables present in dataset
        planktivore_vars = [
            v
            for v, _ in self._get_planktivore_plot_variables()
            if v.startswith("backseat_planktivore_")
        ]
        if not any(var in self.ds for var in planktivore_vars):
            self.logger.warning(
                "No backseat_planktivore plot variables found in dataset, "
                "skipping plot_planktivore_2column",
            )
            return None

        planktivore_plot_vars = [var for var, _ in self._get_planktivore_plot_variables()]
        idist, iz, distnav = self._grid_dims(planktivore_plot_vars)
        if idist.size == 0 or iz.size == 0 or distnav.size == 0:
            self.logger.warning(
                "Skipping plot_planktivore_2column due to missing gridding dimensions"
            )
            return None

        fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(18, 10))
        plt.subplots_adjust(hspace=0.15, wspace=0.04, left=0.05, right=0.97, top=0.96, bottom=0.06)

        best_ctd = None
        if self._is_lrauv():
            self.logger.info("LRAUV mission detected for planktivore 2column plot")
            self._compute_density_lrauv()
        else:
            self.logger.info("Dorado mission detected for planktivore 2column plot")
            best_ctd = self._get_best_ctd()
            self._compute_density(best_ctd)

        self._plot_track_map(ax[0, 0], ax[1, 0], ax[0, 1])

        if self.auv_name and self.mission:
            try:
                gulper_locations = self._get_gulper_locations(distnav)
            except FileNotFoundError as e:
                self.logger.warning("Error retrieving gulper locations: %s", e)  # noqa: TRY400
                gulper_locations = {}
        else:
            try:
                gulper_locations = self._get_sipper_locations(distnav)
            except FileNotFoundError as e:
                self.logger.warning("Error retrieving sipper locations: %s", e)  # noqa: TRY400
                gulper_locations = {}

        try:
            profile_bottoms = self._profile_bottoms(distnav)
        except (TypeError, ValueError) as e:
            self.logger.warning("Error computing profile bottoms: %s", e)  # noqa: TRY400
            profile_bottoms = None

        try:
            bottom_depths = self._get_bathymetry(
                self.ds.cf["longitude"].to_numpy(),
                self.ds.cf["latitude"].to_numpy(),
            )
        except ValueError as e:  # noqa: BLE001
            self.logger.warning("Error retrieving bathymetry: %s", e)  # noqa: TRY400
            bottom_depths = None

        row = 1
        col = 0

        plot_variables = self._get_planktivore_plot_variables()

        for var, scale in plot_variables:
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
                ax[row, col].set_xlabel("Distance along track (km)")

            if row == 4 and col == 0:  # noqa: PLR2004
                row = 0
                col = 1
            else:
                row += 1

        self._plot_nighttime_indicator(fig, ax[0, 1], distnav)
        self._plot_log_file_boundaries(fig, ax[0, 0], ax[1, 0], distnav)

        if self._is_lrauv():
            out_dir = (
                self.output_dir
                if self.output_dir is not None
                else Path(BASE_LRAUV_PATH, f"{Path(self.log_file).parent}")
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = (
                self.plot_name_stem if self.plot_name_stem is not None else Path(self.log_file).stem
            )
            output_file = Path(out_dir, f"{stem}_{self.freq}_2column_planktivore.png")
        else:
            images_dir = Path(BASE_PATH, self.auv_name, MISSIONIMAGES, self.mission)
            Path(images_dir).mkdir(parents=True, exist_ok=True)
            output_file = Path(
                images_dir,
                f"{self.auv_name}_{self.mission}_{self.freq}_2column_planktivore.png",
            )
        plt.savefig(output_file, dpi=100, bbox_inches="tight")
        plt.show()
        plt.close(fig)

        self.logger.info("Saved planktivore 2column plot to %s", output_file)
        return str(output_file)

    def _get_best_ctd(self) -> str:
        """Determine best CTD to use for ODV lookup table based on metadata"""
        # LRAUV doesn't use multiple CTDs, return None
        if self._is_lrauv():
            return None

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
            f"{self.auv_name}_{self.mission}_{self.freq}_Gulper.txt",
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
                        f.write(f"{self.auv_name}_{self.mission}_{self.freq}")
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

    def sipper_odv(self, sec_bnds: int = 1) -> str:  # noqa: C901, PLR0912, PLR0915
        "Create sipper sample numbers and data at sample collection (ODV tab-delimited) file"

        if not self._is_lrauv() or not self.log_file:
            self.logger.info("sipper_odv() is only applicable to LRAUV log_file workflows")
            return None

        sipper = Sipper()
        sipper.args = argparse.Namespace()
        sipper.args.log_file = self.log_file
        sipper.args.local = self.local
        sipper.args.verbose = self.verbose
        sipper.logger.setLevel(self._log_levels[self.verbose])
        sipper.logger.addHandler(self._handler)

        sipper_times = sipper.parse_sippers()
        if not sipper_times:
            self.logger.info("No sipper times found for %s", self.log_file)
            return None

        odv_dir = Path(BASE_LRAUV_PATH, f"{Path(self.log_file).parent}")
        Path(odv_dir).mkdir(parents=True, exist_ok=True)
        sipper_odv_filename = Path(
            odv_dir,
            f"{Path(self.log_file).stem}_{self.freq}_Sipper.txt",
        )

        self._open_ds()

        odv_column_names = self.ODV_COLUMN_NAMES.copy()
        odv_column_names[8] = "Sipper Number [count]"

        # Get bathymetry data for all sipper locations if available
        bathymetry_dict = {}
        if PYGMT_AVAILABLE:
            sipper_lons = []
            sipper_lats = []
            sipper_numbers = []
            for number, esec in sipper_times.items():
                sipper_data = self.ds.sel(
                    time=slice(
                        np.datetime64(int((esec - sec_bnds) * 1e9), "ns"),
                        np.datetime64(int((esec + sec_bnds) * 1e9), "ns"),
                    ),
                )
                if sipper_data.cf["time"].size == 0:
                    continue
                sipper_lons.append(sipper_data.cf["longitude"].to_numpy().mean())
                sipper_lats.append(sipper_data.cf["latitude"].to_numpy().mean())
                sipper_numbers.append(number)

            if sipper_lons and sipper_lats:
                bathymetry = self._get_bathymetry(np.array(sipper_lons), np.array(sipper_lats))
                if bathymetry is not None:
                    for i, number in enumerate(sipper_numbers):
                        bathymetry_dict[number] = bathymetry[i]

        def mean_of(data: xr.Dataset, candidates: list[str]) -> float:
            for candidate in candidates:
                if candidate in data:
                    values = data[candidate].to_numpy()
                    if values.size > 0 and np.isfinite(values).any():
                        return float(np.nanmean(values))
            return np.nan

        with sipper_odv_filename.open("w") as f:
            f.write("\t".join(odv_column_names) + "\n")
            for number, esec in sipper_times.items():
                self.logger.debug("sipper sample: %d of %d", number, len(sipper_times))
                sipper_data = self.ds.sel(
                    time=slice(
                        np.datetime64(int((esec - sec_bnds) * 1e9), "ns"),
                        np.datetime64(int((esec + sec_bnds) * 1e9), "ns"),
                    ),
                )

                if sipper_data.cf["time"].size == 0:
                    self.logger.warning(
                        "No LRAUV data found near sipper sample %s at epoch %.3f", number, esec
                    )
                    continue

                sample_time = pd.to_datetime(sipper_data.cf["time"][0].to_numpy())
                profile_num = mean_of(sipper_data, ["profile_number"])

                for count, name in enumerate(odv_column_names):
                    if name == "Cruise":
                        f.write(f"{Path(self.log_file).stem}_{self.freq}")
                    elif name == "Station":
                        if np.isfinite(profile_num):
                            f.write(f"{int(profile_num):d}")
                        else:
                            f.write("0")
                    elif name == "Type":
                        f.write("B")
                    elif name == "mon/day/yr":
                        f.write(f"{sample_time.month:02d}/{sample_time.day:02d}/{sample_time.year}")
                    elif name == "hh:mm":
                        f.write(f"{sample_time.hour:02d}:{sample_time.minute:02d}")
                    elif name == "Lon (degrees_east)":
                        lon = float(sipper_data.cf["longitude"].to_numpy().mean())
                        f.write(f"{lon + 360.0:9.5f}")
                    elif name == "Lat (degrees_north)":
                        lat = float(sipper_data.cf["latitude"].to_numpy().mean())
                        f.write(f"{lat:8.5f}")
                    elif name == "Bot. Depth [m]":
                        if number in bathymetry_dict:
                            f.write(f"{bathymetry_dict[number]:8.1f}")
                        else:
                            f.write(f"{float(1000):8.1f}")
                    elif name == "Sipper Number [count]":
                        f.write(f"{number}")
                    elif name == "QF":
                        f.write("0")
                    elif name == "DEPTH [m]":
                        f.write(f"{float(sipper_data.cf['depth'].to_numpy().mean()):6.2f}")
                    elif name == "TEMPERATURE [°C]":
                        temp = mean_of(
                            sipper_data,
                            [
                                "ctdseabird_sea_water_temperature",
                                "ctd1_temperature",
                                "ctd2_temperature",
                            ],
                        )
                        f.write(f"{temp:5.2f}" if np.isfinite(temp) else "NaN")
                    elif name == "SALINITY [PSS78]":
                        sal = mean_of(
                            sipper_data,
                            [
                                "ctdseabird_sea_water_salinity",
                                "ctd1_salinity",
                                "ctd2_salinity",
                            ],
                        )
                        f.write(f"{sal:6.3f}" if np.isfinite(sal) else "NaN")
                    elif name == "Oxygen [ml/l]":
                        oxygen = mean_of(
                            sipper_data,
                            [
                                "ctdseabird_sea_water_oxygen",
                                "ctd1_oxygen_mll",
                                "ctd2_oxygen_mll",
                            ],
                        )
                        f.write(f"{oxygen:5.3f}" if np.isfinite(oxygen) else "NaN")
                    elif name == "NITRATE [µmol/kg]":
                        no3 = mean_of(
                            sipper_data,
                            [
                                "isus_mole_concentration_of_nitrate_in_sea_water_time",
                                "isus_nitrate",
                            ],
                        )
                        f.write(f"{no3:6.3f}" if np.isfinite(no3) else "NaN")
                    elif name == "ChlFluor [raw]":
                        chl = mean_of(
                            sipper_data,
                            [
                                "wetlabsbb2fl_mass_concentration_of_chlorophyll_in_sea_water",
                                "hs2_fl700",
                                "hs2_fl676",
                            ],
                        )
                        f.write(f"{chl:11.8f}" if np.isfinite(chl) else "NaN")
                    elif name == "bbp420 [m^{-1}]":
                        bbp_blue = mean_of(
                            sipper_data,
                            [
                                "wetlabsbb2fl_particulatebackscatteringcoeff470nm",
                                "hs2_bb420",
                                "hs2_bbp420",
                            ],
                        )
                        f.write(f"{bbp_blue:8.7f}" if np.isfinite(bbp_blue) else "NaN")
                    elif name == "bbp470 [m^{-1}]":
                        bbp470 = mean_of(sipper_data, ["hs2_bb470"])
                        f.write(f"{bbp470:8.7f}" if np.isfinite(bbp470) else "NaN")
                    elif name == "bbp700 [m^{-1}]":
                        bbp_red = mean_of(
                            sipper_data,
                            [
                                "wetlabsbb2fl_particulatebackscatteringcoeff650nm",
                                "hs2_bb700",
                                "hs2_bbp700",
                            ],
                        )
                        f.write(f"{bbp_red:8.7f}" if np.isfinite(bbp_red) else "NaN")
                    elif name == "bbp676 [m^{-1}]":
                        bbp676 = mean_of(sipper_data, ["hs2_bb676"])
                        f.write(f"{bbp676:8.7f}" if np.isfinite(bbp676) else "NaN")
                    elif name == "PAR [V]":
                        par = mean_of(
                            sipper_data,
                            [
                                "ctd2_par",
                                "surface_downwelling_photosynthetic_photon_flux_in_air",
                            ],
                        )
                        f.write(f"{par:6.3f}" if np.isfinite(par) else "NaN")
                    elif name == "YearDay [day]":
                        fractional_ns = sipper_data.cf["time"][0] - sipper_data.cf["time"][
                            0
                        ].dt.floor("D")
                        fractional_day = float(fractional_ns) / 86400000000000.0
                        f.write(f"{sample_time.dayofyear + fractional_day:9.5f}")

                    if count < len(odv_column_names) - 1:
                        f.write("\t")
                f.write("\n")

        self.logger.info(
            "Wrote %d Sipper data lines to %s",
            len(sipper_times),
            sipper_odv_filename,
        )
        return str(sipper_odv_filename)

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
        parser.add_argument(
            "--log_file",
            help="Path to LRAUV log file (alternative to --auv_name/--mission for LRAUV data)",
            type=str,
        )
        parser.add_argument(
            "--no_scatter",
            dest="use_scatter",
            action="store_false",
            help="Use contour plots instead of scatter plots (default is scatter plots)",
        )
        parser.set_defaults(use_scatter=True)
        # Note: --freq is already defined in get_standard_dorado_parser()

        self.args = parser.parse_args()
        self.commandline = " ".join(sys.argv)

        # Update instance attributes from parsed arguments
        self.auv_name = self.args.auv_name
        self.mission = self.args.mission
        self.base_path = self.args.base_path
        self.start_esecs = self.args.start_esecs
        self.local = self.args.local
        self.verbose = self.args.verbose
        self.log_file = getattr(self.args, "log_file", None)
        self.freq = self.args.freq
        self.use_scatter = self.args.use_scatter

        # Validate that either (auv_name and mission) or log_file is provided
        if self.log_file:
            if self.auv_name or self.mission:
                self.logger.warning(
                    "Both log_file and auv_name/mission provided. Using log_file for LRAUV processing."  # noqa: E501
                )
        elif not (self.auv_name and self.mission):
            parser.error("Either --log_file or both --auv_name and --mission must be provided.")

        self.logger.setLevel(self._log_levels[self.args.verbose])


if __name__ == "__main__":
    cp = CreateProducts()
    cp.process_command_line()
    p_start = time.time()
    cp.plot_2column()
    cp.plot_biolume_2column()
    cp.plot_planktivore_2column()
    if cp.mission and cp.auv_name:
        cp.gulper_odv()
    if cp.log_file:
        cp.sipper_odv()
    cp.logger.info("Time to process: %.2f seconds", (time.time() - p_start))
