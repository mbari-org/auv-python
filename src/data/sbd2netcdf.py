#!/usr/bin/env python
"""
Read realtime SBD shore.nc4 files and produce a resampled netCDF file alongside each.

Each SBD transmission produces one shore.nc4 file under:
    realtime/sbdlogs/YYYY/YYYYMM/YYYYMMDDTHHMMSS/shore.nc4

Shore.nc4 uses a "parallel time" pattern — each variable <var> has a companion
variable <var>_time holding its individual timestamps.  Variables in named groups
are prefixed with the lowercased group name in the output; the "_" (backseat) group
maps to the prefix "backseat"; root-level non-coordinate variables use explicit
prefixes defined in ROOT_VAR_PREFIXES.

Output: shore_{FREQ}.nc written into the same directory as the source shore.nc4.
"""

__author__ = "Mike McCann"
__copyright__ = "Copyright 2026, Monterey Bay Aquarium Research Institute"

import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from socket import gethostname

import git
import numpy as np
import pandas as pd
import xarray as xr
from resample import compute_profile_number

FREQ = "1S"  # used in filenames; pandas requires lowercase — use _PANDAS_FREQ there
_PANDAS_FREQ = FREQ.lower()

# Variables to extract from each group.
# Key "/" = root group; key "_" = backseat group → prefix "backseat".
SBD_PARMS = {
    "/": [
        "depth",
        "latitude",
        "longitude",
        "platform_battery_charge",
        "platform_battery_voltage",
        "platform_average_current",
        "time_fix",
        "latitude_fix",
        "longitude_fix",
        "downwelling_photosynthetic_photon_flux_in_sea_water",
    ],
    "CTD_Seabird": [
        "bin_median_sea_water_temperature",
        "bin_median_sea_water_salinity",
    ],
    "WetLabsBB2FL": [
        "bin_median_mass_concentration_of_chlorophyll_in_sea_water",
        "bin_median_ParticulateBackscatteringCoeff470nm",
        "bin_median_ParticulateBackscatteringCoeff650nm",
    ],
    "_": [
        "planktivore_HM_AvgRois",
        "planktivore_LM_AvgRois",
    ],
    "Science": [
        "PeakPlanktivoreLMavgROI",
        "PeakPlanktivoreLMavgROIDepth",
        "EdgePlanktivoreLMavgROI",
        "EdgePlanktivoreLMavgROIDepth",
        "PeakPlanktivoreHMavgROI",
        "PeakPlanktivoreHMavgROIDepth",
    ],
    "CBIT": [
        "ampHoursUsed",
    ],
}

# Root-level non-coordinate variables: variable name → output prefix
ROOT_VAR_PREFIXES = {
    "platform_battery_charge": "bpc1",
    "platform_battery_voltage": "bpc1",
    "platform_average_current": "onboard",
    "time_fix": "nal9602",
    "latitude_fix": "nal9602",
    "longitude_fix": "nal9602",
    "downwelling_photosynthetic_photon_flux_in_sea_water": "parlicor",
}

# Coordinates from the root group — no prefix applied
ROOT_COORDS = {"depth", "latitude", "longitude"}

# Group name → output variable prefix
GROUP_PREFIX = {
    "CTD_Seabird": "ctdseabird",
    "WetLabsBB2FL": "wetlabsbb2fl",
    "_": "backseat",
    "Science": "science",
    "CBIT": "cbit",
}


class SbdExtract:
    """Process SBD shore.nc4 files for a deployment window, one file at a time."""

    logger = logging.getLogger(__name__)
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        "%(levelname)s %(asctime)s %(filename)s "
        "%(funcName)s():%(lineno)d [%(process)d] %(message)s",
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    _log_levels = (logging.WARN, logging.INFO, logging.DEBUG)

    def __init__(  # noqa: PLR0913
        self,
        auv_name: str,
        start: datetime,
        end: datetime,
        vehicle_dir: str = "/Volumes/LRAUV",
        verbose: int = 0,
        clobber: bool = False,  # noqa: FBT001, FBT002
        commandline: str = "",
    ) -> None:
        self.auv_name = auv_name
        self.start = start
        self.end = end
        self.vehicle_dir = vehicle_dir
        self.verbose = verbose
        self.clobber = clobber
        self.commandline = commandline
        self.logger.setLevel(self._log_levels[min(verbose, 2)])

    def _rel(self, path: Path) -> Path:
        """Return path relative to vehicle_dir, falling back to the full path."""
        try:
            return path.relative_to(self.vehicle_dir)
        except ValueError:
            return path

    def sbd_file_list(self) -> list[Path]:
        """Return sorted list of shore.nc4 paths whose directory timestamp is in [start, end]."""
        sbd_root = Path(self.vehicle_dir) / self.auv_name / "realtime" / "sbdlogs"
        self.logger.info("Searching for shore.nc4 files under %s", sbd_root)
        files = []
        if not sbd_root.exists():
            self.logger.warning("SBD root directory not found: %s", sbd_root)
            return files

        # Build the list of YYYYMM month dirs that overlap [start, end].
        month_dirs: list[Path] = []
        y, m = self.start.year, self.start.month
        while (y, m) <= (self.end.year, self.end.month):
            candidate = sbd_root / str(y) / f"{y}{m:02d}"
            if candidate.is_dir():
                month_dirs.append(candidate)
            m += 1
            if m > 12:  # noqa: PLR2004
                m = 1
                y += 1

        all_files: list[Path] = []
        for month_dir in month_dirs:
            self.logger.info("Scanning %s", month_dir)
            found = list(month_dir.rglob("shore.nc4"))
            self.logger.debug("  rglob found %d shore.nc4 files", len(found))
            all_files.extend(found)

        self.logger.debug("Sorting %d total shore.nc4 candidates", len(all_files))
        for shore_file in sorted(all_files):
            dir_name = shore_file.parent.name  # e.g. 20260406T120000
            self.logger.debug("  checking %s", dir_name)
            try:
                dir_dt = datetime.strptime(dir_name, "%Y%m%dT%H%M%S").replace(tzinfo=UTC)
            except ValueError:
                self.logger.debug("  skipping (unrecognised dir name)")
                continue
            if self.start <= dir_dt <= self.end:
                self.logger.debug("  → in window, adding")
                files.append(shore_file)
            else:
                self.logger.debug("  → outside window [%s, %s]", self.start, self.end)

        self.logger.info("Found %d shore.nc4 files in [%s, %s]", len(files), self.start, self.end)
        return files

    def _output_var_name(self, group: str, var: str) -> str:
        """Return the output variable name for a group/variable pair."""
        if group == "/":
            if var in ROOT_COORDS:
                return var
            prefix = ROOT_VAR_PREFIXES.get(var, "")
            return f"{prefix}_{var}" if prefix else var
        prefix = GROUP_PREFIX.get(group, group.lower().replace("_", ""))
        # Lowercase to match the all-lowercase convention in create_products.py
        # lookup tables (shore.nc4 group variables use mixed case).
        return f"{prefix}_{var}".lower()

    def _read_var_da(self, nc_path: Path, group: str, var: str) -> xr.DataArray | None:
        """Open one variable as a time-indexed DataArray with dim 'time'.

        Shore.nc4 parallel-time pattern: each variable {var} has a companion
        variable {var}_time holding its individual timestamps.  We assign that
        time array as the coordinate and rename the parallel dimension to 'time'
        so every DataArray shares a common dimension name for interp().
        """
        open_kw: dict = {"engine": "netcdf4"}
        if group != "/":
            open_kw["group"] = group
        try:
            ds = xr.open_dataset(nc_path, **open_kw)
        except (OSError, ValueError) as e:
            self.logger.debug("Cannot open group %r in %s: %s", group, nc_path.name, e)
            return None

        time_var = f"{var}_time"
        if var not in ds or time_var not in ds:
            return None

        # Rename the variable-specific parallel time dimension to the canonical "time".
        dim = ds[var].dims[0]
        da = ds[var].assign_coords(**{dim: ds[time_var]}).rename({dim: "time"})
        # Drop rows with NaT time coordinates (NC_FILL_DOUBLE ~9.97e36 and other
        # non-finite fill values that slip through xarray's masking decode as NaT).
        valid_time = pd.notna(pd.DatetimeIndex(da.time.to_numpy()))
        da = da.isel(time=valid_time)
        # Also gate to the deployment window: valid-but-outlier epoch values (e.g. 0 →
        # 1970-01-01) would otherwise corrupt t_min/t_max for the common time grid.
        start_np = np.datetime64(self.start.replace(tzinfo=None), "ns")
        end_np = np.datetime64(self.end.replace(tzinfo=None), "ns")
        in_window = (da.time >= start_np) & (da.time <= end_np)
        da = da.isel(time=in_window.to_numpy()).dropna("time")
        # interp() requires a unique, sorted time index.
        return da.drop_duplicates("time").sortby("time")

    def _is_output_fresh(self, src: Path, out: Path) -> bool:
        """Return True if out exists, is not clobbered, and is newer than src."""
        if not out.exists() or self.clobber:
            return False
        if src.stat().st_mtime <= out.stat().st_mtime:
            self.logger.info("Up to date, skipping: %s", self._rel(out))
            return True
        self.logger.info("%s is newer — reprocessing: %s", self._rel(src), self._rel(out))
        return False

    def process_one_file(self, nc_path: Path) -> Path | None:
        """Process one shore.nc4 → write shore_{FREQ}.nc in the same directory.

        Each variable carries its own independently-sampled time axis.
        All variables are interpolated onto a common 1S grid so the output
        dataset has a single shared 'time' dimension.
        """
        out_path = nc_path.parent / f"shore_{FREQ}.nc"
        if self._is_output_fresh(nc_path, out_path):
            return out_path

        self.logger.info("Reading %s", self._rel(nc_path))

        # Collect a DataArray per output variable, each with its own time axis.
        data_arrays: dict[str, xr.DataArray] = {}
        for group, var_list in SBD_PARMS.items():
            for var in var_list:
                da = self._read_var_da(nc_path, group, var)
                if da is not None and da.size > 0:
                    data_arrays[self._output_var_name(group, var)] = da

        if not data_arrays:
            self.logger.warning("No variables extracted from %s", self._rel(nc_path))
            return None

        # Build a common 1S grid that spans all individually-timed variables.
        all_times = np.concatenate([da.time.to_numpy() for da in data_arrays.values()])
        t_min = pd.Timestamp(all_times.min()).floor("s")
        t_max = pd.Timestamp(all_times.max()).ceil("s")
        common_time = pd.date_range(t_min, t_max, freq=_PANDAS_FREQ)
        common_time_np = common_time.to_numpy().astype("datetime64[ns]")

        # Interpolate each variable (with its own time axis) onto the common grid.
        coords: dict[str, xr.Variable] = {
            "time": xr.Variable(
                "time",
                common_time_np,
                {"standard_name": "time", "long_name": "time"},
            ),
        }
        data_vars: dict[str, xr.Variable] = {}
        for name, da in data_arrays.items():
            interped = da.interp(time=common_time_np, method="linear")
            var_obj = xr.Variable("time", interped.values, dict(da.attrs))
            if name in ROOT_COORDS:
                coords[name] = var_obj
            else:
                data_vars[name] = var_obj

        ds = xr.Dataset(data_vars, coords=coords)
        if "depth" in ds.coords:
            profile_da = compute_profile_number(ds.coords["depth"])
            if profile_da is not None:
                ds["profile_number"] = profile_da
        ds.attrs.update(self._global_metadata(ds))
        self.logger.info("Writing %s", self._rel(out_path))
        ds.to_netcdf(out_path, format="NETCDF4_CLASSIC")
        return out_path

    def _global_metadata(self, ds: xr.Dataset) -> dict:
        """Build CF-compliant global attributes for the output file."""
        if "pytest" in sys.modules:
            return {}

        try:
            repo = git.Repo(search_parent_directories=True)
            gitcommit = repo.head.object.hexsha
        except (git.exc.InvalidGitRepositoryError, git.exc.NoSuchPathError):
            gitcommit = "<unknown>"

        iso_now = datetime.now(UTC).isoformat() + "Z"
        actual_hostname = os.getenv("HOST_NAME", gethostname())
        time_vals = ds.coords["time"].to_numpy()
        depth_vals = ds.coords["depth"].to_numpy() if "depth" in ds.coords else np.array([np.nan])
        lat_vals = (
            ds.coords["latitude"].to_numpy() if "latitude" in ds.coords else np.array([np.nan])
        )
        lon_vals = (
            ds.coords["longitude"].to_numpy() if "longitude" in ds.coords else np.array([np.nan])
        )

        return {
            "netcdf_version": "4",
            "Conventions": "CF-1.6",
            "date_created": iso_now,
            "date_update": iso_now,
            "date_modified": iso_now,
            "featureType": "trajectory",
            "time_coverage_start": str(pd.Timestamp(time_vals.min())),
            "time_coverage_end": str(pd.Timestamp(time_vals.max())),
            "time_coverage_duration": str(
                pd.Timestamp(time_vals.max()) - pd.Timestamp(time_vals.min())
            ),
            "geospatial_vertical_min": float(np.nanmin(depth_vals)),
            "geospatial_vertical_max": float(np.nanmax(depth_vals)),
            "geospatial_lat_min": float(np.nanmin(lat_vals)),
            "geospatial_lat_max": float(np.nanmax(lat_vals)),
            "geospatial_lon_min": float(np.nanmin(lon_vals)),
            "geospatial_lon_max": float(np.nanmax(lon_vals)),
            "title": f"Realtime SBD AUV sensor data from {self.auv_name}",
            "source": "realtime SBD shore.nc4 file",
            "history": f"Created by {self.commandline} on {iso_now}",
            "summary": (
                f"Realtime telemetered LRAUV data from {self.auv_name} "
                f"resampled to {FREQ} grid from a single SBD shore.nc4 transmission."
            ),
            "license": "Any use requires prior approval from MBARI",
            "distribution_statement": "Any use requires prior approval from MBARI",
            "useconst": "Not intended for legal use. Data may contain inaccuracies.",
            "auv_python_source": (
                f"sbd2netcdf.py git commit {gitcommit} on host {actual_hostname}"
            ),
        }

    def process(self) -> list[Path]:
        """Process all shore.nc4 files in [start, end], writing shore_{FREQ}.nc alongside each.

        Returns the list of output paths that were written (or already existed).
        """
        self.logger.info(
            "Starting SBD extraction for %s [%s – %s]", self.auv_name, self.start, self.end
        )
        files = self.sbd_file_list()
        if not files:
            self.logger.warning(
                "No shore.nc4 files found for %s between %s and %s",
                self.auv_name,
                self.start,
                self.end,
            )
            return []

        out_paths: list[Path] = []
        for nc_path in files:
            out = self.process_one_file(nc_path)
            if out is not None:
                out_paths.append(out)

        self.logger.info("Wrote %d/%d shore_%s.nc files", len(out_paths), len(files), FREQ)
        return out_paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auv_name", required=True, help="AUV name, e.g. ahi")
    parser.add_argument(
        "--start",
        required=True,
        help="Start datetime YYYYMMDDTHHMMSS or YYYYMMDD",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="End datetime YYYYMMDDTHHMMSS or YYYYMMDD",
    )
    parser.add_argument(
        "--vehicle_dir",
        default="/Volumes/LRAUV",
        help="Root of LRAUV vehicle directory (default: /Volumes/LRAUV)",
    )
    parser.add_argument("--clobber", action="store_true", help="Overwrite existing output files")
    parser.add_argument(
        "--verbose",
        "-v",
        type=int,
        choices=range(3),
        default=0,
        const=1,
        nargs="?",
        help="Verbosity level: 0=WARN (default), 1=INFO, 2=DEBUG",
    )
    args = parser.parse_args()

    def _parse_dt(s: str) -> datetime:
        for fmt in ("%Y%m%dT%H%M%S", "%Y%m%d"):
            try:
                return datetime.strptime(s, fmt).replace(tzinfo=UTC)
            except ValueError:
                continue
        msg = f"Unrecognised datetime format: {s!r}"
        raise ValueError(msg)

    extractor = SbdExtract(
        auv_name=args.auv_name,
        start=_parse_dt(args.start),
        end=_parse_dt(args.end),
        vehicle_dir=args.vehicle_dir,
        verbose=args.verbose,
        clobber=args.clobber,
        commandline=" ".join(sys.argv),
    )
    out_paths = extractor.process()
    for p in out_paths:
        logging.info(p)
