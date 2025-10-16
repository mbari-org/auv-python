#!/usr/bin/env python
"""
Base classes and utility functions for reading and writing data for MBARI's
Dorado class AUVs.

--
Mike McCann
MBARI 30 March 2020
"""

import logging
from datetime import UTC, datetime

import coards
import numpy as np
import xarray as xr


def monotonic_increasing_time_indices(time_array: np.array) -> np.ndarray:
    monotonic = []
    last_t = 0.0 if isinstance(time_array[0], np.float64) else datetime.min  # noqa: DTZ901
    for t in time_array:
        if t > last_t:
            monotonic.append(True)
            last_t = t
        else:
            monotonic.append(False)
    return np.array(monotonic)


class AUV:
    def add_global_metadata(self):
        iso_now = datetime.now(UTC).isoformat() + "Z"

        self.nc_file.netcdf_version = "4"
        self.nc_file.Conventions = "CF-1.6"
        self.nc_file.date_created = iso_now
        self.nc_file.date_update = iso_now
        self.nc_file.date_modified = iso_now
        self.nc_file.featureType = "trajectory"

        self.nc_file.comment = ""

        self.nc_file.time_coverage_start = (
            coards.from_udunits(self.time[0], self.time.units).isoformat() + "Z"
        )
        self.nc_file.time_coverage_end = (
            coards.from_udunits(self.time[-1], self.time.units).isoformat() + "Z"
        )

        self.nc_file.distribution_statement = "Any use requires prior approval from MBARI"


def nudge_positions(  # noqa: C901, PLR0912, PLR0913, PLR0915
    nav_longitude: xr.DataArray,
    nav_latitude: xr.DataArray,
    gps_longitude: xr.DataArray,
    gps_latitude: xr.DataArray,
    logger: logging.Logger,
    auv_name: str = "",
    mission: str = "",
    max_sec_diff_at_end: int = 10,
    create_plots: bool = False,  # noqa: FBT001, FBT002
) -> tuple[xr.DataArray, xr.DataArray, int, float]:
    """
    Apply linear nudges to underwater latitudes and longitudes so that
    they match the surface GPS positions.

    Parameters:
    -----------
    nav_longitude : xr.DataArray
        Navigation longitude data (dead reckoned)
    nav_latitude : xr.DataArray
        Navigation latitude data (dead reckoned)
    gps_longitude : xr.DataArray
        GPS longitude fixes
    gps_latitude : xr.DataArray
        GPS latitude fixes
    logger : logging.Logger
        Logger for output messages
    auv_name : str, optional
        AUV name for plot titles
    mission : str, optional
        Mission name for plot titles
    max_sec_diff_at_end : int, optional
        Maximum allowable time difference at segment end (default: 10)
    create_plots : bool, optional
        Whether to create debug plots (default: False)

    Returns:
    --------
    tuple[xr.DataArray, xr.DataArray, int, float]
        nudged_longitude, nudged_latitude, segment_count, segment_minsum
    """
    segment_count = None
    segment_minsum = None

    lon = nav_longitude
    lat = nav_latitude
    lon_fix = gps_longitude
    lat_fix = gps_latitude

    logger.info(
        f"{'seg#':5s}  {'end_sec_diff':12s} {'end_lon_diff':12s} {'end_lat_diff':12s}"  # noqa: G004
        f" {'len(segi)':9s} {'seg_min':>9s} {'u_drift (cm/s)':14s} {'v_drift (cm/s)':14s}"
        f" {'start datetime of segment':>29}",
    )

    # Any dead reckoned points before first GPS fix - usually empty
    # as GPS fix happens before dive
    segi = np.where(lat.cf["T"].data < lat_fix.cf["T"].data[0])[0]
    if lon[:][segi].any():
        lon_nudged_array = lon[segi]
        lat_nudged_array = lat[segi]
        dt_nudged = lon.get_index("navigation_time")[segi]
        logger.debug(
            "Filled _nudged arrays with %d values starting at %s "
            "which were before the first GPS fix at %s",
            len(segi),
            lat.get_index("navigation_time")[0],
            lat_fix.get_index("gps_time")[0],
        )
    else:
        lon_nudged_array = np.array([])
        lat_nudged_array = np.array([])
        dt_nudged = np.array([], dtype="datetime64[ns]")
    if segi.any():
        seg_min = (
            lat.get_index("navigation_time")[segi][-1] - lat.get_index("navigation_time")[segi][0]
        ).total_seconds() / 60
    else:
        seg_min = 0
    logger.info(
        f"{' ':5}  {'-':>12} {'-':>12} {'-':>12} {len(segi):-9d} {seg_min:9.2f} {'-':>14} {'-':>14} {'-':>29}",  # noqa: E501, G004
    )

    MIN_SEGMENT_LENGTH = 10
    seg_count = 0
    seg_minsum = 0
    for i in range(len(lat_fix) - 1):
        # Segment of dead reckoned (under water) positions, each surrounded by GPS fixes
        segi = np.where(
            np.logical_and(
                lat.cf["T"].data > lat_fix.cf["T"].data[i],
                lat.cf["T"].data < lat_fix.cf["T"].data[i + 1],
            ),
        )[0]
        if not segi.any():
            logger.debug(
                f"No dead reckoned values found between GPS times of "  # noqa: G004
                f"{lat_fix.cf['T'].data[i]} and {lat_fix.cf['T'].data[i + 1]}",
            )
            continue

        end_sec_diff = float(lat_fix.cf["T"].data[i + 1] - lat.cf["T"].data[segi[-1]]) / 1.0e9

        end_lon_diff = float(lon_fix[i + 1]) - float(lon[segi[-1]])
        end_lat_diff = float(lat_fix[i + 1]) - float(lat[segi[-1]])

        # Compute approximate horizontal drift rate as a sanity check
        try:
            u_drift = (
                end_lon_diff
                * float(np.cos(lat_fix[i + 1] * np.pi / 180))
                * 60
                * 185300
                / (float(lat.cf["T"].data[segi][-1] - lat.cf["T"].data[segi][0]) / 1.0e9)
            )
        except ZeroDivisionError:
            u_drift = 0
        try:
            v_drift = (
                end_lat_diff
                * 60
                * 185300
                / (float(lat.cf["T"].data[segi][-1] - lat.cf["T"].data[segi][0]) / 1.0e9)
            )
        except ZeroDivisionError:
            v_drift = 0

        if abs(end_lon_diff) > 1 or abs(end_lat_diff) > 1:
            # Error handling - same as original
            logger.info(
                f"{i:5d}: {end_sec_diff:12.3f} {end_lon_diff:12.7f}"  # noqa: G004
                f" {end_lat_diff:12.7f} {len(segi):-9d} {seg_min:9.2f}"
                f" {u_drift:14.3f} {v_drift:14.3f} {lat.cf['T'].data[segi][-1]}",
            )
            logger.error(
                "End of underwater segment dead reckoned position is too different "
                "from GPS fix: abs(end_lon_diff) (%s) > 1 or abs(end_lat_diff) (%s) > 1",
                end_lon_diff,
                end_lat_diff,
            )
            logger.info(
                "Fix this error by calling _range_qc_combined_nc() in "
                "_navigation_process() and/or _gps_process() for %s %s",
                auv_name,
                mission,
            )
            error_message = (
                f"abs(end_lon_diff) ({end_lon_diff}) > 1 or abs(end_lat_diff) ({end_lat_diff}) > 1"
            )
            raise ValueError(error_message)
        if abs(end_sec_diff) > max_sec_diff_at_end:
            logger.warning(
                "abs(end_sec_diff) (%s) > max_sec_diff_at_end (%s)",
                end_sec_diff,
                max_sec_diff_at_end,
            )
            logger.info(
                "Overriding end_lon_diff (%s) and end_lat_diff (%s) by setting them to 0",
                end_lon_diff,
                end_lat_diff,
            )
            end_lon_diff = 0
            end_lat_diff = 0

        seg_min = float(lat.cf["T"].data[segi][-1] - lat.cf["T"].data[segi][0]) / 1.0e9 / 60
        seg_minsum += seg_min

        if len(segi) > MIN_SEGMENT_LENGTH:
            logger.info(
                f"{i:5d}: {end_sec_diff:12.3f} {end_lon_diff:12.7f}"  # noqa: G004
                f" {end_lat_diff:12.7f} {len(segi):-9d} {seg_min:9.2f}"
                f" {u_drift:14.3f} {v_drift:14.3f} {lat.cf['T'].data[segi][-1]}",
            )

        # Start with zero adjustment at beginning and linearly ramp up to the diff at the end
        lon_nudge = np.interp(
            lon.cf["T"].data[segi].astype(np.int64),
            [
                lon.cf["T"].data[segi].astype(np.int64)[0],
                lon.cf["T"].data[segi].astype(np.int64)[-1],
            ],
            [0, end_lon_diff],
        )
        lat_nudge = np.interp(
            lat.cf["T"].data[segi].astype(np.int64),
            [
                lat.cf["T"].data[segi].astype(np.int64)[0],
                lat.cf["T"].data[segi].astype(np.int64)[-1],
            ],
            [0, end_lat_diff],
        )

        # Sanity checks
        MAX_LONGITUDE = 180
        MAX_LATITUDE = 90
        if (
            np.max(np.abs(lon[segi] + lon_nudge)) > MAX_LONGITUDE
            or np.max(np.abs(lat[segi] + lon_nudge)) > MAX_LATITUDE
        ):
            logger.warning(
                "Nudged coordinate is way out of reasonable range - segment %d",
                seg_count,
            )
            logger.warning(
                " max(abs(lon)) = %s",
                np.max(np.abs(lon[segi] + lon_nudge)),
            )
            logger.warning(
                " max(abs(lat)) = %s",
                np.max(np.abs(lat[segi] + lat_nudge)),
            )

        lon_nudged_array = np.append(lon_nudged_array, lon[segi] + lon_nudge)
        lat_nudged_array = np.append(lat_nudged_array, lat[segi] + lat_nudge)
        dt_nudged = np.append(dt_nudged, lon.cf["T"].data[segi])
        seg_count += 1

    # Any dead reckoned points after last GPS fix
    segi = np.where(lat.cf["T"].data > lat_fix.cf["T"].data[-1])[0]
    seg_min = 0
    if segi.any():
        lon_nudged_array = np.append(lon_nudged_array, lon[segi])
        lat_nudged_array = np.append(lat_nudged_array, lat[segi])
        dt_nudged = np.append(dt_nudged, lon.cf["T"].data[segi])
        seg_min = float(lat.cf["T"].data[segi][-1] - lat.cf["T"].data[segi][0]) / 1.0e9 / 60

    logger.info(
        f"{seg_count + 1:4d}: {'-':>12} {'-':>12} {'-':>12} {len(segi):-9d} {seg_min:9.2f} {'-':>14} {'-':>14}",  # noqa: E501, G004
    )
    segment_count = seg_count
    segment_minsum = seg_minsum

    logger.info("Points in final series = %d", len(dt_nudged))

    lon_nudged = xr.DataArray(
        data=lon_nudged_array,
        dims=["time"],
        coords={"time": dt_nudged},
        name="longitude",
    )
    lat_nudged = xr.DataArray(
        data=lat_nudged_array,
        dims=["time"],
        coords={"time": dt_nudged},
        name="latitude",
    )

    # Optional plotting code
    if create_plots:
        _create_nudge_plots(
            lat, lon, lat_fix, lon_fix, lat_nudged, lon_nudged, auv_name, mission, logger
        )

    return lon_nudged, lat_nudged, segment_count, segment_minsum


def _create_nudge_plots(  # noqa: PLR0913
    lat, lon, lat_fix, lon_fix, lat_nudged, lon_nudged, auv_name, mission, logger
):
    """Create debug plots for position nudging (separated for clarity)."""
    try:
        import matplotlib.pyplot as plt

        try:
            import cartopy.crs as ccrs  # type: ignore  # noqa: I001, PGH003
            from matplotlib import patches
            from shapely.geometry import LineString  # type: ignore  # noqa: PGH003

            has_cartopy = True
        except ImportError:
            has_cartopy = False

        # Time series plots
        fig, axes = plt.subplots(nrows=2, figsize=(18, 6))
        axes[0].plot(lat_nudged.coords["time"].data, lat_nudged, "-")
        axes[0].plot(lat.cf["T"].data, lat, "--")
        axes[0].plot(lat_fix.cf["T"].data, lat_fix, "*")
        axes[0].set_ylabel("Latitude")
        axes[0].legend(["Nudged", "Original", "GPS Fixes"])
        axes[1].plot(lon_nudged.coords["time"].data, lon_nudged, "-")
        axes[1].plot(lon.cf["T"].data, lon, "--")
        axes[1].plot(lon_fix.cf["T"].data, lon_fix, "*")
        axes[1].set_ylabel("Longitude")
        axes[1].legend(["Nudged", "Original", "GPS Fixes"])
        title = "Corrected nav from nudge_positions()"
        fig.suptitle(title)
        axes[0].grid()
        axes[1].grid()
        logger.debug("Pausing with plot entitled: %s. Close window to continue.", title)
        plt.show()

        # Map plot
        if has_cartopy:
            ax = plt.axes(projection=ccrs.PlateCarree())
            nudged = LineString(zip(lon_nudged.to_numpy(), lat_nudged.to_numpy(), strict=False))
            original = LineString(zip(lon.to_numpy(), lat.to_numpy(), strict=False))
            ax.add_geometries(
                [nudged],
                crs=ccrs.PlateCarree(),
                edgecolor="red",
                facecolor="none",
                label="Nudged",
            )
            ax.add_geometries(
                [original],
                crs=ccrs.PlateCarree(),
                edgecolor="grey",
                facecolor="none",
                label="Original",
            )
            handle_gps = ax.scatter(
                lon_fix.to_numpy(),
                lat_fix.to_numpy(),
                color="green",
                label="GPS Fixes",
            )
            bounds = nudged.buffer(0.02).bounds
            extent = bounds[0], bounds[2], bounds[1], bounds[3]
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            ax.coastlines()

            handle_nudged = patches.Rectangle((0, 0), 1, 0.1, facecolor="red")
            handle_original = patches.Rectangle((0, 0), 1, 0.1, facecolor="gray")
            ax.legend(
                [handle_nudged, handle_original, handle_gps],
                ["Nudged", "Original", "GPS Fixes"],
            )
            ax.gridlines(
                crs=ccrs.PlateCarree(),
                draw_labels=True,
                linewidth=1,
                color="gray",
                alpha=0.5,
            )
            ax.set_title(f"{auv_name} {mission}")
            logger.debug(
                "Pausing map plot (doesn't work well in VS Code debugger)."
                " Close window to continue.",
            )
            plt.show()
        else:
            logger.warning("No map plot, could not import cartopy")

    except ImportError:
        logger.warning("Could not create plots - matplotlib not available")
