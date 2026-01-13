# noqa: INP001
"""
Utility functions for MBARI AUV data processing.

Includes:
- Douglas-Peucker line simplification (pure-Python implementation)
- LRAUV deployment name parsing
- Time series monotonicity checking
- Position nudging for dead reckoning correction

The Douglas-Peucker code was written by Schuyler Erle <schuyler@nocat.net> and is
made available in the public domain. It was ported from a freely-licensed example at
http://www.3dsoftware.com/Cartography/Programming/PolyLineReduction/
(original page no longer available, but mirrored at
http://www.mappinghacks.com/code/PolyLineReduction/)

Example usage of simplify_points:

>>> line = [(0,0),(1,0),(2,0),(2,1),(2,2),(1,2),(0,2),(0,1),(0,0)]
>>> simplify_points(line, 1.0)
[(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)]

>>> line = [(0,0),(0.5,0.5),(1,0),(1.25,-0.25),(1.5,.5)]
>>> simplify_points(line, 0.25)
[(0, 0), (0.5, 0.5), (1.25, -0.25), (1.5, 0.5)]

"""

import logging
import math
from datetime import datetime
from pathlib import Path

import cf_xarray  # Needed for the .cf accessor  # noqa: F401
import numpy as np
import xarray as xr


def get_deployment_name(
    log_file: str, base_lrauv_path: Path, logger: logging.Logger = None
) -> str | None:
    """Parse deployment name from .dlist file in great-grandparent directory.

    Args:
        log_file: Path to log file (e.g., tethys/missionlogs/2012/20120908_20120920/.../.nc4)
        base_lrauv_path: Base path for local LRAUV data
        logger: Optional logger for debug messages

    Returns:
        Deployment name string or None if not found
    """
    try:
        log_path = Path(log_file)
        # Get great-grandparent directory (e.g., tethys/missionlogs/2012)
        great_grandparent_dir = log_path.parent.parent.parent
        # The directory with the .dlist file (e.g., 20120908_20120920)
        deployment_dir = log_path.parent.parent
        # Construct .dlist filename from deployment directory name
        dlist_filename = f"{deployment_dir.name}.dlist"

        # Try file share location first (/Volumes/LRAUV/vehicle/missionlogs/YYYY/...)
        lrauv_share = Path("/Volumes/LRAUV")
        dlist_path = lrauv_share / great_grandparent_dir / dlist_filename

        # If not on file share, try local base_lrauv_path
        if not dlist_path.exists():
            dlist_path = Path(base_lrauv_path, great_grandparent_dir, dlist_filename)

        if not dlist_path.exists():
            if logger:
                logger.debug("No .dlist file found at %s", dlist_path)
            return None

        with dlist_path.open() as f:
            first_line = f.readline().strip()
            # Parse "# Deployment name: <deployment_name>" (case insensitive)
            if first_line.lower().startswith("# deployment name:"):
                deployment_name = first_line.split(":", 1)[1].strip()
                if logger:
                    logger.debug("Found deployment name: %s", deployment_name)
                return deployment_name
            return None
    except (OSError, IndexError) as e:
        if logger:
            logger.debug("Error parsing deployment name: %s", e)
        return None


def monotonic_increasing_time_indices(time_array: np.array) -> np.ndarray:
    """Check which elements in a time array are monotonically increasing.

    Args:
        time_array: Array of time values (datetime or float)

    Returns:
        Boolean array indicating which elements maintain monotonic increase
    """
    monotonic = []
    last_t = 0.0 if isinstance(time_array[0], np.float64) else datetime.min  # noqa: DTZ901
    for t in time_array:
        if t > last_t:
            monotonic.append(True)
            last_t = t
        else:
            monotonic.append(False)
    return np.array(monotonic)


def nudge_positions(  # noqa: C901, PLR0912, PLR0913, PLR0915
    nav_longitude: xr.DataArray,
    nav_latitude: xr.DataArray,
    gps_longitude: xr.DataArray,
    gps_latitude: xr.DataArray,
    logger: logging.Logger,
    auv_name: str = "",
    mission: str = "",
    max_sec_diff_at_end: int = 10,
    log_file: str = "",
    create_plots: bool = False,  # noqa: FBT001, FBT002
    nav_depth: xr.DataArray = None,
    orig_depth: xr.DataArray = None,
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
        dt_nudged = lon.cf["T"][segi]
        logger.debug(
            "Filled _nudged arrays with %d values starting at %s "
            "which were before the first GPS fix at %s",
            len(segi),
            lat.cf["T"].data[0],
            lat_fix.cf["T"].data[0],
        )
    else:
        lon_nudged_array = np.array([])
        lat_nudged_array = np.array([])
        dt_nudged = np.array([], dtype="datetime64[ns]")
    if segi.any():
        # Return difference of numpy timestamps in units of minutes
        seg_min = (lat.cf["T"].data[segi][-1] - lat.cf["T"].data[segi][0]).astype(
            "timedelta64[s]"
        ).astype(float) / 60.0
    else:
        seg_min = 0
    logger.info(
        f"{' ':5}  {'-':>12} {'-':>12} {'-':>12} {len(segi):-9d} {seg_min:9.2f} {'-':>14} {'-':>14} {'-':>29}",  # noqa: E501, G004
    )

    MIN_SEGMENT_LENGTH = 10
    seg_count = 0
    seg_minsum = 0
    error_message = ""
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

        try:
            end_lon_diff = float(lon_fix[i + 1]) - float(lon[segi[-1]])
            end_lat_diff = float(lat_fix[i + 1]) - float(lat[segi[-1]])
        except IndexError as e:
            logger.warning("IndexError computing end_lon_diff/end_lat_diff: %s", e)
            logger.info(
                "Setting end_lon_diff and end_lat_diff to 0 - error likely due "
                "filtering out bad GPS time data in nc42netdefs.py"
            )
            end_lat_diff = 0
            end_lon_diff = 0

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
            if log_file:
                logger.info(
                    "Fix this error by calling _range_qc_combined_nc() in "
                    "_navigation_process() and/or _gps_process() for %s",
                    log_file,
                )
                logger.info("Run to get a plot: combine.py -v 1 --plot --log_file %s", log_file)
            elif auv_name and mission:
                logger.info(
                    "Fix this error by calling _range_qc_combined_nc() in "
                    "_navigation_process() and/or _gps_process() for %s %s",
                    auv_name,
                    mission,
                )
            error_message = (
                f"abs(end_lon_diff) ({end_lon_diff}) > 1 or abs(end_lat_diff) ({end_lat_diff}) > 1"
            )
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
                f"{seg_count:5d}: {end_sec_diff:12.3f} {end_lon_diff:12.7f}"  # noqa: G004
                f" {end_lat_diff:12.7f} {len(segi):-9d} {seg_min:9.2f}"
                f" {u_drift:14.3f} {v_drift:14.3f} {lat.cf['T'].data[segi][-1]}",
            )
            seg_count += 1

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
                float(np.max(np.abs(lon[segi] + lon_nudge))),
            )
            logger.warning(
                " max(abs(lat)) = %s",
                float(np.max(np.abs(lat[segi] + lat_nudge))),
            )

        lon_nudged_array = np.append(lon_nudged_array, lon[segi] + lon_nudge)
        lat_nudged_array = np.append(lat_nudged_array, lat[segi] + lat_nudge)
        dt_nudged = np.append(dt_nudged, lon.cf["T"].data[segi])

    # Any dead reckoned points after last GPS fix
    segi = np.where(lat.cf["T"].data > lat_fix.cf["T"].data[-1])[0]
    seg_min = 0
    if segi.any():
        lon_nudged_array = np.append(lon_nudged_array, lon[segi])
        lat_nudged_array = np.append(lat_nudged_array, lat[segi])
        dt_nudged = np.append(dt_nudged, lon.cf["T"].data[segi])
        seg_min = float(lat.cf["T"].data[segi][-1] - lat.cf["T"].data[segi][0]) / 1.0e9 / 60

    logger.info(
        f"{seg_count + 1:5d}: {'-':>12} {'-':>12} {'-':>12} {len(segi):-9d} {seg_min:9.2f} {'-':>14} {'-':>14}",  # noqa: E501, G004
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

    # Optional plotting code - raise error after opportunity to plot
    if create_plots:
        _create_nudge_plots(
            lat,
            lon,
            lat_fix,
            lon_fix,
            lat_nudged,
            lon_nudged,
            auv_name,
            mission,
            logger,
            nav_depth=nav_depth,
            orig_depth=orig_depth,
        )

    if error_message:
        logger.error("Nudge positions error: %s", error_message)
        raise ValueError(error_message)

    return lon_nudged, lat_nudged, segment_count, segment_minsum


def _create_nudge_plots(  # noqa: PLR0913, PLR0915
    lat,
    lon,
    lat_fix,
    lon_fix,
    lat_nudged,
    lon_nudged,
    auv_name,
    mission,
    logger,
    nav_depth=None,
    orig_depth=None,
):
    """Create debug plots for position nudging (separated for clarity).

    Args:
        lat: Original latitude DataArray
        lon: Original longitude DataArray
        lat_fix: GPS latitude fixes DataArray
        lon_fix: GPS longitude fixes DataArray
        lat_nudged: Nudged latitude DataArray
        lon_nudged: Nudged longitude DataArray
        auv_name: Name of the AUV for plot titles
        mission: Mission identifier for plot titles
        logger: Logger instance for debug messages
        nav_depth: Optional depth from combined file for comparison plotting
        orig_depth: Optional depth from original Group file for comparison plotting
    """
    try:
        import matplotlib.pyplot as plt

        try:
            import cartopy.crs as ccrs  # type: ignore  # noqa: I001, PGH003
            from matplotlib import patches
            from shapely.geometry import LineString  # type: ignore  # noqa: PGH003

            has_cartopy = True
        except ImportError:
            has_cartopy = False

        # Time series plots - include depth if available
        has_depth = nav_depth is not None or orig_depth is not None
        nrows = 3 if has_depth else 2
        fig, axes = plt.subplots(nrows=nrows, figsize=(18, 9 if has_depth else 6))

        axes[0].plot(lat_nudged.coords["time"].data, lat_nudged, "-")
        axes[0].plot(lat.cf["T"].data, lat, "--")
        axes[0].plot(lat_fix.cf["T"].data, lat_fix, "*")
        axes[0].set_ylabel("Latitude")
        axes[0].legend(["Nudged", "Original", "GPS Fixes"])
        axes[0].grid()

        axes[1].plot(lon_nudged.coords["time"].data, lon_nudged, "-")
        axes[1].plot(lon.cf["T"].data, lon, "--")
        axes[1].plot(lon_fix.cf["T"].data, lon_fix, "*")
        axes[1].set_ylabel("Longitude")
        axes[1].legend(["Nudged", "Original", "GPS Fixes"])
        axes[1].grid()

        # Add depth subplot if data is available
        if has_depth:
            if orig_depth is not None:
                axes[2].plot(
                    orig_depth.cf["T"].data,
                    orig_depth,
                    "--",
                    label="Original (Group File)",
                    color="gray",
                )
            if nav_depth is not None:
                axes[2].plot(nav_depth.cf["T"].data, nav_depth, "-", label="Combined", color="red")
            axes[2].set_ylabel("Depth (m)")
            axes[2].set_xlabel("Time")
            axes[2].legend()
            axes[2].grid()
            axes[2].invert_yaxis()  # Depth increases downward

        title = "Corrected nav from nudge_positions()"
        fig.suptitle(title)
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


def simplify_points(pts, tolerance):
    anchor = 0
    floater = len(pts) - 1
    stack = []
    keep = set()

    stack.append((anchor, floater))
    while stack:
        anchor, floater = stack.pop()

        # initialize line segment
        if pts[floater] != pts[anchor]:
            anchorX = float(pts[floater][0] - pts[anchor][0])
            anchorY = float(pts[floater][1] - pts[anchor][1])
            seg_len = math.sqrt(anchorX**2 + anchorY**2)
            # get the unit vector
            anchorX /= seg_len
            anchorY /= seg_len
        else:
            anchorX = anchorY = seg_len = 0.0

        # inner loop:
        max_dist = 0.0
        farthest = anchor + 1
        for i in range(anchor + 1, floater):
            dist_to_seg = 0.0
            # compare to anchor
            vecX = float(pts[i][0] - pts[anchor][0])
            vecY = float(pts[i][1] - pts[anchor][1])
            seg_len = math.sqrt(vecX**2 + vecY**2)
            # dot product:
            proj = vecX * anchorX + vecY * anchorY
            if proj < 0.0:
                dist_to_seg = seg_len
            else:
                # compare to floater
                vecX = float(pts[i][0] - pts[floater][0])
                vecY = float(pts[i][1] - pts[floater][1])
                seg_len = math.sqrt(vecX**2 + vecY**2)
                # dot product:
                proj = vecX * (-anchorX) + vecY * (-anchorY)
                if proj < 0.0:  # noqa: SIM108
                    dist_to_seg = seg_len
                else:  # calculate perpendicular distance to line (pythagorean theorem):
                    dist_to_seg = math.sqrt(abs(seg_len**2 - proj**2))
                if max_dist < dist_to_seg:
                    max_dist = dist_to_seg
                    farthest = i

        if max_dist <= tolerance:  # use line segment
            keep.add(anchor)
            keep.add(floater)
        else:
            stack.append((anchor, farthest))
            stack.append((farthest, floater))

    keep = list(keep)
    keep.sort()
    # Change from original code: add the index from the original line in the return
    return [(pts[i] + (i,)) for i in keep]
