#!/usr/bin/env python
"""
Process realtime SBD shore.nc4 files into resampled netCDF and quick-look plots.

Finds shore.nc4 files from:
    smb://atlas.shore.mbari.org/LRAUV/<vehicle>/realtime/sbdlogs/

and produces:
    data/lrauv_data/<vehicle>/realtime/sbdlogs/<YYYY>/<YYYYMMDD_YYYYMMDD>/
        <vehicle>_<YYYYMMDD_YYYYMMDD>_sbd_1S.nc
        <vehicle>_<YYYYMMDD_YYYYMMDD>_sbd_1S_2column_cmocean.png
        <vehicle>_<YYYYMMDD_YYYYMMDD>_sbd_1S_2column_planktivore.png  (if applicable)

Usage:
    # All vehicles for the current month:
    uv run src/data/process_lrauv_sbd.py --current_month -v

    # All vehicles for the previous month:
    uv run src/data/process_lrauv_sbd.py --previous_month -v

    # Last 7 days for a single vehicle:
    uv run src/data/process_lrauv_sbd.py --last_n_days 7 --auv_name ahi -v

    # Explicit date range for a single vehicle:
    uv run src/data/process_lrauv_sbd.py --start 20260406 --end 20260412 --auv_name ahi -v
"""

__author__ = "Mike McCann"
__copyright__ = "Copyright 2026, Monterey Bay Aquarium Research Institute"

import argparse
import logging
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import xarray as xr
from common_args import ALL_LRAUV_NAMES
from nc42netcdfs import BASE_LRAUV_PATH
from resample import LRAUV_OPENDAP_BASE
from sbd2netcdf import FREQ, SbdExtract

_LOG_LEVELS = (logging.WARN, logging.INFO, logging.DEBUG)
_formatter = logging.Formatter(
    "%(levelname)s %(asctime)s %(filename)s %(funcName)s():%(lineno)d [%(process)d] %(message)s"
)
_handler = logging.StreamHandler()
_handler.setFormatter(_formatter)
logger = logging.getLogger(__name__)
logger.addHandler(_handler)


def _to_opendap_url(path: Path, vehicle_dir: str) -> str:
    """Convert a local shore_1S.nc path to its OPeNDAP URL."""
    for local_base in (Path(vehicle_dir), BASE_LRAUV_PATH):
        try:
            rel = path.relative_to(local_base)
            return LRAUV_OPENDAP_BASE.rstrip("/") + "/" + str(rel)
        except ValueError:
            continue
    return str(path)


def _parse_dt(s: str) -> datetime:
    for fmt in ("%Y%m%dT%H%M%S", "%Y%m%d"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    msg = f"Unrecognised datetime format: {s!r}. Use YYYYMMDDTHHMMSS or YYYYMMDD."
    raise ValueError(msg)


def _rel(path: Path, base: str) -> Path:
    """Return path relative to base, falling back to the full path."""
    try:
        return path.relative_to(base)
    except ValueError:
        return path


def process_command_line() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--auv_name",
        nargs="+",
        default=list(ALL_LRAUV_NAMES),
        metavar="NAME",
        help="One or more AUV names (default: all known LRAUVs)",
    )
    date_mode = parser.add_mutually_exclusive_group(required=True)
    date_mode.add_argument(
        "--current_month",
        action="store_true",
        help="Process from the first of the current month through today",
    )
    date_mode.add_argument(
        "--previous_month",
        action="store_true",
        help="Process the entire previous calendar month",
    )
    date_mode.add_argument(
        "--last_n_days",
        type=int,
        metavar="N",
        help="Process the last N days",
    )
    date_mode.add_argument(
        "--start",
        metavar="YYYYMMDD",
        help="Start datetime: YYYYMMDDTHHMMSS or YYYYMMDD (use with --end)",
    )
    parser.add_argument(
        "--end",
        metavar="YYYYMMDD",
        help="End datetime: YYYYMMDDTHHMMSS or YYYYMMDD (only used with --start)",
    )
    parser.add_argument(
        "--vehicle_dir",
        default="/Volumes/LRAUV",
        help="Root LRAUV vehicle directory (default: /Volumes/LRAUV)",
    )
    parser.add_argument(
        "--clobber",
        action="store_true",
        help="Overwrite existing output files",
    )
    parser.add_argument(
        "--noproducts",
        action="store_true",
        help="Skip create_products step (write netCDF only)",
    )
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
    if args.start and not args.end:
        parser.error("--end is required when --start is used")
    return args


def _resolve_date_range(args: argparse.Namespace) -> tuple[datetime, datetime]:
    """Return (start, end) UTC datetimes from the active date-mode flag."""
    now = datetime.now(tz=UTC)
    if args.current_month:
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end = now
    elif args.previous_month:
        first_of_this = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        last_of_prev = first_of_this - timedelta(days=1)
        start = last_of_prev.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end = last_of_prev.replace(hour=23, minute=59, second=59, microsecond=0)
    elif args.last_n_days is not None:
        start = (now - timedelta(days=args.last_n_days)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        end = now
    else:
        start = _parse_dt(args.start)
        end = _parse_dt(args.end)
    return start, end


def _concat_month_files(output_dir: "Path") -> "tuple[xr.Dataset, list[Path]]":
    """Concatenate all shore_1S.nc files in output_dir into one sorted dataset."""
    import xarray as xr

    month_files = sorted(output_dir.rglob(f"shore_{FREQ}.nc"))
    logger.info("Combining %d shore_%s.nc files from %s", len(month_files), FREQ, output_dir)
    _GEO_COORDS = ("depth", "latitude", "longitude")
    individual = []
    for p in month_files:
        d = xr.open_dataset(p)
        present = [c for c in _GEO_COORDS if c in d.coords]
        if present:
            d = d.reset_coords(present)
        individual.append(d)
    ds = xr.concat(individual, dim="time", join="outer")
    ds = ds.sortby("time").drop_duplicates("time")
    geo_present = [c for c in _GEO_COORDS if c in ds]
    if geo_present:
        ds = ds.set_coords(geo_present)
    return ds, month_files


def _make_per_log_plots(
    args: argparse.Namespace,
    month_files: list,
    mission: str,
    CreateProducts: type,  # noqa: N803
) -> None:
    """Create per-log plots for each shore_1S.nc in month_files."""
    for p in month_files:
        sentinel = p.parent / f"shore_{FREQ}.plotted"
        if sentinel.exists() and not args.clobber:
            if p.stat().st_mtime <= sentinel.stat().st_mtime:
                logger.info("Per-log plots up to date, skipping: %s", _rel(p, args.vehicle_dir))
                continue
            logger.info("shore_1S.nc is newer — replotting: %s", _rel(p, args.vehicle_dir))
        else:
            logger.info("Creating per-log plots for %s", _rel(p, args.vehicle_dir))
        try:
            with xr.open_dataset(p) as ds_log:
                cp = CreateProducts(
                    auv_name=args.auv_name,
                    mission=mission,
                    base_path=args.vehicle_dir,
                    freq=FREQ,
                    verbose=args.verbose,
                    ds=ds_log,
                    log_file=str(p),
                    nc_files=[str(p)],
                    output_dir=p.parent,
                    plot_name_stem="shore",
                )
                cp.plot_2column()
                cp.plot_planktivore_2column()
                cp.plot_engineering_2column()
                cp.plot_cbit_2column()
        except Exception as e:  # noqa: BLE001
            logger.warning("Per-log plot failed for %s: %s", p.name, e)
        sentinel.touch()


def _make_products(
    args: argparse.Namespace, start: datetime, end: datetime, out_paths: list
) -> None:
    """Build plots and HTML index from the shore_1S.nc files produced by SbdExtract."""
    import importlib.util

    if importlib.util.find_spec("create_products") is None:
        logger.warning("create_products not available — skipping plots")
        return

    try:
        from create_products import CreateProducts
    except ImportError as e:
        logger.warning("Cannot import create_products: %s — skipping plots", e)
        return

    date_range = f"{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"
    mission = f"realtime/sbdlogs/{start.strftime('%Y')}/{date_range}"

    output_dir = out_paths[0].parent.parent  # YYYYMM directory
    ds, month_files = _concat_month_files(output_dir)

    monthly_png = output_dir / f"{output_dir.name}_sbd_{FREQ}_2column_cmocean.png"
    if monthly_png.exists() and not args.clobber:
        newest_nc = max(p.stat().st_mtime for p in month_files)
        if newest_nc <= monthly_png.stat().st_mtime:
            logger.info("Monthly plots up to date for %s — skipping", output_dir.name)
            return

    plot_stem = f"{output_dir.name}_sbd"
    cp = CreateProducts(
        auv_name=args.auv_name,
        mission=mission,
        base_path=args.vehicle_dir,
        freq="1S",
        verbose=args.verbose,
        ds=ds,
        log_file="sbd_placeholder.nc4",
        nc_files=[str(p) for p in month_files],
        output_dir=output_dir,
        plot_name_stem=plot_stem,
    )
    _make_per_log_plots(args, month_files, mission, CreateProducts)

    cp.plot_2column()
    cp.plot_planktivore_2column()
    cp.plot_engineering_2column()
    cp.plot_cbit_2column()

    png_paths = cp.sbd_png_paths()
    if png_paths:
        from lrauv_deployment_plots import DeploymentPlotter
        from make_permalink import stoqs_url_from_ds

        dp = DeploymentPlotter()
        dp.logger.setLevel(_LOG_LEVELS[min(args.verbose, 2)])

        stoqs_url = None
        try:
            stoqs_url = stoqs_url_from_ds(ds, auv_name=args.auv_name)
        except Exception as e:  # noqa: BLE001
            logger.debug("Could not generate STOQS URL: %s", e)

        import pandas as pd

        month_year = pd.to_datetime(ds.cf["time"].to_numpy()[0]).strftime("%B %Y")
        html_title = f"Interpolated realtime SBD data for {args.auv_name} in {month_year}"

        nc_file_strs = [_to_opendap_url(p, args.vehicle_dir) for p in month_files]
        nc_durations: dict[str, int] = {}
        for p, url in zip(month_files, nc_file_strs, strict=True):
            with xr.open_dataset(p) as d:
                times = d.cf["time"].to_numpy()
                if len(times) > 1:
                    nc_durations[url] = int(
                        (times[-1] - times[0]).astype("timedelta64[m]").astype(int)
                    )
        html_paths = [p.with_suffix(".html") for p in png_paths]
        for png_path, html_path in zip(png_paths, html_paths, strict=True):
            dp._write_per_png_html(
                html_path=html_path,
                title=html_title,
                png_name=png_path.name,
                png_url="",
                stoqs_url=stoqs_url,
                nc_files=nc_file_strs,
                auv_name=args.auv_name,
                png_file_path=png_path,
                other_png_paths=[str(p) for p in png_paths if p != png_path],
                nc_durations=nc_durations,
            )

        dp._update_index_html(
            output_dir,
            html_paths,
            deployment_name=f"{args.auv_name} {output_dir.name}",
            clobber=True,
        )


def main() -> None:
    args = process_command_line()
    logger.setLevel(_LOG_LEVELS[min(args.verbose, 2)])

    start, end = _resolve_date_range(args)
    logger.info("Date range: %s → %s", start.date(), end.date())

    for auv_name in args.auv_name:
        logger.info("Processing %s", auv_name)
        vehicle_args = argparse.Namespace(**{**vars(args), "auv_name": auv_name})

        extractor = SbdExtract(
            auv_name=auv_name,
            start=start,
            end=end,
            vehicle_dir=args.vehicle_dir,
            verbose=args.verbose,
            clobber=args.clobber,
            commandline=" ".join(sys.argv),
        )

        out_paths = extractor.process()
        if not out_paths:
            logger.warning("No data for %s — skipping", auv_name)
            continue

        if args.noproducts:
            logger.info("Skipping create_products (--noproducts specified)")
            continue

        logger.info("Creating products from %d shore_1S.nc files for %s", len(out_paths), auv_name)
        try:
            _make_products(vehicle_args, start, end, out_paths)
        except RuntimeError as e:
            logger.warning("create_products failed for %s: %s", auv_name, e)


if __name__ == "__main__":
    AUV_NAME = "ahi"
    LRAUV_DIR = "/Volumes/LRAUV"
    main()
