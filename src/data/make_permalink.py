#!/usr/bin/env python3

"""
Generate a permalink for specified mission, log_file, survey, deployment.
Can be used with the stoqs_all_dorado database to zoom in on the data.
"""

import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import lzstring
import numpy as np
import pandas as pd
import requests
import xarray as xr


def get_times_depths(ds_url):
    ds = xr.open_dataset(ds_url)
    stime = datetime.fromisoformat(ds.attrs.get("time_coverage_start").replace("Z", "+00:00"))
    etime = datetime.fromisoformat(ds.attrs.get("time_coverage_end").replace("Z", "+00:00"))

    min_depth = float(ds.attrs.get("geospatial_vertical_min"))
    max_depth = float(ds.attrs.get("geospatial_vertical_max"))

    return (stime, etime), (min_depth, max_depth)


def get_parameter_id(base_url, parameter_name="fl700_uncorr"):
    csv_query = base_url.rstrip("/") + "/api/parameter.csv?name__contains=" + parameter_name
    with requests.Session() as s:
        download = s.get(csv_query)
        decoded_content = download.content.decode("utf-8")
        cr = csv.DictReader(decoded_content.splitlines(), delimiter=",")
        for row in list(cr):
            parm_id = row["id"]
            break

    return parm_id


def gen_permalink(times, depths, platform_name):
    # Create link to examine this Sample in the STOQS UI within the context of other campaign data
    depth_time_platform = {
        # time and depth are simple dictionary entries
        "start-ems": ((min(times) - datetime(1970, 1, 1, tzinfo=UTC)).total_seconds() * 1000),
        "end-ems": ((max(times) - datetime(1970, 1, 1, tzinfo=UTC)).total_seconds() * 1000),
        "start-depth": min(depths) - 1,
        "end-depth": max(depths) + 1,
        # platform_clicks is a list of jQuery selectors to click on the
        # STOQS UI to select the platform(s)
        "platform_clicks": [f'#{platform_name} button.stoqs-toggle:contains("{platform_name}")'],
    }
    compressor = lzstring.LZString()
    return compressor.compressToEncodedURIComponent(
        json.dumps(depth_time_platform, separators=(",", ":")),
    )


def lrauv_stoqs_base_url(ds: xr.Dataset) -> str:
    """Return the STOQS base URL for the LRAUV database covering *ds*'s time range.

    Constructs a URL of the form::

        https://tethysviz.shore.mbari.org/stoqs_lrauv_<mon><yyyy>

    where ``<mon>`` is the 3-letter lowercase month abbreviation and ``<yyyy>``
    is the 4-digit year taken from the first timestamp in the dataset.
    """
    first_time = pd.Timestamp(ds.cf["time"].to_numpy()[0])
    month_str = first_time.strftime("%b").lower()  # e.g. "apr"
    year_str = first_time.strftime("%Y")  # e.g. "2025"
    return f"https://tethysviz.shore.mbari.org/stoqs_lrauv_{month_str}{year_str}"


def stoqs_url_from_ds(ds: xr.Dataset, base_url: str | None = None, auv_name: str = "") -> str:
    """Return a STOQS 'Share this view' URL that zooms to the data in *ds*.

    Derives the time window and depth range from actual data in the dataset,
    queries STOQS for the platform ID, then returns a ready-to-use permalink URL.
    Raises on network failure so callers can catch and degrade gracefully.

    Args:
        ds: Open xarray Dataset (e.g. the combined LRAUV deployment dataset).
        base_url: STOQS database base URL.  If *None* (default), the URL is
            derived automatically from the dataset's time range via
            ``lrauv_stoqs_base_url()``.
        auv_name: AUV/platform name fragment used to look up the platform ID
            (passed to ``name__icontains`` in the API query).
    """
    if base_url is None:
        base_url = lrauv_stoqs_base_url(ds)
    # --- times ---
    times_np = ds.cf["time"].to_numpy()
    stime = pd.Timestamp(times_np[0]).to_pydatetime().replace(tzinfo=UTC)
    etime = pd.Timestamp(times_np[-1]).to_pydatetime().replace(tzinfo=UTC)

    # --- depths ---
    depths_np = ds.cf["depth"].to_numpy()
    min_depth = float(np.nanmin(depths_np))
    max_depth = float(np.nanmax(depths_np))

    compressed = gen_permalink((stime, etime), (min_depth, max_depth), platform_name=auv_name)
    return f"{base_url.rstrip('/')}/query/?permalink_id={compressed}"


if __name__ == "__main__":
    try:
        ds_url = sys.argv[1]
    except IndexError:
        print("Need argement of OPeNDAP url to dataset with ACDD metadata.")  # noqa: T201
        sys.exit(-1)

    base_url = "https://stoqs.shore.mbari.org/stoqs_all_dorado"
    parm_id = get_parameter_id(base_url)
    times, depths = get_times_depths(ds_url)
    print(  # noqa: T201
        Path(
            base_url,
            "query/?permalink_id=" + gen_permalink(times, depths, parm_id),
        ),
    )
