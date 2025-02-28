# cat make_permalink.py
#!/usr/bin/env python3

"""
Generate a permalink for specified mission or survey.
Can be used with the stoqs_all_dorado database to zoom in on the data.
"""

import csv
import json
import lzstring
import os
import requests
import sys
import xarray as xr
from datetime import datetime


def get_times_depths(ds_url):
    ds = xr.open_dataset(ds_url)
    stime = datetime.strptime(ds.attrs.get("time_coverage_start"), "%Y-%m-%dT%H:%M:%SZ")
    etime = datetime.strptime(ds.attrs.get("time_coverage_end"), "%Y-%m-%dT%H:%M:%SZ")

    min_depth = float(ds.attrs.get("geospatial_vertical_min"))
    max_depth = float(ds.attrs.get("geospatial_vertical_max"))

    return (stime, etime), (min_depth, max_depth)


def get_parameter_id(base_url, parameter_name="fl700_uncorr"):
    csv_query = os.path.join(
        base_url, "api/parameter.csv?name__contains=" + parameter_name
    )
    with requests.Session() as s:
        download = s.get(csv_query)
        decoded_content = download.content.decode("utf-8")
        cr = csv.DictReader(decoded_content.splitlines(), delimiter=",")
        for row in list(cr):
            parm_id = row["id"]
            break

    return parm_id


def gen_permalink(times, depths, parm_id):
    # Create link to examine this Sample in the STOQS UI within the context of other campaign data
    depth_time = {
        "start-ems": (min(times) - datetime(1970, 1, 1)).total_seconds() * 1000,
        "end-ems": (max(times) - datetime(1970, 1, 1)).total_seconds() * 1000,
        "start-depth": min(depths) - 1,
        "end-depth": max(depths) + 1,
        "tabs": [["#temporalTabs", 0], ["#spatialTabs", 1]],
        "clicks": [
            'input[type="checkbox"][id="zoomtoextentonupdate"]',
            'input[type="radio"][name="show-mp-3d"][value="all"]',
            'input[type="radio"][name="parameters_plot"][value="' + parm_id + '"]',
            'input[type="radio"][name="colormap_choice"][value="algae"]',
            'input[type="checkbox"][id="showgeox3dmeasurement"]',
            'input[type="checkbox"][id="showplatforms"]',
        ],
    }
    ##print(depth_time)
    compressor = lzstring.LZString()
    permalink = compressor.compressToEncodedURIComponent(
        json.dumps(depth_time, separators=(",", ":"))
    )
    return permalink


if __name__ == "__main__":
    try:
        ds_url = sys.argv[1]
    except IndexError:
        print("Need argement of OPeNDAP url to dataset with ACDD metadata.")
        sys.exit(-1)

    base_url = "https://stoqs.shore.mbari.org/stoqs_all_dorado"
    parm_id = get_parameter_id(base_url)
    times, depths = get_times_depths(ds_url)
    print(
        os.path.join(
            base_url, "query/?permalink_id=" + gen_permalink(times, depths, parm_id)
        )
    )
