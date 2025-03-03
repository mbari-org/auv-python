#!/usr/bin/env python
"""
Base classes and utility functions for reading and writing data for MBARI's
Dorado class AUVs.

--
Mike McCann
MBARI 30 March 2020
"""

import sys
from datetime import datetime, timezone

import coards
import numpy as np


def monotonic_increasing_time_indices(time_array: np.array) -> np.ndarray:
    monotonic = []
    last_t = 0.0 if isinstance(time_array[0], np.float64) else datetime.min
    for t in time_array:
        if t > last_t:
            monotonic.append(True)
            last_t = t
        else:
            monotonic.append(False)
    return np.array(monotonic)


class AUV:
    def add_global_metadata(self):
        iso_now = datetime.now(timezone.utc).isoformat() + "Z"

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
        self.nc_file.license = self.nc_file.distribution_statement
        self.nc_file.useconst = "Not intended for legal use. Data may contain inaccuracies."
        self.nc_file.history = 'Created by "{}" on {}'.format(
            " ".join(sys.argv),
            iso_now,
        )
