# noqa: INP001
"""Tests for Combine_NetCDF._physical_coordinate_qc() — removal of corrupt
(non-physical) longitude/latitude samples before positions are nudged."""

import numpy as np
import xarray as xr
from combine import Combine_NetCDF

_N = 10


def _inst_var(name: str, values: np.ndarray, t0: str) -> tuple[str, xr.DataArray]:
    """A combined_nc style variable: each variable has its own <name>_time coordinate."""
    times = np.datetime64(t0) + np.arange(len(values)) * np.timedelta64(1, "s")
    return name, xr.DataArray(
        values,
        dims=[f"{name}_time"],
        coords={f"{name}_time": times},
        name=name,
    )


def _combined(lon: np.ndarray, lat: np.ndarray) -> Combine_NetCDF:
    c = Combine_NetCDF(log_file="triton/missionlogs/2026/x/y/z.nc4")
    c.combined_nc = xr.Dataset(
        dict(
            [
                _inst_var("universals_longitude", lon, "2026-11-04T13:00:00"),
                _inst_var("universals_latitude", lat, "2026-11-04T13:00:00"),
                _inst_var("universals_depth", np.linspace(1, 10, len(lon)), "2026-11-04T13:00:00"),
            ]
        )
    )
    return c


class TestPhysicalCoordinateQc:
    def test_corrupt_longitude_sample_removed(self, caplog):
        lon = np.full(_N, -122.0)
        lon[4] = 1.3e27
        c = _combined(lon, np.full(_N, 36.7))

        with caplog.at_level("WARNING", logger=c.logger.name):
            c._physical_coordinate_qc()

        assert "outside the physical range" in caplog.text  # noqa: S101
        assert c.combined_nc["universals_longitude"].size == _N - 1  # noqa: S101
        assert np.all(np.abs(c.combined_nc["universals_longitude"].to_numpy()) <= 180)  # noqa: PLR2004, S101
        # Variables of the instrument stay the same length as each other
        assert c.combined_nc["universals_latitude"].size == _N - 1  # noqa: S101
        assert c.combined_nc["universals_depth"].size == _N - 1  # noqa: S101

    def test_corrupt_latitude_sample_removed(self):
        lat = np.full(_N, 36.7)
        lat[2] = -4.0e10
        c = _combined(np.full(_N, -122.0), lat)

        c._physical_coordinate_qc()

        assert c.combined_nc["universals_latitude"].size == _N - 1  # noqa: S101
        assert np.all(np.abs(c.combined_nc["universals_latitude"].to_numpy()) <= 90)  # noqa: PLR2004, S101

    def test_valid_positions_untouched(self, caplog):
        lon = np.linspace(-130.0, 100.0, _N)  # outside Monterey Bay, but physical
        c = _combined(lon, np.linspace(-60.0, 60.0, _N))
        before = c.combined_nc.copy(deep=True)

        with caplog.at_level("WARNING", logger=c.logger.name):
            c._physical_coordinate_qc()

        assert "outside the physical range" not in caplog.text  # noqa: S101
        xr.testing.assert_identical(c.combined_nc, before)

    def test_nan_positions_not_treated_as_out_of_range(self):
        lon = np.full(_N, -122.0)
        lon[3] = np.nan
        c = _combined(lon, np.full(_N, 36.7))

        c._physical_coordinate_qc()

        assert c.combined_nc["universals_longitude"].size == _N  # noqa: S101

    def test_missing_position_variables_ok(self):
        c = Combine_NetCDF(log_file="triton/missionlogs/2026/x/y/z.nc4")
        c.combined_nc = xr.Dataset(dict([_inst_var("ctd_temp", np.ones(_N), "2026-11-04")]))

        c._physical_coordinate_qc()

        assert "ctd_temp" in c.combined_nc  # noqa: S101
