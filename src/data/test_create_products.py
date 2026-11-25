# noqa: INP001
"""Tests for CreateProducts._grid_dims() — the depth-axis max used by the
color-shaded quick look plots (2column, biolume, planktivore, engineering)."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from create_products import CreateProducts

# True max depth in the test fixtures is 297.3m; expect it rounded up to 300m.
_TRUE_MAX_DEPTH = 297.3
_EXPECTED_MAX_DEPTH = 300.0


def _make_grid_dims_ds(depths: np.ndarray, sensor_values: dict[str, np.ndarray]) -> xr.Dataset:
    """Return a minimal Dataset with depth/time/lat/lon plus arbitrary sensor vars.

    Each entry in *sensor_values* becomes a ("time",) data variable, so its
    validity mask lines up with *depths* the way a real resampled dataset does.
    """
    n = len(depths)
    times = pd.date_range("2026-01-01", periods=n, freq="1s")
    lons = np.linspace(-122.0, -121.9, n)
    lats = np.linspace(36.7, 36.8, n)
    data_vars = {
        "depth": ("time", depths),
        "profile_number": ("time", np.arange(1, n + 1)),
    }
    for name, values in sensor_values.items():
        data_vars[name] = ("time", values)
    ds = xr.Dataset(
        data_vars,
        coords={
            "time": times,
            "latitude": ("time", lats),
            "longitude": ("time", lons),
        },
    )
    ds["depth"].attrs["standard_name"] = "depth"
    ds["time"].attrs["standard_name"] = "time"
    ds["latitude"].attrs["standard_name"] = "latitude"
    ds["longitude"].attrs["standard_name"] = "longitude"
    return ds


class TestGridDimsMaxDepth:
    def test_max_depth_rounds_up_not_down(self):
        """A true max depth that isn't a multiple of 10 must not be clipped by
        rounding down (the regression this fix addresses)."""
        depths = np.array([10.0, 50.0, 100.0, 297.3, 150.0])
        sensor = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # valid everywhere
        ds = _make_grid_dims_ds(depths, {"sensor": sensor})
        cp = CreateProducts(ds=ds)

        _, iz, _ = cp._grid_dims(["sensor"])

        assert max(iz) >= _TRUE_MAX_DEPTH  # noqa: S101
        assert max(iz) == _EXPECTED_MAX_DEPTH  # ceil to nearest 10, not floor to 290  # noqa: S101

    def test_deepest_point_from_second_variable_is_not_missed(self):
        """The deepest valid depth may belong to any one of several plotted
        variables — the union across all of them must still find it."""
        depths = np.array([10.0, 50.0, 100.0, 297.3, 150.0])
        # var_a is NaN at the deepest point; var_b is valid there.
        var_a = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
        var_b = np.array([np.nan, np.nan, np.nan, 4.0, np.nan])
        ds = _make_grid_dims_ds(depths, {"var_a": var_a, "var_b": var_b})
        cp = CreateProducts(ds=ds)

        _, iz, _ = cp._grid_dims(["var_a", "var_b"])

        assert max(iz) == _EXPECTED_MAX_DEPTH  # noqa: S101

    def test_mismatched_length_variable_is_skipped_without_crashing(self):
        """A plot_vars entry whose variable doesn't share depth's shape (e.g. a
        differently-gridded product) must be safely ignored, not raise."""
        depths = np.array([10.0, 50.0, 100.0, 297.3, 150.0])
        sensor = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ds = _make_grid_dims_ds(depths, {"sensor": sensor})
        ds["odd_shape"] = ("z", np.array([1.0, 2.0, 3.0]))

        cp = CreateProducts(ds=ds)
        _, iz, _ = cp._grid_dims(["sensor", "odd_shape"])

        assert max(iz) == _EXPECTED_MAX_DEPTH  # noqa: S101


class TestGridDimsCorruptPositions:
    """A few corrupt longitude/latitude values (e.g. ~1e26 degrees from a bad log
    sample) must not break the UTM zone or the distance along track."""

    def _ds_with_bad_positions(self):
        n = 50
        depths = np.linspace(10.0, 100.0, n)
        ds = _make_grid_dims_ds(depths, {"sensor": np.ones(n)})
        lons = ds["longitude"].to_numpy().copy()
        lats = ds["latitude"].to_numpy().copy()
        lons[20:28] = 1.0e26
        lats[30] = 5.0e30
        return ds.assign_coords(longitude=("time", lons), latitude=("time", lats))

    def test_corrupt_positions_do_not_raise(self, caplog):
        cp = CreateProducts(ds=self._ds_with_bad_positions())

        with caplog.at_level("WARNING", logger=cp.logger.name):
            _, _, distnav = cp._grid_dims(["sensor"])

        assert "Ignoring 9 position(s)" in caplog.text  # noqa: S101
        assert np.isfinite(distnav.to_numpy()).all()  # noqa: S101
        # ~0.1 deg lon (~9 km) + ~0.1 deg lat (~11 km) across the track
        assert 10_000 < distnav.to_numpy()[-1] < 30_000  # noqa: PLR2004, S101

    def test_distance_matches_clean_data(self):
        """Interpolating across the removed samples gives nearly the clean distance."""
        n = 50
        clean = _make_grid_dims_ds(np.linspace(10.0, 100.0, n), {"sensor": np.ones(n)})
        _, _, clean_dist = CreateProducts(ds=clean)._grid_dims(["sensor"])
        _, _, bad_dist = CreateProducts(ds=self._ds_with_bad_positions())._grid_dims(["sensor"])

        assert bad_dist.to_numpy()[-1] == pytest.approx(clean_dist.to_numpy()[-1], rel=0.01)  # noqa: S101

    def test_all_positions_corrupt_returns_empty_result(self):
        n = 10
        ds = _make_grid_dims_ds(np.linspace(10.0, 100.0, n), {"sensor": np.ones(n)})
        ds = ds.assign_coords(longitude=("time", np.full(n, 1.0e26)))

        idist, iz, distnav = CreateProducts(ds=ds)._grid_dims(["sensor"])

        # The existing "cannot compute" return value: empty arrays and a scalar NaN
        assert idist.size == 0  # noqa: S101
        assert iz.size == 0  # noqa: S101
        assert distnav.ndim == 0  # noqa: S101
