# noqa: INP001
"""Tests for CreateProducts._grid_dims() — the depth-axis max used by the
color-shaded quick look plots (2column, biolume, planktivore, engineering)."""

import numpy as np
import pandas as pd
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
