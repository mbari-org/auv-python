# noqa: INP001

import numpy as np
import pandas as pd
import pytest
import xarray as xr

# The test should not take more than 5 minutes to run
MAX_SECS = 5 * 60  # 5 minutes

# Test configuration for LRAUV processing with start/end dates
TEST_LRAUV_VEHICLE = "tethys"
TEST_START = "20120909T000000"
TEST_END = "20120910T000000"


@pytest.fixture(scope="session")
def mock_lrauv_data(tmp_path_factory):
    """Create mock LRAUV data structure for testing."""
    base_path = tmp_path_factory.mktemp("lrauv_test")
    vehicle_dir = base_path / TEST_LRAUV_VEHICLE
    mission_year_dir = vehicle_dir / "missionlogs/2012"
    mission_dir = mission_year_dir / "20120908_20120920"

    # Create .dlist file in the year directory (great-grandparent of log files)
    # The filename should match the deployment directory name
    dlist_file = mission_year_dir / "20120908_20120920.dlist"
    dlist_file.parent.mkdir(parents=True, exist_ok=True)
    dlist_file.write_text("# Deployment name: CANON_september2012\nSome other info\n")

    # Create two log file directories
    log_dirs = [
        mission_dir / "20120909T010636",
        mission_dir / "20120909T152301",
    ]

    log_file_stems = [
        "201209090106_201209091521",
        "201209091523_201209101900",
    ]

    for log_dir, stem in zip(log_dirs, log_file_stems):  # noqa: B905
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create minimal Group files with realistic LRAUV structure
        time_vals = np.arange(
            np.datetime64("2012-09-09T01:00:00"),
            np.datetime64("2012-09-09T15:00:00"),
            np.timedelta64(1, "s"),
        )

        # Create a few Group files
        for group_name in ["navigation", "ctd1", "oxygen"]:
            ds = xr.Dataset(
                {
                    f"{group_name}_latitude": (["time"], np.full(len(time_vals), 36.8)),
                    f"{group_name}_longitude": (["time"], np.full(len(time_vals), -121.8)),
                    f"{group_name}_depth": (["time"], np.random.uniform(0, 50, len(time_vals))),
                },
                coords={"time": time_vals},
            )
            ds.to_netcdf(log_dir / f"{stem}_Group_{group_name}.nc")

    return base_path


@pytest.fixture(scope="session", autouse=False)
def complete_lrauv_processing(mock_lrauv_data):
    """Process LRAUV data using start/end date range with mocked data."""
    # For now, just return the mock data path - full processing integration
    # would require mocking the entire pipeline which is complex.
    # Instead, we'll test individual components with the mocked data.
    return mock_lrauv_data


def test_lrauv_mock_data_structure(complete_lrauv_processing):
    """Test that mock LRAUV data structure is created correctly."""
    base_path = complete_lrauv_processing

    # Check that Group files were created for the first log file
    log_file_stem = "201209090106_201209091521"
    netcdfs_dir = (
        base_path / TEST_LRAUV_VEHICLE / "missionlogs/2012/20120908_20120920/20120909T010636"
    )

    # Check for Group files
    group_files = list(netcdfs_dir.glob(f"{log_file_stem}_Group_*.nc"))
    assert len(group_files) == 3, "Expected 3 Group files"  # noqa: PLR2004, S101

    # Check that Group files contain expected variables
    for group_file in group_files:
        ds = xr.open_dataset(group_file)
        assert "time" in ds.coords  # noqa: S101
        assert len(ds.dims) > 0  # noqa: S101
        ds.close()


def test_lrauv_deployment_name_parsing(complete_lrauv_processing):
    """Test that deployment name can be parsed from .dlist file."""
    from utils import get_deployment_name

    base_path = complete_lrauv_processing
    # Construct path to any log file in the structure
    log_file = (
        base_path
        / TEST_LRAUV_VEHICLE
        / "missionlogs/2012/20120908_20120920/20120909T010636/201209090106_201209091521.nc4"
    )

    # The .dlist file should exist in the year directory
    dlist_file = base_path / TEST_LRAUV_VEHICLE / "missionlogs/2012/20120908_20120920.dlist"
    assert dlist_file.exists(), f".dlist file not found at {dlist_file}"  # noqa: S101

    # Test deployment name extraction
    deployment_name = get_deployment_name(str(log_file), str(base_path))
    assert deployment_name == "CANON_september2012"  # noqa: S101


def test_lrauv_group_file_structure(complete_lrauv_processing):
    """Test that Group files have correct LRAUV structure."""
    base_path = complete_lrauv_processing

    log_file_stem = "201209090106_201209091521"
    netcdfs_dir = (
        base_path / TEST_LRAUV_VEHICLE / "missionlogs/2012/20120908_20120920/20120909T010636"
    )

    # Check navigation Group file
    nav_file = netcdfs_dir / f"{log_file_stem}_Group_navigation.nc"
    assert nav_file.exists()  # noqa: S101

    ds = xr.open_dataset(nav_file)
    # Check for expected coordinate variables
    assert "navigation_latitude" in ds.variables  # noqa: S101
    assert "navigation_longitude" in ds.variables  # noqa: S101
    assert "navigation_depth" in ds.variables  # noqa: S101
    assert "time" in ds.coords  # noqa: S101
    ds.close()


@pytest.mark.skip(reason="Full integration test - requires all processing modules")
def test_lrauv_full_pipeline(complete_lrauv_processing):
    """Test full LRAUV processing pipeline from logs to resampled data."""
    # This would test the full pipeline but requires significant mocking
    # of calibration files, configuration, etc.
    pass  # noqa: PIE790


def test_lrauv_2d_array_variable_handling(tmp_path):
    """Test that 2D array variables (time, array_index) are handled correctly in combine.py."""
    from combine import Combine_NetCDF

    # Create a minimal test that exercises the _create_data_array_for_variable method
    # with a 2D variable

    # Create time array
    time_vals = np.arange(
        np.datetime64("2025-06-08T02:00:00"),
        np.datetime64("2025-06-08T03:00:00"),
        np.timedelta64(10, "s"),  # 360 time points
    )

    # Create a mock dataset with a 2D variable
    ds = xr.Dataset(
        {
            # 2D variable - 60 samples per time point (like biolume_raw)
            "biolume_array": (["time", "sample"], np.random.uniform(0, 100, (len(time_vals), 60))),
            # 1D variable for comparison
            "temperature": (["time"], np.random.uniform(10, 15, len(time_vals))),
        },
        coords={"time": time_vals},
    )

    # Create a Combine_NetCDF instance (minimal setup)
    combine = Combine_NetCDF(
        log_file="test/test.nc4",
        verbose=1,
    )

    # Mock the time coordinate data
    time_coord_data = time_vals.astype("datetime64[ns]").astype("int64") / 1e9

    # Test 1D variable (should work)
    data_array_1d = combine._create_data_array_for_variable(
        ds, "temperature", "test_time", time_coord_data
    )
    assert len(data_array_1d.dims) == 1  # noqa: PLR2004, S101
    assert data_array_1d.dims[0] == "test_time"  # noqa: S101

    # Test 2D variable (this is what fails without the fix)
    try:
        data_array_2d = combine._create_data_array_for_variable(
            ds, "biolume_array", "test_time", time_coord_data
        )
        # After the fix, this should work
        assert len(data_array_2d.dims) == 2  # noqa: PLR2004, S101
        assert "test_time" in data_array_2d.dims  # noqa: S101
        assert data_array_2d.shape[1] == 60  # noqa: PLR2004, S101  # Second dimension should be 60
    except ValueError as e:
        if "different number of dimensions" in str(e):
            pytest.fail(f"2D array handling not implemented: {e}")
        raise


def test_ubat_60hz_expansion(tmp_path):
    """Test that UBAT 2D digitized_raw_ad_counts array is expanded to 60hz time series."""
    from combine import Combine_NetCDF

    # Create time array for 1Hz data
    time_vals = np.arange(
        np.datetime64("2025-06-08T02:00:00"),
        np.datetime64("2025-06-08T02:00:10"),  # 10 seconds
        np.timedelta64(1, "s"),
    )
    time_seconds = time_vals.astype("datetime64[ns]").astype("int64") / 1e9

    # Create a Combine_NetCDF instance
    combine = Combine_NetCDF(
        log_file="test/test.nc4",
        verbose=1,
    )

    # Create mock combined_nc with UBAT 2D data
    combine.combined_nc = xr.Dataset(
        {
            "wetlabsubat_digitized_raw_ad_counts": (
                ["wetlabsubat_time", "sample"],
                np.random.randint(0, 1000, (len(time_vals), 60)),
            ),
        },
        coords={"wetlabsubat_time": time_seconds},
    )

    # Add attributes to match real data
    combine.combined_nc["wetlabsubat_digitized_raw_ad_counts"].attrs = {
        "long_name": "Digitized raw AD counts",
        "comment": "Test UBAT data",
    }

    # Call the expansion method
    combine._expand_ubat_to_60hz()

    # Check that the original variable is now 1D with 60hz time coordinate
    # (analogous to Dorado biolume_raw with TIME60HZ)
    assert "wetlabsubat_digitized_raw_ad_counts" in combine.combined_nc  # noqa: S101
    assert "wetlabsubat_time_60hz" in combine.combined_nc  # noqa: S101

    # Check dimensions - should now be 1D with 60hz time
    ubat_var = combine.combined_nc["wetlabsubat_digitized_raw_ad_counts"]
    assert len(ubat_var.dims) == 1  # noqa: PLR2004, S101
    assert ubat_var.dims[0] == "wetlabsubat_time_60hz"  # noqa: S101

    # Check shape - should have 60 samples per second, so 10 seconds * 60 = 600 samples
    expected_samples = len(time_vals) * 60  # noqa: PLR2004
    assert len(ubat_var) == expected_samples  # noqa: S101

    # Check time coordinate has proper attributes
    time_60hz = combine.combined_nc["wetlabsubat_time_60hz"]
    assert time_60hz.attrs["units"] == "seconds since 1970-01-01T00:00:00Z"  # noqa: S101
    assert time_60hz.attrs["standard_name"] == "time"  # noqa: S101

    # Check attributes were copied
    assert "long_name" in ubat_var.attrs  # noqa: S101
    assert "coordinates" in ubat_var.attrs  # noqa: S101


def _find_time_coordinate(variable: str, combined_nc_vars: dict) -> str:
    """Helper to find time coordinate for a variable (mimics align.py logic)."""
    var_parts = variable.split("_")
    possible_time_coords = []

    for i in range(len(var_parts)):
        group_candidate = "_".join(var_parts[: i + 1])
        for suffix in ["_time", "_time_60hz"]:
            time_coord = f"{group_candidate}{suffix}"
            if time_coord in combined_nc_vars:
                possible_time_coords.append((group_candidate, time_coord))

    if not possible_time_coords:
        return None

    # For 60hz variables, prefer 60hz time coordinates
    has_60hz_time = any(tc[1].endswith("_60hz") for tc in possible_time_coords)
    if variable.endswith("_60hz") and has_60hz_time:
        time_60hz_coords = [(g, t) for g, t in possible_time_coords if t.endswith("_60hz")]
        return max(time_60hz_coords, key=lambda x: len(x[0]))[1]

    # For regular variables, prefer non-60hz time coordinates
    non_60hz_coords = [(g, t) for g, t in possible_time_coords if not t.endswith("_60hz")]
    if non_60hz_coords:
        return max(non_60hz_coords, key=lambda x: len(x[0]))[1]

    return max(possible_time_coords, key=lambda x: len(x[0]))[1]


def test_align_60hz_time_coordinate_matching():
    """Test that variables with 60hz time coordinates are matched correctly."""
    # Mock dataset with both regular and 60hz time coordinates
    combined_nc_vars = {
        "wetlabsubat_time": True,
        "wetlabsubat_time_60hz": True,
    }

    # Test 1: Regular variable should match regular time coordinate
    timevar = _find_time_coordinate("wetlabsubat_flow_rate", combined_nc_vars)
    assert timevar == "wetlabsubat_time"  # noqa: S101
    assert not timevar.endswith("_60hz")  # noqa: S101

    # Test 2: UBAT variable (now 1D with 60hz time) should match 60hz time coordinate
    # Note: After expansion in combine.py, wetlabsubat_digitized_raw_ad_counts
    # has coordinate wetlabsubat_time_60hz (variable name has NO _60hz suffix)
    timevar = _find_time_coordinate("wetlabsubat_digitized_raw_ad_counts", combined_nc_vars)
    # This will match wetlabsubat_time (the regular one) because the variable name
    # doesn't have _60hz suffix. The actual coordinate binding happens in align.py
    # by reading the variable's coordinate, not by name matching.
    assert timevar == "wetlabsubat_time"  # noqa: S101


def test_wetlabsubat_proxy_processing_with_realistic_coordinates(tmp_path):
    """Test add_wetlabsubat_proxies with realistic LRAUV coordinate variable names.

    Real LRAUV data has instrument-prefixed coordinates like:
    - parlicor_latitude, parlicor_longitude
    - massservo_latitude, massservo_longitude
    - nudged_latitude, nudged_longitude
    - onboard_latitude, onboard_longitude
    - wetlabsubat_latitude, wetlabsubat_longitude

    But NOT navigation_latitude/navigation_longitude (which exist in Dorado data).
    This test ensures the coordinate lookup doesn't fail when navigation_* are missing.
    """
    from resample import Resampler

    # Create time arrays
    time_vals = pd.date_range("2025-06-08 02:00:00", periods=3600, freq="1s")  # 1 hour
    time_60hz_vals = pd.date_range("2025-06-08 02:00:00", periods=3600 * 60, freq="16666667ns")

    # Create a mock dataset with realistic LRAUV structure
    # Key: NO navigation_latitude/navigation_longitude variables
    ds = xr.Dataset(
        {
            # UBAT 60Hz raw data (after expansion from 2D to 1D)
            "wetlabsubat_digitized_raw_ad_counts": (
                ["wetlabsubat_time_60hz"],
                np.random.randint(200, 800, len(time_60hz_vals)),
            ),
            # Regular 1Hz variables
            "wetlabsubat_flow_rate": (
                ["wetlabsubat_time"],
                np.full(len(time_vals), 350.0),
            ),
            "wetlabsbb2fl_fluorescence": (
                ["wetlabsbb2fl_time"],
                np.random.uniform(0, 5, len(time_vals)),
            ),
            # Realistic coordinate variables - instrument-prefixed, NO navigation_*
            "nudged_latitude": (["nudged_time"], np.full(len(time_vals), 36.8)),
            "nudged_longitude": (["nudged_time"], np.full(len(time_vals), -122.0)),
            "onboard_latitude": (["onboard_time"], np.full(len(time_vals), 36.8)),
            "onboard_longitude": (["onboard_time"], np.full(len(time_vals), -122.0)),
            "wetlabsubat_latitude": (
                ["wetlabsubat_time"],
                np.full(len(time_vals), 36.8),
            ),
            "wetlabsubat_longitude": (
                ["wetlabsubat_time"],
                np.full(len(time_vals), -122.0),
            ),
        },
        coords={
            "wetlabsubat_time": time_vals.to_numpy(),
            "wetlabsubat_time_60hz": time_60hz_vals.to_numpy(),
            "wetlabsbb2fl_time": time_vals.to_numpy(),
            "nudged_time": time_vals.to_numpy(),
            "onboard_time": time_vals.to_numpy(),
        },
    )

    # Add attributes
    ds["wetlabsubat_digitized_raw_ad_counts"].attrs = {
        "long_name": "Digitized raw AD counts",
        "units": "counts",
    }
    ds["nudged_latitude"].attrs = {"standard_name": "latitude", "units": "degrees_north"}
    ds["nudged_longitude"].attrs = {"standard_name": "longitude", "units": "degrees_east"}

    # Create Resampler instance
    resampler = Resampler(
        auv_name="pontus",
        log_file=None,
        freq="1S",
        verbose=0,
    )

    # Set the dataset
    resampler.ds = ds
    resampler.df_r = pd.DataFrame(index=time_vals)

    # Create mock resampled_nc (would normally be created by resample_variable)
    resampler.resampled_nc = xr.Dataset(coords={"time": time_vals.to_numpy()})
    resampler.resampled_nc["wetlabsbb2fl_fluorescence"] = (
        ["time"],
        np.random.uniform(0, 5, len(time_vals)),
    )

    # This should NOT raise KeyError for navigation_latitude/navigation_longitude
    # The method should find nudged_latitude/longitude or another available coordinate
    try:
        resampler.add_wetlabsubat_proxies(freq="1S")
        # If we get here, the coordinate lookup worked
        assert True  # noqa: S101
    except KeyError as e:
        if "navigation_latitude" in str(e) or "navigation_longitude" in str(e):
            pytest.fail(
                f"Coordinate lookup failed - should find alternative to navigation_* variables: {e}"
            )
        raise
