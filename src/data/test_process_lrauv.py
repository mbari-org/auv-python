# noqa: INP001

import numpy as np
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
