# noqa: INP001
from pathlib import Path
from time import time

from logs2netcdfs import MISSIONNETCDFS

# The test should not take more than 5 minutes to run, so this is as old as the _1S.nc file can be
MAX_SECS = 5 * 60  # 5 minutes


def test_process_i2map(complete_i2map_processing):
    """Test that Dorado processing runs without error"""
    # complete_processing ia a fixture from the conftest.py module;
    # it is automatically loaded by pytest.  We need this complete_processing
    # to run withut error to trust that tthe production processing is working.
    proc = complete_i2map_processing

    # Check that the _1S.nc file was created recently and is the correct size
    nc_file = Path(
        proc.args.base_path,
        proc.args.auv_name,
        MISSIONNETCDFS,
        proc.args.mission,
        f"{proc.args.auv_name}_{proc.args.mission}_1S.nc",
    )
    assert nc_file.exists()  # noqa: S101
    assert time() - nc_file.stat().st_mtime < MAX_SECS  # noqa: S101
    assert nc_file.stat().st_size > 0  # noqa: S101
    # Testing that the file size matches a specific value is crude,
    # but it will alert us if a code change unexpectedly changes the file size.
    # If code changes are expected to change the file size then we should
    # update the expected size here.
    EXPECTED_SIZE = 60130
    EXPECTED_SIZE_LOCAL = 58896
    if str(proc.args.base_path).startswith("/home/runner"):
        # The size is different in GitHub Actions, maybe due to different metadata
        assert nc_file.stat().st_size == EXPECTED_SIZE  # noqa: S101
    else:
        # The size is different locally, maybe due to different metadata
        # It's likely that the size will be different on different machines
        # as these kind of metadata items are added to nc_file:
        # NC_GLOBAL.history: Created by /Users/mccann/GitHub/auv-python/src/data/process_dorado.py ... # noqa: E501
        assert nc_file.stat().st_size == EXPECTED_SIZE_LOCAL  # noqa: S101
