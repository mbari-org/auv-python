# noqa: INP001
import hashlib
from pathlib import Path
from time import time

from logs2netcdfs import MISSIONNETCDFS

# The test should not take more than 2 minutes to run, so this is as old as the _1S.nc file can be
MAX_SECS = 2 * 60  # 2 minutes


def test_process_dorado(complete_dorado_processing):
    """Test that Dorado processing runs without error"""
    # complete_processing ia a fixture from the conftest.py module;
    # it is automatically loaded by pytest.  We need this complete_processing
    # to run withut error to trust that tthe production processing is working.
    proc = complete_dorado_processing

    # Check that the _1S.nc file was created and is the correct size
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
    EXPECTED_SIZE_GITHUB = 621298
    EXPECTED_SIZE_ACT = 621298
    EXPECTED_SIZE_LOCAL = 621286
    if str(proc.args.base_path).startswith("/home/runner"):
        # The size is different in GitHub Actions, maybe due to different metadata
        assert nc_file.stat().st_size == EXPECTED_SIZE_GITHUB  # noqa: S101
    elif str(proc.args.base_path).startswith("/root"):
        # The size is different in act, maybe due to different metadata
        assert nc_file.stat().st_size == EXPECTED_SIZE_ACT  # noqa: S101
    else:
        # The size is different locally, maybe due to different metadata
        # It's likely that the size will be different on different machines
        # as these kind of metadata items are added to nc_file:
        # NC_GLOBAL.history: Created by /Users/mccann/GitHub/auv-python/src/data/process_dorado.py ... # noqa: E501
        assert nc_file.stat().st_size == EXPECTED_SIZE_LOCAL  # noqa: S101

    check_md5 = True
    if check_md5:
        # Check that the MD5 hash has not changed
        EXPECTED_MD5_GITHUB = "11f9a455dfae3bad24aa485181ef5384"
        EXPECTED_MD5_ACT = "ec95eabdc1450aefee4667808bc46d92"
        EXPECTED_MD5_LOCAL = "d9754a20d1c8ac6ddbb1a62d75aa507e"
        if str(proc.args.base_path).startswith("/home/runner"):
            # The MD5 hash is different in GitHub Actions, maybe due to different metadata
            assert hashlib.md5(open(nc_file, "rb").read()).hexdigest() == EXPECTED_MD5_GITHUB  # noqa:  PTH123, S101, S324, SIM115
        elif str(proc.args.base_path).startswith("/root"):
            # The MD5 hash is different in act, maybe due to different metadata
            assert hashlib.md5(open(nc_file, "rb").read()).hexdigest() == EXPECTED_MD5_ACT  # noqa: PTH123, S101, S324, SIM115
        else:
            # The MD5 hash is different locally, maybe due to different metadata
            # It's likely that the hash will be different on different machines
            # as these kind of metadata items are added to nc_file:
            # NC_GLOBAL.history: Created by /Users/mccann/GitHub/auv-python/src/data/process_dorado.py ... # noqa: E501
            assert hashlib.md5(open(nc_file, "rb").read()).hexdigest() == EXPECTED_MD5_LOCAL  # noqa:  PTH123, S101, S324, SIM115
