# noqa: INP001
from pathlib import Path

from logs2netcdfs import MISSIONNETCDFS


def test_process_dorado(complete_processing):
    """Test that Dorado processing runs without error"""
    # complete_processing ia a fixture from the conftest.py module;
    # it is automatically loaded by pytest.  We need this complete_processing
    # to run withut error to trust that tthe production processing is working.
    proc = complete_processing

    # Check that the _1S.nc file was created and is the correct size
    nc_file = Path(
        proc.args.base_path,
        proc.args.auv_name,
        MISSIONNETCDFS,
        proc.args.mission,
        f"{proc.args.auv_name}_{proc.args.mission}_1S.nc",
    )
    assert nc_file.exists()  # noqa: S101
    assert nc_file.stat().st_size > 0  # noqa: S101
    assert nc_file.stat().st_size == 621235  # noqa: PLR2004, S101
