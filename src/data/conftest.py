# noqa: INP001
import logging
import os
from argparse import Namespace
from pathlib import Path

import pytest
from calibrate import Calibrate_NetCDF
from hs2_proc import hs2_read_cal_file
from logs2netcdfs import BASE_PATH, MISSIONLOGS
from process import Processor
from resample import FLASH_THRESHOLD, FREQ, MF_WIDTH


def create_test_namespace(vehicle_overrides=None, processing_overrides=None):
    """Create a standardized test namespace using Processor's CONFIG_SCHEMA.

    Args:
        vehicle_overrides: Dict of vehicle-specific overrides (mission, auv_name, etc.)
        processing_overrides: Dict of processing-specific overrides (verbose, clobber, etc.)

    Returns:
        argparse.Namespace with all CONFIG_SCHEMA attributes properly set
    """
    # Start with Processor's config schema defaults
    config = dict(Processor._CONFIG_SCHEMA)

    # Apply common test defaults
    test_defaults = {
        "base_path": os.getenv("BASE_PATH", BASE_PATH),
        "local": True,
        "noinput": True,
        "noreprocess": False,
        "use_portal": False,
        "freq": FREQ,
        "mf_width": MF_WIDTH,
        "flash_threshold": FLASH_THRESHOLD,
        "clobber": False,
        "no_cleanup": True,
        "num_cores": 1,
        "verbose": 1,
    }
    config.update(test_defaults)

    # Apply vehicle-specific overrides
    if vehicle_overrides:
        config.update(vehicle_overrides)

    # Apply processing-specific overrides
    if processing_overrides:
        config.update(processing_overrides)

    # Create namespace and set all attributes
    ns = Namespace()
    for key, value in config.items():
        setattr(ns, key, value)

    return ns


bootstrap_mission = """The working directory on a development machine must be bootstrapped
with some mission data. Process the mission used for testing with:

uv run src/data/process_Dorado389.py --no_cleanup --download --mission 2011.256.02 -v

This uses the legacy name "Dorado389" for the vehicle - the new name is "dorado".
"""

TEST_VEHICLE = "Dorado389"
TEST_MISSION = "2011.256.02"  # http://stoqs.mbari.org/p/DmHOaxI
# Set TEST_VEHICLE_DIR to local path for testing
TEST_VEHICLE_DIR = Path(os.getenv("BASE_PATH", BASE_PATH), TEST_VEHICLE, "missionlogs")
TEST_CALIBRATION_DIR = ""
TEST_MOUNT_DIR = ""
TEST_START_YEAR = 2011

TEST_I2MAP_VEHICLE = "i2map"
TEST_I2MAP_MISSION = "2018.348.01"
# Set TEST_VEHICLE_DIR to local path for testing
TEST_I2MAP_VEHICLE_DIR = Path(os.getenv("BASE_PATH", BASE_PATH), TEST_I2MAP_VEHICLE, "missionlogs")
TEST_I2MAP_CALIBRATION_DIR = Path(
    os.getenv("CAL_DIR", "/Volumes/DMO/MDUC_CORE_CTD_200103/Calibration Files")
)
TEST_I2MAP_MOUNT_DIR = ""
TEST_I2MAP_START_YEAR = 2018


@pytest.fixture(scope="session", autouse=False)
def mission_data():
    if not Path(TEST_VEHICLE_DIR).exists():
        pytest.fail(f"\n\n{bootstrap_mission}\n")
    """Load a short recent mission to have some real data to work with"""
    cal_netcdf = Calibrate_NetCDF()
    ns = Namespace()
    # The BASE_PATH environment variable can be set in ci.yml for running in GitHub Actions
    ns.base_path = os.getenv("BASE_PATH", BASE_PATH)
    ns.auv_name = TEST_VEHICLE
    ns.mission = TEST_MISSION
    ns.plot = None
    cal_netcdf.args = ns
    cal_netcdf.logger.setLevel(logging.DEBUG)
    cal_netcdf.process_logs(process_gps=False)
    return cal_netcdf


@pytest.fixture(scope="session", autouse=False)
def calibration(mission_data):
    md = mission_data
    logs_dir = Path(
        md.args.base_path,
        md.args.auv_name,
        MISSIONLOGS,
        md.args.mission,
    )
    cal_fn = Path(logs_dir, md.sinfo["hs2"]["cal_filename"])
    return hs2_read_cal_file(cal_fn)


@pytest.fixture(scope="session", autouse=False)
def complete_dorado_processing():
    """Load a short mission to have some real data to work with"""
    # Create namespace with vehicle-specific settings
    vehicle_overrides = {
        "auv_name": TEST_VEHICLE,
        "mission": TEST_MISSION,
        "start_year": TEST_START_YEAR,
    }

    ns = create_test_namespace(vehicle_overrides=vehicle_overrides)

    # Create processor using new factory method
    proc = Processor.from_args(
        TEST_VEHICLE, TEST_VEHICLE_DIR, TEST_MOUNT_DIR, TEST_CALIBRATION_DIR, ns
    )
    proc.process_missions(TEST_START_YEAR)
    return proc


@pytest.fixture(scope="session", autouse=False)
def complete_i2map_processing():
    """Load a short mission to have some real data to work with"""
    # Create namespace with i2map-specific settings
    vehicle_overrides = {
        "auv_name": TEST_I2MAP_VEHICLE,
        "mission": TEST_I2MAP_MISSION,
        "start_year": TEST_I2MAP_START_YEAR,
        "last_n_days": 0,  # i2map-specific setting
    }

    ns = create_test_namespace(vehicle_overrides=vehicle_overrides)

    # Create processor using new factory method
    proc = Processor.from_args(
        TEST_I2MAP_VEHICLE,
        TEST_I2MAP_VEHICLE_DIR,
        TEST_I2MAP_MOUNT_DIR,
        TEST_I2MAP_CALIBRATION_DIR,
        ns,
    )
    proc.process_missions(TEST_I2MAP_START_YEAR)
    return proc
