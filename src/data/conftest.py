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
    proc = Processor(TEST_VEHICLE, TEST_VEHICLE_DIR, TEST_MOUNT_DIR, TEST_CALIBRATION_DIR)
    ns = Namespace()
    ns.base_path = os.getenv("BASE_PATH", BASE_PATH)
    ns.auv_name = TEST_VEHICLE
    ns.mission = TEST_MISSION
    ns.start_year = TEST_START_YEAR
    # There are several options that need to be set to run the full processing
    ns.clobber = False
    proc.commandline = "args set in conftest.py::complete_dorado_processing()"
    ns.local = True
    ns.noinput = True
    ns.noreprocess = False
    ns.use_portal = False
    ns.freq = FREQ
    ns.mf_width = MF_WIDTH
    ns.flash_threshold = FLASH_THRESHOLD
    # Set step flags to false to force all steps to run as the logic in
    # process_mission() is not fully implemented.
    ns.download_process = False
    ns.calibrate = False
    ns.align = False
    ns.resample = False
    ns.create_products = False
    ns.archive = False
    ns.archive_only_products = False
    ns.email_to = None
    ns.cleanup = False
    ns.no_cleanup = True
    ns.skip_download_process = False
    ns.num_cores = 1
    ns.verbose = 1
    proc.args = ns
    proc.process_missions(TEST_START_YEAR)
    return proc


@pytest.fixture(scope="session", autouse=False)
def complete_i2map_processing():
    """Load a short mission to have some real data to work with"""
    proc = Processor(
        TEST_I2MAP_VEHICLE,
        TEST_I2MAP_VEHICLE_DIR,
        TEST_I2MAP_MOUNT_DIR,
        TEST_I2MAP_CALIBRATION_DIR,
    )
    ns = Namespace()
    ns.base_path = os.getenv("BASE_PATH", BASE_PATH)
    ns.auv_name = TEST_I2MAP_VEHICLE
    ns.mission = TEST_I2MAP_MISSION
    ns.start_year = TEST_I2MAP_START_YEAR
    # There are several options that need to be set to run the full processing
    ns.clobber = False
    proc.commandline = "args set in conftest.py::complete_i2map_processing()"
    ns.local = True
    ns.noinput = True
    ns.noreprocess = False
    ns.use_portal = False
    ns.freq = FREQ
    ns.mf_width = MF_WIDTH
    ns.flash_threshold = FLASH_THRESHOLD
    # Set step flags to false to force all steps to run as the logic in
    # process_mission() is not fully implemented.
    ns.download_process = False
    ns.calibrate = False
    ns.align = False
    ns.resample = False
    ns.create_products = False
    ns.archive = False
    ns.archive_only_products = False
    ns.email_to = None
    ns.cleanup = False
    ns.no_cleanup = True
    ns.skip_download_process = False
    ns.last_n_days = 0
    ns.num_cores = 1
    ns.verbose = 1
    proc.args = ns
    proc.process_missions(TEST_START_YEAR)
    return proc
