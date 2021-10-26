import logging
import os
import pytest
from calibrate_align import CalAligned_NetCDF
from argparse import Namespace
from hs2_proc import hs2_read_cal_file
from logs2netcdfs import MISSIONLOGS

@pytest.fixture(scope='session', autouse=False)
def mission_data():
    '''Load a short recent mission to have some real data to work with
    '''
    cal_netcdf = CalAligned_NetCDF()
    ns = Namespace()
    ns.base_path = 'auv_data'
    ns.auv_name = 'Dorado389'
    ns.mission = '2020.245.00'
    ns.plot = None
    cal_netcdf.args = ns
    cal_netcdf.logger.setLevel(logging.DEBUG)
    cal_netcdf.process_logs()
    return cal_netcdf

@pytest.fixture(scope='session', autouse=False)
def calibration(mission_data):
    md = mission_data
    logs_dir = os.path.join(md.args.base_path, md.args.auv_name, 
                            MISSIONLOGS, md.args.mission)
    cal_fn = os.path.join(logs_dir, md.sinfo['hs2']['cal_filename'])
    return hs2_read_cal_file(cal_fn)
