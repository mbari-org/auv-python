import numpy as np
import os
import pytest
from calibrate_align import CalAligned_NetCDF
from argparse import Namespace
from hs2_proc import (hs2_read_cal_file, typ_absorption, purewater_scatter, 
                      _get_gains, hs2_calc_bb)
from logs2netcdfs import MISSIONLOGS

@pytest.fixture()
def mission_data():
    '''Load a short recent mission to have some real data to work with
    '''
    cal_netcdf = CalAligned_NetCDF()
    ns = Namespace()
    ns.base_path = 'auv_data'
    ns.auv_name = 'dorado389'
    ns.mission = '2020.245.00'
    ns.plot = None
    cal_netcdf.args = ns
    cal_netcdf.process_logs()
    return cal_netcdf

@pytest.fixture()
def calibration(mission_data):
    md = mission_data
    logs_dir = os.path.join(md.args.base_path, md.args.auv_name, 
                            MISSIONLOGS, md.args.mission)
    cal_fn = os.path.join(logs_dir, md.sinfo['hs2']['cal_filename'])
    return hs2_read_cal_file(cal_fn)

def test_typ_absorption():
    assert round(typ_absorption(420), 4) == 0.0235
    assert round(typ_absorption(470), 4) == 0.0179

def test_purewater_scatter():

    # Matlab RunReprocess on 2020.245.00

    # K>> [a, b] = purewater_scatter(420)
    # a =
    #    5.7162e-04
    # b =
    #     0.0031

    # K>> [a, b] = purewater_scatter(700)
    # a =
    #    6.2910e-05
    # b =
    #    3.3764e-04
    
    assert np.allclose(purewater_scatter(420), (5.7162e-04, 0.0031), atol=1e-4)
    assert np.allclose(purewater_scatter(700), (6.2910e-05, 3.3764e-04), atol=1e-4)

def test_get_gains(mission_data, calibration):
    md = mission_data
    cals = calibration
    hs2 = _get_gains(md.hs2.orig_data, cals, md.hs2)

    # Matlab RunReprocess on 2020.245.00

    # K>> hs2.Gain1(1:5)
    # ans =
    #    1.0e+03 *
    #     1.0077
    #     0.0982
    #     1.0077
    #     1.0077
    #     1.0077

    # K>> hs2.Gain2(1:5)
    # ans =
    #    1.0e+03 *
    #     9.7439
    #     0.9325
    #     0.9325
    #     0.9325
    #     0.9325

    # K>> hs2.Gain3(1:5)
    # ans =
    #    1.0e+03 *
    #     9.7439
    #     0.9325
    #     0.9325
    #     0.9325
    #     0.9325

    assert np.all(hs2.Gain1[:5] == np.array([1007.7, 98.1811, 1007.7, 1007.7, 1007.7]))
    assert np.all(hs2.Gain2[:5] == np.array([9743.9 ,  932.46,  932.46,  932.46,  932.46]))
    assert np.all(hs2.Gain3[:5] == np.array([9743.9 ,  932.46,  932.46,  932.46,  932.46]))

def test_hs2_calc_bb(mission_data, calibration):
    md = mission_data
    cals = calibration
    hs2 = hs2_calc_bb(md.hs2.orig_data, cals)

    # Matlab RunReprocess on 2020.245.00

    # K>> denom = ((1 + str2num(CAL.Ch(1).TempCoeff).*(hs2.Temp-str2num(CAL.General.CalTemp))).*hs2.Gain1.*str2num(CAL.Ch(1).RNominal));
    # K>> denom(1:5)
    #
    # ans =
    #
    # 1.0e+06 *
    #
    # 6.0244
    # 0.5870
    # 6.0244
    # 6.0244
    # 6.0244

    # K>> hs2.bb420(1:10)
    #
    # ans =
    #
    #     0.0060
    #     0.0052
    #     0.0060
    #     0.0055
    #     0.0062
    #     0.0061
    #     0.0057
    #     0.0058
    #     0.0056
    #     0.0054

    assert np.allclose(hs2.bb420[:10], (0.0060, 0.0052, 0.0060, 0.0055, 0.0062,
                                        0.0061, 0.0057, 0.0058, 0.0056, 0.0054))
    pass

    # K>> hs2.bbp420(1:10)
    #
    # ans =
    #
    #     0.0029
    #     0.0022
    #     0.0030
    #     0.0024
    #     0.0031
    #     0.0030
    #     0.0026
    #     0.0027
    #     0.0025
    #     0.0023

    ##assert 