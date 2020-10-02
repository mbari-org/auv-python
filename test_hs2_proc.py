import logging
import numpy as np
import pytest
from calibrate_align import CalAligned_NetCDF
from hs2_proc import typ_absorption, purewater_scatter, _get_gains, hs2_calc_bb

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

    # K>> format long
    # K>> hs2.bb420(1:5)
    # ans =
    #    0.005975077727995
    #    0.005245234803124
    #    0.006044696205554
    #    0.005488825740040
    #    0.006203063319423

    # K>> hs2.bbp420(1:5)
    # ans =
    #    0.002907206559870
    #    0.002177363634998
    #    0.002976825037428
    #    0.002420954571914
    #    0.003135192151297

    # K>> hs2.bb700(1:5)
    # ans =
    #    0.012280996975942
    #    0.012087616414187
    #    0.012479078951715
    #    0.011599823683877
    #    0.012087616414187

    # K>> hs2.bbp700(1:5)
    # ans =
    #    0.011943359597475
    #    0.011749979035720
    #    0.012141441573249
    #    0.011262186305410
    #    0.011749979035720

    # K>> hs2.fl700_uncorr(1:5)
    # ans =
    #    0.002448326645388
    #   -0.001983999313643
    #    0.002104647920554
    #    0.001990702014027
    #    0.001943783111340

    assert np.allclose(hs2.bb420[:5], (0.005975077727995, 0.005245234803124,
                                       0.006044696205554, 0.005488825740040,
                                       0.006203063319423, ))
    assert np.allclose(hs2.bbp420[:5], (0.002907206559870, 0.002177363634998,
                                        0.002976825037428, 0.002420954571914,
                                        0.003135192151297, ))
    assert np.allclose(hs2.bb700[:5], (0.012280996975942, 0.012087616414187,
                                       0.012479078951715, 0.011599823683877,
                                       0.012087616414187, ))
    assert np.allclose(hs2.bbp700[:5], (0.011943359597475, 0.011749979035720,
                                        0.012141441573249, 0.011262186305410,
                                        0.01174997903572, ))
    assert np.allclose(hs2.fl700[:5], (0.002448326645388, -0.001983999313643,
                                       0.002104647920554, 0.001990702014027,
                                       0.001943783111340, ))
