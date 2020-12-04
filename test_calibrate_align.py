import logging
import numpy as np
import pytest
from calibrate_align import align_geom

def test_align_geom():
    # https://www.onlinemathlearning.com/math-trick-unit-circle.html
    y30 = 0.5
    y45 = np.sqrt(2) / 2.0
    y60 = np.sqrt(3) / 2.0
    test_angles = [  0,   30,   45,   60,   90,  120,  135,  150, 
                   180,  210,  225,  240,  270,  300,  315,  330,  360]
    test_off_00 = [  0,    0,    0,    0,    0,    0,    0,    0,
                     0,    0,    0,    0,    0,    0,    0,    0,    0]
    test_off_10 = [  0,  y30,  y45,  y60,  1.0,  y60,  y45,  y30,
                     0, -y30, -y45, -y60, -1.0, -y60, -y45, -y30,    0]

    offsets_00 = align_geom([0, 0], test_angles)
    np.testing.assert_allclose(offsets_00, test_off_00, atol=1e-15)

    offsets_10 = align_geom([1, 0], test_angles)
    np.testing.assert_allclose(offsets_10, test_off_10, atol=1e-15)

