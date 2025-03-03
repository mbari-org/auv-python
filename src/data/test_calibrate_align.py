# noqa: INP001
import numpy as np
from calibrate import align_geom


def test_align_geom():
    # https://www.onlinemathlearning.com/math-trick-unit-circle.html
    y30 = 0.5
    y45 = np.sqrt(2) / 2.0
    y60 = np.sqrt(3) / 2.0
    test_angles = [  0,   30,   45,   60,   90,  120,  135,  150, 
                   180,  210,  225,  240,  270,  300,  315,  330,  360]
    test_depth_00 = [  0,    0,    0,    0,    0,    0,    0,    0,
                       0,    0,    0,    0,    0,    0,    0,    0,    0]
    test_depth_10 = [  0,  y30,  y45,  y60,  1.0,  y60,  y45,  y30,
                       0, -y30, -y45, -y60, -1.0, -y60, -y45, -y30,    0]

    depths_00 = align_geom([0, 0], test_angles)
    np.testing.assert_allclose(depths_00, test_depth_00, atol=1e-15)

    depths_10 = align_geom([1, 0], test_angles)
    np.testing.assert_allclose(depths_10, test_depth_10, atol=1e-15)

    # Test with an x & y offset
    depths_11 = align_geom([1 * y45, 1 * y45], [0, 45])
    np.testing.assert_allclose(depths_11, [y45, 1], atol=1e-15)

    # Test from Matlab processHS2.m debugging session:
    #
    # K>> align_geom([0.1397, -0.2794], [0.2928, 0.2939, 0.2952])  # noqa: ERA001
    #
    # ans =
    #
    #   -0.2272   -0.2270   -0.2267
    test_angles_radians = [0.2928, 0.2939, 0.2952]
    test_angles_degrees = [a * 180 / np.pi for a in test_angles_radians]
    test_depth_hs2 = [-0.2272, -0.2270, -0.2267]

    offset_hs2 = [0.1397, -0.2794]
    depths_hs2 =  align_geom(offset_hs2, test_angles_degrees)
    np.testing.assert_allclose(depths_hs2, test_depth_hs2, atol=1e-4)
