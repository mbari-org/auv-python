import numpy as np  # noqa: INP001
from calibrate import _calibrated_O2_from_volts, _oxsat


def test_oxsat(mission_data):
    # mission_data and calibration are fixtures from the conftest.py module;
    # they are automatically loaded by pytest
    md = mission_data
    oxsat = _oxsat(md.ctd1.orig_data["temperature"], md.ctd1.orig_data["salinity"])

    # K>> T(1:5)
    #
    # ans =
    #
    #    15.2244
    #    15.2208
    #    15.2009
    #    15.1985
    #    15.1965
    #
    # K>> S(1:5)
    #
    # ans =
    #
    #    33.6782
    #    33.6813
    #    33.6980
    #    33.7001
    #    33.7018
    #
    # K>> dbstep
    # 24  OXSAT = exp(A1 + A2*(100./TK) + A3*log(TK/100) + A4*(TK/100) + [S .* (B1 + B2*(TK/100) + (B3*(TK/100).*(TK/100)))] );  # noqa: E501
    #
    # K>> format long
    # K>> OXSAT(1:5)
    #
    # ans =
    #
    #    5.708586520895623
    #    5.708890280128260
    #    5.710582691938066
    #    5.710788414255579
    #    5.710956592071382

    assert np.allclose(  # noqa: S101
        oxsat.to_numpy()[:5],
        np.array([5.74781474, 5.73779594, 5.72991797, 5.72753987, 5.72684232]),
        atol=1e-3,
    )


def test_calibrated_O2_from_volts(mission_data):
    # mission_data and calibration are fixtures from the conftest.py module;
    # they are automatically loaded by pytest
    md = mission_data
    oxy_mll, oxy_umolkg = _calibrated_O2_from_volts(
        md.combined_nc,
        md.ctd1.cals,
        md.ctd1.orig_data,
        "dissolvedO2",
        md.ctd1.orig_data["temperature"],
        # First mission with salinity in orig_data is 2010.081.02
        md.ctd1.orig_data["salinity"],
    )

    # K>> dbstep
    # 32  O2 = [O2cal.SOc * ((O2V+O2cal.offset)+(tau*docdt)) + O2cal.BOc * exp(-0.03*T)].*exp(O2cal.Tcor*T + O2cal.Pcor*P).*OXSAT;  # noqa: E501
    # K>> O2(1:5)
    #
    # ans =
    #
    #                  NaN
    #    6.408202411463399
    #    6.405165617687661
    #    6.391949575962099
    #    6.382043909703177

    assert np.allclose(  # noqa: S101
        oxy_mll[1:5],  # Remove NaN as it doesn't compare
        np.array([6.44962584, 6.42498158, 6.41836539, 6.40601656]),
        atol=1e-3,
    )

    # K>> dbstep
    # 43   O2 = (O2 * 1.4276) .* (1e6./(dens*32));
    # K>> O2(1:5)
    #
    # ans =
    #
    #    1.0e+02 *
    #
    #                  NaN
    #    2.789381686158782
    #    2.788012883132636
    #    2.782254353946028
    #    2.777937837511869

    assert np.allclose(  # noqa: S101
        oxy_umolkg[1:5],  # Remove NaN as it doesn't compare
        np.array([280.79173073, 279.73990631, 279.45802189, 278.92183826]),
        atol=1e-1,
    )
