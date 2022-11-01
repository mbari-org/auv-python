import numpy as np

from calibrate import Calibrate_NetCDF
from ctd_proc import _calibrated_O2_from_volts, _oxsat


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
    # 24  OXSAT = exp(A1 + A2*(100./TK) + A3*log(TK/100) + A4*(TK/100) + [S .* (B1 + B2*(TK/100) + (B3*(TK/100).*(TK/100)))] );
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

    assert np.allclose(
        oxsat.values[:5],
        np.array([5.7085865, 5.7088902, 5.7105826, 5.7107884, 5.7109565]),
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
        md.ctd1.orig_data["salinity"],
    )

    # K>> dbstep
    # 32  O2 = [O2cal.SOc * ((O2V+O2cal.offset)+(tau*docdt)) + O2cal.BOc * exp(-0.03*T)].*exp(O2cal.Tcor*T + O2cal.Pcor*P).*OXSAT;
    # K>> O2(1:5)
    #
    # ans =
    #
    #                  NaN
    #    6.408202411463399
    #    6.405165617687661
    #    6.391949575962099
    #    6.382043909703177

    assert np.allclose(
        oxy_mll[1:5],  # Remove NaN as it doesn't compare
        np.array([6.4082024, 6.4051656, 6.3919495, 6.3820439]),
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

    assert np.allclose(
        oxy_umolkg[1:5],  # Remove NaN as it doesn't compare
        np.array([278.93817, 278.80129, 278.22544, 277.79378]),
        atol=1e-1,
    )
