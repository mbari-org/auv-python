# noqa: INP001

from collections import defaultdict
from math import exp
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d


class SensorInfo:
    pass


class HS2:
    pass


def hs2_read_cal_file(cal_filename: Path):
    cals = defaultdict(dict)
    channels = []

    with cal_filename.open() as fh:
        for line in fh:
            if "[General]" in line:
                category = "General"
            elif "[Channel" in line:
                # Before 2005103/2005.103.01/hs2Calibration.dat there was no space
                ch_num = line.split("[Channel")[1].split("]")[0].strip()
                category = f"Ch{int(ch_num)}"
                channels.append(category)
            elif "=" in line:
                name, value = [s.strip() for s in line.split("=")]
                cals[category][name] = value
            elif line in ("\n", "[End]\n"):
                pass
            else:
                error_message = f"Unexpected line: {line}"
                raise ValueError(error_message)

    # From the CVS log: revision 1.3
    # date: 2007/07/13 17:59:06;  author: ssdsadmin;  state: Exp;  lines: +11 -0
    # Added approximate values for SigmaExp for those Cals that don't have it.
    #
    # NB:  The range of values for SigmaExp is relatively small and the error
    #      in the final output that results from our choice of SigmaExp (when a
    #      value is not provided in the calibration file) is likely to be <1%
    #      Value needs to be a string as hs2_calc_bb calls float() on it.
    if "SigmaExp" not in cals["Ch1"] and "SigmaExp" not in cals["Ch2"]:
        if cals["General"]["Serial"] == "H2000325":
            # value obtained from subsequent calibrations of this instrument
            cals["Ch1"]["SigmaExp"] = "0.1460"
            cals["Ch2"]["SigmaExp"] = "0.1600"
        else:
            # Average SigmaExp value for sensors H2000325 and H2D021004
            # from 2001-2007: [(0.153+0.145+0.153+0.146+0.146)/5]
            cals["Ch1"]["SigmaExp"] = "0.1486"
            cals["Ch2"]["SigmaExp"] = "0.1522"

    return cals


def _get_gains(orig_nc, cals, hs2):
    # % FIND REAL GAIN NUMBER FROM CAL FILE AND HS2 POINTERS
    # See section 9.2.7 in HS2ManualRevI-2011-12.pdf (between >> << below):
    # ---------------------------------------------------------------------
    # There is one 4-bit gain/status value for each of the optical data channels.
    # The three LSB's comprise a gain setting, and the MSB indicates status.
    # A gain setting of zero indicates that the channel is disabled and its
    # data should be ignored. >> Gains of 1 to 5 are used to select one of five
    # coefficients to be applied to Snorm. Gains 6 and 7 are undefined. <<
    # The status bit is zero under normal conditions. The status bit for a
    # channel may be set to 1 if the HydroScat detects a condition that may
    # affect the quality of the data on that channel. However a status value
    # of 1 does not necessarily indicate invalid data.
    for chan, gs in ((1, "Gain_Status_1"), (2, "Gain_Status_2"), (3, "Gain_Status_3")):
        gv = orig_nc[gs]
        for gnum in range(1, 6):
            if chan <= 2:  # noqa: PLR2004
                gc = float(cals[f"Ch{chan}"][f"Gain{gnum}"])
            elif chan == 3:  # noqa: PLR2004
                gc = float(cals[f"Ch{chan - 1}"][f"Gain{gnum}"])

            gv = np.where(orig_nc[gs] == gnum, gc, gv)
            setattr(hs2, f"Gain{chan}", gv)

    return hs2


def _int_signer(ints_in):
    # -% signed_int = int_in - 65536*(int_in > 32767);
    signed_ints = []
    for int_in in ints_in.values:
        if int_in > 32767:  # noqa: PLR2004
            signed_ints.append(int_in - 65536)
        else:
            signed_ints.append(int_in)
    return np.array(signed_ints)


def compute_backscatter(wavelength_nm: float, salinity: float, volScat: float):  # noqa: N803
    # Cribbed from https://mbari.slack.com/archives/C04ETLY6T7V/p1710457297254969?thread_ts=1710348431.316509&cid=C04ETLY6T7V
    # This is  the same computation used for LRAUV ecopucks. Used here for Dorado ecopuck
    # following the conversion to "scaled" output using scale_factor and dark counts.
    theta = 117.0 / 57.29578  # radians
    d = 0.09

    # These calculations are from the Triplet Puck User's Guide, Revision H
    Bw = (
        1.38
        * (wavelength_nm / 500.0) ** (-4.32)
        * (1.0 + 0.3 * salinity / 37.0)
        * 1e-4
        * (1.0 + np.cos(theta) ** 2.0 * (1.0 - d) / (1.0 + d))
    )
    Bp = volScat - Bw
    if salinity < 35.0:  # noqa: PLR2004
        bw = 0.0022533 * (wavelength_nm / 500.0) ** (-4.23) * 1e-4
    else:
        bw = 0.0029308 * (wavelength_nm / 500.0) ** (-4.24) * 1e-4
    bbw = bw / 2.0
    bbp = 2.0 * np.pi * 1.1 * Bp

    return bbw, bbp


def hs2_calc_bb(orig_nc, cals):
    # Some original comments from hs2_calc_bb.m
    # % Date Created:  June 21, 2007
    # % Date Modified: June 26, 2007
    # %
    # % Brandon Sackmann
    # % Postdoctoral Fellow
    # % Monterey Bay Aquarium Research Institute
    # % 7700 Sandholdt Road
    # % Moss Landing, California  95039
    # %
    # % Tel: (831) 775-1958
    # % Fax: (831) 775-1620
    # % Email: sackmann@mbari.org
    #

    hs2 = HS2()

    hs2 = _get_gains(orig_nc, cals, hs2)

    # BACKSCATTERING COEFFICIENT CALCULATION
    # Ch1 is blue backscatter, either beta420 or beta470
    # Ch2 is red backscatter, either beta676 or beta700
    # Ch3 is fluorescence, either fl676 or fl700
    # Item cals[f'Ch{chan}']['Name'] identifies which one
    for chan in (1, 2):
        # -% RAW SIGNAL CONVERSION
        # -% hs2.beta420_uncorr = (hs2.Snorm1.*str2num(CAL.Ch(1).Mu))./((1 + str2num(CAL.Ch(1).TempCoeff).*(hs2.Temp-str2num(CAL.General.CalTemp))).*hs2.Gain1.*str2num(CAL.Ch(1).RNominal));'  # noqa: E501
        denom = np.multiply(
            (
                1
                + float(cals[f"Ch{chan}"]["TempCoeff"])
                * ((orig_nc["RawTempValue"] / 5 - 10) - float(cals["General"]["CalTemp"]))
            ),
            (getattr(hs2, f"Gain{chan}") * float(cals[f"Ch{chan}"]["RNominal"])),
        )
        beta_uncorr = np.divide(
            (_int_signer(orig_nc[f"Snorm{chan}"]) * float(cals[f"Ch{chan}"]["Mu"])),
            denom,
        )
        # Replaces "RawTempValue" as the name, helpful when looking at things in the debugger
        beta_uncorr.name = f"beta_uncorr_Ch{chan}"
        wavelength = int(cals[f"Ch{chan}"]["Name"][2:])

        # Use compute_backscatter - same as used for ecopucks - to calculate bbp
        _, bbp = compute_backscatter(wavelength, 35.2, beta_uncorr)
        setattr(hs2, f"bbp{wavelength}", bbp)

    # Fluorescence
    # -% 'hs2.fl700_uncorr = (hs2.Snorm3.*50)./((1 + str2num(CAL.Ch(3).TempCoeff).*(hs2.Temp-str2num(CAL.General.CalTemp))).*hs2.Gain3.*str2num(CAL.Ch(3).RNominal));'  # noqa: E501
    denom = (
        (
            1
            + float(cals["Ch3"]["TempCoeff"])
            * ((orig_nc["RawTempValue"] / 5 - 10) - float(cals["General"]["CalTemp"]))
        )
        * hs2.Gain3
        * float(cals["Ch3"]["RNominal"])
    )
    snorm3 = _int_signer(orig_nc["Snorm3"])
    setattr(hs2, f"fl{wavelength}", np.divide(snorm3 * 50, denom))

    hs2.caldepth = float(cals["General"]["DepthCal"]) * orig_nc["RawDepthValue"] - float(
        cals["General"]["DepthOff"]
    )

    return hs2


def purewater_scatter(lamda):
    beta_w_ref = 2.18e-04  # for seawater
    b_bw_ref = 1.17e-03  # for seawater
    # beta_w_ref  =   1.67E-04   # for freshwater
    # b_bw_ref    =   8.99E-04   # for freshwater
    lamda_ref = 525
    gamma = 4.32

    beta_w = beta_w_ref * (lamda_ref / lamda) ** gamma
    b_bw = b_bw_ref * (lamda_ref / lamda) ** gamma

    return beta_w, b_bw


def typ_absorption(lamda):
    C = 0.1
    gamma_y = 0.014
    a_d_400 = 0.01
    gamma_d = 0.011

    # -% Embed the lookup table from the AStar.CSV file here
    # -%%a_star    =   load('AStar.csv');
    a_star_values = np.array(
        [
            [400, 0.687],
            [410, 0.828],
            [420, 0.913],
            [430, 0.973],
            [440, 1.000],
            [450, 0.944],
            [460, 0.917],
            [470, 0.870],
            [480, 0.798],
            [490, 0.750],
            [500, 0.668],
            [510, 0.618],
            [520, 0.528],
            [530, 0.474],
            [540, 0.416],
            [550, 0.357],
            [560, 0.294],
            [570, 0.276],
            [580, 0.291],
            [590, 0.282],
            [600, 0.236],
            [610, 0.252],
            [620, 0.276],
            [630, 0.317],
            [640, 0.334],
            [650, 0.356],
            [660, 0.441],
            [670, 0.595],
            [680, 0.502],
            [690, 0.329],
            [700, 0.215],
        ],
    )

    a_interp = interp1d(a_star_values[:, 0], a_star_values[:, 1])
    a_star = a_interp(lamda)

    return (0.06 * a_star * (C**0.65)) * (1 + 0.2 * exp(-gamma_y * (lamda - 440))) + (
        a_d_400 * exp(-gamma_d * (lamda - 400))
    )
