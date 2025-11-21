#!/usr/bin/env python
"""
Calibrate original data and produce NetCDF file for mission

Read original data from netCDF files created by logs2netcdfs.py, apply
calibration information in .cfg and .xml files associated with the
original .log files and write out a single netCDF file with the important
variables at original sampling intervals. Geometric alignment and plumbing lag
corrections are also done during this step. The file will contain combined
variables (the combined_nc member variable) and be analogous to the original
netCDF4 files produced by MBARI's LRAUVs. Rather than using groups in netCDF-4
the data will be written in classic netCDF-CF with a naming syntax that mimics
the LRAUV group naming convention with the coordinates for each sensor:
```
    <sensor>_<variable_1>
    <sensor>_<..........>
    <sensor>_<variable_n>
    <sensor>_time
    <sensor>_depth
    <sensor>_latitude
    <sensor>_longitude
```
Note: The name "sensor" is used here, but it's really more aligned
with the concept of "instrument" in SSDS parlance.
"""

__author__ = "Mike McCann"
__copyright__ = "Copyright 2020, Monterey Bay Aquarium Research Institute"

import logging  # noqa: I001
import os
import shlex
import shutil
import subprocess
import sys
import time
from collections import OrderedDict
from datetime import UTC, datetime
from pathlib import Path
from socket import gethostname
from typing import NamedTuple

import cf_xarray  # Needed for the .cf accessor  # noqa: F401
import defusedxml.ElementTree as ET  # noqa: N817
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import xarray as xr
from scipy import signal
from scipy.interpolate import interp1d

from AUV import monotonic_increasing_time_indices, nudge_positions
from common_args import get_standard_dorado_parser
from hs2_proc import compute_backscatter, hs2_calc_bb, hs2_read_cal_file
from logs2netcdfs import AUV_NetCDF, MISSIONLOGS, MISSIONNETCDFS, TIME, TIME60HZ
from seawater import eos80

AVG_SALINITY = 33.6  # Typical value for upper 100m of Monterey Bay


class Range(NamedTuple):
    min: float
    max: float


# Using lower case vehicle names, modify in _define_sensor_info() for changes over time
# Used to reduce ERROR & WARNING log messages for expected missing sensor data
EXPECTED_SENSORS = {
    "dorado": [
        "navigation",
        "gps",
        "depth",
        "ecopuck",
        "hs2",
        "ctd1",
        "ctd2",
        "isus",
        "biolume",
        "lopc",
        "tailcone",
    ],
    "i2map": [
        "navigation",
        "gps",
        "depth",
        "seabird25p",
        "transmissometer",
        "tailcone",
    ],
}
# Used in test fixture in conftetst.py
EXPECTED_SENSORS["Dorado389"] = EXPECTED_SENSORS["dorado"]


def align_geom(sensor_offset, pitches):
    """Use x & y sensor_offset values in meters from sensor_info and
    pitch in degrees to compute and return actual depths of the sensor
    based on the geometry relative to the vehicle's depth sensor.
    """
    # See https://en.wikipedia.org/wiki/Rotation_matrix
    #
    #                        * instrument location with pitch applied
    #                      / |
    #                     /  |
    #                    /   |
    #                   /    |
    #                  /     |
    #                 /      |
    #                /       |
    #               /        |
    #              /         |
    #             /
    #            /
    #           /            y
    #          /             _
    #         /              o
    #        /               f
    #       /                f
    #      /                                 *  instrument location
    #     /                                  |
    #    / \                 |               |
    #   /   \                |               y
    #  / pitch (theta)       |               |
    # /        \             |               |
    # --------------------x------------------+    --> nose
    #
    # [ cos(pitch) -sin(pitch) ]    [x]   [x']
    #                             X     =
    # [ sin(pitch)  cos(pitch) ]    [y]   [y']
    offsets = []
    for pitch in pitches:
        theta = pitch * np.pi / 180.0
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        x_off, y_off = np.matmul(R, sensor_offset)
        offsets.append(y_off)

    return offsets


class Coeffs:
    pass


# History of seabird25p.cfg file changes:

# [mccann@elvis i2MAP]$ pwd
# /mbari/M3/master/i2MAP
# [mccann@elvis i2MAP]$ ls -l */*/*/*/seabird25p.cfg
# -rwx------. 1        519 games  3050 Sep 20  2016 2017/01/20170117/2017.017.00/seabird25p.cfg
# -rwx------. 1        519 games  3050 Sep 20  2016 2017/01/20170117/2017.017.01/seabird25p.cfg
# -rwx------. 1 lonny      nobody 3050 Sep 20  2016 2017/04/20170407/2017.097.00/seabird25p.cfg
# -rwx------. 1 robs       games  3050 Sep 20  2016 2017/05/20170508/2017.128.00/seabird25p.cfg
# -rwx------. 1 robs       games  3109 May 11  2017 2017/05/20170512/2017.132.00/seabird25p.cfg
# -rwx------. 1 robs       games  3109 May 11  2017 2017/06/20170622/2017.173.00/seabird25p.cfg
# -rwx------. 1        519 games  3109 May 11  2017 2017/08/20170824/2017.236.00/seabird25p.cfg
# -rwx------. 1        519 games  3109 May 11  2017 2017/09/20170914/2017.257.00/seabird25p.cfg
# -rwx------. 1 etrauschke games  3109 Jan 29  2018 2018/01/20180125/2018.025.00/seabird25p.cfg
# -rwx------. 1 henthorn   games  3109 Feb 15  2018 2018/02/20180214/2018.045.03/seabird25p.cfg
# -rwx------. 1 lonny      games  3667 Mar  2  2018 2018/03/20180306/2018.065.02/seabird25p.cfg
# -rwx------. 1 lonny      games  3667 Mar  2  2018 2018/04/20180404/2018.094.00/seabird25p.cfg
# -rwx------. 1 lonny      games  3667 Mar  2  2018 2018/06/20180618/2018.169.01/seabird25p.cfg
# -rwx------. 1 lonny      games  3667 Jul 19  2018 2018/07/20180718/2018.199.00/seabird25p.cfg
# -rwx------. 1 jana       games  3667 Aug 30  2018 2018/08/20180829/2018.241.01/seabird25p.cfg
# -rwx------. 1 lonny      games  3667 Oct 25  2018 2018/10/20181023/2018.296.00/seabird25p.cfg
# -rwx------. 1 jana       games  3667 Mar  2  2018 2018/12/20181203/2018.337.00/seabird25p.cfg
# -rwx------. 1 jana       games  3667 Mar  2  2018 2018/12/20181210/2018.344.01/seabird25p.cfg
# -rwx------. 1 jana       games  3667 Mar  2  2018 2018/12/20181210/2018.344.05/seabird25p.cfg
# -rwx------. 1 jana       games  3667 Mar  2  2018 2018/12/20181210/2018.344.06/seabird25p.cfg
# -rwx------. 1 jana       games  3667 Mar  2  2018 2018/12/20181210/2018.344.07/seabird25p.cfg
# -rwx------. 1 jana       games  3667 Mar  2  2018 2018/12/20181210/2018.344.08/seabird25p.cfg
# -rwx------. 1 jana       games  3667 Mar  2  2018 2018/12/20181210/2018.344.09/seabird25p.cfg
# -rwx------. 1 jana       games  3667 Mar  2  2018 2018/12/20181210/2018.344.10/seabird25p.cfg
# -rwx------. 1 jana       games  3667 Mar  2  2018 2018/12/20181210/2018.344.11/seabird25p.cfg
# -rwx------. 1 jana       games  3667 Mar  2  2018 2018/12/20181210/2018.344.12/seabird25p.cfg
# -rwx------. 1 jana       games  3667 Mar  2  2018 2018/12/20181210/2018.344.13/seabird25p.cfg
# -rwx------. 1 jana       games  3667 Mar  2  2018 2018/12/20181214/2018.348.00/seabird25p.cfg
# -rwx------. 1 jana       games  3667 Mar  2  2018 2018/12/20181214/2018.348.01/seabird25p.cfg
# -rwx------. 1 jana       games  3667 Mar  2  2018 2018/12/20181214/2018.348.02/seabird25p.cfg
# -rwx------. 1 jana       games  3667 Mar  2  2018 2018/12/20181214/2018.348.03/seabird25p.cfg
# -rwx------. 1 lonny      games  3667 Mar  2  2018 2019/01/20190107/2019.007.07/seabird25p.cfg
# -rwx------. 1 lonny      games  3667 Mar  2  2018 2019/01/20190107/2019.007.09/seabird25p.cfg
# -rwx------. 1 lonny      nobody 3667 Mar  2  2018 2019/02/20190204/2019.035.10/seabird25p.cfg
# -rwx------. 1 lonny      nobody 3667 Mar  2  2018 2019/02/20190226/2019.057.00/seabird25p.cfg
# -rwx------. 1 lonny      nobody 3667 Mar  2  2018 2019/02/20190226/2019.057.01/seabird25p.cfg
# -rwx------. 1 lonny      nobody 3667 Mar  2  2018 2019/02/20190226/2019.057.02/seabird25p.cfg
# -rwx------. 1 lonny      nobody 3667 Mar  2  2018 2019/02/20190226/2019.057.03/seabird25p.cfg
# -rwx------. 1 lonny      nobody 3667 Mar  2  2018 2019/02/20190226/2019.057.04/seabird25p.cfg
# -rwx------. 1 lonny      nobody 3667 Mar  2  2018 2019/02/20190226/2019.057.05/seabird25p.cfg
# -rwx------. 1 lonny      nobody 3667 Mar  2  2018 2019/02/20190226/2019.057.06/seabird25p.cfg
# -rwx------. 1 lonny      nobody 3667 Mar  2  2018 2019/02/20190226/2019.057.07/seabird25p.cfg
# -rwx------. 1 lonny      nobody 3667 Mar  2  2018 2019/02/20190226/2019.057.08/seabird25p.cfg
# -rwx------. 1 lonny      nobody 3667 Mar  2  2018 2019/02/20190228/2019.059.01/seabird25p.cfg
# -rwx------. 1 lonny      nobody 3667 Mar  2  2018 2019/04/20190408/2019.098.01/seabird25p.cfg
# -rwx------. 1 lonny      nobody 3667 Mar  2  2018 2019/06/20190606/2019.157.00/seabird25p.cfg
# -rwx------. 1 lonny      nobody 3667 Mar  2  2018 2019/06/20190606/2019.157.01/seabird25p.cfg
# -rwx------. 1 lonny      nobody 3667 Mar  2  2018 2019/06/20190606/2019.157.02/seabird25p.cfg
# -rwx------. 1 lonny      nobody 3667 Mar  2  2018 2019/07/20190709/2019.190.00/seabird25p.cfg
# -rwx------. 1 lonny      nobody 3667 Mar  2  2018 2019/09/20190916/2019.259.01/seabird25p.cfg
# -rwx------. 1 lonny      nobody 3667 Mar  2  2018 2019/10/20191007/2019.280.02/seabird25p.cfg
# -rwx------. 1 lonny      nobody 3667 Mar  2  2018 2019/10/20191021/2019.294.00/seabird25p.cfg
# -rwx------. 1 lonny      nobody 3667 Mar  2  2018 2019/11/20191107/2019.311.00/seabird25p.cfg
# -rwx------. 1 lonny      nobody 3667 Mar  2  2018 2019/12/20191210/2019.344.06/seabird25p.cfg
# -rwx------. 1 lonny      nobody 3667 Mar  2  2018 2020/01/20200108/2020.008.00/seabird25p.cfg
# -rwx------. 1 mbassett   nobody 3667 Mar  2  2018 2020/02/20200210/2020.041.02/seabird25p.cfg
# -rwx------. 1 lonny      nobody 3667 Mar  2  2018 2020/02/20200224/2020.055.01/seabird25p.cfg
# -rwx------. 1 lonny      nobody 3667 Mar  2  2018 2020/06/20200629/2020.181.02/seabird25p.cfg
# -rwx------. 1 lonny      nobody 3667 Mar  2  2018 2020/07/20200728/2020.210.03/seabird25p.cfg
# -rwx------. 1 lonny      nobody 3667 Mar  2  2018 2020/08/20200811/2020.224.04/seabird25p.cfg
# -rwx------. 1 lonny      nobody 3899 Sep 11  2020 2020/09/20200914/2020.258.01/seabird25p.cfg
# -rwx------. 1 lonny      nobody 3919 Sep 21  2020 2020/09/20200922/2020.266.01/seabird25p.cfg
# -rwxr-xr-x. 1 brian      games  4267 Mar  1  2021 2021/03/20210303/2021.062.01/seabird25p.cfg
# -rwxr-xr-x. 1 robs       games  4267 Mar  1  2021 2021/03/20210330/2021.089.00/seabird25p.cfg
# -rwxr-xr-x. 1 robs       games  4267 Mar  1  2021 2021/05/20210512/2021.132.01/seabird25p.cfg
# -rwxr-xr-x. 1 robs       games  4267 Mar  1  2021 2021/06/20210624/2021.175.03/seabird25p.cfg
# -rwx------. 1 lonny      nobody 4267 Mar  1  2021 2021/09/20210921/2021.264.03/seabird25p.cfg
# -rwx------. 1 lonny      nobody 4267 Mar  1  2021 2021/10/20211018/2021.291.00/seabird25p.cfg
# -rwx------. 1 lonny      nobody 4267 Mar  1  2021 2021/11/20211103/2021.307.02/seabird25p.cfg
# -rwx------. 1 lonny      nobody 4267 Mar  1  2021 2022/03/20220302/2022.061.01/seabird25p.cfg


def _calibrated_temp_from_frequency(cf, nc):
    # From processCTD.m:
    # TC = 1./(t_a + t_b*(log(t_f0./temp_frequency)) + t_c*((log(t_f0./temp_frequency)).^2) + t_d*((log(t_f0./temp_frequency)).^3)) - 273.15;  # noqa: E501
    # From Seabird25p.cc:
    # if (*_t_coefs == 'A') {
    #   f = ::log(T_F0/f);
    #   T = 1/(T_A + (T_B + (T_C + T_D*f)*f)*f) - 273.15;
    # }
    # else if (*_t_coefs == 'G') {
    #   f = ::log(T_GF0/f);
    #   T = 1/(T_G + (T_H + (T_I + T_J*f)*f)*f) - 273.15;
    # }
    K2C = 273.15
    if cf.t_coefs == "A":
        calibrated_temp = (
            1.0
            / (
                cf.t_a
                + cf.t_b * np.log(cf.t_f0 / nc["temp_frequency"].to_numpy())
                + cf.t_c * np.power(np.log(cf.t_f0 / nc["temp_frequency"]), 2)
                + cf.t_d * np.power(np.log(cf.t_f0 / nc["temp_frequency"]), 3)
            )
            - K2C
        )
    elif cf.t_coefs == "G":
        calibrated_temp = (
            1.0
            / (
                cf.t_g
                + cf.t_h * np.log(cf.t_gf0 / nc["temp_frequency"].to_numpy())
                + cf.t_i * np.power(np.log(cf.t_gf0 / nc["temp_frequency"]), 2)
                + cf.t_j * np.power(np.log(cf.t_gf0 / nc["temp_frequency"]), 3)
            )
            - K2C
        )
    else:
        error_message = f"Unknown t_coefs: {cf.t_coefs}"
        raise ValueError(error_message)

    return calibrated_temp


def _calibrated_sal_from_cond_frequency(args, combined_nc, logger, cf, nc, temp):  # noqa: PLR0913
    # Comments carried over from doradosdp's processCTD.m:
    # Note that recalculation of conductivity and correction for thermal mass
    # are possible, however, their magnitude results in salinity differences
    # of less than 10^-4.
    # In other regions where these corrections are more significant, the
    # corrections can be turned on.
    # conductivity at S=35 psu , T=15 C [ITPS 68] and P=0 db) ==> 42.914
    sw_c3515 = 42.914
    eps = np.spacing(1)

    f_interp = interp1d(
        combined_nc["depth_time"].to_numpy().tolist(),
        combined_nc["depth_filtpres"].to_numpy(),
        fill_value=(
            combined_nc["depth_filtpres"].to_numpy()[0],
            combined_nc["depth_filtpres"].to_numpy()[-1],
        ),
        bounds_error=False,
    )
    p1 = f_interp(nc["time"].to_numpy().tolist())
    if args.plot:
        pbeg = 0
        pend = len(combined_nc["depth_time"])
        if args.plot.startswith("first"):
            pend = int(args.plot.split("first")[1])
        plt.figure(figsize=(18, 6))
        plt.plot(
            combined_nc["depth_time"][pbeg:pend],
            combined_nc["depth_filtpres"][pbeg:pend],
            ":o",
            nc["time"][pbeg:pend],
            p1[pbeg:pend],
            "o",
        )
        plt.legend(("Pressure from parosci", "Interpolated to ctd time"))
        title = "Comparing Interpolation of Pressure to CTD Time"
        title += f" - First {pend} Points from each series"
        plt.title(title)
        plt.grid()
        logger.debug("Pausing with plot entitled: %s. Close window to continue.", title)
        plt.show()

    # Conductivity Calculation
    # cfreq=cond_frequency/1000;
    # c1 = (c_a*(cfreq.^c_m)+c_b*(cfreq.^2)+c_c+c_d*TC)./(10*(1+eps*p1));
    #
    # seabird25p.cc: https://bitbucket.org/mbari/dorado-auv-qnx/src/master/auv/altex/onboard/seabird25p/Seabird25p.cc
    # if(*_c_coefs == 'A') {
    # C = (C_A*pow(f,C_M) + C_B*f*f +C_C +C_D*t)/(10*(1+EPS*p));
    # }
    # else if(*_c_coefs == 'G') {
    # C = (C_G +(C_H +(C_I + C_J*f)*f)*f*f) / (10.*(1+C_TCOR*t+C_PCOR*p)) ;
    # }
    # else {
    # Syslog::write("Seabird25p::calculate_Cond(): no c_coefs set selected.\n");
    # C=0;
    # }
    cfreq = nc["cond_frequency"].to_numpy() / 1000.0

    if cf.c_coefs == "A":
        calibrated_conductivity = (
            cf.c_a * np.power(cfreq, cf.c_m)
            + cf.c_b * np.power(cfreq, 2)
            + cf.c_c
            + cf.c_d * temp.to_numpy()
        ) / (10 * (1 + eps * p1))
    elif cf.c_coefs == "G":
        # C = (C_G +(C_H +(C_I + C_J*f)*f)*f*f) / (10.*(1+C_TCOR*t+C_PCOR*p)) ;
        calibrated_conductivity = (
            cf.c_g + (cf.c_h + (cf.c_i + cf.c_j * cfreq) * cfreq) * np.power(cfreq, 2)
        ) / (10 * (1 + cf.c_tcor * temp.to_numpy() + cf.c_pcor * p1))
    else:
        error_message = f"Unknown c_coefs: {cf.c_coefs}"
        raise ValueError(error_message)

    # % Calculate Salinty
    # cratio = c1*10/sw_c3515; % sw_C is conductivity value at 35,15,0
    # CTD.salinity = sw_salt(cratio,CTD.temperature,p1); % (psu)
    # seabird25p.cc: https://bitbucket.org/mbari/dorado-auv-qnx/src/master/auv/altex/onboard/seabird25p/Seabird25p.cc
    # //
    # // rsm 28 Mar 07: Compute salinity from conductivity, temperature and
    # // presssure:
    # cndr      = 10.*read_cond/sw_c3515();
    # salinity  = sw_salt( cndr, read_temp, depthSensor_pres);
    cratio = calibrated_conductivity * 10 / sw_c3515
    calibrated_salinity = eos80.salt(cratio, temp, p1)

    return calibrated_conductivity, calibrated_salinity


def _oxsat(temperature, salinity):
    #
    # ----------------------------------
    # Oxygen saturation: f(T,S); ml/l
    # ----------------------------------
    # TK = 273.15+T;  % degrees Kelvin
    # A1 = -173.4292; A2 = 249.6339; A3 = 143.3483; A4 = -21.8492;
    # B1 = -0.033096; B2 = 0.014259; B3 =  -0.00170;
    # OXSAT = exp(A1 + A2*(100./TK) + A3*log(TK/100) + A4*(TK/100) + [S .* (B1 + B2*(TK/100) + (B3*(TK/100).*(TK/100)))] );  # noqa: E501
    tk = 273.15 + temperature  # degrees Kelvin
    a1 = -173.4292
    a2 = 249.6339
    a3 = 143.3483
    a4 = -21.8492
    b1 = -0.033096
    b2 = 0.014259
    b3 = -0.00170
    return np.exp(
        a1
        + a2 * (100 / tk)
        + a3 * np.log(tk / 100)
        + a4 * (tk / 100)
        + np.multiply(
            salinity,
            b1 + b2 * (tk / 100) + np.multiply(b3 * (tk / 100), (tk / 100)),
        ),
    )


def _calibrated_O2_from_volts(  # noqa: PLR0913
    combined_nc: np.array,
    cf: Coeffs,
    nc: xr.Dataset,
    var_name: str,
    temperature: xr.DataArray,
    salinity: xr.DataArray,
) -> tuple[np.array, np.array, str, str]:
    # Contents of doradosdp's calc_O2_SBE43.m:
    # ----------------------------------------
    # function [O2] = calc_O2_SBE43(O2V,T,S,P,O2cal,time,units);
    # To calculate Oxygen from sbe voltage
    # Reference: W.B. Owens and R.C. Millard, 1985. A new algorithm for CTD oxygen
    # calibration, J. Phys. Oceanogr. 15:621-631.
    # Also, described in SeaBird application note.
    # pltit = 'n';
    # % disp(['   Pressure should be in dB']);
    f_interp = interp1d(
        combined_nc["depth_time"].to_numpy().tolist(),
        combined_nc["depth_filtpres"].to_numpy(),
        fill_value=(
            combined_nc["depth_filtpres"].to_numpy()[0],
            combined_nc["depth_filtpres"].to_numpy()[-1],
        ),
        bounds_error=False,
    )
    pressure = f_interp(nc["time"].to_numpy().tolist())

    #
    # ----------------------------------
    # Oxygen voltage
    # ----------------------------------
    # % disp(['   Minimum of oxygen voltage ' num2str(min(O2V)) ' V']);
    # % disp(['   Maximum of oxygen voltage ' num2str(max(O2V)) ' V']);
    # % disp(['   Mean of oxygen voltage ' num2str(mean(O2V)) ' V']);
    # docdt = [NaN;[diff(O2V)./diff(time)]];  % slope of oxygen current (uA/sec);
    docdt = np.append(
        np.nan,
        np.divide(
            np.diff(nc[var_name]),
            np.diff(nc["time"].astype(np.int64).to_numpy() / 1e9),
        ),
    )

    oxsat = _oxsat(temperature, salinity)

    # Owens-Millard equation
    #
    # ----------------------------------
    # Oxygen concentration (mL/L)
    # ----------------------------------
    # Constants
    # tau=0;
    #
    # O2 = [O2cal.SOc * ((O2V+O2cal.offset)+(tau*docdt)) + O2cal.BOc * exp(-0.03*T)].*exp(O2cal.Tcor*T + O2cal.Pcor*P).*OXSAT;  # noqa: E501
    tau = 0.0
    try:
        o2_mll = np.multiply(
            cf.SOc * ((nc[var_name].to_numpy() + cf.Voff) + (tau * docdt))
            + cf.BOc * np.exp(-0.03 * temperature.to_numpy()),
            np.multiply(
                np.exp(cf.TCor * temperature.to_numpy() + cf.PCor * pressure),
                oxsat.to_numpy(),
            ),
        )
    except AttributeError as e:
        error_message = f"Cannot calculate o2_mll: {e}"
        raise ValueError(error_message) from e

    #
    # if strcmp(units,'umolkg')==1
    # ----------------------------------
    # Convert to umol/kg
    # ----------------------------------
    # SeaBird equations are for ml/l computations
    #  Can convert OXSAT at atmospheric pressure to mg/l by 1.4276
    #  Convert dissolved O2 to mg/l using density of oxygen = 1.4276 kg/m^3
    # dens=sw_dens(S,T,P);
    # O2 = (O2 * 1.4276) .* (1e6./(dens*32));
    dens = eos80.dens(salinity.to_numpy(), temperature.to_numpy(), pressure)
    o2_umolkg = np.multiply(o2_mll * 1.4276, (1.0e6 / (dens * 32)))

    return o2_mll, o2_umolkg


def _calibrated_O2_from_volts_SBE43(  # noqa: PLR0913
    combined_nc: np.array,
    cf: Coeffs,
    nc: xr.Dataset,
    var_name: str,
    temperature: xr.DataArray,
    salinity: xr.DataArray,
) -> tuple[np.array, np.array]:
    # Written to handle the seabird25p O2 sensor from the i2map vehicle - October 2023
    # - Uses Equation 1 from the SeaBird 25p manual
    #
    # See for example: "/Volumes/DMO/MDUC_CORE_CTD_200103/Calibration Files/SBE-43/2510/2014_sep/SBE 43 O2510 09Sep14.pdf"  # noqa: E501
    # Soc = oxygen calibration coefficient (ml/l/V)
    # V = measured voltage (V)
    # Voffset = voltage offset (V)
    # A = temperature compensation coefficient (1/°C)
    # B = temperature compensation coefficient (1/°C)
    # C = temperature compensation coefficient (1/°C)
    # T = temperature (°C, ITS-90)
    # E = pressure compensation coefficient (1/dbar)
    # K = temperature (°K)
    # P = pressure (dbar)

    f_interp = interp1d(
        combined_nc["depth_time"].to_numpy().tolist(),
        combined_nc["depth_filtpres"].to_numpy(),
        fill_value=(
            combined_nc["depth_filtpres"].to_numpy()[0],
            combined_nc["depth_filtpres"].to_numpy()[-1],
        ),
        bounds_error=False,
    )
    pressure = f_interp(nc["time"].to_numpy().tolist())

    # Oxsol(T,S) = oxygen saturation (ml/l); P = pressure (dbar)
    oxsat = _oxsat(temperature, salinity)

    # Oxygen concentration (ml/l) = Soc * (V + Voffset) * (1.0 + A * T + B * T**2 + C * T**3 ) * Oxsol(T,S) * exp(E * P / K)  # noqa: E501
    o2_mll = np.multiply(
        cf.Soc * (nc[var_name].to_numpy() + cf.offset),
        np.multiply(
            (
                1.0
                + cf.A * temperature.to_numpy()
                + cf.B * np.power(temperature.to_numpy(), 2)
                + cf.C * np.power(temperature.to_numpy(), 3)
            ),
            np.multiply(
                oxsat.to_numpy(),
                np.exp(np.divide(cf.E * pressure, (273.15 + temperature.to_numpy()))),
            ),
        ),
    )

    # if strcmp(units,'umolkg')==1
    # ----------------------------------
    # Convert to umol/kg
    # ----------------------------------
    # SeaBird equations are for ml/l computations
    #  Can convert OXSAT at atmospheric pressure to mg/l by 1.4276
    #  Convert dissolved O2 to mg/l using density of oxygen = 1.4276 kg/m^3
    # dens=sw_dens(S,T,P);
    # O2 = (O2 * 1.4276) .* (1e6./(dens*32));
    dens = eos80.dens(salinity.to_numpy(), temperature.to_numpy(), pressure)
    o2_umolkg = np.multiply(o2_mll * 1.4276, (1.0e6 / (dens * 32)))

    return o2_mll, o2_umolkg


def _beam_transmittance_from_volts(combined_nc, nc) -> tuple[float, float]:
    # ----------------------------------------------
    # From: robs <robs@mbari.org>
    # Subject: Fwd: Merging i2MAP nav and CTD with VARS
    # Date: November 14, 2022 at 10:53:04 AM PST
    # To: Mike McCann <mccann@mbari.org>
    #
    # Oops, I'm sorry! Apparently I sent this to myself (ah, Monday)….
    #
    # Begin forwarded message:
    #
    # From: robs <robs@mbari.org>
    # Subject: Re: Merging i2MAP nav and CTD with VARS
    # Date: November 14, 2022 at 8:34:22 AM PST
    # To: Rob Sherlock <robs@mbari.org>
    #
    # Here is the Cal-sheet for the Transmissometer if you need it:
    # <pdf file transcribed>
    # C-Star Calibration
    # Date 11.25.14
    # S/N# CST-1694DR
    # Pathlength 25 cm
    #                     Analog Output   Digital Output
    # Vd                        0.006 V         0 counts
    # Vair                      4.830 V     15867 counts
    # Vref                      4.701 V     15443 counts

    # Relationship of transmittance (Tr) to beam attenuation coefficient (c),
    # and pathlength (x, in meters): Tr = exp(-c*x)

    # To determine beam transmittance: Tr = (Vsig - Vd) / (Vref - Vd)
    # To determine beam attenuation coefficient: c = -1/x * ln (Tr)

    # Vd Meter output with the beam blocked. This is the offset.
    # Vair Meter output in air with a clear beam path.
    # Vref Meter output with clean water in the path.
    # Temperature of calibration water: temperature of clean water used to obtain Vref.
    # Ambient temperature: meter temperature in air during the calibration.
    # Vsig Measured signal output of meter.
    # </pdf file transcribed>

    # Hard-coded values from the calibration sheet, but when they are available
    # in the .cfg file, they should be read from cf instead.
    Vd = 0.006
    Vref = 4.701
    #
    # Return beam transmittance (Tr) and beam attenuation coefficient (c)
    Tr = (nc["transmissometer"] - Vd) / (Vref - Vd)
    with np.errstate(invalid="ignore"):
        c = -1 / 0.25 * np.log(Tr)

    return Tr, c


class SensorInfo:
    pass


class Calibrate_NetCDF:
    logger = logging.getLogger(__name__)
    _handler = logging.StreamHandler()
    _handler.setFormatter(AUV_NetCDF._formatter)
    logger.addHandler(_handler)
    _log_levels = (logging.WARN, logging.INFO, logging.DEBUG)

    def global_metadata(self):
        """Use instance variables to return a dictionary of
        metadata specific for the data that are written
        """
        from datetime import datetime

        iso_now = datetime.now(tz=UTC).isoformat() + "Z"

        metadata = {}
        metadata["netcdf_version"] = "4"
        metadata["Conventions"] = "CF-1.6"
        metadata["date_created"] = iso_now
        metadata["date_update"] = iso_now
        metadata["date_modified"] = iso_now
        metadata["featureType"] = "trajectory"
        try:
            metadata["time_coverage_start"] = str(
                self.combined_nc["depth_time"].to_pandas().iloc[0].isoformat(),
            )
        except KeyError:
            error_message = "No depth_time variable in combined_nc"
            raise EOFError(error_message) from None
        metadata["time_coverage_end"] = str(
            self.combined_nc["depth_time"].to_pandas().iloc[-1].isoformat(),
        )
        metadata["distribution_statement"] = "Any use requires prior approval from MBARI"
        metadata["license"] = metadata["distribution_statement"]
        metadata["useconst"] = "Not intended for legal use. Data may contain inaccuracies."
        metadata["history"] = f"Created by {self.commandline} on {iso_now}"

        metadata["title"] = (
            f"Calibrated AUV sensor data from {self.args.auv_name} mission {self.args.mission}"
        )
        metadata["summary"] = (
            "Observational oceanographic data obtained from an Autonomous"
            " Underwater Vehicle mission with measurements at"
            " original sampling intervals. The data have been calibrated"
            " by MBARI's auv-python software."
        )
        if self.summary_fields:
            # Should be just one item in set, but just in case join them
            metadata["summary"] += " " + ". ".join(self.summary_fields)
        metadata["comment"] = (
            f"MBARI Dorado-class AUV data produced from original data"
            f" with execution of '{self.commandline}'' at {iso_now} on"
            f" host {gethostname()}. Software available at"
            f" 'https://github.com/mbari-org/auv-python'"
        )

        return metadata

    def _get_file(self, download_url, local_filename, session):
        with session.get(download_url, timeout=60) as resp:
            HTTP_OK = 200
            if resp.status != HTTP_OK:
                self.logger.warning(
                    "Cannot read %s, status = %s",
                    download_url,
                    resp.status,
                )
            else:
                self.logger.info("Started download to %s...", local_filename)
                with Path(local_filename).open("wb") as handle:
                    for chunk in resp.content.iter_chunked(1024):
                        handle.write(chunk)
                    if self.args.verbose > 1:
                        self.logger.info("%s(done)", Path(local_filename).name)

    def _define_sensor_info(self, start_datetime):
        # Using lower case vehicle names, modify below for changes over time
        # Used to reduce ERROR log messages for missing sensor data
        self.expected_sensors = {
            "dorado": [
                "navigation",
                "gps",
                "depth",
                "ecopuck",
                "hs2",
                "ctd1",
                "ctd2",
                "isus",
                "biolume",
                "lopc",
                "tailcone",
            ],
            "i2map": [
                "navigation",
                "gps",
                "depth",
                "seabird25p",
                "transmissometer",
                "tailcone",
            ],
        }

        # Horizontal and vertical distance from origin in meters
        # The origin of the x, y coordinate system is location of the
        # vehicle's paroscientific depth sensor in the tailcone.
        class SensorOffset(NamedTuple):
            x: float
            y: float

        # Original configuration of Dorado389 - Modify below with changes over time
        # This code uses pandas.shift() to apply a lag to the data. Posivite lag_secs
        # shifts the data forward in time to account for plumbing delays for the sensor.
        # As of April 2023 only integer lag_secs are supported because of pandas.shift().
        self.sinfo = OrderedDict(
            [
                (
                    "navigation",
                    {
                        "data_filename": "navigation.nc",
                        "cal_filename": None,
                        "lag_secs": None,
                        "sensor_offset": None,
                    },
                ),
                (
                    "gps",
                    {
                        "data_filename": "gps.nc",
                        "cal_filename": None,
                        "lag_secs": None,
                        "sensor_offset": None,
                    },
                ),
                (
                    "depth",
                    {
                        "data_filename": "parosci.nc",
                        "cal_filename": None,
                        "lag_secs": None,
                        "sensor_offset": SensorOffset(-0.927, -0.076),
                    },
                ),
                (
                    "hs2",
                    {
                        "data_filename": "hydroscatlog.nc",
                        "cal_filename": "hs2Calibration.dat",
                        "lag_secs": None,
                        "sensor_offset": SensorOffset(0.1397, -0.2794),
                    },
                ),
                (
                    "ctd1",
                    {
                        "data_filename": "ctdDriver.nc",
                        "cal_filename": "ctdDriver.cfg",
                        "lag_secs": None,
                        "sensor_offset": SensorOffset(1.003, 0.0001),
                    },
                ),
                (
                    "ctd2",
                    {
                        "data_filename": "ctdDriver2.nc",
                        "cal_filename": "ctdDriver2.cfg",
                        "lag_secs": None,
                        "sensor_offset": SensorOffset(1.003, 0.0001),
                    },
                ),
                (
                    "seabird25p",
                    {
                        "data_filename": "seabird25p.nc",
                        "cal_filename": "seabird25p.cfg",
                        "lag_secs": None,
                        "sensor_offset": SensorOffset(4.04, 0.0),
                    },
                ),
                (
                    "isus",
                    {
                        "data_filename": "isuslog.nc",
                        "cal_filename": None,
                        "lag_secs": 6,
                        "sensor_offset": None,
                    },
                ),
                (
                    "biolume",
                    {
                        "data_filename": "biolume.nc",
                        "cal_filename": None,
                        # See Slack thread https://mbari.slack.com/archives/C04ETLY6T7V/p1682439517159249?thread_ts=1682128534.742919&cid=C04ETLY6T7V
                        "lag_secs": 0.5,
                        "sensor_offset": SensorOffset(4.04, 0.0),
                        # From https://bitbucket.org/messiem/matlab_libraries/src/master/
                        # data_access/donnees_insitu/MBARI/AUV/charge_Dorado.m
                        # % UBAT flow conversion
                        # if time>=datenum(2010,6,29), flow_conversion=4.49E-04;
                        # else, flow_conversion=4.5E-04;			% calibration on 2/2/2009 but unknown before  # noqa: E501
                        # end
                        # flow_conversion=flow_conversion*1E3;	% using flow in mL/s
                        # flow1Hz=rpm*flow_conversion;
                        "flow_conversion": 4.5e-4 * 1e3,  # conversion to mL/s
                    },
                ),
                (
                    "lopc",
                    {
                        "data_filename": "lopc.nc",
                        "cal_filename": None,
                        "lag_secs": None,
                        "sensor_offset": None,
                    },
                ),
                (
                    "ecopuck",
                    {
                        "data_filename": "FLBBCD2K.nc",
                        "cal_filename": "FLBBCD2K-3695.dev",
                        "lag_secs": None,
                        "sensor_offset": None,
                    },
                ),
                (
                    "tailcone",
                    {
                        "data_filename": "tailCone.nc",
                        "cal_filename": None,
                        "lag_secs": None,
                        "sensor_offset": None,
                    },
                ),
            ],
        )

        # Changes over time
        if self.args.auv_name.lower().startswith("dorado"):
            self.sinfo["depth"]["sensor_offset"] = None
            if start_datetime >= datetime(2007, 4, 30, tzinfo=UTC):
                # First missions with 10 Gulpers: 2007.120.00 & 2007.120.01
                for instr in ("ctd1", "ctd2", "hs2", "lopc", "ecopuck", "isus"):
                    # TODO: Verify the length of the 10-Gulper midsection
                    self.sinfo[instr]["sensor_offset"] = SensorOffset(4.5, 0.0)
            if start_datetime >= datetime(2014, 9, 21, tzinfo=UTC):
                # First mission with 20 Gulpers: 2014.265.03
                for instr in ("ctd1", "ctd2", "hs2", "lopc", "ecopuck", "isus"):
                    self.sinfo[instr]["sensor_offset"] = SensorOffset(4.5, 0.0)
            if start_datetime >= datetime(2010, 6, 29, tzinfo=UTC):
                self.sinfo["biolume"]["flow_conversion"] = 4.49e-4 * 1e3

    def _range_qc_combined_nc(
        self,
        instrument: str,
        variables: list[str],
        ranges: dict,
        set_to_nan: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """For variables in combined_nc remove values that fall outside
        of specified min, max range.  Meant to be called by instrument so
        that the union of bad values from a set of variables can be removed.
        Use set_to_nan=True to set values outside of range to NaN instead of
        removing all variables from the instrument.  Setting set_to_nan=True
        makes sense for record (data) variables - such as ctd1_salinity,
        but not for coordinate variables."""
        out_of_range_indices = np.array([], dtype=int)
        vars_checked = []
        for var in variables:
            if var in self.combined_nc.variables:
                if var in ranges:
                    out_of_range = np.where(
                        (self.combined_nc[var] < ranges[var].min)
                        | (self.combined_nc[var] > ranges[var].max),
                    )[0]
                    self.logger.debug(
                        "%s: %d out of range values = %s",
                        var,
                        len(self.combined_nc[var][out_of_range].to_numpy()),
                        self.combined_nc[var][out_of_range].to_numpy(),
                    )
                    out_of_range_indices = np.union1d(
                        out_of_range_indices,
                        out_of_range,
                    )
                    if len(out_of_range_indices) > 500:  # noqa: PLR2004
                        self.logger.warning(
                            "More than 500 (%d) %s values found outside of range. "
                            "This may indicate a problem with the %s data.",
                            len(self.combined_nc[var][out_of_range_indices].to_numpy()),
                            var,
                            instrument,
                        )
                    if set_to_nan and var not in self.combined_nc.coords:
                        self.logger.info(
                            "Setting %s %s values to NaN", len(out_of_range_indices), var
                        )
                        self.combined_nc[var][out_of_range_indices] = np.nan
                    vars_checked.append(var)
                else:
                    self.logger.debug("No Ranges set for %s", var)
            else:
                self.logger.warning("%s not in self.combined_nc", var)
        inst_vars = [
            str(var) for var in self.combined_nc.variables if str(var).startswith(f"{instrument}_")
        ]
        self.logger.info(
            "Checked for data outside of these variables and ranges: %s",
            [(v, ranges[v]) for v in vars_checked],
        )
        if not set_to_nan:
            for var in inst_vars:
                self.logger.info(
                    "%s: deleting %d values found outside of above ranges: %s",
                    var,
                    len(self.combined_nc[var][out_of_range_indices].to_numpy()),
                    self.combined_nc[var][out_of_range_indices].to_numpy(),
                )
                coord = next(iter(self.combined_nc[var].coords))
                self.combined_nc[f"{var}_qced"] = (
                    self.combined_nc[var]
                    .drop_isel({coord: out_of_range_indices})
                    .rename({f"{instrument}_time": f"{instrument}_time_qced"})
                )
            self.combined_nc = self.combined_nc.drop_vars(inst_vars)
            for var in inst_vars:
                self.logger.debug("Renaming %s_qced to %s", var, var)
                self.combined_nc[var] = self.combined_nc[f"{var}_qced"].rename(
                    {f"{coord}_qced": coord},
                )
            qced_vars = [f"{var}_qced" for var in inst_vars]
            self.combined_nc = self.combined_nc.drop_vars(qced_vars)
        self.logger.info("Done range checking %s", instrument)

    def _read_data(self, logs_dir, netcdfs_dir):  # noqa: C901, PLR0912
        """Read in all the instrument data into member variables named by "sensor"
        Access xarray.Dataset like: self.ctd.data, self.navigation.data, ...
        Access calibration coefficients like: self.ctd.cals.t_f0, or as a
        dictionary for hs2 data.  Collect summary metadata fields that should
        describe the source of the data if copied from M3.
        """
        self.summary_fields = set()
        for sensor, info in self.sinfo.items():
            sensor_info = SensorInfo()
            orig_netcdf_filename = Path(netcdfs_dir, info["data_filename"])
            self.logger.debug(
                "Reading data from %s into self.%s.orig_data",
                orig_netcdf_filename,
                sensor,
            )
            try:
                sensor_info.orig_data = xr.open_dataset(
                    orig_netcdf_filename, decode_timedelta=False
                )
            except (FileNotFoundError, ValueError) as e:
                self.logger.debug(
                    "%-10s: Cannot open file %s: %s",
                    sensor,
                    orig_netcdf_filename,
                    e,
                )
            except OverflowError:
                self.logger.exception(
                    "%-10s: Cannot open file %s",
                    sensor,
                    orig_netcdf_filename,
                )
                self.logger.info(
                    "Perhaps _remove_bad_values() needs to be called for it in logs2netcdfs.py",
                )
            if info["cal_filename"]:
                cal_filename = Path(logs_dir, info["cal_filename"])
                self.logger.debug(
                    "Reading calibrations from %s into self.%s.cals",
                    orig_netcdf_filename,
                    sensor,
                )
                if str(cal_filename).endswith(".cfg"):
                    try:
                        sensor_info.cals = self._read_cfg(cal_filename)
                    except FileNotFoundError as e:
                        self.logger.debug("%s", e)
                elif str(cal_filename).endswith(".dev"):
                    try:
                        sensor_info.cals = self._read_eco_dev(cal_filename)
                    except FileNotFoundError as e:
                        self.logger.debug("%s", e)

            setattr(self, sensor, sensor_info)
            if hasattr(sensor_info, "orig_data"):
                try:
                    self.summary_fields.add(
                        getattr(self, sensor).orig_data.attrs["summary"],
                    )
                except KeyError:
                    self.logger.warning("%s: No summary field", orig_netcdf_filename)

        # TODO: Warn if no data found and if logs2netcdfs.py should be run

    def _read_cfg(self, cfg_filename):
        """Emulate what get_auv_cal.m and processCTD.m do in the
        Matlab doradosdp toolbox
        """
        self.logger.debug("Opening %s", cfg_filename)
        coeffs = Coeffs()
        # Default for non-i2map data
        coeffs.t_coefs = "A"
        coeffs.c_coefs = "A"
        with Path(cfg_filename).open() as fh:
            for line in fh:
                ##self.logger.debug(line)
                if line.startswith("//"):
                    continue
                # From get_auv_cal.m - Handle CTD calibration parameters
                if line[:2] in (
                    "t_",
                    "c_",
                    "ep",
                    "SO",
                    "BO",
                    "Vo",
                    "TC",
                    "PC",
                    "Sc",
                    "Da",
                ):
                    coeff, value = (s.strip() for s in line.split("="))
                    try:
                        self.logger.debug("Saving %s", line)
                        # Like in Seabird25p.cc use ?_coefs to determine which
                        # calibration scheme to use for i2map data
                        if coeff in {"t_coefs", "c_coefs"}:
                            setattr(coeffs, coeff, str(value.split(";")[0]))
                        else:
                            setattr(coeffs, coeff, float(value.split(";")[0]))
                    except ValueError as e:
                        self.logger.debug("%s", e)
        return coeffs

    def _cal_date_xml_files(
        self,
        sensor_dir: str,
        cal_date_dirs: list,
        serial_number: int,
    ) -> dict:
        cal_date_xml_files = {}
        for cal_date_dir in cal_date_dirs:
            find_cmd = f'find "{Path(sensor_dir, cal_date_dir)}" -iname "*.xml"'
            self.logger.debug("Executing %s", find_cmd)
            import subprocess

            safe_sensor_dir = Path(sensor_dir).resolve()
            safe_cal_date_dir = Path(sensor_dir, cal_date_dir).resolve()

            find_cmd = f'find "{safe_sensor_dir}" "{safe_cal_date_dir}" -iname "*.xml"'
            if not safe_sensor_dir.is_dir() or not safe_cal_date_dir.is_dir():
                error_message = "Invalid directory paths provided."
                raise ValueError(error_message)
            if not safe_sensor_dir.is_dir() or not safe_cal_date_dir.is_dir():
                error_message = "Invalid directory paths provided."
                raise ValueError(error_message)
            result = subprocess.run(  # noqa: S603
                shlex.split(find_cmd),  # noqa: S603
                capture_output=True,
                text=True,
                check=True,
            )
            xml_files = [x for x in result.stdout.split("\n") if x]
            if len(xml_files) == 0:
                self.logger.debug(
                    "Cannot find %s.xml in %s/%s",
                    serial_number,
                    sensor_dir,
                    cal_date_dir,
                )
                continue
            if len(xml_files) > 1:
                self.logger.warning(
                    "Found %d xml files in %s/%s",
                    len(xml_files),
                    sensor_dir,
                    cal_date_dir,
                )
                self.logger.info("{xml_files}")
            cal_xml_filename = xml_files[0]

            # The .xml file looks like:
            # <?xml version="1.0" encoding="UTF-8"?>
            # <OxygenSensor SensorID="38" SB_ConfigCTD_FileVersion="7.23.0.2" >
            #   <SerialNumber>2510</SerialNumber>
            #   <CalibrationDate>06-May-22</CalibrationDate>
            #   <Use2007Equation>1</Use2007Equation>
            #   <CalibrationCoefficients equation="0" >
            #     <!-- Coefficients for Owens-Millard equation. -->
            #     <Boc>0.0000</Boc>
            #     <Soc>0.0000e+000</Soc>
            #     ....
            try:
                root = ET.parse(cal_xml_filename).getroot()
            except ET.ParseError as e:
                self.logger.warning(
                    "Cannot parse %s: %s",
                    cal_xml_filename,
                    e,
                )
                continue
            try:
                cal_date = datetime.strptime(
                    root.find("CalibrationDate").text,
                    "%d-%b-%y",
                ).replace(tzinfo=UTC)
            except ValueError as e:
                self.logger.warning(
                    "Cannot parse CalibrationDate, %s",
                    root.find("CalibrationDate").text,
                )
                # "/Volumes/DMO/MDUC_CORE_CTD_200103/Calibration Files/SBE-43/143/2011_June/Oxygen_SBE43_0143.XML"  # noqa: E501
                # has:  <CalibrationDate>08-Jun-11p</CalibrationDate>
                if root.find("CalibrationDate").text.endswith("p"):
                    self.logger.info("Trying to parse CalibrationDate without 'p'")
                    cal_date = datetime.strptime(
                        root.find("CalibrationDate").text[:-1],
                        "%d-%b-%y",
                    ).replace(tzinfo=UTC)
                else:
                    error_message = (
                        f"Cannot parse CalibrationDate {root.find('CalibrationDate').text}"
                    )
                    raise ValueError(error_message) from e
            cal_date_xml_files[cal_date] = cal_xml_filename

        return OrderedDict(sorted(cal_date_xml_files.items()))

    def _read_oxy_coeffs(  # noqa: C901, PLR0912, PLR0915
        self,
        cfg_filename: Path,
        portstbd: str = "",
    ) -> tuple[Coeffs, str]:
        """Based on the serial number found as a comment in the .cfg file find
        the approriate calibration coefficients for the oxygen sensor within the
        '/DMO/MDUC_CORE_CTD_200103/Calibration Files' shared drive folder.
        portstbd is either "", "port" or "stbd".
        """
        # For i2map .cfg file lines look like:
        # //OxygenSerialNumber = 2510;
        # //note - this is the sensor in line with the C & T sensors. Goes to voltage channel 3
        #
        # //OxygenSerialNumber = 3968;
        # //note - this sensor is installed on the stbd side of the vehicle in line with the
        # //       transmissometer. Goes to voltage channel 5
        # //note - seabird has adopted a new DO calibration with a polynomial for temp correction
        # //A = -3.0812e-003
        # //B =  7.8442e-005
        # //C = -9.0601e-007
        # //E = 0.036
        # SOc = 0.4466;
        # BOc = 0.0000;
        # Voff = -0.5070;
        # TCor = -0.0000;
        # PCor = 1.3500e-04; //not given in new calibration sheet

        # Read from .cfg file to get the serial numbers of the oxygen sensors
        self.logger.debug("Opening %s", cfg_filename)
        coeffs = Coeffs()

        portstbd_order = {
            "port": 0,
            "stbd": 1,
        }  # Typical order of oxygen sensors in seabird25p.cfg file
        with cfg_filename.open() as fh:
            sensor_count = 0
            serial_numbers = []
            for line in fh:
                self.logger.debug(line)
                if line.startswith("//OxygenSerialNumber = "):
                    serial_numbers.append(int(line.split()[-1].strip(";")))
                    sensor_count += 1
        if len(serial_numbers) == 0:
            error_message = f"No oxygen sensor serial number found in {cfg_filename}"
            raise ValueError(error_message)
        if len(serial_numbers) > 2:  # noqa: PLR2004
            error_message = f"More than 2 oxygen sensor serial numbers found in {cfg_filename}"
            raise ValueError(error_message)
        if portstbd:
            serial_number = serial_numbers[portstbd_order[portstbd]]
            self.logger.info(
                "Looking for calibration file for O2 sensor serial number %s on %s side",
                serial_number,
                portstbd,
            )
        elif len(serial_numbers) == 1:
            self.logger.info(
                "Looking for calibration file for O2 sensor serial number %s",
                serial_numbers[0],
            )
            serial_number = serial_numbers[0]
        else:
            error_message = (
                f"Multiple oxygen sensor serial numbers found in {cfg_filename} "
                "with no port or stbd specified"
            )
            raise ValueError(error_message)

        # Find the calibration file for the oxygen sensor
        self.logger.debug(
            "Finding calibration file for oxygen serial number = %s on mission %s",
            serial_number,
            self.args.mission,
        )

        safe_calibration_dir = Path(self.calibration_dir).resolve()
        if not safe_calibration_dir.is_dir():
            error_message = f"Calibration directory '{self.calibration_dir}' does not exist"
            raise LookupError(error_message)
        find_cmd = f'find "{safe_calibration_dir}" -name "{serial_number}"'
        self.logger.info("Executing: %s ", find_cmd)
        safe_find_cmd = shlex.split(find_cmd)
        sensor_dir = subprocess.run(  # noqa: S603
            safe_find_cmd,  # noqa: S603
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        self.logger.debug("%s", sensor_dir)

        safe_sensor_dir = Path(sensor_dir).resolve()
        if not safe_sensor_dir.is_dir():
            error_message = f"Sensor directory '{sensor_dir}' does not exist"
            raise LookupError(error_message)
        # Find only the direct child directories: https://stackoverflow.com/a/20103980
        # Unable to use subprocess.run() with find an "*" in the command, apparently
        dir_find_cmd = f'find "{safe_sensor_dir}"/* -maxdepth 0 -type d'
        self.logger.debug("Executing: dir_find_cmd = %s", dir_find_cmd)
        cal_date_dirs = [x.split("/")[-1] for x in os.popen(dir_find_cmd).read().split("\n") if x]  # noqa: S605
        self.logger.info("Found calibration date dirs: %s", " ".join(cal_date_dirs))
        cal_dates = self._cal_date_xml_files(sensor_dir, cal_date_dirs, serial_number)
        mission_start = self.seabird25p.orig_data.cf["time"].to_numpy()[0]
        cal_date_to_use = next(iter(cal_dates))  # Default to first calibration date
        for cal_date in cal_dates:
            # Find the most recent calibration date just before the mission start
            self.logger.debug(
                "Comparing cal_date=%s with mission_start=%s", cal_date, mission_start
            )
            self.logger.info(
                "File %s has CalibrationDate %s",
                cal_dates[cal_date],
                cal_date,
            )
            if np.datetime64(cal_date.replace(tzinfo=None)) > mission_start:
                self.logger.info(
                    "Breaking from loop as %s is after %s with mission_start=%s",
                    cal_dates[cal_date],
                    self.args.mission,
                    mission_start,
                )
                break
            cal_date_to_use = cal_date

        if np.datetime64(cal_date_to_use.replace(tzinfo=None)) < mission_start:
            self.logger.info(
                "File %s is just before %s with mission_start=%s",
                cal_dates[cal_date_to_use],
                self.args.mission,
                mission_start,
            )
        else:
            self.logger.info(
                "File %s is the first calibration file, but is after %s with mission_start=%s",
                cal_dates[cal_date_to_use],
                self.args.mission,
                mission_start,
            )

        # Read the calibration coefficients from the .cal file which looks like:
        # INSTRUMENT_TYPE=SBE43
        # SERIALNO=2510
        # OCALDATE=09-Sep-14
        # SOC= 4.533809e-001
        # VOFFSET=-5.191352e-001
        # A=-5.251956e-003
        # B= 2.762519e-004
        # C=-4.164687e-006
        # E= 3.600000e-002
        # Tau20= 1.030000e+000

        # parse the .xml file to get the "equation 1" calibration coefficients:
        #  <CalibrationCoefficients equation="1" >
        #    <!-- Coefficients for Sea-Bird equation - SBE calibration in 2007 and later. -->
        #    <Soc>5.0544e-001</Soc>
        #    <offset>-0.5124</offset>
        #    <A>-4.8460e-003</A>
        #    <B> 2.2670e-004</B>
        #    <C>-3.2013e-006</C>
        #    <D0> 2.5826e+000</D0>
        #    <D1> 1.92634e-004</D1>
        #    <D2>-4.64803e-002</D2>
        #    <E> 3.6000e-002</E>
        #    <Tau20> 1.5600</Tau20>
        #    <H1>-3.3000e-002</H1>
        #    <H2> 5.0000e+003</H2>
        #    <H3> 1.4500e+003</H3>
        #  </CalibrationCoefficients>
        root = ET.parse(cal_dates[cal_date_to_use]).getroot()
        cal_xml_serial_number = int(root.find("SerialNumber").text)
        if cal_xml_serial_number != serial_number:
            self.logger.warning(
                "Serial number in %s = %s does not match %s",
                cal_dates[cal_date_to_use],
                cal_xml_serial_number,
                serial_number,
            )
        for elem in root.findall("CalibrationCoefficients[@equation]"):
            if elem.attrib["equation"] == "1":
                eq1 = elem
        for child in eq1:
            try:
                setattr(coeffs, child.tag, float(child.text))
            except ValueError:
                setattr(coeffs, child.tag, child.text)

        return coeffs, cal_dates[cal_date_to_use]

    def _read_eco_dev(self, dev_filename):
        """Read calibration information from the file associated with the
        ecopuck log data. The number match what are in the cal sheets in
        https://bitbucket.org/messiem/auv-biolum/src/master/DATA/sensors%20%26%20calibration/FLBBCD2K_Dorado/

        As of 13 January 2023 the contents of all the FLBBCD2K-3695.dev files are the same:
        ECO 	FLBBCD2K-3695
        Created on: 	10/29/2014

        COLUMNS=9
        N/U=1
        N/U=2
        N/U=3
        CHL=4		0.0073		45
        N/U=5
        Lambda=6	1.633E-06	46	700	700
        N/U=7
        CDOM=8		0.0909		45
        N/U=9
        """
        # Read the calibration coefficients from the .dev file, in case they change
        coeffs = Coeffs()
        with dev_filename.open() as fh:
            for line in fh:
                if line.startswith("CHL"):
                    # CHL (μg/l) = Scale Factor * (Output - Dark counts)
                    coeffs.chl_scale_factor = float(line.split()[1])
                    coeffs.chl_dark_counts = float(line.split()[2])
                elif line.startswith("Lambda"):
                    # From Scattering Meter Calibration Sheet - wavelength 700 nm
                    # "Lambda" == "bbp700" ?
                    # β(θc) m-1 sr-1 = Scale Factor x (Output - Dark Counts)
                    coeffs.bbp700_scale_factor = float(line.split()[1])
                    coeffs.bbp700_dark_counts = float(line.split()[2])
                elif line.startswith("CDOM"):
                    # CDOM (ppb) = Scale Factor x (Output - Dark Counts)
                    coeffs.cdom_scale_factor = float(line.split()[1])
                    coeffs.cdom_dark_counts = float(line.split()[2])
        return coeffs

    def _navigation_process(self, sensor):  # noqa: C901, PLR0912, PLR0915
        # AUV navigation data, which comes from a process on the vehicle that
        # integrates data from several instruments.  We use it to grab the DVL
        # data to help determine vehicle position when it is below the surface.
        #
        #  Nav.depth is used to compute pressure for salinity and oxygen computations
        #  Nav.latitude and Nav.longitude converted to degrees were added to
        #                                 the log file at end of 2004
        #  Nav.roll, Nav.pitch, Nav.yaw, Nav.Xpos and Nav.Ypos are extracted for
        #                                 3-D mission visualization
        try:
            orig_nc = getattr(self, sensor).orig_data
        except FileNotFoundError as e:
            self.logger.error("%s", e)  # noqa: TRY400
            return
        except AttributeError:
            error_message = (
                f"{sensor} has no orig_data - likely a missing or zero-sized .log file"
                f" in {Path(MISSIONLOGS, self.args.mission)}"
            )
            raise EOFError(error_message) from None

        # Remove non-monotonic times
        self.logger.debug("Checking for non-monotonic increasing times")
        monotonic = monotonic_increasing_time_indices(orig_nc.get_index("time"))
        if (~monotonic).any():
            self.logger.debug(
                "Removing non-monotonic increasing times at indices: %s",
                np.argwhere(~monotonic).flatten(),
            )
        orig_nc = orig_nc.sel(time=monotonic)

        source = self.sinfo[sensor]["data_filename"]
        coord_str = f"{sensor}_time {sensor}_depth {sensor}_latitude {sensor}_longitude"
        vars_to_qc = []
        # Units of these angles are radians in the original files, we want degrees
        vars_to_qc.append("navigation_roll")
        self.combined_nc["navigation_roll"] = xr.DataArray(
            orig_nc["mPhi"].to_numpy() * 180 / np.pi,
            coords=[orig_nc.get_index("time")],
            dims={f"{sensor}_time"},
            name=f"{sensor}_roll",
        )
        self.combined_nc["navigation_roll"].attrs = {
            "long_name": "Vehicle roll",
            "standard_name": "platform_roll_angle",
            "units": "degree",
            "coordinates": coord_str,
            "comment": f"mPhi from {source}",
        }

        vars_to_qc.append("navigation_pitch")
        self.combined_nc["navigation_pitch"] = xr.DataArray(
            orig_nc["mTheta"].to_numpy() * 180 / np.pi,
            coords=[orig_nc.get_index("time")],
            dims={"navigation_time"},
            name="pitch",
        )
        self.combined_nc["navigation_pitch"].attrs = {
            "long_name": "Vehicle pitch",
            "standard_name": "platform_pitch_angle",
            "units": "degree",
            "coordinates": coord_str,
            "comment": f"mTheta from {source}",
        }

        vars_to_qc.append("navigation_yaw")
        self.combined_nc["navigation_yaw"] = xr.DataArray(
            orig_nc["mPsi"].to_numpy() * 180 / np.pi,
            coords=[orig_nc.get_index("time")],
            dims={"navigation_time"},
            name="yaw",
        )
        self.combined_nc["navigation_yaw"].attrs = {
            "long_name": "Vehicle yaw",
            "standard_name": "platform_yaw_angle",
            "units": "degree",
            "coordinates": coord_str,
            "comment": f"mPsi from {source}",
        }

        self.combined_nc["navigation_posx"] = xr.DataArray(
            orig_nc["mPos_x"].to_numpy() - orig_nc["mPos_x"].to_numpy()[0],
            coords=[orig_nc.get_index("time")],
            dims={"navigation_time"},
            name="posx",
        )
        self.combined_nc["navigation_posx"].attrs = {
            "long_name": "Relative lateral easting",
            "units": "m",
            "coordinates": coord_str,
            "comment": f"mPos_x (minus first position) from {source}",
        }

        self.combined_nc["navigation_posy"] = xr.DataArray(
            orig_nc["mPos_y"].to_numpy() - orig_nc["mPos_y"].to_numpy()[0],
            coords=[orig_nc.get_index("time")],
            dims={"navigation_time"},
            name="posy",
        )
        self.combined_nc["navigation_posy"].attrs = {
            "long_name": "Relative lateral northing",
            "units": "m",
            "coordinates": coord_str,
            "comment": f"mPos_y (minus first position) from {source}",
        }

        vars_to_qc.append("navigation_depth")
        self.combined_nc["navigation_depth"] = xr.DataArray(
            orig_nc["mDepth"].to_numpy(),
            coords=[orig_nc.get_index("time")],
            dims={"navigation_time"},
            name="navigation_depth",
        )
        self.combined_nc["navigation_depth"].attrs = {
            "long_name": "Depth from Nav",
            "standard_name": "depth",
            "units": "m",
            "comment": f"mDepth from {source}",
        }

        self.combined_nc["navigation_mWaterSpeed"] = xr.DataArray(
            orig_nc["mWaterSpeed"].to_numpy(),
            coords=[orig_nc.get_index("time")],
            dims={"navigation_time"},
            name="navigation_mWaterSpeed",
        )
        self.combined_nc["navigation_mWaterSpeed"].attrs = {
            "long_name": "Current speed based upon DVL data",
            "standard_name": "platform_speed_wrt_sea_water",
            "units": "m/s",
            "comment": f"mWaterSpeed from {source}",
        }

        if "latitude" in orig_nc:
            navlat_var = "latitude"
        elif "latitudeNav" in orig_nc:
            # Starting with 2022.243.00 the latitude variable name was changed
            navlat_var = "latitudeNav"
        else:
            navlat_var = None  # noqa: F841
            self.logger.debug(
                "Likely before 2004.167.04 when latitude was added to navigation.log",
            )

        navlons = None
        navlats = None
        if "longitude" in orig_nc:
            # starting with 2004.167.04 latitude & longitude were added to navigation.log
            navlons = orig_nc["longitude"].to_numpy()
            navlats = orig_nc["latitude"].to_numpy()
        elif "longitudeNav" in orig_nc:
            # Starting with 2022.243.00 the longitude variable name was changed
            navlons = orig_nc["longitudeNav"].to_numpy()
            navlats = orig_nc["latitudeNav"].to_numpy()
        else:
            # Up through 2004.112.02 we converted from Easting/Northing to lat/lon
            # - all missions in Monterey Bay (Zone 10)
            self.logger.info(
                "Converting from Easting/Northing to lat/lon for mission %s",
                self.args.mission,
            )
            proj = pyproj.Proj(proj="utm", zone=10, ellps="WGS84", radians=False)
            navlons, navlats = proj(
                orig_nc["mPos_y"].to_numpy(),
                orig_nc["mPos_x"].to_numpy(),
                inverse=True,
            )
            navlons = navlons * np.pi / 180.0
            navlats = navlats * np.pi / 180.0

        if navlons.any() and navlats.any():
            vars_to_qc.append("navigation_latitude")
            self.combined_nc["navigation_latitude"] = xr.DataArray(
                navlats * 180 / np.pi,
                coords=[orig_nc.get_index("time")],
                dims={"navigation_time"},
                name="latitude",
            )
            self.combined_nc["navigation_latitude"].attrs = {
                "long_name": "latitude",
                "standard_name": "latitude",
                "units": "degrees_north",
                "comment": f"latitude (converted from radians) from {source}",
            }
            vars_to_qc.append("navigation_longitude")
            self.combined_nc["navigation_longitude"] = xr.DataArray(
                navlons * 180 / np.pi,
                coords=[orig_nc.get_index("time")],
                dims={"navigation_time"},
                name="longitude",
            )
            # Setting standard_name attribute here once sets it for all variables
            self.combined_nc["navigation_longitude"].coords[f"{sensor}_time"].attrs = {
                "standard_name": "time",
            }
            self.combined_nc["navigation_longitude"].attrs = {
                "long_name": "longitude",
                "standard_name": "longitude",
                "units": "degrees_east",
                "comment": f"longitude (converted from radians) from {source}",
            }
        else:
            # Setting standard_name attribute here once sets it for all variables
            self.combined_nc["navigation_depth"].coords[f"{sensor}_time"].attrs = {
                "standard_name": "time",
            }

        # % Remove obvious outliers that later disrupt the section plots.
        # % (First seen on mission 2008.281.03)
        # % In case we ever use this software for the D Allan B mapping vehicle determine
        # % the good depth range from the median of the depths
        # % From mission 2011.250.11 we need to first eliminate the near surface values
        # % before taking the median.
        # pdIndx = find(Nav.depth > 1);
        # posDepths = Nav.depth(pdIndx);
        pos_depths = np.where(self.combined_nc["navigation_depth"].to_numpy() > 1)
        if self.args.mission in {"2013.301.02", "2009.111.00"}:
            self.logger.info("Bypassing Nav QC depth check")
            maxGoodDepth = 1250
        else:
            if pos_depths[0].size == 0:
                self.logger.warning(
                    "No positive depths found in %s/navigation.nc",
                    self.args.mission,
                )
                maxGoodDepth = 1250
            else:
                maxGoodDepth = 7 * np.median(pos_depths)
                self.logger.debug("median of positive valued depths = %s", np.median(pos_depths))
            if maxGoodDepth < 0:
                maxGoodDepth = 100  # Fudge for the 2009.272.00 mission where median was -0.1347!
            if self.args.mission == "2010.153.01":
                maxGoodDepth = 1250  # Fudge for 2010.153.01 where the depth was bogus, about 1.3

        self.logger.debug("Finding depths less than '%s' and times > 0'", maxGoodDepth)

        if self.args.mission == "2010.172.01":
            self.logger.info(
                "Performing special QC for %s/navigation.nc",
                self.args.mission,
            )
            self._range_qc_combined_nc(
                instrument="navigation",
                variables=vars_to_qc,
                ranges={
                    "navigation_depth": Range(0, 1000),
                    "navigation_roll": Range(-180, 180),
                    "navigation_pitch": Range(-180, 180),
                    "navigation_yaw": Range(-360, 360),
                    "navigation_longitude": Range(-360, 360),
                    "navigation_latitude": Range(-90, 90),
                },
            )

        missions_to_check = {
            "2004.345.00",
            "2005.240.00",
            "2007.134.09",
            "2010.293.00",
            "2011.116.00",
            "2013.227.00",
            "2016.348.00",
            "2017.121.00",
            "2017.269.01",
            "2017.297.00",
            "2017.347.00",
            "2017.304.00",
            "2011.166.00",
        }
        if self.args.mission in missions_to_check:
            self.logger.info(
                "Removing points outside of Monterey Bay for %s/navigation.nc", self.args.mission
            )
            self._range_qc_combined_nc(
                instrument="navigation",
                variables=vars_to_qc,
                ranges={
                    "navigation_longitude": Range(-122.1, -121.7),
                    "navigation_latitude": Range(36, 37),
                },
            )
        if self.args.mission == "2010.284.00":
            self.logger.info(
                "Removing points outside of time range for %s/navigation.nc",
                self.args.mission,
            )
            self._range_qc_combined_nc(
                instrument="navigation",
                variables=[v for v in self.combined_nc.variables if v.startswith(sensor)],
                ranges={
                    f"{sensor}_time": Range(
                        pd.Timestamp(2010, 10, 11, 20, 0, 0),
                        pd.Timestamp(2010, 10, 12, 3, 28, 0),
                    ),
                },
            )

    def _nudge_pos(self, max_sec_diff_at_end=10):
        """Apply linear nudges to underwater latitudes and longitudes so that
        they match the surface gps positions.
        """
        try:
            lon = self.combined_nc["navigation_longitude"]
        except KeyError:
            error_message = "No navigation_longitude data in combined_nc"
            raise EOFError(error_message) from None
        lat = self.combined_nc["navigation_latitude"]
        lon_fix = self.combined_nc["gps_longitude"]
        lat_fix = self.combined_nc["gps_latitude"]

        # Use the shared function from AUV module
        lon_nudged, lat_nudged, segment_count, segment_minsum = nudge_positions(
            nav_longitude=lon,
            nav_latitude=lat,
            gps_longitude=lon_fix,
            gps_latitude=lat_fix,
            logger=self.logger,
            auv_name=self.args.auv_name,
            mission=self.args.mission,
            max_sec_diff_at_end=max_sec_diff_at_end,
            create_plots=False,
        )

        # Store results in instance variables for compatibility
        self.segment_count = segment_count
        self.segment_minsum = segment_minsum

        return lon_nudged, lat_nudged

    def _gps_process(self, sensor):
        try:
            orig_nc = getattr(self, sensor).orig_data
        except FileNotFoundError as e:
            self.logger.exception("%s", e)  # noqa: TRY401
            return
        except AttributeError:
            if self.args.mission == "2010.151.04":
                # Gulf of Mexico mission - use data from usbl.dat file(s)
                usbl_file = Path(
                    self.args.base_path,
                    self.args.auv_name,
                    MISSIONNETCDFS,
                    self.args.mission,
                    "usbl.nc",
                )
                if not usbl_file.exists():
                    # Copy from archive AUVCTD/missionnetcdfs/YYYY/YYYYJJJ the usbl.nc file
                    from archive import AUVCTD_VOL

                    year = self.args.mission.split(".")[0]
                    YYYYJJJ = "".join(self.args.mission.split(".")[:2])
                    missionnetcdfs_dir = Path(
                        AUVCTD_VOL,
                        MISSIONNETCDFS,
                        year,
                        YYYYJJJ,
                        self.args.mission,
                    )
                    shutil.copyfile(
                        Path(missionnetcdfs_dir, "usbl.nc"),
                        usbl_file,
                    )
                self.logger.info(
                    "Just for the GoMx mission 2010.151.04 use data from %s "
                    "that came from the missionlogs/usbl.dat file",
                    usbl_file,
                )
                orig_nc = xr.open_dataset(usbl_file)

                # Subsample usbl so that it has similar frequency to gps data
                # and convert to radians so that it matches the gps data
                orig_nc = orig_nc.isel(time=slice(None, None, 10))
                orig_nc["latitude"] = orig_nc["latitude"] * np.pi / 180.0
                orig_nc["longitude"] = orig_nc["longitude"] * np.pi / 180.0
            else:
                error_message = (
                    f"{sensor} has no orig_data - likely a missing or zero-sized .log file"
                    f" in {Path(MISSIONLOGS, self.args.mission)}"
                )
                raise EOFError(error_message) from None

        lat = orig_nc["latitude"] * 180.0 / np.pi
        if not lat.any():
            error_message = f"No latitude data found in {sensor}.log"
            raise ValueError(error_message)
        if orig_nc["longitude"][0] > 0:
            lon = -1 * orig_nc["longitude"] * 180.0 / np.pi
        else:
            lon = orig_nc["longitude"] * 180.0 / np.pi

        gps_time_to_save = orig_nc.get_index("time")
        lat_to_save = lat
        lon_to_save = lon

        source = self.sinfo[sensor]["data_filename"]
        vars_to_qc = []
        vars_to_qc.append("gps_latitude")
        self.combined_nc["gps_latitude"] = xr.DataArray(
            lat_to_save.to_numpy(),
            coords=[gps_time_to_save],
            dims={"gps_time"},
            name="gps_latitude",
        )
        self.combined_nc["gps_latitude"].attrs = {
            "long_name": "GPS Latitude",
            "standard_name": "latitude",
            "units": "degrees_north",
            "comment": f"latitude from {source}",
        }

        vars_to_qc.append("gps_longitude")
        self.combined_nc["gps_longitude"] = xr.DataArray(
            lon_to_save.to_numpy(),
            coords=[gps_time_to_save],
            dims={"gps_time"},
            name="gps_longitude",
        )
        # Setting standard_name attribute here once sets it for all variables
        self.combined_nc["gps_longitude"].coords[f"{sensor}_time"].attrs = {
            "standard_name": "time",
        }
        self.combined_nc["gps_longitude"].attrs = {
            "long_name": "GPS Longitude",
            "standard_name": "longitude",
            "units": "degrees_east",
            "comment": f"longitude from {source}",
        }
        if self.args.mission in {
            "2004.345.00",
            "2005.240.00",
            "2007.134.09",
            "2010.293.00",
            "2011.116.00",
            "2013.227.00",
            "2016.348.00",
            "2017.121.00",
            "2017.269.01",
            "2017.297.00",
            "2017.347.00",
            "2017.304.00",
            "2011.166.00",
        }:
            self.logger.info(
                "Removing points outside of Monterey Bay for %s/gps.nc", self.args.mission
            )
            self._range_qc_combined_nc(
                instrument="gps",
                variables=vars_to_qc,
                ranges={
                    "gps_latitude": Range(36, 37),
                    "gps_longitude": Range(-122.1, -121.7),
                },
            )

        # TODO: Put this in a separate module like match_to_gps.py or something
        # With navigation dead reckoned positions available in self.combined_nc
        # and the gps positions added we can now match the underwater inertial
        # (dead reckoned) positions to the surface gps positions.
        nudged_longitude, nudged_latitude = self._nudge_pos()
        self.combined_nc["nudged_latitude"] = nudged_latitude
        self.combined_nc["nudged_latitude"].attrs = {
            "long_name": "Nudged Latitude",
            "standard_name": "latitude",
            "units": "degrees_north",
            "comment": "Dead reckoned latitude nudged to GPS positions",
        }
        self.combined_nc["nudged_longitude"] = nudged_longitude
        self.combined_nc["nudged_longitude"].attrs = {
            "long_name": "Nudged Longitude",
            "standard_name": "longitude",
            "units": "degrees_east",
            "comment": "Dead reckoned longitude nudged to GPS positions",
        }

    def _depth_process(self, sensor, latitude=36, cutoff_freq=1):  # noqa: PLR0915
        """Depth data (from the Parosci) is 10 Hz - Use a butterworth window
        to filter recorded pressure to values that are appropriately sampled
        at 1 Hz (when matched with other sensor data).  cutoff_freq is in
        units of Hz.
        """
        try:
            orig_nc = getattr(self, sensor).orig_data
        except (FileNotFoundError, AttributeError) as e:
            self.logger.debug("Original data not found for %s: %s", sensor, e)
            return

        # Remove non-monotonic times
        self.logger.debug("Checking for non-monotonic increasing times")
        monotonic = monotonic_increasing_time_indices(orig_nc.get_index("time"))
        if (~monotonic).any():
            self.logger.debug(
                "Removing non-monotonic increasing times at indices: %s",
                np.argwhere(~monotonic).flatten(),
            )
        orig_nc = orig_nc.sel(time=monotonic)

        depths = orig_nc["depth"]
        # Remove egregious outliers before filtering seen form 2008 through 2012
        # ad hoc corrections for depth after testing stoqs_all_dorado load in July 2023
        mission_depth_ranges = {
            "2006.054.00": Range(-1, 150),  # Soquel Canyon
            "2007.120.00": Range(-0.5, 32),  # Shallow N. Monterey Bay
            "2007.120.01": Range(-0.5, 32),  # Shallow  N. Monterey Bay
            "2007.123.05": Range(-0.5, 32),  # Shallow  N. Monterey Bay
            "2008.281.03": Range(-1, 30),  # Shallow (< 30 m depth ) Soquel Bight
            "2009.084.02": Range(-1, 60),  # Diamond - lots of bad depths
            "2009.085.02": Range(-1, 60),  # Monterey Bay - lots of bad depths
            "2009.112.07": Range(-1, 30),  # Shallow Monterey Bay
            "2009.113.00": Range(-1, 30),  # Shallow Monterey Bay
            "2009.154.00": Range(-1, 50),  # Shallow Monterey Bay
            "2009.155.03": Range(-1, 50),  # Shallow Monterey Bay
            "2009.272.00": Range(-1, 40),  # Shallow Monterey Bay
            "2010.118.00": Range(-1, 260),  # Monterey Canyon transect
            "2010.181.01": Range(-0.5, 22),  # Shallow  N. Monterey Bay
            "2010.181.02": Range(-0.5, 22),  # Shallow  N. Monterey Bay
            # ESP drifter missions out at station 67-70 with Flyer doing casts and ESP
            # drifting south toward Davidson Seamount - no gulpers (Frederic sent me note about survey grouping)  # noqa: E501
            # Faulty parosci lead to several mission depth aborts at beginning of this set of volume surveys  # noqa: E501
            "2010.258.00": Range(-1, 110),  # Offshore CANON 2010
            "2010.258.01": Range(-1, 110),  # Offshore CANON 2010
            "2010.258.02": Range(-1, 110),  # Offshore CANON 2010
            "2010.258.03": Range(-1, 110),  # Offshore CANON 2010
            "2010.258.04": Range(-1, 110),  # Offshore CANON 2010
            "2010.259.01": Range(-1, 110),  # Offshore CANON 2010
            "2010.259.02": Range(-1, 110),  # Offshore CANON 2010
            "2011.061.00": Range(-1, 50),  # Shallow Monterey Bay
            "2011.171.01": Range(-1, 55),  # Shallow Monterey Bay
            "2011.250.01": Range(-1, 60),  # Shallow Monterey Bay
            "2011.263.00": Range(-1, 30),  # Shallow Monterey Bay
            "2011.285.01": Range(-1, 25),  # Shallow Monterey Bay
            "2012.258.00": Range(-1, 160),  # Shallow Monterey Bay
            "2012.270.04": Range(-1, 30),  # Shallow Monterey Bay
        }
        if self.args.mission in mission_depth_ranges:
            valid_depth_range = mission_depth_ranges[self.args.mission]
            self.logger.info(
                "Removing depths outside of valid_depth_range=%s for self.args.mission=%s",
                valid_depth_range,
                self.args.mission,
            )
            out_of_range = np.where(
                (depths < valid_depth_range.min) | (depths > valid_depth_range.max),
            )[0]
            self.logger.debug(
                "depths: %d out of range values = %s",
                len(depths[out_of_range].to_numpy()),
                depths[out_of_range].to_numpy(),
            )
            self.logger.info("Setting %d depths values to NaN", len(out_of_range))
            depths[out_of_range] = np.nan
        depths = depths.dropna("time", how="all")

        # From initial CVS commit in 2004 the processDepth.m file computed
        # pres from depth this way.  I don't know what is done on the vehicle
        # side where a latitude of 36 is not appropriate: GoM, SoCal, etc.
        self.logger.debug("Converting depth to pressure using latitude = %s", latitude)
        pres = eos80.pres(depths, latitude)

        # See https://docs.scipy.org/doc/scipy-1.0.0/reference/generated/scipy.signal.filtfilt.html#scipy.signal.filtfilt
        # and https://docs.scipy.org/doc/scipy-1.0.0/reference/generated/scipy.signal.butter.html#scipy.signal.butter
        # Sample rate should be 10 - calcuate it to be sure
        sample_rate = 1.0 / np.round(
            np.mean(np.diff(depths["time"])) / np.timedelta64(1, "s"),
            decimals=2,
        )
        if sample_rate != 10:  # noqa: PLR2004
            self.logger.warning(
                "Expected sample_rate to be 10 Hz, instead it's %.2f Hz",
                sample_rate,
            )

        # The Wn parameter for butter() is fraction of the Nyquist frequency
        Wn = cutoff_freq / (sample_rate / 2.0)
        b, a = signal.butter(8, Wn)
        try:
            depth_filtpres_butter = signal.filtfilt(b, a, pres)
        except ValueError as e:
            error_message = "Likely short or empty file"
            raise EOFError(error_message) from e
        depth_filtdepth_butter = signal.filtfilt(b, a, depths)

        # Use 10 points in boxcar as in processDepth.m
        a = 10
        b = signal.windows.boxcar(a)
        depth_filtpres_boxcar = signal.filtfilt(b, a, pres)
        pres_plot = True  # Set to False for debugging other plots
        if self.args.plot and pres_plot:
            # Use Pandas to plot multiple columns of data
            # to validate that the filtering works as expected
            pbeg = 0
            pend = len(depths.get_index("time"))
            if self.args.plot.startswith("first"):
                pend = int(self.args.plot.split("first")[1])
            df_plot = pd.DataFrame(index=depths.get_index("time")[pbeg:pend])
            df_plot["pres"] = pres[pbeg:pend]
            df_plot["depth_filtpres_butter"] = depth_filtpres_butter[pbeg:pend]
            df_plot["depth_filtpres_boxcar"] = depth_filtpres_boxcar[pbeg:pend]
            title = (
                f"First {pend} points from"
                f" {self.args.mission}/{self.sinfo[sensor]['data_filename']}"
            )
            ax = df_plot.plot(title=title, figsize=(18, 6))
            ax.grid("on")
            self.logger.debug("Pausing with plot entitled: %s. Close window to continue.", title)
            plt.show()

        depth_filtdepth = xr.DataArray(
            depth_filtdepth_butter,
            coords=[depths.get_index("time")],
            dims={"depth_time"},
            name="depth_filtdepth",
        )
        depth_filtdepth.attrs = {
            "long_name": "Filtered Depth",
            "standard_name": "depth",
            "units": "m",
            "comment": (
                f"Original {sample_rate:.3f} Hz depth data filtered using"
                f" Butterworth window with cutoff frequency of {cutoff_freq} Hz"
            ),
        }

        depth_filtpres = xr.DataArray(
            depth_filtpres_butter,
            coords=[depths.get_index("time")],
            dims={"depth_time"},
            name="depth_filtpres",
        )
        depth_filtpres.attrs = {
            "long_name": "Filtered Pressure",
            "standard_name": "sea_water_pressure",
            "units": "dbar",
            "comment": (
                f"Original {sample_rate:.3f} Hz pressure data filtered using"
                f" Butterworth window with cutoff frequency of {cutoff_freq} Hz"
            ),
        }

        self.combined_nc["depth_filtdepth"] = depth_filtdepth
        self.combined_nc["depth_filtpres"] = depth_filtpres

    def _hs2_process(self, sensor, logs_dir):  # noqa: C901, PLR0912, PLR0915
        try:
            orig_nc = getattr(self, sensor).orig_data
        except (FileNotFoundError, AttributeError) as e:
            self.logger.debug("Original data not found for %s: %s", sensor, e)
            return

        # Remove non-monotonic times
        self.logger.debug("Checking for non-monotonic increasing times")
        monotonic = monotonic_increasing_time_indices(orig_nc.get_index("time"))
        if (~monotonic).any():
            self.logger.debug(
                "Removing non-monotonic increasing times at indices: %s",
                np.argwhere(~monotonic).flatten(),
            )
        orig_nc = orig_nc.sel(time=monotonic)

        try:
            cal_fn = Path(logs_dir, self.sinfo["hs2"]["cal_filename"])
            cals = hs2_read_cal_file(cal_fn)
        except FileNotFoundError as e:
            self.logger.warning("Cannot process HS2 data: %s", e)
            return

        hs2 = hs2_calc_bb(orig_nc, cals)

        source = self.sinfo[sensor]["data_filename"]
        coord_str = f"{sensor}_time {sensor}_depth {sensor}_latitude {sensor}_longitude"

        # Blue backscatter
        if hasattr(hs2, "bbp420"):
            blue_bs = xr.DataArray(
                hs2.bbp420.to_numpy(),
                coords=[hs2.bbp420.get_index("time")],
                dims={"hs2_time"},
                name="hs2_bbp420",
            )
            blue_bs.attrs = {
                "long_name": "Particulate backscattering coefficient at 420 nm",
                "coordinates": coord_str,
                "units": "m-1",
                "comment": (f"Computed by hs2_calc_bb() from data in {source}"),
            }
        if hasattr(hs2, "bbp470"):
            blue_bs = xr.DataArray(
                hs2.bbp470.to_numpy(),
                coords=[hs2.bbp470.get_index("time")],
                dims={"hs2_time"},
                name="hs2_bbp470",
            )
            blue_bs.attrs = {
                "long_name": "Particulate backscattering coefficient at 470 nm",
                "coordinates": coord_str,
                "units": "m-1",
                "comment": (f"Computed by hs2_calc_bb() from data in {source}"),
            }

        # Red backscatter
        if hasattr(hs2, "bbp676"):
            red_bs = xr.DataArray(
                hs2.bbp676.to_numpy(),
                coords=[hs2.bbp676.get_index("time")],
                dims={"hs2_time"},
                name="hs2_bbp676",
            )
            red_bs.attrs = {
                "long_name": "Particulate backscattering coefficient at 676 nm",
                "coordinates": coord_str,
                "units": "m-1",
                "comment": (f"Computed by hs2_calc_bb() from data in {source}"),
            }
        if hasattr(hs2, "bbp700"):
            red_bs = xr.DataArray(
                hs2.bbp700.to_numpy(),
                coords=[hs2.bbp700.get_index("time")],
                dims={"hs2_time"},
                name="hs2_bbp700",
            )
            red_bs.attrs = {
                "long_name": "Particulate backscattering coefficient at 700 nm",
                "coordinates": coord_str,
                "units": "m-1",
                "comment": (f"Computed by hs2_calc_bb() from data in {source}"),
            }

        # Fluorescence
        if hasattr(hs2, "fl676"):
            fl676 = xr.DataArray(
                hs2.fl676.to_numpy(),
                coords=[hs2.fl676.get_index("time")],
                dims={"hs2_time"},
                name="hs2_fl676",
            )
            fl676.attrs = {
                "long_name": "Fluorescence at 676 nm",
                "coordinates": coord_str,
                "comment": (f"Computed by hs2_calc_bb() from data in {source}"),
            }
            fl = fl676
        if hasattr(hs2, "fl700"):
            fl700 = xr.DataArray(
                hs2.fl700.to_numpy(),
                coords=[hs2.fl700.get_index("time")],
                dims={"hs2_time"},
                name="hs2_fl700",
            )
            fl700.attrs = {
                "long_name": "Fluorescence at 700 nm",
                "coordinates": coord_str,
                "comment": (f"Computed by hs2_calc_bb() from data in {source}"),
            }
            fl = fl700

        # Zeroth level quality control - same as in legacy Matlab
        mblue = np.ma.masked_invalid(blue_bs)
        mblue = np.ma.masked_greater(mblue, 0.1)
        mred = np.ma.masked_invalid(red_bs)
        mred = np.ma.masked_greater(mred, 0.1)
        mfl = np.ma.masked_invalid(fl)
        mfl = np.ma.masked_greater(mfl, 0.02)

        bad_hs2 = [
            f"{b}, {r}, {f}"
            for b, r, f in zip(
                blue_bs.to_numpy()[:][mblue.mask],
                red_bs.to_numpy()[:][mred.mask],
                fl.to_numpy()[:][mfl.mask],
                strict=False,
            )
        ]

        if bad_hs2:
            self.logger.info(
                "Number of bad %s points: %d of %d",
                sensor,
                len(blue_bs.to_numpy()[:][mblue.mask]),
                len(blue_bs),
            )
            self.logger.debug(
                "Removing bad %s points (indices, (blue, red, fl)): %s, %s",
                sensor,
                np.where(mred.mask)[0],
                bad_hs2,
            )
            blue_bs = blue_bs[:][~mblue.mask]
            red_bs = red_bs[:][~mfl.mask]

        red_blue_plot = True  # Set to False for debugging other plots
        if self.args.plot and red_blue_plot:
            # Use Pandas to more easiily plot multiple columns of data
            pbeg = 0
            pend = len(blue_bs.get_index("hs2_time"))
            if self.args.plot.startswith("first"):
                pend = int(self.args.plot.split("first")[1])
            df_plot = pd.DataFrame(index=blue_bs.get_index("hs2_time")[pbeg:pend])
            df_plot["blue_bs"] = blue_bs[pbeg:pend]
            df_plot["red_bs"] = red_bs[pbeg:pend]
            ## df_plot["fl"] = fl[pbeg:pend]
            title = (
                f"First {pend} points from"
                f" {self.args.mission}/{self.sinfo[sensor]['data_filename']}"
            )
            ax = df_plot.plot(title=title, figsize=(18, 6), ylim=(-0.003, 0.004))
            ax.grid("on")
            self.logger.debug("Pausing with plot entitled: %s. Close window to continue.", title)
            plt.show()

        # Save blue, red, & fl to combined_nc, also
        if hasattr(hs2, "bbp420"):
            self.combined_nc["hs2_bbp420"] = blue_bs
        if hasattr(hs2, "bbp470"):
            self.combined_nc["hs2_bbp470"] = blue_bs
        if hasattr(hs2, "bbp676"):
            self.combined_nc["hs2_bbp676"] = red_bs
        if hasattr(hs2, "bbp700"):
            self.combined_nc["hs2_bbp700"] = red_bs
        if hasattr(hs2, "fl676"):
            self.combined_nc["hs2_fl676"] = fl
        if hasattr(hs2, "fl700"):
            self.combined_nc["hs2_fl700"] = fl

        # For missions before 2009.055.05 hs2 will have attributes like bbp470, bbp676, and fl676
        # Hobilabs modified the instrument in 2009 to now give:         bbp420, bbp700, and fl700,
        # apparently giving a better measurement of chlorophyll.
        #
        # Detect the difference in this code and keep the member names descriptive in the survey
        # data so the the end user knows the difference.

        # Align Geometry, correct for pitch
        self.combined_nc[f"{sensor}_depth"] = self._geometric_depth_correction(
            sensor,
            orig_nc,
        )
        out_fn = f"{self.args.auv_name}_{self.args.mission}_cal.nc"
        self.combined_nc[f"{sensor}_depth"].attrs = {
            "long_name": "Depth",
            "units": "m",
            "comment": (
                f"Variable depth_filtdepth from {out_fn} linearly interpolated"
                f" to {sensor}_time and corrected for pitch using"
                f" {self.sinfo[sensor]['sensor_offset']}"
            ),
        }

        # Coordinates latitude & longitude are interpolated to the sensor time
        # in the align.py code.  Here we add the sensor depths as this is where
        # the sensor offset is applied with _geometric_depth_correction().

    def _calibrated_oxygen(  # noqa: PLR0913
        self,
        logs_dir,
        sensor,
        cf,
        orig_nc,
        var_name,
        temperature,
        salinity,
        portstbd="",
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Calibrate oxygen data, returning DataArrays."""

        if sensor == "seabird25p":
            cf, cal_file = self._read_oxy_coeffs(
                Path(logs_dir, self.sinfo[sensor]["cal_filename"]),
                portstbd,
            )
            (
                oxy_mll,
                oxy_umolkg,
            ) = _calibrated_O2_from_volts_SBE43(
                self.combined_nc,
                cf,
                orig_nc,
                var_name,
                temperature,
                salinity,
            )
            mll_comment = (
                f"Derived from {var_name} from {sensor}.nc and eq 1 calibration coefficients "
                f"{vars(cf)} from {cal_file = }"
            )
            umolkg_comment = (
                f"Computed from oxygen_mll_{portstbd} with "
                "'np.multiply(o2_mll * 1.4276, (1.0e6 / (dens * 32)))'"
            )
            self.logger.info("%s: parsed from %s file: %s", var_name, cal_file, vars(cf))
        else:
            (
                oxy_mll,
                oxy_umolkg,
            ) = _calibrated_O2_from_volts(
                self.combined_nc,
                cf,
                orig_nc,
                var_name,
                temperature,
                salinity,
            )
            mll_comment = (
                f"Derived from {var_name} from {sensor}.nc using calibration "
                f"coefficients {vars(cf)}"
            )
            umolkg_comment = (
                "Computed from oxygen_mll with "
                "'np.multiply(o2_mll * 1.4276, (1.0e6 / (dens * 32)))'"
            )
        oxygen_mll = xr.DataArray(
            oxy_mll,
            coords=[orig_nc.get_index("time")],
            dims={f"{sensor}_time"},
            name="oxygen_mll" + portstbd,
        )
        oxygen_mll.attrs = {
            "long_name": "Dissolved Oxygen",
            "units": "ml/l",
            "comment": mll_comment,
        }

        oxygen_umolkg = xr.DataArray(
            oxy_umolkg,
            coords=[orig_nc.get_index("time")],
            dims={f"{sensor}_time"},
            name="oxygen_umolkg" + portstbd,
        )
        oxygen_umolkg.attrs = {
            "long_name": "Dissolved Oxygen",
            "units": "umol/kg",
            "comment": umolkg_comment,
        }
        return oxygen_mll, oxygen_umolkg

    def _ctd_process(self, logs_dir, sensor, cf):  # noqa: C901, PLR0912, PLR0915
        # Don't be put off by the length of this method.
        # It's lengthy because of all the possible netCDF variables and
        # attribute metadata that need to be added to the combined_nc.
        try:
            orig_nc = getattr(self, sensor).orig_data
        except FileNotFoundError as e:
            self.logger.exception("%s", e)  # noqa: TRY401
            return
        except AttributeError:
            error_message = (
                f"{sensor} has no orig_data - likely a missing or zero-sized .log file"
                f" in {Path(MISSIONLOGS, self.args.mission)}"
            )
            raise EOFError(error_message) from None

        # Remove non-monotonic times
        self.logger.debug("Checking for non-monotonic increasing times")
        monotonic = monotonic_increasing_time_indices(orig_nc.get_index("time"))
        if (~monotonic).any():
            self.logger.debug(
                "Removing non-monotonic increasing times at indices: %s",
                np.argwhere(~monotonic).flatten(),
            )
        orig_nc = orig_nc.sel(time=monotonic)

        # Need to do this zeroth-level QC to calibrate temperature
        orig_nc["temp_frequency"][orig_nc["temp_frequency"] == 0.0] = np.nan
        source = self.sinfo[sensor]["data_filename"]

        # === Temperature and salinity variables ===
        # Seabird specific calibrations
        vars_to_qc = []
        self.logger.debug("Calling _calibrated_temp_from_frequency()")
        temperature = xr.DataArray(
            _calibrated_temp_from_frequency(cf, orig_nc),
            coords=[orig_nc.get_index("time")],
            dims={f"{sensor}_time"},
            name="temperature",
        )
        temperature.attrs = {
            "long_name": "Temperature",
            "standard_name": "sea_water_temperature",
            "units": "degree_Celsius",
            "comment": (
                f"Derived from temp_frequency from {source} via calibration parms: {cf.__dict__}"
            ),
        }
        self.combined_nc[f"{sensor}_temperature"] = temperature

        self.logger.debug("Calling _calibrated_sal_from_cond_frequency()")
        cal_conductivity, cal_salinity = _calibrated_sal_from_cond_frequency(
            self.args,
            self.combined_nc,
            self.logger,
            cf,
            orig_nc,
            temperature,
        )
        conductivity = xr.DataArray(
            cal_conductivity,
            coords=[orig_nc.get_index("time")],
            dims={f"{sensor}_time"},
            name="conductivity",
        )
        conductivity.attrs = {
            "long_name": "Conductivity",
            "standard_name": "sea_water_conductivity",
            "units": "Siemens/meter",
            "comment": (
                f"Derived from cond_frequency from {source} via calibration parms: {cf.__dict__}"
            ),
        }
        self.combined_nc[f"{sensor}_conductivity"] = conductivity
        vars_to_qc.append(f"{sensor}_salinity")
        salinity = xr.DataArray(
            cal_salinity,
            coords=[orig_nc.get_index("time")],
            dims={f"{sensor}_time"},
            name="salinity",
        )
        salinity.attrs = {
            "long_name": "Salinity",
            "standard_name": "sea_water_salinity",
            "units": "",
            "comment": (
                f"Derived from cond_frequency from {source} via calibration parms: {cf.__dict__}"
            ),
        }
        self.combined_nc[f"{sensor}_salinity"] = salinity

        # Variables computed onboard the vehicle that are recomputed here
        self.logger.debug("Collecting temperature_onboard")
        temperature_onboard = xr.DataArray(
            orig_nc["temperature"],
            coords=[orig_nc.get_index("time")],
            dims={f"{sensor}_time"},
            name="temperature_onboard",
        )
        # Onboard software sets bad values to absolute zero - replace with NaN
        temperature_onboard[temperature_onboard <= -273] = np.nan  # noqa: PLR2004
        temperature_onboard.attrs = {
            "long_name": "Temperature computed onboard the vehicle",
            "units": "degree_Celsius",
            "comment": (
                "Temperature computed onboard the vehicle from"
                " calibration parameters installed on the vehicle"
                " at the time of deployment."
            ),
        }
        self.combined_nc[f"{sensor}_temperature_onboard"] = temperature_onboard

        self.logger.debug("Collecting conductivity_onboard")
        conductivity_onboard = xr.DataArray(
            orig_nc["conductivity"],
            coords=[orig_nc.get_index("time")],
            dims={f"{sensor}_time"},
            name="conductivity_onboard",
        )
        conductivity_onboard.attrs = {
            "long_name": "Conductivity computed onboard the vehicle",
            "units": "Siemens/meter",
            "comment": (
                "Temperature computed onboard the vehicle from"
                " calibration parameters installed on the vehicle"
                " at the time of deployment."
            ),
        }
        self.combined_nc[f"{sensor}_conductivity_onboard"] = conductivity_onboard

        if "salinity" in orig_nc:
            self.logger.debug("Collecting salinity_onboard")
            salinity_onboard = xr.DataArray(
                orig_nc["salinity"],
                coords=[orig_nc.get_index("time")],
                dims={f"{sensor}_time"},
                name="salinity_onboard",
            )
            salinity_onboard.attrs = {
                "long_name": "Salinity computed onboard the vehicle",
                "units": "",
                "comment": (
                    "Salinity computed onboard the vehicle from"
                    " calibration parameters installed on the vehicle"
                    " at the time of deployment."
                ),
            }
            self.combined_nc[f"{sensor}_salinity_onboard"] = salinity_onboard

        # === Oxygen variables ===
        # original values in units of volts
        self.logger.debug("Collecting dissolvedO2")
        try:
            dissolvedO2 = xr.DataArray(
                orig_nc["dissolvedO2"],
                coords=[orig_nc.get_index("time")],
                dims={f"{sensor}_time"},
                name="dissolvedO2",
            )
            dissolvedO2.attrs = {
                "long_name": "Dissolved Oxygen sensor",
                "units": "Volts",
                "comment": ("Analog Voltage Channel 6 - to be converted to umol/kg"),
            }
            self.combined_nc[f"{sensor}_dissolvedO2"] = dissolvedO2
            (
                self.combined_nc[f"{sensor}_oxygen_mll"],
                self.combined_nc[f"{sensor}_oxygen_umolkg"],
            ) = self._calibrated_oxygen(
                logs_dir,
                sensor,
                cf,
                orig_nc,
                "dissolvedO2",
                temperature,
                salinity,
                "",
            )
        except KeyError:
            self.logger.debug("No dissolvedO2 data in %s", self.args.mission)
        except ValueError as e:
            cfg_file = Path(
                MISSIONLOGS,
                "".join(self.args.mission.split(".")[:2]),
                self.args.mission,
                self.sinfo["ctd"]["cal_filename"],
            )
            self.logger.exception("Likely missing a calibration coefficient in %s", cfg_file)
            self.logger.error(e)  # noqa: TRY400
        self.logger.debug("Collecting dissolvedO2_port")
        try:
            dissolvedO2_port = xr.DataArray(
                orig_nc["dissolvedO2_port"],
                coords=[orig_nc.get_index("time")],
                dims={f"{sensor}_time"},
                name="dissolvedO2_port",
            )
            dissolvedO2_port.attrs = {
                "long_name": "Dissolved Oxygen port side sensor",
                "units": "Volts",
                "comment": ("Analog Voltage Channel 3 - to be converted to umol/kg"),
            }
            self.combined_nc[f"{sensor}_dissolvedO2_port"] = dissolvedO2_port
            (
                self.combined_nc[f"{sensor}_oxygen_mll_port"],
                self.combined_nc[f"{sensor}_oxygen_umolkg_port"],
            ) = self._calibrated_oxygen(
                logs_dir,
                sensor,
                cf,
                orig_nc,
                "dissolvedO2_port",
                temperature,
                salinity,
                "port",
            )
        except KeyError:
            self.logger.debug("No dissolvedO2_port data in %s", self.args.mission)
        self.logger.debug("Collecting dissolvedO2_port")
        try:
            dissolvedO2_stbd = xr.DataArray(
                orig_nc["dissolvedO2_stbd"],
                coords=[orig_nc.get_index("time")],
                dims={f"{sensor}_time"},
                name="dissolvedO2_stbd",
            )
            dissolvedO2_stbd.attrs = {
                "long_name": "Dissolved Oxygen stbd side sensor",
                "units": "Volts",
                "comment": ("Analog Voltage Channel 5 - to be converted to umol/kg"),
            }
            self.combined_nc[f"{sensor}_dissolvedO2_stbd"] = dissolvedO2_stbd
            (
                self.combined_nc[f"{sensor}_oxygen_mll_stbd"],
                self.combined_nc[f"{sensor}_oxygen_umolkg_stbd"],
            ) = self._calibrated_oxygen(
                logs_dir,
                sensor,
                cf,
                orig_nc,
                "dissolvedO2_stbd",
                temperature,
                salinity,
                "stbd",
            )
        except KeyError:
            self.logger.debug("No dissolvedO2_port data in %s", self.args.mission)

        # === flow variables ===
        # A lot of 0.0 values in Dorado missions until about 2020.282.01
        self.logger.debug("Collecting flow1")
        try:
            flow1 = xr.DataArray(
                orig_nc["flow1"],
                coords=[orig_nc.get_index("time")],
                dims={f"{sensor}_time"},
                name="flow1",
            )
            flow1.attrs = {
                "long_name": "Flow sensor on ctd1",
                "units": "Volts",
                "comment": f"flow1 from {source}",
            }
            self.combined_nc[f"{sensor}_flow1"] = flow1
        except KeyError:
            self.logger.debug("No flow1 data in %s", self.args.mission)
        self.logger.debug("Collecting flow2")
        try:
            flow2 = xr.DataArray(
                orig_nc["flow2"],
                coords=[orig_nc.get_index("time")],
                dims={f"{sensor}_time"},
                name="flow2",
            )
            flow2.attrs = {
                "long_name": "Flow sensor on ctd1",
                "units": "Volts",
                "comment": f"flow2 from {source}",
            }
            self.combined_nc[f"{sensor}_flow2"] = flow2
        except KeyError:
            self.logger.debug("No flow2 data in %s", self.args.mission)

        # === beam_transmittance variable from seabird25p on i2map vehicle ===
        try:
            beam_transmittance, _ = _beam_transmittance_from_volts(
                self.combined_nc,
                orig_nc,
            )
            beam_transmittance = xr.DataArray(
                beam_transmittance * 100.0,
                coords=[orig_nc.get_index("time")],
                dims={f"{sensor}_time"},
                name="beam_transmittance",
            )
            beam_transmittance.attrs = {
                "long_name": "Beam Transmittance",
                "units": "%",
                "comment": (
                    f"Calibrated Beam Transmittance from {source}'s transmissometer variable"
                ),
            }
            self.combined_nc[f"{sensor}_beam_transmittance"] = beam_transmittance

        except KeyError:
            self.logger.debug(
                "No transmissometer data in %s/%s.nc",
                self.args.mission,
                sensor,
            )

        self.combined_nc[f"{sensor}_depth"] = self._geometric_depth_correction(
            sensor,
            orig_nc,
        )
        out_fn = f"{self.args.auv_name}_{self.args.mission}_cal.nc"
        self.combined_nc[f"{sensor}_depth"].attrs = {
            "long_name": "Depth",
            "units": "m",
            "comment": (
                f"Variable depth_filtdepth from {out_fn} linearly interpolated"
                f" to {sensor}_time and corrected for pitch using"
                f" {self.sinfo[sensor]['sensor_offset']}"
            ),
        }

        # === PAR variable from ctd2 on dorado vehicle ===
        try:
            par = xr.DataArray(
                orig_nc["par"],
                coords=[orig_nc.get_index("time")],
                dims={f"{sensor}_time"},
                name="par",
            )
            par.attrs = {
                "long_name": "Photosynthetically Available Radiation",
                "units": "Volts",
                "comment": f"PAR from {source}'s par variable",
            }
            self.combined_nc[f"{sensor}_par"] = par

        except KeyError:
            self.logger.debug("No par data in %s/%s.nc", self.args.mission, sensor)

        self.combined_nc[f"{sensor}_depth"] = self._geometric_depth_correction(
            sensor,
            orig_nc,
        )
        out_fn = f"{self.args.auv_name}_{self.args.mission}_cal.nc"
        self.combined_nc[f"{sensor}_depth"].attrs = {
            "long_name": "Depth",
            "units": "m",
            "comment": (
                f"Variable depth_filtdepth from {out_fn} linearly interpolated"
                f" to {sensor}_time and corrected for pitch using"
                f" {self.sinfo[sensor]['sensor_offset']}"
            ),
        }

        # === ad hoc Range checking ===
        self.logger.info(
            "Performing range checking of %s in %s/%s.nc", vars_to_qc, self.args.mission, sensor
        )
        self._range_qc_combined_nc(
            instrument=sensor,
            variables=vars_to_qc,
            ranges={f"{sensor}_salinity": Range(30, 40)},
            set_to_nan=True,
        )
        if self.args.mission == "2010.284.00":
            self.logger.info(
                "Removing points outside of time range for %s/%s.nc", self.args.mission, sensor
            )
            self._range_qc_combined_nc(
                instrument=sensor,
                variables=[v for v in self.combined_nc.variables if v.startswith(sensor)],
                ranges={
                    f"{sensor}_time": Range(
                        pd.Timestamp(2010, 10, 11, 20, 0, 0),
                        pd.Timestamp(2010, 10, 12, 3, 28, 0),
                    ),
                },
            )

    def _tailcone_process(self, sensor):
        # As requested by Rob Sherlock capture propRpm for comparison with
        # mWaterSpeed from navigation.log
        try:
            orig_nc = getattr(self, sensor).orig_data
        except FileNotFoundError as e:
            self.logger.error("%s", e)  # noqa: TRY400
            return
        except AttributeError:
            error_message = (
                f"{sensor} has no orig_data - likely a missing or zero-sized .log file"
                f" in {Path(MISSIONLOGS, self.args.mission)}"
            )
            raise EOFError(error_message) from None

        # Remove non-monotonic times
        self.logger.debug("Checking for non-monotonic increasing times")
        try:
            monotonic = monotonic_increasing_time_indices(orig_nc.get_index("time"))
        except IndexError:
            error_message = "No data in tailcone.nc - likely empty tailcone.log file"
            raise ValueError(error_message) from None
        if (~monotonic).any():
            self.logger.debug(
                "Removing non-monotonic increasing times at indices: %s",
                np.argwhere(~monotonic).flatten(),
            )
        orig_nc = orig_nc.sel(time=monotonic)

        source = self.sinfo[sensor]["data_filename"]
        coord_str = f"{sensor}_time {sensor}_depth {sensor}_latitude {sensor}_longitude"
        self.combined_nc["tailcone_propRpm"] = xr.DataArray(
            orig_nc["propRpm"].to_numpy(),
            coords=[orig_nc.get_index("time")],
            dims={f"{sensor}_time"},
            name=f"{sensor}_propRpm",
        )
        self.combined_nc["tailcone_propRpm"].attrs = {
            "long_name": "Vehicle propeller speed",
            # Don't be confused by its name - propeller speed is logged in radians/sec.
            "units": "rad/s",
            "coordinates": coord_str,
            "comment": f"propRpm from {source} (convert to RPM by multiplying by 9.549297)",
        }

    def _ecopuck_process(self, sensor, cf):
        # ecpouck's first mission 2020.245.00 - email dialog on 5 Dec 2022 discussing
        # using it for developing an HS2 transfer function and comparison with LRAUV data
        try:
            orig_nc = getattr(self, sensor).orig_data
        except FileNotFoundError as e:
            self.logger.error("%s", e)  # noqa: TRY400
            return
        except AttributeError:
            error_message = (
                f"{sensor} has no orig_data - likely a missing or zero-sized .log file"
                f" in {Path(MISSIONLOGS, self.args.mission)}"
            )
            raise EOFError(error_message) from None

        # Remove non-monotonic times
        self.logger.debug("Checking for non-monotonic increasing times")
        monotonic = monotonic_increasing_time_indices(orig_nc.get_index("time"))
        if (~monotonic).any():
            self.logger.debug(
                "Removing non-monotonic increasing times at indices: %s",
                np.argwhere(~monotonic).flatten(),
            )
        orig_nc = orig_nc.sel(time=monotonic)

        source = self.sinfo[sensor]["data_filename"]
        coord_str = f"{sensor}_time {sensor}_depth {sensor}_latitude {sensor}_longitude"
        beta_700 = cf.bbp700_scale_factor * (orig_nc["BB_Sig"].to_numpy() - cf.bbp700_dark_counts)
        _, bbp = compute_backscatter(700, AVG_SALINITY, beta_700)  # 33.6

        self.combined_nc["ecopuck_bbp700"] = xr.DataArray(
            bbp,
            coords=[orig_nc.get_index("time")],
            dims={f"{sensor}_time"},
            name=f"{sensor}_bbp700",
        )
        self.combined_nc["ecopuck_bbp700"].attrs = {
            "long_name": "Particulate backscattering coefficient at 700 nm",
            "units": "m-1",
            "coordinates": coord_str,
            "comment": (
                f"BB_Sig from {source} converted to beta_700 using scale factor "
                f"{cf.bbp700_scale_factor} and dark counts {cf.bbp700_dark_counts}, "
                "then converted to bbp700 by the compute_backscatter() function."
            ),
        }

        self.combined_nc["ecopuck_cdom"] = xr.DataArray(
            cf.cdom_scale_factor * (orig_nc["CDOM_Sig"].to_numpy() - cf.cdom_dark_counts),
            coords=[orig_nc.get_index("time")],
            dims={f"{sensor}_time"},
            name=f"{sensor}_cdom",
        )
        self.combined_nc["ecopuck_cdom"].attrs = {
            "long_name": "Colored Dissolved Organic Matter",
            "units": "ppb",
            "coordinates": coord_str,
            "comment": (
                f"CDOM_Sig from {source} converted to cdom using scale factor "
                f"{cf.cdom_scale_factor} and dark counts {cf.cdom_dark_counts}"
            ),
        }

        self.combined_nc["ecopuck_chl"] = xr.DataArray(
            cf.chl_scale_factor * (orig_nc["Chl_Sig"].to_numpy() - cf.chl_dark_counts),
            coords=[orig_nc.get_index("time")],
            dims={f"{sensor}_time"},
            name=f"{sensor}_chl",
        )

        # From: FLBBCD2K-3695_(CHL)CharSheet.pdf
        # The relationship between fluorescence and chlorophyll-a concentrations in-situ is
        # highly variable. The scale factor listed on this document was determined using a
        # mono-culture of phytoplankton (Thalassiosira weissflogii). The population was
        # assumed to be reasonably healthy and the concentration was determined by using the
        # absorption method. To accurately determine chlorophyll concentration using a
        # fluorometer, you must perform secondary measurements on the populations of
        # interest. This is typically done using extraction-based measurement techniques on
        # discrete samples. For additional information on determining chlorophyll
        # concentration see "Standard Methods for the Examination of Water and Wastewater"
        # part 10200 H, published jointly by the American Public Health Association,
        # American Water Works Association, and the Water Environment ,)deration.
        self.combined_nc["ecopuck_chl"].attrs = {
            "long_name": "Chlorophyll",
            "units": "ug/l",
            "coordinates": coord_str,
            "comment": (
                f"Chl_Sig from {source} converted to chl using scale factor "
                f"{cf.chl_scale_factor} and dark counts {cf.chl_dark_counts}"
            ),
        }

    def _apply_plumbing_lag(
        self,
        sensor: str,
        time_index: pd.DatetimeIndex,
        time_name: str,
    ) -> tuple[xr.DataArray, str]:
        """
        Apply plumbing lag to a time index in the combined netCDF file.
        """
        # Convert lag_secs to milliseconds as np.timedelta64 neeeds an integer
        lagged_time = time_index - np.timedelta64(
            int(self.sinfo[sensor]["lag_secs"] * 1000),
            "ms",
        )
        # Need to update the sensor's time coordinate in the combined netCDF file
        # so that DataArrays created with lagged_time fit onto the coordinate
        self.combined_nc.coords[f"{sensor}_{time_name}"] = xr.DataArray(
            lagged_time,
            coords=[lagged_time],
            dims={f"{sensor}_{time_name}"},
            name=f"{sensor}_{time_name}",
        )
        lag_info = f"with plumbing lag correction of {self.sinfo[sensor]['lag_secs']} seconds"
        return lagged_time, lag_info

    def _biolume_process(self, sensor):
        try:
            orig_nc = getattr(self, sensor).orig_data
        except FileNotFoundError as e:
            self.logger.error("%s", e)  # noqa: TRY400
            return
        except AttributeError:
            error_message = (
                f"{sensor} has no orig_data - likely a missing or zero-sized .log file"
                f" in {Path(MISSIONLOGS, self.args.mission)}"
            )
            raise EOFError(error_message) from None

        # Remove non-monotonic times
        self.logger.debug("Checking for non-monotonic increasing time")
        monotonic = monotonic_increasing_time_indices(orig_nc.get_index("time"))
        if (~monotonic).any():
            self.logger.debug(
                "Removing non-monotonic increasing time at indices: %s",
                np.argwhere(~monotonic).flatten(),
            )
        orig_nc = orig_nc.sel({TIME: monotonic})

        self.logger.info("Checking for non-monotonic increasing %s", TIME60HZ)
        monotonic = monotonic_increasing_time_indices(orig_nc.get_index(TIME60HZ))
        if (~monotonic).any():
            self.logger.info(
                "Removing non-monotonic increasing %s at indices: %s",
                TIME60HZ,
                np.argwhere(~monotonic).flatten(),
            )
        orig_nc = orig_nc.sel({TIME60HZ: monotonic})

        self.combined_nc[f"{sensor}_depth"] = self._geometric_depth_correction(
            sensor,
            orig_nc,
        )

        source = self.sinfo[sensor]["data_filename"]
        self.combined_nc["biolume_flow"] = xr.DataArray(
            orig_nc["flow"].to_numpy() * self.sinfo["biolume"]["flow_conversion"],
            coords=[orig_nc.get_index("time")],
            dims={f"{sensor}_time"},
            name=f"{sensor}_flow",
        )
        self.combined_nc["biolume_flow"].attrs = {
            "long_name": "Bioluminesence pump flow rate",
            "units": "mL/s",
            "coordinates": f"{sensor}_time {sensor}_depth",
            "comment": f"flow from {source}",
        }

        lagged_time, lag_info = self._apply_plumbing_lag(
            sensor,
            orig_nc.get_index(TIME),
            TIME,
        )
        self.combined_nc["biolume_avg_biolume"] = xr.DataArray(
            orig_nc["avg_biolume"].to_numpy(),
            coords=[lagged_time],
            dims={f"{sensor}_{TIME}"},
            name=f"{sensor}_avg_biolume",
        )
        self.combined_nc["biolume_avg_biolume"].attrs = {
            "long_name": "Bioluminesence Average of 60Hz data",
            "units": "photons s^-1",
            "coordinates": f"{sensor}_{TIME} {sensor}_depth",
            "comment": f"avg_biolume from {source} {lag_info}",
        }

        lagged_time, lag_info = self._apply_plumbing_lag(
            sensor,
            orig_nc.get_index(TIME60HZ),
            TIME60HZ,
        )
        self.combined_nc["biolume_raw"] = xr.DataArray(
            orig_nc["raw"].to_numpy(),
            coords=[lagged_time],
            dims={f"{sensor}_{TIME60HZ}"},
            name=f"{sensor}_raw",
        )
        self.combined_nc["biolume_raw"].attrs = {
            "long_name": "Raw 60 hz biolume data",
            # xarray writes out its own units attribute
            "coordinates": f"{sensor}_{TIME60HZ} {sensor}_depth60hz",
            "comment": f"raw values from {source} {lag_info}",
        }
        if self.args.mission == "2010.284.00":
            self.logger.info(
                "Removing points outside of time range for %s/biolume.nc", self.args.mission
            )
            for time_axis in (TIME, TIME60HZ):
                self._range_qc_combined_nc(
                    instrument=sensor,
                    variables=[
                        "biolume_time",
                        "biolume_time60hz",
                        "biolume_depth",
                        "biolume_flow",
                        "biolume_avg_biolume",
                        "biolume_raw",
                    ],
                    ranges={
                        f"{sensor}_{time_axis}": Range(
                            pd.Timestamp(2010, 10, 11, 20, 0, 0),
                            pd.Timestamp(2010, 10, 12, 3, 28, 0),
                        ),
                    },
                    set_to_nan=True,
                )

    def _lopc_process(self, sensor):
        try:
            orig_nc = getattr(self, sensor).orig_data
        except FileNotFoundError as e:
            self.logger.error("%s", e)  # noqa: TRY400
            return
        except AttributeError:
            error_message = (
                f"{sensor} has no orig_data - likely a missing or zero-sized .log file"
                f" in {Path(MISSIONLOGS, self.args.mission)}"
            )
            raise EOFError(error_message) from None

        source = self.sinfo[sensor]["data_filename"]
        coord_str = f"{sensor}_time {sensor}_depth {sensor}_latitude {sensor}_longitude"

        # A lopc.nc file without a time variable will return a RangeIndex object
        # from orig_nc.get_index('time') - test for presence of actual 'time' coordinate
        if "time" not in orig_nc.coords:
            error_message = (
                f"{sensor} has no time coordinate - likely an incomplete lopc.nc file"
                f" in {Path(MISSIONLOGS, self.args.mission)}"
            )
            raise EOFError(error_message)

        self.combined_nc["lopc_countListSum"] = xr.DataArray(
            orig_nc["countListSum"].to_numpy(),
            coords=[orig_nc.get_index("time")],
            dims={f"{sensor}_time"},
            name=f"{sensor}_countListSum",
        )
        self.combined_nc["lopc_countListSum"].attrs = {
            "long_name": orig_nc["countListSum"].attrs["long_name"],
            "units": orig_nc["countListSum"].attrs["units"],
            "coordinates": coord_str,
            "comment": f"Sum of countListSum values by size class from {source}",
        }

        self.combined_nc["lopc_transCount"] = xr.DataArray(
            orig_nc["transCount"].to_numpy(),
            coords=[orig_nc.get_index("time")],
            dims={f"{sensor}_time"},
            name=f"{sensor}_transCount",
        )
        self.combined_nc["lopc_transCount"].attrs = {
            "long_name": orig_nc["transCount"].attrs["long_name"],
            "units": orig_nc["transCount"].attrs["units"],
            "coordinates": coord_str,
            "comment": f"transCount from {source}",
        }

        self.combined_nc["lopc_nonTransCount"] = xr.DataArray(
            orig_nc["nonTransCount"].to_numpy(),
            coords=[orig_nc.get_index("time")],
            dims={f"{sensor}_time"},
            name=f"{sensor}_nonTransCount",
        )
        self.combined_nc["lopc_nonTransCount"].attrs = {
            "long_name": orig_nc["nonTransCount"].attrs["long_name"],
            "units": orig_nc["nonTransCount"].attrs["units"],
            "coordinates": coord_str,
            "comment": f"nonTransCount from {source}",
        }

        self.combined_nc["lopc_LCcount"] = xr.DataArray(
            orig_nc["LCcount"].to_numpy(),
            coords=[orig_nc.get_index("time")],
            dims={f"{sensor}_time"},
            name=f"{sensor}_LCcount",
        )
        self.combined_nc["lopc_LCcount"].attrs = {
            "long_name": orig_nc["LCcount"].attrs["long_name"],
            "units": orig_nc["LCcount"].attrs["units"],
            "coordinates": coord_str,
            "comment": f"LCcount from {source}",
        }

        self.combined_nc["lopc_flowSpeed"] = xr.DataArray(
            orig_nc["flowSpeed"].to_numpy(),
            coords=[orig_nc.get_index("time")],
            dims={f"{sensor}_time"},
            name=f"{sensor}_flowSpeed",
        )
        self.combined_nc["lopc_flowSpeed"].attrs = {
            "long_name": orig_nc["flowSpeed"].attrs["long_name"],
            "units": orig_nc["flowSpeed"].attrs["units"],
            "coordinates": coord_str,
            "comment": f"flowSpeed from {source}",
        }

    def _isus_process(self, sensor):
        try:
            orig_nc = getattr(self, sensor).orig_data
        except FileNotFoundError as e:
            self.logger.error("%s", e)  # noqa: TRY400
            return
        except AttributeError:
            error_message = (
                f"{sensor} has no orig_data - likely a missing or zero-sized .log file"
                f" in {Path(MISSIONLOGS, self.args.mission)}"
            )
            raise EOFError(error_message) from None

        # Remove non-monotonic times
        self.logger.debug("Checking for non-monotonic increasing times")
        monotonic = monotonic_increasing_time_indices(orig_nc.get_index("time"))
        if (~monotonic).any():
            self.logger.debug(
                "Removing non-monotonic increasing times at indices: %s",
                np.argwhere(~monotonic).flatten(),
            )
        orig_nc = orig_nc.sel(time=monotonic)

        source = self.sinfo[sensor]["data_filename"]
        coord_str = f"{sensor}_time {sensor}_depth {sensor}_latitude {sensor}_longitude"

        self.combined_nc["isus_nitrate"] = xr.DataArray(
            orig_nc["isusNitrate"].to_numpy(),
            coords=[orig_nc.get_index("time")],
            dims={f"{sensor}_time"},
            name=f"{sensor}_nitrate",
        )
        self.combined_nc["isus_nitrate"].attrs = {
            "long_name": "Nitrate",
            "units": "micromoles/liter",
            "coordinates": coord_str,
            "comment": f"isusNitrate from {source}",
        }
        self.combined_nc["isus_temp"] = xr.DataArray(
            orig_nc["isusTemp"].to_numpy(),
            coords=[orig_nc.get_index("time")],
            dims={f"{sensor}_time"},
            name=f"{sensor}_temp",
        )
        self.combined_nc["isus_temp"].attrs = {
            "long_name": "Temperature from ISUS",
            "units": "Celsius",
            "coordinates": coord_str,
            "comment": f"isusTemp from {source}",
        }
        self.combined_nc["isus_quality"] = xr.DataArray(
            orig_nc["isusQuality"].to_numpy(),
            coords=[orig_nc.get_index("time")],
            dims={f"{sensor}_time"},
            name=f"{sensor}_quality",
        )
        self.combined_nc["isus_quality"].attrs = {
            "long_name": "Fit Residuals from ISUS",
            "units": "",
            "coordinates": coord_str,
            "comment": f"isusQuality from {source}",
        }

    def _geometric_depth_correction(self, sensor, orig_nc):
        """Performs the align_geom() function from the legacy Matlab.
        Works for any sensor, but requires navigation being processed first
        as its variables in combined_nc are required. Returns corrected depth
        array.
        """
        # Fix pitch values to first and last points for interpolation to time
        # values outside the range of the pitch values.
        # See https://stackoverflow.com/a/45446546
        # and https://github.com/scipy/scipy/issues/12707#issuecomment-672794335
        try:
            p_interp = interp1d(
                self.combined_nc["navigation_time"].to_numpy().tolist(),
                self.combined_nc["navigation_pitch"].to_numpy(),
                fill_value=(
                    self.combined_nc["navigation_pitch"].to_numpy()[0],
                    self.combined_nc["navigation_pitch"].to_numpy()[-1],
                ),
                bounds_error=False,
            )
        except KeyError:
            error_message = "No navigation_time or navigation_pitch in combined_nc."
            raise EOFError(error_message) from None
        pitch = p_interp(orig_nc["time"].to_numpy().tolist())

        d_interp = interp1d(
            self.combined_nc["depth_time"].to_numpy().tolist(),
            self.combined_nc["depth_filtdepth"].to_numpy(),
            fill_value=(
                self.combined_nc["depth_filtdepth"].to_numpy()[0],
                self.combined_nc["depth_filtdepth"].to_numpy()[-1],
            ),
            bounds_error=False,
        )
        orig_depth = d_interp(orig_nc["time"].to_numpy().tolist())
        offs_depth = align_geom(self.sinfo[sensor]["sensor_offset"], pitch)

        corrected_depth = xr.DataArray(
            (orig_depth - offs_depth).astype(np.float64).tolist(),
            coords=[orig_nc.get_index("time")],
            dims={f"{sensor}_time"},
            name=f"{sensor}_depth",
        )
        # 2008.289.03 has self.combined_nc["depth_time"][-1] (2008-10-16T15:42:32)
        # at lot less than               orig_nc["time"][-1] (2008-10-16T16:24:43)
        # which, with "extrapolate" causes wildly incorrect depths to -359 m
        # There may be other cases where this happens, in which case we'd like
        # a general solution. For now, we'll just correct this mission.
        d_beg_time_diff = (
            orig_nc["time"].to_numpy()[0] - self.combined_nc["depth_time"].to_numpy()[0]
        )
        d_end_time_diff = (
            orig_nc["time"].to_numpy()[-1] - self.combined_nc["depth_time"].to_numpy()[-1]
        )
        self.logger.info(
            "%s: d_beg_time_diff: %s, d_end_time_diff: %s",
            sensor,
            d_beg_time_diff.astype("timedelta64[s]"),
            d_end_time_diff.astype("timedelta64[s]"),
        )
        if self.args.mission in (
            "2008.289.03",
            "2010.259.01",
            "2010.259.02",
        ):
            # This could be a more general check for all missions, but let's restrict it
            # to known problematic missions for now.  The above info message can help
            # determine if this is needed for other missions.
            self.logger.info(
                "%s: Special QC for mission %s: Setting corrected_depth to NaN for times after %s",
                sensor,
                self.args.mission,
                self.combined_nc["depth_time"][-1].to_numpy(),
            )
            corrected_depth[
                np.where(
                    orig_nc.get_index("time") > self.combined_nc["depth_time"].to_numpy()[-1],
                )
            ] = np.nan
        if self.args.plot:
            plt.figure(figsize=(18, 6))
            plt.plot(
                orig_nc["time"].to_numpy(),
                orig_depth,
                "-",
                orig_nc["time"].to_numpy(),
                corrected_depth,
                "--",
                orig_nc["time"].to_numpy(),
                pitch,
                ".",
            )
            plt.ylabel("Depth (m) & Pitch (deg)")
            plt.legend(("Original depth", "Pitch corrected depth", "Pitch"))
            plt.title(
                f"Original and pitch corrected depth for {self.args.auv_name} {self.args.mission}",
            )
            plt.show()

        return corrected_depth

    def _process(self, sensor, logs_dir, netcdfs_dir):  # noqa: C901, PLR0912
        coeffs = None
        try:
            coeffs = getattr(self, sensor).cals
        except AttributeError as e:
            self.logger.debug("No calibration information for %s: %s", sensor, e)

        if sensor == "navigation":
            self._navigation_process(sensor)
        elif sensor == "gps":
            self._gps_process(sensor)
        elif sensor == "depth":
            self._depth_process(sensor)
        elif sensor == "ecopuck":
            self._ecopuck_process(sensor, coeffs)
        elif sensor == "hs2":
            self._hs2_process(sensor, logs_dir)
        elif sensor == "tailcone":
            self._tailcone_process(sensor)
        elif sensor == "lopc":
            self._lopc_process(sensor)
        elif sensor == "isus":
            self._isus_process(sensor)
        elif sensor in ("ctd1", "ctd2", "seabird25p"):
            if coeffs is not None:
                self._ctd_process(logs_dir, sensor, coeffs)
            elif hasattr(getattr(self, sensor), "orig_data"):
                self.logger.warning("No calibration information for %s", sensor)
        elif sensor == "biolume":
            self._biolume_process(sensor)
        elif hasattr(getattr(self, sensor), "orig_data"):
            self.logger.warning("No method (yet) to process %s", sensor)

    def write_netcdf(self, netcdfs_dir, vehicle: str = "", name: str = "") -> None:
        name = name or self.args.mission
        vehicle = vehicle or self.args.auv_name
        self.combined_nc.attrs = self.global_metadata()
        out_fn = Path(netcdfs_dir, f"{vehicle}_{name}_cal.nc")
        self.logger.info("Writing calibrated instrument data to %s", out_fn)
        if Path(out_fn).exists():
            Path(out_fn).unlink()
        self.combined_nc.to_netcdf(out_fn)
        self.logger.info(
            "Data variables written: %s",
            ", ".join(sorted(self.combined_nc.variables)),
        )

    def process_logs(self, vehicle: str = "", name: str = "", process_gps: bool = True) -> None:  # noqa: FBT001, FBT002
        name = name or self.args.mission
        vehicle = vehicle or self.args.auv_name
        logs_dir = Path(self.args.base_path, vehicle, MISSIONLOGS, name)
        netcdfs_dir = Path(self.args.base_path, vehicle, MISSIONNETCDFS, name)
        start_datetime = datetime.strptime(".".join(name.split(".")[:2]), "%Y.%j").astimezone(
            UTC,
        )
        self._define_sensor_info(start_datetime)
        self._read_data(logs_dir, netcdfs_dir)
        self.combined_nc = xr.Dataset()

        for sensor in self.sinfo:
            if not process_gps and sensor == "gps":
                continue  # to skip gps processing in conftest.py fixture
            getattr(self, sensor).cal_align_data = xr.Dataset()
            self.logger.debug("Processing %s %s %s", vehicle, name, sensor)
            try:
                self._process(sensor, logs_dir, netcdfs_dir)
            except EOFError as e:
                short_name = vehicle.lower()
                if vehicle == "Dorado389":
                    # For supporting pytest & conftest.py fixture
                    short_name = "dorado"
                if sensor in EXPECTED_SENSORS[short_name]:
                    self.logger.error("Error processing %s: %s", sensor, e)  # noqa: TRY400
                else:
                    self.logger.debug("Error processing %s: %s", sensor, e)
            except ValueError:
                self.logger.exception("Error processing %s", sensor)
            except KeyError as e:
                self.logger.error("Error processing %s: missing variable %s", sensor, e)  # noqa: TRY400

        return netcdfs_dir

    def process_command_line(self):
        """Process command line arguments using shared parser infrastructure."""
        examples = "Examples:" + "\n\n"
        examples += "  Calibrate original data for some missions:\n"
        examples += "    " + sys.argv[0] + " --mission 2020.064.10\n"
        examples += "    " + sys.argv[0] + " --auv_name i2map --mission 2020.055.01\n"

        # Use shared parser with calibrate-specific additions
        parser = get_standard_dorado_parser(
            description=__doc__,
            epilog=examples,
        )

        # Add calibrate-specific arguments
        parser.add_argument(
            "--plot",
            action="store",
            help="Create intermediate plots"
            " to validate data operations. Use first<n> to plot <n>"
            " points, e.g. first2000. Program blocks upon show.",
        )

        self.args = parser.parse_args()
        self.logger.setLevel(self._log_levels[self.args.verbose])
        self.commandline = " ".join(sys.argv)


if __name__ == "__main__":
    cal_netcdf = Calibrate_NetCDF()
    cal_netcdf.process_command_line()
    cal_netcdf.calibration_dir = "/Volumes/DMO/MDUC_CORE_CTD_200103/Calibration Files"
    p_start = time.time()
    # Set process_gps=False to skip time consuming _nudge_pos() processing
    # netcdf_dir = cal_netcdf.process_logs(process_gps=False)
    netcdf_dir = cal_netcdf.process_logs()
    cal_netcdf.write_netcdf(netcdf_dir)
    cal_netcdf.logger.info("Time to process: %.2f seconds", (time.time() - p_start))
