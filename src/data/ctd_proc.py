import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from seawater import eos80

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
    # TC = 1./(t_a + t_b*(log(t_f0./temp_frequency)) + t_c*((log(t_f0./temp_frequency)).^2) + t_d*((log(t_f0./temp_frequency)).^3)) - 273.15;
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
                + cf.t_b * np.log(cf.t_f0 / nc["temp_frequency"].values)
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
                + cf.t_h * np.log(cf.t_gf0 / nc["temp_frequency"].values)
                + cf.t_i * np.power(np.log(cf.t_gf0 / nc["temp_frequency"]), 2)
                + cf.t_j * np.power(np.log(cf.t_gf0 / nc["temp_frequency"]), 3)
            )
            - K2C
        )
    else:
        raise ValueError(f"Unknown t_coefs: {cf.t_coefs}")

    return calibrated_temp


def _calibrated_sal_from_cond_frequency(args, combined_nc, logger, cf, nc, temp):
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
        combined_nc["depth_time"].values.tolist(),
        combined_nc["depth_filtpres"].values,
        fill_value="extrapolate",
    )
    p1 = f_interp(nc["time"].values.tolist())
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
        logger.debug(
            f"Pausing with plot entitled: {title}." " Close window to continue."
        )
        plt.show()

    # %% Conductivity Calculation
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
    cfreq = nc["cond_frequency"].values / 1000.0

    if cf.c_coefs == "A":
        calibrated_conductivity = (
            cf.c_a * np.power(cfreq, cf.c_m)
            + cf.c_b * np.power(cfreq, 2)
            + cf.c_c
            + cf.c_d * temp.values
        ) / (10 * (1 + eps * p1))
    elif cf.c_coefs == "G":
        # C = (C_G +(C_H +(C_I + C_J*f)*f)*f*f) / (10.*(1+C_TCOR*t+C_PCOR*p)) ;
        calibrated_conductivity = (
            cf.c_g + (cf.c_h + (cf.c_i + cf.c_j * cfreq) * cfreq) * np.power(cfreq, 2)
        ) / (10 * (1 + cf.c_tcor * temp.values + cf.c_pcor * p1))
    else:
        raise ValueError(f"Unknown c_coefs: {cf.c_coefs}")

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
    # %%----------------------------------
    # %% Oxygen saturation: f(T,S); ml/l
    # %%----------------------------------
    # TK = 273.15+T;  % degrees Kelvin
    # A1 = -173.4292; A2 = 249.6339; A3 = 143.3483; A4 = -21.8492;
    # B1 = -0.033096; B2 = 0.014259; B3 =  -0.00170;
    # OXSAT = exp(A1 + A2*(100./TK) + A3*log(TK/100) + A4*(TK/100) + [S .* (B1 + B2*(TK/100) + (B3*(TK/100).*(TK/100)))] );
    tk = 273.15 + temperature  # degrees Kelvin
    a1 = -173.4292
    a2 = 249.6339
    a3 = 143.3483
    a4 = -21.8492
    b1 = -0.033096
    b2 = 0.014259
    b3 = -0.00170
    oxsat = np.exp(
        a1
        + a2 * (100 / tk)
        + a3 * np.log(tk / 100)
        + a4 * (tk / 100)
        + np.multiply(
            salinity, b1 + b2 * (tk / 100) + np.multiply(b3 * (tk / 100), (tk / 100))
        )
    )
    return oxsat


def _calibrated_O2_from_volts(combined_nc, cf, nc, var_name, temperature, salinity):
    # Contents of doradosdp's calc_O2_SBE43.m:
    # ----------------------------------------
    # function [O2] = calc_O2_SBE43(O2V,T,S,P,O2cal,time,units);
    # %% To calculate Oxygen from sbe voltage
    # %% Reference: W.B. Owens and R.C. Millard, 1985. A new algorithm for CTD oxygen
    # %%  calibration, J. Phys. Oceanogr. 15:621-631.
    # %%  Also, described in SeaBird application note.
    # pltit = 'n';
    # % disp(['   Pressure should be in dB']);
    f_interp = interp1d(
        combined_nc["depth_time"].values.tolist(),
        combined_nc["depth_filtpres"].values,
        fill_value="extrapolate",
    )
    pressure = f_interp(nc["time"].values.tolist())

    #
    # %%----------------------------------
    # %% Oxygen voltage
    # %%----------------------------------
    # % disp(['   Minimum of oxygen voltage ' num2str(min(O2V)) ' V']);
    # % disp(['   Maximum of oxygen voltage ' num2str(max(O2V)) ' V']);
    # % disp(['   Mean of oxygen voltage ' num2str(mean(O2V)) ' V']);
    # docdt = [NaN;[diff(O2V)./diff(time)]];  % slope of oxygen current (uA/sec);
    docdt = np.append(
        np.nan,
        np.divide(
            np.diff(nc[var_name]), np.diff(nc["time"].astype(np.int64).values / 1e9)
        ),
    )

    oxsat = _oxsat(temperature, salinity)

    #
    # %%----------------------------------
    # %% Oxygen concentration (mL/L)
    # %%----------------------------------
    # %% Constants
    # tau=0;
    #
    # O2 = [O2cal.SOc * ((O2V+O2cal.offset)+(tau*docdt)) + O2cal.BOc * exp(-0.03*T)].*exp(O2cal.Tcor*T + O2cal.Pcor*P).*OXSAT;
    tau = 0.0
    o2_mll = np.multiply(
        cf.SOc * ((nc[var_name].values + cf.Voff) + (tau * docdt))
        + cf.BOc * np.exp(-0.03 * temperature.values),
        np.multiply(
            np.exp(cf.TCor * temperature.values + cf.PCor * pressure), oxsat.values
        ),
    )

    #
    # if strcmp(units,'umolkg')==1
    # %%----------------------------------
    # %% Convert to umol/kg
    # %%----------------------------------
    # %% SeaBird equations are for ml/l computations
    # %%  Can convert OXSAT at atmospheric pressure to mg/l by 1.4276
    # %%  Convert dissolved O2 to mg/l using density of oxygen = 1.4276 kg/m^3
    # dens=sw_dens(S,T,P);
    # O2 = (O2 * 1.4276) .* (1e6./(dens*32));
    dens = eos80.dens(salinity.values, temperature.values, pressure)
    o2_umolkg = np.multiply(o2_mll * 1.4276, (1.0e6 / (dens * 32)))

    return o2_mll, o2_umolkg
