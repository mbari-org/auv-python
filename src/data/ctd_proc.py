import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from seawater import eos80

def _calibrated_temp_from_frequency(cf, nc):
    # From processCTD.m:
    # TC = 1./(t_a + t_b*(log(t_f0./temp_frequency)) + t_c*((log(t_f0./temp_frequency)).^2) + t_d*((log(t_f0./temp_frequency)).^3)) - 273.15;
    K2C = 273.15
    calibrated_temp = (1.0 / 
            (cf.t_a + 
                cf.t_b * np.log(cf.t_f0 / nc['temp_frequency'].values) + 
                cf.t_c * np.power(np.log(cf.t_f0 / nc['temp_frequency']),2) + 
                cf.t_d * np.power(np.log(cf.t_f0 / nc['temp_frequency']),3)
            ) - K2C)

    return calibrated_temp

def _calibrated_sal_from_cond_frequency(args, combined_nc, logger, cf, nc, temp, depth):
    # Comments carried over from doradosdp's processCTD.m:
    # Note that recalculation of conductivity and correction for thermal mass
    # are possible, however, their magnitude results in salinity differences
    # of less than 10^-4.  
    # In other regions where these corrections are more significant, the
    # corrections can be turned on.
    # conductivity at S=35 psu , T=15 C [ITPS 68] and P=0 db) ==> 42.914
    sw_c3515 = 42.914
    eps = np.spacing(1)

    f_interp = interp1d(combined_nc['depth_time'].values.tolist(), 
                        combined_nc['depth_filtpres'].values,
                        fill_value="extrapolate")
    p1 = f_interp(nc['time'].values.tolist())
    if args.plot:
        pbeg = 0
        pend = len(combined_nc['depth_time'])
        if args.plot.startswith('first'):
            pend = int(args.plot.split('first')[1])
        plt.figure(figsize=(18,6))
        plt.plot(combined_nc['depth_time'][pbeg:pend],
                    combined_nc['depth_filtpres'][pbeg:pend], ':o',
                    nc['time'][pbeg:pend], p1[pbeg:pend], 'o')
        plt.legend(('Pressure from parosci', 'Interpolated to ctd time'))
        title = "Comparing Interpolation of Pressure to CTD Time"
        title += f" - First {pend} Points"
        plt.title(title)
        plt.grid()
        logger.debug(f"Pausing with plot entitled: {title}."
                            " Close window to continue.")
        plt.show()

    # %% Conductivity Calculation
    # cfreq=cond_frequency/1000;
    # c1 = (c_a*(cfreq.^c_m)+c_b*(cfreq.^2)+c_c+c_d*TC)./(10*(1+eps*p1));            
    cfreq = nc['cond_frequency'].values / 1000.0
    c1 = (cf.c_a * np.power(cfreq, cf.c_m) +
            cf.c_b * np.power(cfreq, 2) +
            cf.c_c + 
            cf.c_d * temp.values) / (10 * (1 + eps * p1))

    # % Calculate Salinty
    # cratio = c1*10/sw_c3515; % sw_C is conductivity value at 35,15,0
    # CTD.salinity = sw_salt(cratio,CTD.temperature,p1); % (psu)
    cratio = c1 * 10 / sw_c3515
    calibrated_salinity = eos80.salt(cratio, temp, p1)

    return calibrated_salinity
