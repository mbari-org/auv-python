# noqa: INP001

# Module for providing mission specific information for the Dorado missions.
# What has been added as comments in reprocess_surveys.m is now here in
# the form of a dictionary.  This dictionary is then used in the workflow
# processing code to add appropriate metadata to the data products.
#
# Align attributes with ACDD 1.3 standard
# http://wiki.esipfed.org/index.php/Attribute_Convention_for_Data_Discovery_1-3

# Attempt to use a controlled vocabulary for the program names


OCCO = "OCCO"
BIOLUME = "BIOLUME"
LOCO = "LOCO"
AUVCTD = "AUVCTD"
AOSN = "AOSN"
DIAMOND = "Monterey Bay Diamond"
CANON = "CANON"
CANONSEP2010 = "CANON September 2010"
CANONOCT2010 = "CANON October 2010"
CANONAPR2011 = "CANON April 2011"
CANONJUN2011 = "CANON June 2011"
CANONSEP2011 = "CANON September 2011"
CANONMAY2012 = "CANON May 2012"
CANONSEP2012 = "CANON September 2012"
CANONMAR2013 = "CANON March 2013"
CANONSEP2013 = "CANON September 2013"
CANONAPR2014 = "CANON April 2014"
CANONSEP2014 = "CANON September 2014"
CANONMAY2015 = "CANON May 2015"
CANONSEP2015 = "CANON September 2015"
CANONOS2016 = "CANON Off Season 2016"
CANONSEP2016 = "CANON September 2016"
CANONAPR2017 = "CANON April 2017"
CANONPS2017 = "CANON Post Season 2017"
CANONSEP2017 = "CANON September 2017"
CANONMAY2018 = "CANON May 2018"
CANONSEP2018 = "CANON September 2018"
CANONMAY2019 = "CANON May 2019"
CANONFALL2019 = "CANON Fall 2019"
CANONJUL2020 = "CANON July 2020"
CANONOCT2020 = "CANON October 2020"
CANONAPR2021 = "CANON April 2021"
CANONOCT2022 = "CANON October 2022"
MBTSLINE = "MBTS Line"
ALLSALINITYBAD = "Nearly all salinity is bad"  # REMOVE color in Google sheet
SIMZAUG2013 = "SIMZ August 2013"
SIMZOCT2013 = "SIMZ October 2013"
SIMZSPRING2014 = "SIMZ Spring 2014"
SIMZJUL2014 = "SIMZ July 2014"
SIMZOCT2014 = "SIMZ October 2014"

# TEST and FAILED are recognized by process.py to not be processed
TEST = "TEST"
FAILED = "FAILED"

# ----------------------------- 2003 ---------------------------------------
dorado_info = {}
for mission_number in range(2, 14):
    dorado_info[f"2003.106.{mission_number:02d}"] = {
        "program": OCCO,
        "comment": (
            "OCCO2003106: missions {02,03,04,05,06,07,08,09,10,11,12,13}"
            " - Very first Dorado missions in the archive - does not process with legacy Matlab code"
            " - ctdToUse = ctd1 "
        ),
    }
for mission_number in range(3, 8):
    dorado_info[f"2003.111.{mission_number:02d}"] = {
        "program": OCCO,
        "comment": (
            "OCCO2003111: missions {03,04,05,06,07}"
            " - Second set of Dorado missions in the archive - does not process with legacy Matlab code"
            " - ctdToUse = ctd1 "
        ),
    }
for mission_number in range(2, 4):
    dorado_info[f"2003.115.{mission_number:02d}"] = {
        "program": OCCO,
        "comment": (
            "OCCO2003115: missions {02,03}"
            " - Missing log files - are they on an AUVCTD project computer? "
            " - (JR found them on old RH laptop & copied to my tempbox - I'll put them in place)"
            " - does not process with legacy Matlab code"
            " - ctdToUse = ctd1 "
        ),
    }
for mission_number in range(2, 5):
    dorado_info[f"2003.118.{mission_number:02d}"] = {
        "program": OCCO,
        "comment": (
            "OCCO2003118: missions {02,03,04}"
            " - 2003.118 Hydroscat cal changes - does not process with legacy Matlab code"
            " - ctdToUse = ctd1 "
        ),
    }
# ------------ Lagacy Matlab processing works for these missions on... ---------------
dorado_info["2003.125.02"] = {
    "program": AUVCTD,
    "comment": (
        "Requested by Patrick McEnaney 5/20/05 (deviceIDs copied from later mission)"
        " - Needed to copy ctdDriver.cfg and ctdDriver2.cfg from 2003.207.07 too"
        " - calibrations may NOT be right"
        " - ctdToUse = ctd1 "
    ),
}
for mission_number in range(2, 5):
    dorado_info[f"2003.164.{mission_number:02d}"] = {
        "program": OCCO,
        "comment": ("OCCO2003164: missions {02,03,04} - ctdToUse = ctd1 "),
    }
for mission_number in range(2, 5):
    dorado_info[f"2003.167.{mission_number:02d}"] = {
        "program": OCCO,
        "comment": ("OCCO2003167: missions {02,03,04} - ctdToUse = ctd1 "),
    }
for mission_number in range(4, 7):
    dorado_info[f"2003.169.{mission_number:02d}"] = {
        "program": OCCO,
        "comment": ("OCCO2003169: missions {04,05,06} - ctdToUse = ctd1 "),
    }
for mission_number in range(3, 6):
    dorado_info[f"2003.174.{mission_number:02d}"] = {
        "program": OCCO,
        "comment": ("OCCO2003174: missions {03,04,05} - ctdToUse = ctd1 "),
    }
for mission_number in range(2, 5):
    dorado_info[f"2003.176.{mission_number:02d}"] = {
        "program": OCCO,
        "comment": ("OCCO2003176: missions {02,03,04} - ctdToUse = ctd1 "),
    }
dorado_info["2003.177.02"] = {
    "program": AUVCTD,
    "comment": (
        "Portaled files exist in archive, but not processed by reprocess_surveys.m"
        " - ctdToUse = ctd1 "
    ),
}
for mission_number in range(2, 4):
    dorado_info[f"2003.178.{mission_number:02d}"] = {
        "program": AUVCTD,
        "comment": (
            "Portaled files exist in archive, but not processed by reprocess_surveys.m"
            " - ctdToUse = ctd1 "
        ),
    }
dorado_info["2003.191.00"] = {
    "program": AUVCTD,
    "comment": (
        "Portaled files exist in archive, but not processed by reprocess_surveys.m"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2003.199.03"] = {
    "program": AUVCTD,
    "comment": (
        "Portaled files exist in archive, but not processed by reprocess_surveys.m"
        " - ctdToUse = ctd1 "
    ),
}
for mission_number in (7, 8, 9, 11, 13):
    dorado_info[f"2003.205.{mission_number:02d}"] = {
        "program": AUVCTD,
        "comment": (
            "Portaled files exist in archive, but not processed by reprocess_surveys.m"
            " - ctdToUse = ctd1 "
        ),
    }
for mission_number in range(2, 5):
    dorado_info[f"2003.206.{mission_number:02d}"] = {
        "program": AUVCTD,
        "comment": (
            "Portaled files exist in archive, but not processed by reprocess_surveys.m"
            " - ctdToUse = ctd1 "
        ),
    }
dorado_info["2003.216.03"] = {
    "program": AUVCTD,
    "comment": (
        "Portaled files exist in archive, but not processed by reprocess_surveys.m"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2003.218.04"] = {
    "program": AUVCTD,
    "comment": (
        "Portaled files exist in archive, but not processed by reprocess_surveys.m"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2003.220.11"] = {
    "program": AUVCTD,
    "comment": (
        "Portaled files exist in archive, but not processed by reprocess_surveys.m"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2003.222.02"] = {
    "program": AUVCTD,
    "comment": (
        "Portaled to SSDS on 12 March 2014, but did not process for missing hydroscat cal file"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2003.223.02"] = {
    "program": AUVCTD,
    "comment": (
        "Re-portaled 7/1/05, added oxygen coefficients to ctdDriver2.cfg on 11 March 2015 - no change in oxygen values"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2003.224.05"] = {
    "program": BIOLUME,
    "comment": ("Re-portaled 7/1/05: 2 separate surveys close to the beach - ctdToUse = ctd1 "),
}
dorado_info["2003.224.08"] = {
    "program": AUVCTD,
    "comment": ("Re-portaled 7/1/05: 2 separate surveys close to the beach - ctdToUse = ctd1 "),
}
dorado_info["2003.225.02"] = {
    "program": BIOLUME,
    "comment": ("Re-portaled 7/7/05 - ctdToUse = ctd1 "),
}
dorado_info["2003.226.02"] = {
    "program": BIOLUME,
    "comment": ("Re-portaled 7/7/05 - ctdToUse = ctd1 "),
}
dorado_info["2003.226.05"] = {
    "program": AUVCTD,
    "comment": ("Re-portaled 7/7/05 - ctdToUse = ctd1 "),
}
dorado_info["2003.227.02"] = {
    "program": BIOLUME,
    "comment": ("Re-portaled 7/7/05 - ctdToUse = ctd1 "),
}
dorado_info["2003.228.02"] = {
    "program": BIOLUME,
    "comment": ("Re-portaled 7/7/05 - ctdToUse = ctd1 "),
}
dorado_info["2003.228.05"] = {
    "program": AOSN,
    "comment": ("Re-portaled 7/7/05 - ctdToUse = ctd1 "),
}
for mission_number in range(2, 6):
    dorado_info[f"2003.230.{mission_number:02d}"] = {
        "program": AUVCTD,
        "comment": (
            "Re-portaled 7/7/05: maybe these 4 missions are one survey ? - ctdToUse = ctd1 "
        ),
    }
dorado_info["2003.232.02"] = {
    "program": AUVCTD,
    "comment": ("Re-portaled 7/8/05 - ctdToUse = ctd1 "),
}
for mission_number in range(2, 7):
    dorado_info[f"2003.233.{mission_number:02d}"] = {
        "program": OCCO,
        "comment": ("Re-portaled 7/8/05.  Increased tDiffCrit to 800 on 8/3/06 - ctdToUse = ctd1 "),
    }
dorado_info["2003.234.02"] = {
    "program": AUVCTD,
    "comment": ("Re-portaled 7/8/05 - ctdToUse = ctd1 "),
}
for mission_number in range(3, 5):
    dorado_info[f"2003.237.{mission_number:02d}"] = {
        "program": OCCO,
        "comment": ("OCCO2003237: missions {03,04}: Re-portaled 7/11/05 - ctdToUse = ctd1 "),
    }
dorado_info["2003.238.02"] = {
    "program": AUVCTD,
    "comment": ("Re-portaled 7/11/05 - ctdToUse = ctd1 "),
}
dorado_info["2003.241.02"] = {
    "program": AUVCTD,
    "comment": ("Re-portaled 7/11/05 - ctdToUse = ctd1 "),
}
for mission_number in range(2, 5):
    dorado_info[f"2003.245.{mission_number:02d}"] = {
        "program": OCCO,
        "comment": ("OCCO2003245: missions {02,03,04}: Re-portaled 7/11/05 - ctdToUse = ctd1 "),
    }
dorado_info["2003.246.02"] = {
    "program": OCCO,
    "comment": ("OCCO2003246: missions {02} - ctdToUse = ctd1 "),
}
for mission_number in range(2, 4):
    dorado_info[f"2003.280.{mission_number:02d}"] = {
        "program": AUVCTD,
        "comment": ("Portaled 7/11/05 - ctdToUse = ctd1 "),
    }
dorado_info["2003.281.02"] = {
    "program": AUVCTD,
    "comment": ("Portaled 7/11/05 - ctdToUse = ctd1 "),
}
for mission_number in (2, 4):
    dorado_info[f"2003.308.{mission_number:02d}"] = {
        "program": AUVCTD,
        "comment": ("Portaled 7/12/05 - ctdToUse = ctd1 "),
    }
for mission_number in [0, 1, 2, 4]:
    dorado_info[f"2003.309.{mission_number:02d}"] = {
        "program": AUVCTD,
        "comment": ("Portaled 7/12/05 - ctdToUse = ctd1 "),
    }
for mission_number in range(2, 5):
    dorado_info[f"2003.317.{mission_number:02d}"] = {
        "program": "DMOSeaTrials",
        "comment": ("Portaled 7/12/05 - ctdToUse = ctd1 "),
    }
dorado_info["2003.336.02"] = {
    "program": BIOLUME,
    "comment": ("Portaled 7/12/05 - ctdToUse = ctd1 "),
}
dorado_info["2003.337.04"] = {
    "program": BIOLUME,
    "comment": ("Portaled 7/12/05 - ctdToUse = ctd1 "),
}
for mission_number in range(15, 23):
    dorado_info[f"2003.338.{mission_number:02d}"] = {
        "program": BIOLUME,
        "comment": ("Portaled 7/12/05 - ctdToUse = ctd1 "),
    }
dorado_info["2003.339.04"] = {
    "program": BIOLUME,
    "comment": ("Portaled 7/12/05 - ctdToUse = ctd1 "),
}
dorado_info["2003.340.02"] = {
    "program": BIOLUME,
    "comment": ("Portaled 7/12/05 - ctdToUse = ctd1 "),
}

# ----------------------------- 2004 ---------------------------------------
dorado_info["2004.028.05"] = {
    "program": AUVCTD,
    "comment": (
        "AUVCTD: Processed 10/4/04 - no metadata loaded anywhere, re-portaled & processed 7/26/05"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2004.029.03"] = {
    "program": AUVCTD,
    "comment": (
        "2004.029.03: parosci interp1 problem - multigenerateNetcdf failed to create navigation.nc file because of '#' problem"
        " # begin"
        " #<binary_data>"
        " Will need to rerun the portal with the fix to set byteOffset."
        " Re-portaled & processed 7/26/05"
        " According to GPS data AUV is in Mexico beginning: 29 Jan 2004 19:37:36 GMT	1.0754050561900759E9	0.4733113223544617"
        " *** Need to reprocess with just the .03 mission and not include the data equal to and after this time ***"
        " Had bad nav point - processed 8/10/05"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2004.029.05"] = {
    "program": AUVCTD,
    "comment": dorado_info["2004.029.03"]["comment"],
}
dorado_info["2004.035.05"] = {
    "program": AUVCTD,
    "comment": ("Portaled 8/9/05 - ctdToUse = ctd1 "),
}
dorado_info["2004.069.03"] = {
    "program": AUVCTD,
    "comment": (
        "Isus data (nitrate) is missing for 069.03 - re-portaled 8/9/05 - ctdToUse = ctd1 "
    ),
}
dorado_info["2004.094.04"] = {
    "program": BIOLUME,
    "comment": (
        "Something whaky with lat lon calculations start at (237.77744, 36.94086) and end at (255.1847, 53.14181)"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2004.095.02"] = {
    "program": BIOLUME,
    "comment": ("Missing netCDF files, cant process yet - re-portaled 8/10/05 - ctdToUse = ctd1 "),
}
for mission_number in (3, 5, 6):
    dorado_info[f"2004.096.{mission_number:02d}"] = {
        "program": BIOLUME,
        "comment": ("Portaled 8/10/05 - ctdToUse = ctd1 "),
    }
dorado_info["2004.111.04"] = {
    "program": AUVCTD,
    "comment": ("Portaled 8/11/05 - ctdToUse = ctd1 "),
}
dorado_info["2004.112.02"] = {
    "program": AUVCTD,
    "comment": ("Portaled 8/11/05 - ctdToUse = ctd1 "),
}
dorado_info["2004.167.04"] = {
    "program": AUVCTD,
    "comment": (
        "??? Reference to non-existent field 'depth'."
        " Error in ==> /u/ssdsadmin/dev/auv_ctd/src/matlab/doradosdp/process/build_auv_survey.m"
        " On line 118  ==>                hs2depth=[hs2depth;HS2.depth];"
        " No variables in the hydroscatlog.nc file, hydroscat.log file is 0 size"
        " Will need to get those data"
        " Portaled 8/11/05 - no hs2 data"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2004.168.01"] = {
    "program": AUVCTD,
    "comment": (" - ctdToUse = ctd1 "),
}
dorado_info["2004.168.03"] = {
    "program": AUVCTD,
    "comment": ("Portaled 8/11/05 - ctdToUse = ctd1 "),
}
for mission_number in range(1, 4):
    dorado_info[f"2004.169.{mission_number:02d}"] = {
        "program": BIOLUME,
        "comment": ("SDP 10/6/04 - re-portaled 8/11/05 - ctdToUse = ctd1 "),
    }
for mission_number in range(1, 4):
    dorado_info[f"2004.170.{mission_number:02d}"] = {
        "program": AUVCTD,
        "comment": ("SDP 10/6/04 - re-portaled 8/11/05 - ctdToUse = ctd1 "),
    }
for mission_number in range(2, 4):
    dorado_info[f"2004.196.{mission_number:02d}"] = {
        "program": AUVCTD,
        "comment": ("re-portaled 8/11/05 - ctdToUse = ctd1 "),
    }
for mission_number in range(1, 3):
    dorado_info[f"2004.197.{mission_number:02d}"] = {
        "program": AUVCTD,
        "comment": (
            "Not processed 11/1/04 - Re-portaled 8/11/05"
            "hydroscatlog.log is 0 size."
            " - ctdToUse = ctd1 "
        ),
    }
dorado_info["2004.208.01"] = {
    "program": AUVCTD,
    "comment": ("Not processed 11/1/04 - Re-portaled 8/12/05 - ctdToUse = ctd1 "),
}
dorado_info["2004.209.03"] = {
    "program": AUVCTD,
    "comment": ("Not processed 11/1/04 - Re-portaled 8/12/05 - ctdToUse = ctd1 "),
}
dorado_info["2004.211.00"] = {
    "program": BIOLUME,
    "comment": ("Re-portaled 8/12/05 - ctdToUse = ctd1 "),
}
dorado_info["2004.211.01"] = {
    "program": BIOLUME,
    "comment": ("Re-portaled 8/12/05 - ctdToUse = ctd1 "),
}
dorado_info["2004.233.01"] = {
    "program": OCCO,
    "comment": ("Re-portaled 8/16/05 - ctdToUse = ctd1 "),
}
dorado_info["2004.236.00"] = {
    "program": OCCO,
    "comment": ("Re-portaled 8/16/05 - ctdToUse = ctd1 "),
}
dorado_info["2004.237.00"] = {
    "program": OCCO,
    "comment": ("Re-portaled 8/16/05 - ctdToUse = ctd1 "),
}
dorado_info["2004.238.00"] = {
    "program": OCCO,
    "comment": ("Re-portaled 8/16/05 - ctdToUse = ctd1 "),
}
dorado_info["2004.239.01"] = {
    "program": OCCO,
    "comment": ("Re-portaled 8/16/05 - ctdToUse = ctd1 "),
}
dorado_info["2004.240.00"] = {
    "program": OCCO,
    "comment": ("Re-portaled 8/18/05 - ctdToUse = ctd1 "),
}
dorado_info["2004.251.00"] = {
    "program": AUVCTD,
    "comment": (
        "Added 20 December 2004 in response to Carole's email question"
        " Fails in plotting because of missing O2 (?)"
        " Re-portaled 8/18/05"
        " Added 20 Dec 2004 in response to Carole's email question. Fails in plotting because of missing O2 (?)"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2004.252.00"] = {
    "program": AUVCTD,
    "comment": ("Re-portaled 8/18/05 - ctdToUse = ctd1 "),
}
dorado_info["2004.254.00"] = {
    "program": BIOLUME,
    "comment": (
        "Problem with navigation.nc time array - all bogus because of '#' as first byte of data - fixed."
        " Re-portaled 8/18/05"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2004.261.00"] = {
    "program": OCCO,
    "comment": (
        "Mistake made at initial portal run, should be OCCO, not LOCO"
        " Reportaled 3/9/05 with OCCO"
        " Re-portaled 8/18/05"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2004.271.00"] = {
    "program": OCCO,
    "comment": (
        "Must have missed this one, added 8 March 2005 w/ recacled isuslog"
        " Re-portaled 8/18/05"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2004.272.00"] = {
    "program": AUVCTD,
    "comment": ("SDP on 29 Sep 2004 Re-portaled 8/18/05 - ctdToUse = ctd1 "),
}
dorado_info["2004.272.01"] = {
    "program": AUVCTD,
    "comment": ("SDP on 29 Sep 2004 Re-portaled 8/18/05 - ctdToUse = ctd1 "),
}
for mission_number in range(2):
    dorado_info[f"2004.273.{mission_number:02d}"] = {
        "program": AUVCTD,
        "comment": ("Added 10 March 2005 Re-portaled 8/19/05 - ctdToUse = ctd1 "),
    }
dorado_info["2004.274.00"] = {
    "program": AUVCTD,
    "comment": ("Added 11 March 2005 Re-portaled 8/19/05 - ctdToUse = ctd1 "),
}
dorado_info["2004.275.00"] = {
    "program": AUVCTD,
    "comment": ("Added 11 March 2005 Re-portaled 8/19/05 - ctdToUse = ctd1 "),
}
dorado_info["2004.295.03"] = {
    "program": AUVCTD,
    "comment": ("SDP on 1 Nov 2004 Re-portaled 8/19/05 - ctdToUse = ctd1 "),
}
for mission_number in range(3, 5):
    dorado_info[f"2004.296.{mission_number:02d}"] = {
        "program": BIOLUME,
        "comment": (
            "Hardware failure, no ctdDriver2 data.  Flow was bad for both CTDs."
            " Re-portaled 8/19/05"
            " Failed because no ctdDriver2.cfg for 296.03."
            " - ctdToUse = ctd1 "
        ),
    }
dorado_info["2004.314.00"] = {
    "program": AUVCTD,
    "comment": ("Portaled 8/19/05 - ctdToUse = ctd1 "),
}
for mission_number in range(3):
    dorado_info[f"2004.315.{mission_number:02d}"] = {
        "program": AUVCTD,
        "comment": (
            "Processed fine on prey, a volume survey Re-portaled 8/21/05 - ctdToUse = ctd1 "
        ),
    }
for mission_number in range(3):
    dorado_info[f"2004.317.{mission_number:02d}"] = {
        "program": BIOLUME,
        "comment": ("Re-portaled 8/21/05 - ctdToUse = ctd1 "),
    }
dorado_info["2004.321.00"] = {
    "program": BIOLUME,
    "comment": ("Re-portaled 8/22/05 - ctdToUse = ctd1 "),
}
for mission_number in range(5, 8):
    dorado_info[f"2004.344.{mission_number:02d}"] = {
        "program": OCCO,
        "comment": ("Re-portaled 8/22/05 - ctdToUse = ctd1 "),
    }
for mission_number in range(10):
    dorado_info[f"2004.345.{mission_number:02d}"] = {
        "program": OCCO,
        "comment": ("Re-portaled 8/22/05 - ctdToUse = ctd1 "),
    }
dorado_info["2004.348.00"] = {
    "program": BIOLUME,
    "comment": ("Re-portaled 8/22/05 - ctdToUse = ctd1 "),
}
dorado_info["2004.349.00"] = {
    "program": BIOLUME,
    "comment": ("Re-portaled 8/22/05 - ctdToUse = ctd1 "),
}
for mission_number in range(2):
    dorado_info[f"2004.352.{mission_number:02d}"] = {
        "program": BIOLUME,
        "comment": ("Re-portaled 8/22/05 - ctdToUse = ctd1 "),
    }

# ---------------------------- 2005 ----------------------------
dorado_info["2005.006.21"] = {
    "program": OCCO,
    "comment": ("One mission on Elkhorn tidal plume race track - ctdToUse = ctd1 "),
}
dorado_info["2005.010.00"] = {
    "program": OCCO,
    "comment": ("OFirst one portaled to predator with new directory structure - ctdToUse = ctd1 "),
}
dorado_info["2005.012.00"] = {
    "program": OCCO,
    "comment": (
        "Noticed that log files aren't being read through completion by MultgenerateNetcdf"
        " - ctdToUse = ctd1 "
    ),
}
for mission_number in range(5):
    dorado_info[f"2005.014.{mission_number:02d}"] = {
        "program": BIOLUME,
        "comment": ("BIOLUME night missions, touched bottom on mission .00 - ctdToUse = ctd1 "),
    }
for mission_number in (0, 2, 3, 4, 5):
    dorado_info[f"2005.020.{mission_number:02d}"] = {
        "program": BIOLUME,
        "comment": (
            "Elkhorn Slough plume racetrack Monterey triangle (missing short mission 01)"
            " - ctdToUse = ctd1 "
        ),
    }
for mission_number in range(5):
    dorado_info[f"2005.021.{mission_number:02d}"] = {
        "program": OCCO,
        "comment": ("Elkhorn Slough plume racetrack Monterey triangle - ctdToUse = ctd1 "),
    }
for mission_number in range(18):
    dorado_info[f"2005.027.{mission_number:02d}"] = {
        "program": BIOLUME,
        "comment": (
            "Depth abort problems - lots of missions for this biolume survey - ctdToUse = ctd1 "
        ),
    }
dorado_info["2005.061.00"] = {
    "program": FAILED,
    "comment": (
        "Portaled files exist in archive, but not processed by reprocess_surveys.m."
        " Downcast to 960 m, then failed. No GPS fix upon surfacing so nav is bad."
    ),
}
for mission_number in range(7, 9):
    dorado_info[f"2005.096.{mission_number:02d}"] = {
        "program": AUVCTD,
        "comment": (
            "M1.5 to C1.5.for Dr. Chavez - first missions since drop weight installed"
            " - ctdToUse = ctd1 "
        ),
    }
for mission_number in range(1, 3):
    dorado_info[f"2005.103.{mission_number:02d}"] = {
        "program": BIOLUME,
        "comment": ("Night-time survey that hit mud near Pt. Pinos. - ctdToUse = ctd1 "),
    }
dorado_info["2005.118.05"] = {
    "program": AUVCTD,
    "comment": ("M1.5 to M1 Hans mentioned goofy data from hydroscat on this - ctdToUse = ctd1 "),
}
for mission_number in (16, 19):
    dorado_info[f"2005.119.{mission_number:02d}"] = {
        "program": AUVCTD,
        "comment": ("ctdToUse = ctd1"),
    }
dorado_info["2005.122.00"] = {
    "program": AUVCTD,
    "comment": ("ctdToUse = ctd1"),
}
dorado_info["2005.123.01"] = {
    "program": AUVCTD,
    "comment": ("ctdToUse = ctd1"),
}
dorado_info["2005.125.00"] = {
    "program": AUVCTD,
    "comment": ("ctdToUse = ctd1"),
}
dorado_info["2005.133.00"] = {
    "program": BIOLUME,
    "comment": ("ctdToUse = ctd1"),
}
dorado_info["2005.137.00"] = {
    "program": BIOLUME,
    "comment": ("ctdToUse = ctd1"),
}
dorado_info["2005.140.00"] = {
    "program": BIOLUME,
    "comment": ("ctdToUse = ctd1"),
}
dorado_info["2005.159.00"] = {
    "program": AUVCTD,
    "comment": ("ctdToUse = ctd1"),
}
for mission_number in range(3):
    dorado_info[f"2005.161.{mission_number:02d}"] = {
        "program": BIOLUME,
        "comment": (
            "First mission (161.00) failed to make netCDFs by SSDS (??) - ctdToUse = ctd1 "
        ),
    }
dorado_info["2005.175.00"] = {
    "program": BIOLUME,
    "comment": (
        "dynamicControl log data has device info.  isusLog.cfg has information now."
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2005.179.01"] = {
    "program": AUVCTD,
    "comment": ("Portaled 7/8/05 - ctdToUse = ctd1 "),
}
dorado_info["2005.180.00"] = {
    "program": LOCO,
    "comment": (
        "Portaled 7/8/05: 4 missions after this aborted then had bad GPS, on;y this one portaled over."
        " - ctdToUse = ctd1 "
    ),
}
for mission_number in range(3):
    dorado_info[f"2005.182.{mission_number:02d}"] = {
        "program": BIOLUME,
        "comment": ("Portaled 7/8/05 - ctdToUse = ctd1 "),
    }
dorado_info["2005.194.00"] = {
    "program": AUVCTD,
    "comment": ("Portaled 7/26/05 - ctdToUse = ctd1 "),
}
dorado_info["2005.196.00"] = {
    "program": BIOLUME,
    "comment": (
        "Portaled 7/26/05:  196.01 had battery failure - do not include in survey"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2005.196.01"] = {
    "program": FAILED,
    "comment": ("Portaled 7/26/05:  196.01 had battery failure - do not include in survey"),
}
dorado_info["2005.201.00"] = {
    "program": AUVCTD,
    "comment": ("Portaled 7/26/05 - ctdToUse = ctd1 "),
}
for mission_number in range(1, 3):
    dorado_info[f"2005.203.{mission_number:02d}"] = {
        "program": BIOLUME,
        "comment": ("Portaled 7/26/05 - ctdToUse = ctd1 "),
    }
dorado_info["2005.217.00"] = {
    "program": BIOLUME,
    "comment": ("Portaled 7/26/05 - ctdToUse = ctd1 "),
}
for mission_number in range(2):
    dorado_info[f"2005.222.{mission_number:02d}"] = {
        "program": AUVCTD,
        "comment": ("Portaled 8/30/05 - ctdToUse = ctd1 "),
    }
dorado_info["2005.224.00"] = {
    "program": BIOLUME,
    "comment": ("Portaled 8/16/05 - ctdToUse = ctd1 "),
}
for day, mission_number in [
    (230, 0),
    (237, 0),
    (237, 2),
    (238, 0),
    (238, 1),
    (238, 2),
    (239, 0),
    (239, 1),
    (239, 2),
    (240, 0),
    (240, 1),
    (240, 2),
    (241, 2),
    (241, 3),
    (241, 5),
    (242, 0),
    (242, 1),
    (243, 0),
    (243, 2),
    (243, 3),
    (244, 1),
    (244, 2),
]:
    dorado_info[f"2005.{day:03d}.{mission_number:02d}"] = {
        "program": LOCO,
        "comment": ("Conducted from the Sproul on John Ryan's LOCO project - ctdToUse = ctd1 "),
    }
dorado_info["2005.249.00"] = {
    "program": LOCO,
    "comment": ("Portaled by DMO 9/6/05 - ctdToUse = ctd1 "),
}
dorado_info["2005.250.00"] = {
    "program": LOCO,
    "comment": ("Portaled by DMO 9/7/05 - ctdToUse = ctd1 "),
}
dorado_info["2005.257.00"] = {
    "program": AUVCTD,
    "comment": ("Portaled by DMO 9/14/05 - ctdToUse = ctd1 "),
}
dorado_info["2005.259.00"] = {
    "program": BIOLUME,
    "comment": ("Portaled by DMO 9/16/05 - ctdToUse = ctd1 "),
}
dorado_info["2005.299.12"] = {
    "program": AUVCTD,
    "comment": ("Portaled by DMO 10/26/05 - ctdToUse = ctd1 "),
}
dorado_info["2005.301.00"] = {
    "program": BIOLUME,
    "comment": ("Portaled by DMO 10/28/05 - ctdToUse = ctd1 "),
}
dorado_info["2005.306.00"] = {
    "program": AUVCTD,
    "comment": ("Portaled by DMO 10/28/05 - ctdToUse = ctd1 "),
}
for mission_number in range(2):
    dorado_info[f"2005.308.{mission_number:02d}"] = {
        "program": BIOLUME,
        "comment": ("Portaled by DMO 10/28/05 - ctdToUse = ctd1 "),
    }
dorado_info["2005.324.03"] = {
    "program": AUVCTD,
    "comment": ("Portaled by DMO 11/22/05 - ctdToUse = ctd1 "),
}
for mission_number in range(5):
    dorado_info[f"2005.332.{mission_number:02d}"] = {
        "program": AUVCTD,
        "comment": ("Portaled by DMO 11/28/05 - ctdToUse = ctd1 "),
    }
dorado_info["2005.334.03"] = {
    "program": AUVCTD,
    "comment": ("Portaled by DMO 11/28/05 - ctdToUse = ctd1 "),
}
dorado_info["2005.342.00"] = {
    "program": BIOLUME,
    "comment": ("Portaled by DMO 11/28/05 - ctdToUse = ctd1 "),
}
dorado_info["2005.348.00"] = {
    "program": AUVCTD,
    "comment": ("Portaled by DMO 12/14/05 - ctdToUse = ctd1 "),
}
dorado_info["2005.350.03"] = {
    "program": BIOLUME,
    "comment": (
        "Portaled by DMO 12/16/05  'Error in plot_survey_stats (line 139)' when reprocessing"
        " - ctdToUse = ctd1 "
    ),
}

# ----------------------------- 2006 ---------------------------------------
dorado_info["2006.052.06"] = {
    "program": AUVCTD,
    "comment": (
        "Portal  executed from Eclipse to get through ingest/ruminate - multiple Ruminate proble"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2006.054.00"] = {
    "program": BIOLUME,
    "comment": (
        "Portal by Duane on 2/23/06 failed, executed from Eclipse to get through ingest/ruminate - multiple Ruminate problem"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2006.065.12"] = {
    "program": AUVCTD,
    "comment": (
        "First unattended mission done overnight!  Multiple transects in this one mission file"
        "Core dumps on _2column.png with headless print - 18 July 2007"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2006.073.00"] = {
    "program": TEST,
    "comment": ("No useful data from this 4 minute long mission"),
}
dorado_info["2006.073.01"] = {
    "program": AUVCTD,
    "comment": (
        "Second unattended mission done- this one to M2!  Outbound at depth, return yo-yo."
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2006.088.03"] = {
    "program": AUVCTD,
    "comment": (
        "Third unattended mission done- this one to M2.  Outbound at depth, return yo-yo."
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2006.108.03"] = {
    "program": AUVCTD,
    "comment": (
        "Fourth unattended mission done- this one to M2.  Outbound at depth , return yo-yo.  (A little slower than normal.)"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2006.117.05"] = {
    "program": BIOLUME,
    "comment": (
        "Unattended BIOLUME mission - Monterey to Soquel, then sawtooth at canyon edge back to M0"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2006.130.00"] = {
    "program": BIOLUME,
    "comment": (
        "Unattended BIOLUME mission - Monterey to Soquel, then one sawtooth at canyon picked up early for next mission"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2006.130.01"] = {
    "program": BIOLUME,
    "comment": (
        "Unattended BIOLUME mission - C1 to M2 and back - a lot of data collecetd in the last 48 hours. (After strong wind event.)"
        " - ctdToUse = ctd1 "
    ),
}
for day, mission_number in [(135, 3), (135, 4), (137, 0), (138, 1)]:
    dorado_info[f"2006.{day:03d}.{mission_number:02d}"] = {
        "program": AUVCTD,
        "comment": ("Davidson Seamount - John Ryan - ctdToUse = ctd1 "),
    }
dorado_info["2006.200.01"] = {
    "program": LOCO,
    "comment": ("LOCO mission - as part of MB06 - John Ryan - ctdToUse = ctd1 "),
}
dorado_info["2006.206.38"] = {
    "program": AUVCTD,
    "comment": ("AUVCTD mission - as part of MB06 - John Ryan - ctdToUse = ctd1 "),
}
dorado_info["2006.214.01"] = {
    "program": BIOLUME,
    "comment": (
        "BIOLUME mission - as part of MB06 - Finished up on LOCO line, Missed some Iridium points (???))"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2006.219.08"] = {
    "program": LOCO,
    "comment": (
        "LOCO mission - as part of MB06 - Ryan line on way to a Chavez survey for MB06"
        " - ctdToUse = ctd1 "
    ),
}
for day, mission_number in [(219, 10), (220, 0), (221, 0)]:
    dorado_info[f"2006.{day:03d}.{mission_number:02d}"] = {
        "program": AUVCTD,
        "comment": (
            "AUVCTD mission - as part of MB06 - Chavez survey for MB06 to fill in spots where glider's aren't going"
            " - ctdToUse = ctd1 "
        ),
    }
dorado_info["2006.227.08"] = {
    "program": BIOLUME,
    "comment": (
        "BIOUME mission - as part of MB06 - Standard Haddock section (split out DAllanB iridium messages)"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2006.227.09"] = {
    "program": BIOLUME,
    "comment": ("AUVCTD mission - as part of MB06 - Standard C1 to M2 and back - ctdToUse = ctd1 "),
}
dorado_info["2006.234.02"] = {
    "program": AUVCTD,
    "comment": (
        "AUVCTD mission - as part of MB06 - Chavez survey for MB06 to fill in spots where glider's aren't going"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2006.235.01"] = {
    "program": AUVCTD,
    "comment": (
        "AUVCTD mission - as part of MB06 - Chavez survey for MB06 to fill in spots where glider's aren't going"
        " Had problem processing O2, no ISUS data (maybe some wierd bug whend there is no ISUS data)"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2006.249.01"] = {
    "program": BIOLUME,
    "comment": ("BIOLUME mission - ctdToUse = ctd1 "),
}
dorado_info["2006.249.03"] = {
    "program": AUVCTD,
    "comment": ("AUVCTD mission - out to between M1 & M2 - ctdToUse = ctd1 "),
}
dorado_info["2006.264.00"] = {
    "program": AUVCTD,
    "comment": ("AUVCTD mission - out to between M1 & M2 - ctdToUse = ctd1 "),
}
dorado_info["2006.270.00"] = {
    "program": BIOLUME,
    "comment": ("AUVCTD mission - Standard Biolume run.  CTD1 was noisey - ctdToUse = ctd1 "),
}
dorado_info["2006.270.01"] = {
    "program": AUVCTD,
    "comment": ("AUVCTD mission - out to between M1 & M2.  CTD1 was noisey - ctdToUse = ctd1 "),
}
dorado_info["2006.291.03"] = {
    "program": BIOLUME,
    "comment": ("AUVCTD mission - Standard Biolume run.  CTD1 was noisey - ctdToUse = ctd1 "),
}
dorado_info["2006.291.04"] = {
    "program": AUVCTD,
    "comment": ("AUVCTD mission - out to M1 and back - ctdToUse = ctd1 "),
}
dorado_info["2006.324.11"] = {
    "program": AUVCTD,
    "comment": ("AUVCTD mission - out to between M1 & M2 - ctdToUse = ctd1 "),
}
dorado_info["2006.331.02"] = {
    "program": AUVCTD,
    "comment": ("AUVCTD mission - out to between M1 & M2 - ctdToUse = ctd1 "),
}
dorado_info["2006.338.11"] = {
    "program": AUVCTD,
    "comment": (
        "Portaled files exist in archive, but not processed by reprocess_surveys.m"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2006.346.03"] = {
    "program": AUVCTD,
    "comment": (
        "Portaled files exist in archive, but not processed by reprocess_surveys.m"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2006.347.05"] = {
    "program": LOCO,
    "comment": ("LOCO mission - HAB incubator area (off Soquel) - ctdToUse = ctd1 "),
}
dorado_info["2006.348.01"] = {
    "program": LOCO,
    "comment": (
        "LOCO mission - HAB incubator area (off Soquel) - Zephyr prop damage to nosecone"
        " Got 'Error in plot_sections (line 852)' when reprocessed on 30 May 2018"
        " - ctdToUse = ctd1 "
    ),
}

# ----------------------------- 2007 ---------------------------------------
dorado_info["2007.009.06"] = {
    "program": LOCO,
    "comment": (
        "LOCO mission - 12 m isobath - after Iridium SBD service turned off"
        " Got 'Error in plot_sections (line 852)' when reprocessing attempted on 30 May 2018"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2007.029.11"] = {
    "program": BIOLUME,
    "comment": (
        "BIOLUME mission - Filled sensors with sand half-way through survey at 17m; data after that is bad"
        " Got 'Error in plot_sections (line 852)' when reprocessing attempted on 30 May 2018"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2007.061.02"] = {
    "program": AUVCTD,
    "comment": (
        "Test AUVCTD mission - First mission with new blunt nose cone"
        " Got 'Error in plot_sections (line 852)' when reprocessing attempted on 30 May 2018"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2007.067.01"] = {
    "program": BIOLUME,
    "comment": (
        "BIOLUME mission - First real mission with new blunt nose cone - Zephr attended to dorado at M1 in the morning"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2007.085.01"] = {
    "program": AUVCTD,
    "comment": ("AUVCTD mission - Short survey at M1 - ctdToUse = ctd1 "),
}
for mission_number in [0, 1]:
    dorado_info[f"2007.120.{mission_number:02d}"] = {
        "program": AUVCTD,
        "comment": ("AUVCTD mission - Volume survey around M0 for Scholin - ctdToUse = ctd1 "),
    }
dorado_info["2007.093.12"] = {
    "program": AUVCTD,
    "comment": ("Portaled files exist in archive, but not processed by reprocess_surveys.m"),
}
dorado_info["2007.094.02"] = {
    "program": AUVCTD,
    "comment": ("Portaled files exist in archive, but not processed by reprocess_surveys.m"),
}
for mission_number in [5, 7]:
    dorado_info[f"2007.123.{mission_number:02d}"] = {
        "program": AUVCTD,
        "comment": ("AUVCTD mission - Volume survey around M0 for Scholin - ctdToUse = ctd1 "),
    }
for mission_number in [4, 6, 7, 9]:
    dorado_info[f"2007.134.{mission_number:02d}"] = {
        "program": AUVCTD,
        "comment": (
            "AUVCTD missions - Bellingham looking for internal waves in Southern MB"
            " - ctdToUse = ctd1 "
        ),
    }
dorado_info["2007.142.02"] = {
    "program": AUVCTD,
    "comment": (
        "AUVCTD mission - C1 to M2 & back - gulper tripped at nephaloid layers (we hope!)"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2007.143.01"] = {
    "program": AUVCTD,
    "comment": ("Portaled files exist in archive, but not processed by reprocess_surveys.m"),
}
dorado_info["2007.144.02"] = {
    "program": BIOLUME,
    "comment": ("BIOLUME mission - ctdToUse = ctd1 "),
}
dorado_info["2007.149.02"] = {
    "program": BIOLUME,
    "comment": ("BIOLUME mission - ctdToUse = ctd1 "),
}
dorado_info["2007.171.04"] = {
    "program": BIOLUME,
    "comment": (
        "BIOLUME mission - reverse path, starting from the north.  Stopped near M1."
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2007.211.02"] = {
    "program": BIOLUME,
    "comment": (
        "AUVCTD mission - Canyon Axis with Gulper, no flow cell on main CTD1 line"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2007.234.05"] = {
    "program": AUVCTD,
    "comment": ("AUVCTD mission - Canyon Axis - ctdToUse = ctd1 "),
}
dorado_info["2007.235.00"] = {
    "program": AUVCTD,
    "comment": ("AUVCTD mission - Northern shelf, then cross-bay - ctdToUse = ctd1 "),
}
dorado_info["2007.239.03"] = {
    "program": AUVCTD,
    "comment": ("AUVCTD mission - Internal wave study near Monterey - ctdToUse = ctd1 "),
}
dorado_info["2007.239.05"] = {
    "program": AUVCTD,
    "comment": ("AUVCTD mission - Internal wave study near Monterey - ctdToUse = ctd1 "),
}
dorado_info["2007.247.05"] = {
    "program": OCCO,
    "comment": (
        "OCCO mission - Incubation area flux track - CTD1 clogged or something at end - in conjunction with ESP deployments"
        " - ctdToUse = ctd2 "
    ),
}
dorado_info["2007.248.00"] = {
    "program": OCCO,
    "comment": (
        "OCCO mission - Nighttime run, Iridium not working - no data from Isus - in conjunction with ESP deployments"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2007.249.00"] = {
    "program": OCCO,
    "comment": (
        " OCCO mission - Iridium not working - in conjunction with ESP deployments"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2007.253.02"] = {
    "program": OCCO,
    "comment": ("OCCO mission - Tight box survey about ESP site - ctdToUse = ctd1 "),
}
dorado_info["2007.254.01"] = {
    "program": OCCO,
    "comment": ("OCCO mission - Canyon axis survey - ctdToUse = ctd1 "),
}
dorado_info["2007.255.01"] = {
    "program": OCCO,
    "comment": ("OCCO mission - Northern shelf and cross-bay chevron survey - ctdToUse = ctd1 "),
}
dorado_info["2007.260.00"] = {
    "program": OCCO,
    "comment": ("OCCO mission - Northern shelf survey about ESP deployments - ctdToUse = ctd2 "),
}
dorado_info["2007.261.00"] = {
    "program": OCCO,
    "comment": (
        "OCCO mission - Northern shelf survey about ESP deployments - got tied up in kelp near Natural Bridges"
        " - ctdToUse = ctd1 "
    ),
}
for day, mission_number in [(261, 1), (263, 1)]:
    dorado_info[f"2007.{day:02d}.{mission_number:02d}"] = {
        "program": OCCO,
        "comment": (
            "OCCO mission - Northern shelf survey about ESP deployments - ctdToUse = ctd1 "
        ),
    }
dorado_info["2007.264.09"] = {
    "program": AUVCTD,
    "comment": ("AUVCTD mission - Kanna control tests around M0 - ctdToUse = ctd1 "),
}
for day, mission_number in [
    (267, 4),
    (268, 1),
]:
    dorado_info[f"2007.{day:02d}.{mission_number:02d}"] = {
        "program": OCCO,
        "comment": ("OCCO mission - Tight box survey about ESP site - ctdToUse = ctd1 "),
    }
dorado_info["2007.269.00"] = {
    "program": OCCO,
    "comment": ("OCCO mission - Last one for Sept 2007 - ctdToUse = ctd1 "),
}
dorado_info["2007.325.16"] = {
    "program": AUVCTD,
    "comment": ("AUVCTD mission - Kanna control tests south of M0 - ctdToUse = ctd1 "),
}
dorado_info["2007.330.05"] = {
    "program": AUVCTD,
    "comment": ("AUVCTD mission - Bellingham internal waves near MISO - ctdToUse = ctd1 "),
}
dorado_info["2007.331.01"] = {
    "program": AUVCTD,
    "comment": ("AUVCTD mission - Canyon Axis with Gulper - ctdToUse = ctd1 "),
}

# From: Monique Messié <monique@mbari.org>
# Subject: Dorado changes
# Date: November 15, 2022 at 3:08:37 PM PST
# To: Mike McCann <mccann@mbari.org>
#
# Hi Mike,
#
# Sorry - I realized I forgot to send you the changes I know of for Dorado. Here it is:
#
# From Hans's email (on Dec 14, 2016):
# The change from side mounting to nose mounting occurred when we change to the new nose
# with the LOPC and other instruments. The first mission for the new nose was 2007.344.00,
# so any data prior to that was in the side mounted configuration. The last mission with
# the UCSB biolume sensor was 2010.081.02. It developed pump problems after this mission
# and was removed from the vehicle. We took delivery of the UBAT from Wetlabs on July 29 2010
# (this was per an email shipment notification I received from Wetlabs). The first mission
# I show data from this UBAT was 2010.277.01, on October 4 2010. This can be determined by
# inspecting the biolume.log files for the mission. The Wetlabs log files have some extra
# fields in their log files (avg_biolume, system_voltage, HV, and reserved).
#
# Monique


dorado_info["2007.344.00"] = {
    "program": AUVCTD,
    "comment": (
        "AUVCTD mission - Canyon Axis with Gulper"
        " New nose with the LOPC and other instruments"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2007.346.00"] = {
    "program": BIOLUME,
    "comment": (
        "AUVCTD mission -  Northern shelf then chevron pattern across bay to Monterey"
        " - ctdToUse = ctd1 "
    ),
}

# ----------------------------- 2008 ---------------------------------------
for mission_number in [7, 8, 10]:
    dorado_info[f"2008.010.{mission_number:02d}"] = {
        "program": AUVCTD,
        "comment": (
            "AUVCTD mission - Autonomy group missions over canyon axis between M0 & M1 (gulper trips in nice layers)"
            " - ctdToUse = ctd1 "
        ),
    }
dorado_info["2008.107.02"] = {
    "program": AUVCTD,
    "comment": ("AUVCTD mission - Canyon axis (after 3 months off) - ctdToUse = ctd1 "),
}
for mission_number in [4, 5]:
    dorado_info[f"2008.133.{mission_number:02d}"] = {
        "program": AUVCTD,
        "comment": ("AUVCTD mission - Autonomy TREX mission - ctdToUse = ctd1 "),
    }
dorado_info["2008.140.00"] = {
    "program": AUVCTD,
    "comment": ("AUVCTD mission - Autonomy TREX mission - ctdToUse = ctd1 "),
}
dorado_info["2008.154.01"] = {
    "program": AUVCTD,
    "comment": ("AUVCTD mission - Ryan - ESP inner northern shelf - ctdToUse = ctd1 "),
}
dorado_info["2008.155.01"] = {
    "program": AUVCTD,
    "comment": ("AUVCTD mission - Ryan - ESP inner northern shelf - ctdToUse = ctd1 "),
}
dorado_info["2008.161.02"] = {
    "program": AUVCTD,
    "comment": (
        "AUVCTD mission - Ryan - ESP inner northern shelf (something hit AUV just before midnight)"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2008.168.00"] = {
    "program": AUVCTD,
    "comment": ("AUVCTD mission - Ryan - ESP inner northern shelf - ctdToUse = ctd1 "),
}
dorado_info["2008.178.01"] = {
    "program": AUVCTD,
    "comment": ("AUVCTD mission - Autonomy TREX around M0 - ctdToUse = ctd1 "),
}
for mission_number in [0, 1, 2]:
    dorado_info[f"2008.261.{mission_number:02d}"] = {
        "program": AUVCTD,
        "comment": (
            "AUVCTD mission - Autonomy TREX around M0 - CTD2 wasn't pumped, there was a piece of tubing that came off. "
            "ctdToUse = ctd1 "
        ),
    }
dorado_info["2008.281.03"] = {
    "program": AUVCTD,
    "comment": (
        "AUVCTD mission - Rob McEwen's software with gulper trigger at 2e-3 Chlor"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2008.281.04"] = {
    "program": AUVCTD,
    "comment": (
        "AUVCTD mission - Rob McEwen's software with gulper trigger at 2e-3 Chlor"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2008.287.05"] = {
    "program": AUVCTD,
    "comment": ("AUVCTD mission - North bay shelf survey  - ctdToUse = ctd2 "),
}
for day, mission_number in [(288, 0), (295, 1), (296, 0), (297, 1)]:
    dorado_info[f"2008.{day:02d}.{mission_number:02d}"] = {
        "program": AUVCTD,
        "comment": ("AUVCTD mission - North bay shelf survey - ctdToUse = ctd2 "),
    }
for day, mission_number in [(289, 3), (297, 0)]:
    dorado_info[f"2008.{day:02d}.{mission_number:02d}"] = {
        "program": AUVCTD,
        "comment": ("AUVCTD mission - North bay shelf survey - ctdToUse = ctd1 "),
    }
for day, mission_number in [(315, 1), (318, 1), (318, 2)]:
    dorado_info[f"2008.{day:02d}.{mission_number:02d}"] = {
        "program": AUVCTD,
        "comment": ("AUVCTD mission - Kanna trex mission - ctdToUse = ctd1 "),
    }

# ----------------------------- 2009 ---------------------------------------
dorado_info["2009.055.05"] = {
    "program": AUVCTD,
    "comment": (
        "AUVCTD mission - Ryan along canyon transect - CTD1 had some issues we're looking into (software probably)"
        " - ctdToUse = ctd2 "
    ),
}
dorado_info["2009.084.00"] = {
    "program": AUVCTD,
    "comment": ("AUVCTD mission - Ryan short mission, Gulpers fired - ctdToUse = ctd2 "),
}
dorado_info["2009.084.02"] = {
    "program": AUVCTD,
    "comment": ("AUVCTD mission - Ryan short mission, Gulpers fired - ctdToUse = ctd2 "),
}
dorado_info["2009.085.02"] = {
    "program": AUVCTD,
    "comment": ("AUVCTD mission - Ryan along north coast across bay overnight - ctdToUse = ctd1 "),
}
dorado_info["2009.111.00"] = {
    "program": AUVCTD,
    "comment": (
        "AUVCTD mission - On N-S line by M0 for ESP deployment - very short mission - one Gulper fired"
        " Returns 'Error in plot_sections (line 852)' when reprocessing attemped on 29 May 2018"
        " - ctdToUse = ctd2 "
    ),
}
dorado_info["2009.111.01"] = {
    "program": AUVCTD,
    "comment": ("AUVCTD mission - On N-S line by M0 for ESP deployment - ctdToUse = ctd2 "),
}
dorado_info["2009.112.07"] = {
    "program": AUVCTD,
    "comment": ("AUVCTD mission - On N-S line by M0 for ESP deployment - ctdToUse = ctd2 "),
}
dorado_info["2009.113.00"] = {
    "program": AUVCTD,
    "comment": ("AUVCTD mission - On N-S line by M0 for ESP deployment - ctdToUse = ctd2 "),
}
dorado_info["2009.113.08"] = {
    "program": AUVCTD,
    "comment": ("AUVCTD mission - On N-S line by M0 for ESP deployment - ctdToUse = ctd2 "),
}
dorado_info["2009.124.03"] = {
    "program": AUVCTD,
    "comment": ("AUVCTD mission - Autonomy mission - ctdToUse = ctd2 "),
}
dorado_info["2009.125.00"] = {
    "program": AUVCTD,
    "comment": ("AUVCTD mission - Canyon transect - ctdToUse = ctd2 "),
}
dorado_info["2009.126.00"] = {
    "program": AUVCTD,
    "comment": ("AUVCTD mission - Single dive to trip bottles at M1 - ctdToUse = ctd2 "),
}
dorado_info["2009.127.02"] = {
    "program": FAILED,
    "comment": (
        "AUVCTD mission - Autonomy mission - mvc crashed, some files may be corrupted - fails with 'Error in processNav (line 43)'"
        " - ctdToUse = ctd2 "
    ),
}
for day, mission_number in [(152, 0), (153, 0), (153, 1), (154, 0), (155, 3)]:
    dorado_info[f"2009.{day:02d}.{mission_number:02d}"] = {
        "program": AUVCTD,
        "comment": ("AUVCTD mission - Ryan - around ESP - ctdToUse = ctd2 "),
    }
dorado_info["2009.182.01"] = {
    "program": AUVCTD,
    "comment": ("UVCTD mission - TREX - ctdToUse = ctd2 "),
}
dorado_info["2009.272.00"] = {
    "program": AUVCTD,
    "comment": (
        "AUVCTD mission - Yanwu's detection - bottle tripping algorithm - ctdToUse = ctd2 "
    ),
}
dorado_info["2009.274.03"] = {
    "program": AUVCTD,
    "comment": (
        "AUVCTD mission - Yanwu's detection - bottle tripping algorithm - ctdToUse = ctd1 "
    ),
}
for mission_number in [1, 2]:
    dorado_info[f"2009.278.{mission_number:02d}"] = {
        "program": AUVCTD,
        "comment": (
            "AUVCTD mission - Short missions, The .01 mission was very short but two gulpers fired"
            " - ctdToUse = ctd1 "
        ),
    }
dorado_info["2009.279.00"] = {
    "program": AUVCTD,
    "comment": ("AUVCTD mission - Truncated section for some reason... - ctdToUse = ctd1 "),
}
dorado_info["2009.280.00"] = {
    "program": AUVCTD,
    "comment": (
        "AUVCTD mission -  Problem in mergeDVL with GPS reoutine where survey distance got shortened"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2009.281.01"] = {
    "program": AUVCTD,
    "comment": ("AUVCTD mission -  - ctdToUse = ctd1 "),
}
dorado_info["2009.308.04"] = {
    "program": AUVCTD,
    "comment": ("AUVCTD mission - Kanna mission - ctdToUse = ctd1 "),
}
for mission_number in range(4):
    dorado_info[f"2009.309.{mission_number:02d}"] = {
        "program": AUVCTD,
        "comment": (
            "AUVCTD mission - Kanna mission - Even though the thursdays runs (3 of them) are very short (~30mn  each) the first one at leas had the gulper fired so it is of relevance  for further science analysis."
            " - ctdToUse = ctd1 "
        ),
    }
dorado_info["2009.313.02"] = {
    "program": AUVCTD,
    "comment": ("AUVCTD mission - Kanna mission - ctdToUse = ctd1 "),
}
dorado_info["2009.342.04"] = {
    "program": AUVCTD,
    "comment": ("AUVCTD mission - Kanna mission - ctdToUse = ctd1 "),
}
dorado_info["2009.348.05"] = {
    "program": AUVCTD,
    "comment": ("AUVCTD mission - Kanna mission - ctdToUse = ctd1 "),
}

# ----------------------------- 2010 ---------------------------------------
dorado_info["2010.081.02"] = {
    "program": AUVCTD,
    "comment": (
        "AUVCTD mission - Autonomy mission"
        " Last mission with the UCSB biolume sensor"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2010.082.02"] = {
    "program": AUVCTD,
    "comment": ("AUVCTD mission - Autonomy mission - ctdToUse = ctd1 "),
}
dorado_info["2010.083.03"] = {
    "program": AUVCTD,
    "comment": (
        "AUVCTD mission - Processing requested by Yanwu (Kevin helped with SSDS ingest/portal)"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2010.083.08"] = {
    "program": AUVCTD,
    "comment": (
        "AUVCTD mission - Processing requested by Yanwu (Kevin helped with SSDS ingest/portal)"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2010.118.00"] = {
    "program": AUVCTD,
    "comment": ("AUVCTD mission - Ryan mission - ctdToUse = ctd2 "),
}
dorado_info["2010.119.01"] = {
    "program": AUVCTD,
    "comment": ("AUVCTD mission - Ryan mission - ctdToUse = ctd1 "),
}
dorado_info["2010.151.04"] = {
    "program": AUVCTD,
    "comment": (
        "AUVCTD mission - Gulf of Mexico mission without GPS data (USBL fixes provided in a separate file)"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2010.153.01"] = {
    "program": AUVCTD,
    "comment": (
        "AUVCTD mission - Gulf of Mexico mission - down to 1200 m a few yo-yos at depth strong backscatter signal"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2010.154.01"] = {
    "program": AUVCTD,
    "comment": (
        "AUVCTD mission - Gulf of Mexico mission - perhaps a better volume survey than on the previous day"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2010.172.01"] = {
    "program": FAILED,
    "comment": (
        "AUVCTD mission - Ryan mission in Monterey Bay rhodamine sensor replaces CDOM -"
        " bad times in Nav.time ( -4.7749e+307) - won't process (bad mission?)"
        " Really bad longitude and latitude - best to ignore this mission."
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2010.173.00"] = {
    "program": AUVCTD,
    "comment": (
        "AUVCTD mission - Ryan mission in Monterey Bay rhodamine sensor replaces CDOM  - won't process"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2010.174.00"] = {
    "program": AUVCTD,
    "comment": (
        "AUVCTD mission - Ryan mission in Monterey Bay rhodamine sensor replaces CDOM  - won't process"
        " - ctdToUse = ctd1 "
    ),
}
for day, mission_number in [(180, 5), (181, 1), (181, 2), (182, 2)]:
    dorado_info[f"2010.{day:02d}.{mission_number:02d}"] = {
        "program": AUVCTD,
        "comment": (
            "AUVCTD mission - Kanna missions in Monterey Bay rhodamine sensor replaces CDOM  - around time of Board meeting"
            " - ctdToUse = ctd1 "
        ),
    }
for day, mission_number in [
    (257, 0),
    (257, 1),
    (257, 2),
    (257, 3),
    (257, 4),
    (257, 5),
    (257, 6),
    (257, 7),
    (257, 8),
    (257, 9),
    (257, 10),
    (257, 11),
    (258, 0),
    (258, 1),
    (258, 2),
    (258, 3),
    (258, 4),
    (258, 5),
    (258, 6),
    (258, 7),
    (258, 8),
    (259, 0),
    (259, 1),
    (259, 2),
    (259, 3),
    (260, 0),
    (261, 0),
]:
    dorado_info[f"2010.{day:02d}.{mission_number:02d}"] = {
        "program": AUVCTD,
        "comment": (
            "ESP drifter missions out at station 67-70 with Flyer doing casts and ESP drifting south"
            " toward Davidson Seamount - no gulpers (Frederic sent me note about survey grouping)"
            " Faulty parosci lead to several mission depth aborts at beginning of this set of volume surveys"
            " - ctdToUse = ctd1 "
        ),
    }
dorado_info["2010.265.00"] = {
    "program": AUVCTD,
    "comment": ("Cross-bay survey in preparation for BloomEx in October 2010 - ctdToUse = ctd2 "),
}
dorado_info["2010.277.01"] = {
    "program": AUVCTD,
    "comment": (
        "Inner-bay survey first one for BloomEx in October 2010 -"
        " Doug said there was a lot of stuff out there.  It was trailing kelp when they picked it up."
        " The first mission with the UBAT"
        " - ctdToUse = ctd2 "
    ),
}
dorado_info["2010.278.01"] = {
    "program": AUVCTD,
    "comment": (
        "Inner-bay survey second one for BloomEx in October 2010 - Aborted at midnight local for not geting sufficient GPS fixes, possibly because of surfacing into a big jellyfish"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2010.279.02"] = {
    "program": AUVCTD,
    "comment": (
        "Inner-bay survey third one for BloomEx in October 2010 - Tight survey around stell drifters"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2010.280.01"] = {
    "program": AUVCTD,
    "comment": (
        "Inner-bay survey fourth day-time one for BloomEx in October 2010 - ctdToUse = ctd1 "
    ),
}
dorado_info["2010.284.00"] = {
    "program": AUVCTD,
    "comment": (
        "Inner-bay survey fifth - M0 to LatMix, aborted at 2000 local - serial interface hardware failure first rhodamin.log file"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2010.285.00"] = {
    "program": AUVCTD,
    "comment": (
        "Inner-bay survey sixth - M0 to LatMix, aborted at 0500 local - 8 laps rqn out of battery"
        " - ctdToUse = ctd1 "
    ),
}
for mission_number in [1, 2]:
    dorado_info[f"2010.286.{mission_number:02d}"] = {
        "program": AUVCTD,
        "comment": (
            "Inner-bay survey seventh - TREX lagrangian mission near M0 and stella104 drifter"
            " - ctdToUse = ctd1 "
        ),
    }
dorado_info["2010.287.00"] = {
    "program": AUVCTD,
    "comment": ("Inner-bay survey eigth - Lap around LatMix array - ctdToUse = ctd1 "),
}
dorado_info["2010.291.00"] = {
    "program": AUVCTD,
    "comment": ("Inner-bay survey ninth - Overnight after the bay was flushed - ctdToUse = ctd1 "),
}
dorado_info["2010.292.01"] = {
    "program": AUVCTD,
    "comment": ("Inner-bay survey tenth - Overnight - ctdToUse = ctd1 "),
}
dorado_info["2010.293.00"] = {
    "program": AUVCTD,
    "comment": (
        "Inner-bay survey eleveth - Overnight TREX mission following CHL_PATCH - ctdToUse = ctd1 "
    ),
}
dorado_info["2010.294.01"] = {
    "program": AUVCTD,
    "comment": (
        "Inner-bay survey twelfth - Daytime short misison around LatMix - ctdToUse = ctd1 "
    ),
}
dorado_info["2010.298.01"] = {
    "program": AUVCTD,
    "comment": (
        "Inner-bay survey thirteenth - Overnight mirroring waveglider N-S across Monterey Bay"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2010.299.00"] = {
    "program": AUVCTD,
    "comment": (
        "Inner-bay survey fourteenth - Overnight north end of Bay looking for the bloom"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2010.300.00"] = {
    "program": AUVCTD,
    "comment": (
        "Inner-bay survey fifteenth - Overnight north end of Bay looking for the bloom"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2010.301.00"] = {
    "program": AUVCTD,
    "comment": ("Inner-bay survey fifteenth - Daytime run - ctdToUse = ctd1 "),
}
for day, mission_number in [(340, 0), (340, 1), (341, 0), (341, 1)]:
    dorado_info[f"2010.{day:02d}.{mission_number:02d}"] = {
        "program": AUVCTD,
        "comment": ("Out and back from M1 - ctdToUse = ctd1 "),
    }
dorado_info["2010.342.04"] = {
    "program": AUVCTD,
    "comment": (
        "Kanna mission with automated forwarding of stella103 position to TREX via MBARITracking/scripts/drifter.py software"
        " - ctdToUse = ctd1 "
    ),
}

# ----------------------------- 2011 ---------------------------------------
dorado_info["2011.060.01"] = {
    "program": AUVCTD,
    "comment": (
        "Kanna mission with automated forwarding of stella104 position to TREX via MBARITracking/scripts/drifter.py RabbitMQ and Tom O'Reilly's Tracking APp"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2011.061.00"] = {
    "program": AUVCTD,
    "comment": (
        "Kanna mission with automated forwarding of stella104 position to TREX via MBARITracking/scripts/drifter.py RabbitMQ and Tom O'Reilly's Tracking APp"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2011.062.05"] = {
    "program": AUVCTD,
    "comment": (
        "Kanna mission with automated forwarding of simulated tethys_cen position to TREX via MBARITracking/scripts/drifter.py  and McCann' tres.py code"
        " - ctdToUse = ctd2 "
    ),
}
dorado_info["2011.074.02"] = {
    "program": AUVCTD,
    "comment": ("Cross-bay Gulper mission for Vrijenhouk Lab - John Ryan - ctdToUse = ctd1 "),
}
dorado_info["2011.110.12"] = {
    "program": AUVCTD,
    "comment": (
        "Autonomy mission, first follow stella102 then switch to do a diamond around Moe after the stellas were heading toward the beach"
        " CTD1 had its intake tube pushed into the fairing so the pumped data are bad - the flow was impeded and there is a lot of 'hystersis'.  Use CTD2 with the proper new .cfg."
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2011.111.00"] = {
    "program": AUVCTD,
    "comment": (
        "Autonomy mission, Do diamonds around Moe with Tethys doing a circle around M0."
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2011.115.10"] = {
    "program": AUVCTD,
    "comment": (
        "Yanwu trigger on front servey mission in north bay. Hans said it aborted earlly.  Maybe it hit something."
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2011.116.00"] = {
    "program": AUVCTD,
    "comment": (
        "Yanwu trigger on front servey mission in north bay. Repeat of yesterday's survey.  It went longer this time."
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2011.117.01"] = {
    "program": AUVCTD,
    "comment": (
        "Autonomy mission - follow stella102 with Tom O'Reilly's Drifter Tracking App then go to M0"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2011.118.00"] = {
    "program": AUVCTD,
    "comment": (
        "Autonomy mission - follow tethysCtr (generated by tethys front detection code) with Tom O'Reilly's Drifter Tracking App - a windy day on deployment"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2011.155.04"] = {
    "program": AUVCTD,
    "comment": (
        "Brewer mission - Mapping vehicle tail plus Dorado nosecone - In Santa Monica Basin UTM Zone 11S"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2011.157.01"] = {
    "program": AUVCTD,
    "comment": (
        "Brewer mission - Mapping vehicle tail plus Dorado nosecone - In Santa Monica Basin UTM Zone 11S"
        " - Incomplete depth data for CTD, unable to create section plots"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2011.158.00"] = {
    "program": AUVCTD,
    "comment": (
        "Brewer mission - Mapping vehicle tail plus Dorado nosecone - In Santa Monica Basin UTM Zone 11S"
        " - Incomplete depth data for CTD, unable to create section plots"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2011.164.05"] = {
    "program": f"{CANONJUN2011}",
    "comment": (
        "First mission for June 2011 CANON - only down to 25m not to the bottom on the outbound leg"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2011.165.00"] = {
    "program": f"{CANONJUN2011}",
    "comment": (
        "Second mission for June 2011 CANON - strong upwelling front, still not all the way to the bottom"
        " - ctdToUse = ctd2 "
    ),
}
dorado_info["2011.166.00"] = {
    "program": f"{CANONJUN2011}",
    "comment": (
        "Third mission for June 2011 CANON - strong upwelling front, still not all the way to the bottom"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2011.171.01"] = {
    "program": f"{CANONJUN2011}",
    "comment": (
        "TREX mission going back and forth on 36.9 line around weak salinity front - aborted early, maybe because of a trex cron job"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2011.172.00"] = {
    "program": f"{CANONJUN2011}",
    "comment": (
        "TREX mission going back and forth on 36.9 line around weak salinity front"
        " - ctdToUse = ctd2 "
    ),
}
dorado_info["2011.249.00"] = {
    "program": f"{CANONSEP2011}",
    "comment": (
        "TREX mission - first of Sept 2011 CANON aborted due to software - ctdToUse = ctd1 "
    ),
}
dorado_info["2011.250.01"] = {
    "program": f"{CANONSEP2011}",
    "comment": (
        "TREX mission - M0 around M1 corner then out to upwelling region - ctdToUse = ctd2 "
    ),
}
dorado_info["2011.255.00"] = {
    "program": f"{CANONSEP2011}",
    "comment": ("TREX mission - Following SIO_WWL_1 for September 2011 CANON - ctdToUse = ctd2 "),
}
for mission_num in [2, 3]:
    dorado_info[f"2011.256.{mission_num:02}"] = {
        "program": f"{CANONSEP2011}",
        "comment": (
            "TREX mission - Following SIO_WWL_1 for September 2011 CANON - ctdToUse = ctd1 "
        ),
    }
dorado_info["2011.257.00"] = {
    "program": f"{CANONSEP2011}",
    "comment": (
        "TREX mission - Following Lagrangian mission - no upwelling yet - ctdToUse = ctd1 "
    ),
}
dorado_info["2011.262.00"] = {
    "program": f"{CANONSEP2011}",
    "comment": (
        "TREX mission - Following Lagrangian mission around M0 - no upwelling yet"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2011.263.00"] = {
    "program": f"{CANONSEP2011}",
    "comment": (
        "TREX mission - Following Lagrangian mission around M0 - no upwelling yet"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2011.264.00"] = {
    "program": f"{CANONSEP2011}",
    "comment": (
        "TREX mission - Following Lagrangian mission around M0 - no upwelling yet"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2011.285.01"] = {
    "program": f"{CANONSEP2011}",
    "comment": ("TREX mission - Maria Fox's algorithm in Monterey Bay - ctdToUse = ctd1 "),
}
dorado_info["2011.286.00"] = {
    "program": f"{CANONSEP2011}",
    "comment": ("TREX mission - Maria Fox's algorithm in Monterey Bay - ctdToUse = ctd1 "),
}
# ----------------------------- 2012 ---------------------------------------

for day, mission_num in [(142, 1), (142, 2), (143, 7), (143, 8)]:
    dorado_info[f"2012.{day:03}.{mission_num:02}"] = {
        "program": f"{AUVCTD}",
        "comment": (
            "TREX missions - Initial engineering test from Yanwu & Frederic - ctdToUse = ctd1 "
        ),
    }
for day, mission_num in [(150, 0), (151, 0), (152, 0), (157, 7), (158, 0)]:
    dorado_info[f"2012.{day:03}.{mission_num:02}"] = {
        "program": f"{CANONMAY2012}",
        "comment": ("For CANON May 2012 - yoyo along 36.9N - ctdToUse = ctd2 "),
    }
dorado_info["2012.256.00"] = {
    "program": f"{CANONSEP2012}",
    "comment": (
        "CANON September 2012 - laps around ESP drifter - battery problems - less than 1/3 of a normal mission"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2012.257.01"] = {
    "program": f"{CANONSEP2012}",
    "comment": ("CANON September 2012 - laps around ESP drifter - ctdToUse = ctd1 "),
}
dorado_info["2012.258.00"] = {
    "program": f"{CANONSEP2012}",
    "comment": (
        "CANON September 2012 - laps around ESP drifter - last one for this campaign"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2012.268.07"] = {
    "program": f"{CANONSEP2012}",
    "comment": (
        "CANON September 2012 - laps around ESP drifter - last one for this campaign"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2012.269.00"] = {
    "program": f"{CANONSEP2012}",
    "comment": ("CANON September 2012 - Distributed Autonomy - ctdToUse = ctd1 "),
}
dorado_info["2012.270.04"] = {
    "program": f"{CANONSEP2012}",
    "comment": ("CANON September 2012 - Distributed Autonomy - ctdToUse = ctd1 "),
}

# ----------------------------- 2013 ---------------------------------------
dorado_info["2013.074.02"] = {
    "program": f"{CANONMAR2013}",
    "comment": (
        "ECOHAB - CANON March 2013 - outbound mission, Gulpers did not fire - ctdToUse = ctd1 "
    ),
}
for mission_num in [5, 6]:
    dorado_info[f"2013.075.{mission_num:02}"] = {
        "program": f"{CANONMAR2013}",
        "comment": (
            "ECOHAB - CANON March 2013 - drifter tracking T-REX mission, 2 Gulpers on .05 , 8 Gulpers on .06 - stitched together"
            " - ctdToUse = ctd1 "
        ),
    }
for mission_num in [1, 2]:
    dorado_info[f"2013.076.{mission_num:02}"] = {
        "program": f"{CANONMAR2013}",
        "comment": (
            "ECOHAB - CANON March 2013 - inbound mission, Gulper 0 fired on .01 mission and Gulpers 1 through 6 fired on .02"
            " - ctdToUse = ctd1 "
        ),
    }
dorado_info["2013.079.00"] = {
    "program": f"{CANONMAR2013}",
    "comment": (
        "ECOHAB - CANON March 2013 - Tiny test mission with change in parosci.log format. Nav data on deck, did not process"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2013.079.04"] = {
    "program": f"{CANONMAR2013}",
    "comment": ("ECOHAB - CANON March 2013 - Outbound mission, 7 Gulpers fired - ctdToUse = ctd1 "),
}
dorado_info["2013.080.02"] = {
    "program": f"{CANONMAR2013}",
    "comment": ("ECOHAB - CANON March 2013 - Outbound mission, 9 Gulpers fired - ctdToUse = ctd1 "),
}
dorado_info["2013.081.05"] = {
    "program": f"{CANONMAR2013}",
    "comment": (
        "ECOHAB - CANON March 2013 - Outbound mission, JD's adaptive samping algorithm (niche model), 7 Gulpers fired"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2013.081.06"] = {
    "program": f"{CANONMAR2013}",
    "comment": (
        "ECOHAB - CANON March 2013 - Inbound mission, Yanwu's adaptive samping algorithm, all 9 Gulpers fired"
        " - ctdToUse = ctd1 "
    ),
}
for day, mission_num in [
    (224, 2),
    (225, 0),
    (225, 1),
    (226, 1),
    (226, 3),
    (227, 0),
    (227, 1),
    (228, 1),
]:
    dorado_info[f"2013.{day:03}.{mission_num:02}"] = {
        "program": f"{SIMZAUG2013}",
        "comment": (
            "SIMZ Molecular Ecology - August 2013 - North Monterey Bay  - ctdToUse = ctd1 "
        ),
    }
for day, mission_num in [(228, 0)]:
    dorado_info[f"2013.{day:03}.{mission_num:02}"] = {
        "program": f"{SIMZAUG2013}",
        "comment": (
            "SIMZ Molecular Ecology - August 2013 - North Monterey Bay  - ctdToUse = ctd2 "
        ),
    }
for day, mission_num in [
    (259, 0),
    (260, 0),
    (261, 1),
    (262, 0),
    (262, 1),
    (268, 0),
    (273, 0),
    (274, 0),
    (274, 1),
    (275, 0),
    (275, 1),
    (276, 0),
    (280, 1),
]:
    dorado_info[f"2013.{day:03}.{mission_num:02}"] = {
        "program": f"{CANONSEP2013}",
        "comment": ("CANON September 2013 in Monterey Bay  - ctdToUse = ctd1 "),
    }
for day, mission_num in [(282, 0), (283, 0), (287, 0), (287, 1), (287, 2), (290, 0)]:
    dorado_info[f"2013.{day:03}.{mission_num:02}"] = {
        "program": f"{AUVCTD}",
        "comment": ("Autonomy group's short missions in Soquel Bight - ctdToUse = ctd1 "),
    }
for day, mission_num in [
    (295, 0),
    (295, 1),
    (296, 0),
    (296, 1),
    (297, 1),
    (297, 2),
    (298, 0),
    (298, 1),
    (301, 2),
    (301, 3),
    (301, 4),
]:
    dorado_info[f"2013.{day:03}.{mission_num:02}"] = {
        "program": f"{SIMZOCT2013}",
        "comment": ("SIMZ October 2013 in Soquel Bight - ctdToUse = ctd1 "),
    }
# ----------------------------- 2014 ---------------------------------------
for day, mission_num in [(50, 0), (50, 1), (71, 1), (71, 2), (72, 0), (72, 1)]:
    dorado_info[f"2014.{day:03}.{mission_num:02}"] = {
        "program": f"{SIMZSPRING2014}",
        "comment": ("SIMZ Spring 2014 in Soquel Bight - ctdToUse = ctd1 "),
    }
for day, mission_num in [
    (102, 0),
    (103, 0),
    (103, 1),
    (104, 1),
    (107, 0),
    (108, 1),
    (108, 2),
    (109, 0),
    (109, 1),
]:
    dorado_info[f"2014.{day:03}.{mission_num:02}"] = {
        "program": f"{CANONAPR2014}",
        "comment": ("San Pedro Bay CANON-ECOHAB - ctdToUse = ctd1 "),
    }
for day, mission_num in [(211, 2), (212, 0)]:
    dorado_info[f"2014.{day:03}.{mission_num:02}"] = {
        "program": f"{SIMZJUL2014}",
        "comment": ("SIMZ Summer 2014 in Bodega Bay - ctdToUse = ctd1 "),
    }
for day, mission_num in [(210, 1), (210, 2), (211, 3)]:
    dorado_info[f"2014.{day:03}.{mission_num:02}"] = {
        "program": f"{SIMZJUL2014}",
        "comment": ("SIMZ Summer 2014 in Bodega Bay - ctdToUse = ctd2 "),
    }
for day, mission_num in [
    (265, 3),
    (266, 4),
    (266, 5),
    (267, 7),
    (268, 5),
    (280, 1),
    (281, 0),
    (281, 8),
    (282, 2),
    (282, 3),
]:
    dorado_info[f"2014.{day:03}.{mission_num:02}"] = {
        "program": f"{CANONSEP2014}",
        "comment": (
            "CANON-ECOHAB September 2014 with rhodamine sensor on ctdDriver2, channel 1"
            " - ctdToUse = ctd1 "
        ),
    }
for day, mission_num in [(289, 4), (290, 0), (293, 0), (294, 0)]:
    dorado_info[f"2014.{day:03}.{mission_num:02}"] = {
        "program": f"{SIMZOCT2014}",
        "comment": ("SIMZ October 2014 Northern Monterey Bay gulper missions - ctdToUse = ctd1 "),
    }
for day, mission_num in [(295, 0)]:
    dorado_info[f"2014.{day:03}.{mission_num:02}"] = {
        "program": f"{SIMZOCT2014}",
        "comment": ("SIMZ October 2014 Northern Monterey Bay gulper missions - ctdToUse = ctd2 "),
    }

# ----------------------------- 2015 ---------------------------------------
for day, mission_num in [(132, 4), (148, 1), (156, 0)]:
    dorado_info[f"2015.{day:03}.{mission_num:02}"] = {
        "program": f"{CANONMAY2015}",
        "comment": ("CANON-ECOHAB Spring 2015 - ctdToUse = ctd1 "),
    }
for day, mission_num in [(265, 3), (267, 1), (285, 0), (286, 0), (287, 0)]:
    dorado_info[f"2015.{day:03}.{mission_num:02}"] = {
        "program": f"{CANONSEP2015}",
        "comment": ("CANON September 2015 - ctdToUse = ctd1 "),
    }

# ----------------------------- 2016 ---------------------------------------
dorado_info["2016.090.01"] = {
    "program": f"{CANONOS2016}",
    "comment": ("Mini CANON - 20 gulper fires C1 to M1 - ctdToUse = ctd1 "),
}
dorado_info["2016.161.00"] = {
    "program": DIAMOND,
    "comment": ("Mini CANON - 1/4 mission completed: diamond between C1 and M1 - ctdToUse = ctd1 "),
}
dorado_info["2016.179.01"] = {
    "program": DIAMOND,
    "comment": (
        "Mini CANON - 1/4 mission completed: diamond between C1 and M1, recovered in Monterey"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2016.181.00"] = {
    "program": DIAMOND,
    "comment": ("QC notes: Best CTD is ctd1 - ctdToUse = ctd1 "),
}
dorado_info["2016.270.00"] = {
    "program": f"{CANONSEP2016} {DIAMOND}",
    "comment": ("CANON QC notes: Best CTD is ctd1, no HS2 data - ctdToUse = ctd1 "),
}
dorado_info["2016.307.00"] = {
    "program": DIAMOND,
    "comment": (
        "Successful around-the-bay overnight tow-out mission"
        " QC notes: Best CTD is ctd1"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2016.348.00"] = {
    "program": DIAMOND,
    "comment": (
        "Around-the-bay overnight tow-out mission QC notes: Best CTD is ctd1 - ctdToUse = ctd1 "
    ),
}

# ----------------------------- 2017 ---------------------------------------
dorado_info["2017.044.00"] = {
    "program": DIAMOND,
    "comment": (
        "Around-the-bay overnight tow-out mission - messed up CTD data on ctd1 and ctd2 - flow problems?"
        f" QC notes: Best CTD is ctd1, Temp is bad, ctd2 is bad too (high sediments survey), {ALLSALINITYBAD}"
        " - ctdToUse = ctd2 "
    ),
}
dorado_info["2017.068.00"] = {
    "program": DIAMOND,
    "comment": (
        "Around-the-bay overnight tow-out mission QC notes: Best CTD is ctd1 - ctdToUse = ctd1 "
    ),
}
dorado_info["2017.108.01"] = {
    "program": f"{CANONAPR2017} {DIAMOND}",
    "comment": (
        "Around-the-bay diamond overnight tow-out mission for CANON April 2017"
        " QC notes: Best CTD is ctd1"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2017.121.00"] = {
    "program": f"{CANONAPR2017} {DIAMOND}",
    "comment": (
        "Around-the-bay diamond overnight tow-out mission for CANON April 2017"
        " QC notes: Best CTD is ctd1"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2017.124.00"] = {
    "program": f"{CANONAPR2017} {DIAMOND}",
    "comment": (
        "Around-the-bay diamond overnight tow-out mission for CANON April 2017"
        " QC note: Best CTD is ctd1, not great but ctd2 worse"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2017.157.00"] = {
    "program": f"{CANONPS2017} {DIAMOND}",
    "comment": (
        "Around-the-bay diamond overnight tow-out mission for June 2017"
        " QC note: Best CTD is ctd2"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2017.248.01"] = {
    "program": f"{CANONSEP2017} {DIAMOND}",
    "comment": (
        "Around-the-bay diamond overnight tow-out mission for September 2017"
        " QC note: Best CTD is ctd2"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2017.269.01"] = {
    "program": f"{CANONSEP2017}",
    "comment": (
        "Overnight lawn-mower pattern during CPF deployment for CANON September 2017"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2017.275.01"] = {
    "program": f"{CANONSEP2017} {DIAMOND}",
    "comment": (
        "Overnight diamond pattern during CPF deployment for CANON September 2017"
        " QC note: Best CTD is ctd2"
        " - ctdToUse = ctd1 "
    ),
}
# This mission will not be found in the directory scan because it has been renamed
# to AUVCTD/missionlogs/2017/2017284_incorrect_times_use_2017297
# dorado_info["2017.284.00"] = {
#    "program": f"{CANONSEP2017} {DIAMOND}",
#    "comment": (
#        "Overnight diamond pattern for CANON September 2017"
#        " *** TIME IS WRONG ON THE MVC FOR THE 284.00 mission !!! ***"
#        " Do not use this mission for any analysis - instead use 2017.297.00"
#    ),
# }

dorado_info["2017.297.00"] = {
    "program": f"{CANONSEP2017} {DIAMOND}",
    "comment": (
        "Overnight diamond pattern for CANON September 2017."
        " The 2017.297.00 logs were converted from logs originally collected in 2017.284.00 -"
        " those original logs are in /mbari/AUVCTD/missionlogs/2017/2017284_incorrect_times_use_2017297."
        " Times corrected with auv-python/correct_log_times.py, see"
        " https://bitbucket.org/mbari/auv-python/issues/6/dorado_2017_284_00-clock-is-wrong"
        "QC note: Best CTD is ctd2"
        " - ctdToUse = ctd1 "
    ),
}

dorado_info["2017.304.00"] = {
    "program": f"{DIAMOND}",
    "comment": (
        "Overnight diamond pattern for CANON September 2017"
        " Bad blocks in hs2 data"
        " QC note: Best CTD is ctd2, ctd2 not great but better for salt although a couple screwey profiles in temp"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2017.347.00"] = {
    "program": f"{DIAMOND}",
    "comment": (
        "December Dorado run with 60 cartridge ESP in the water"
        f" QC note: Best CTD is ctd2, ctd2 not great but better for salt although a couple screwey profiles in temp, {ALLSALINITYBAD}"
        " - ctdToUse = ctd1 "
    ),
}

# ----------------------------- 2018 ---------------------------------------
dorado_info["2018.030.00"] = {
    "program": f"{DIAMOND}",
    "comment": (
        "MBTS mission - Overnight diamond in Monterey Bay - Gulpers failed to fire"
        f" QC note: Best CTD is ctd2?, still issues in ctd2 so would loose temp data and would need to be cleaned up for salt, {ALLSALINITYBAD}"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2018.059.00"] = {
    "program": f"{DIAMOND}",
    "comment": (
        "MBTS mission - Overnight diamond in Monterey Bay"
        " QC note: Best CTD is ctd2, still issues in ctd2 so would loose temp data and would need to be cleaned up for salt."
        f" Only the first half is good for ctd2 salt, but 1 is screwy. {ALLSALINITYBAD}"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2018.079.00"] = {
    "program": f"{DIAMOND}",
    "comment": (
        "MBTS mission - Overnight diamond in Monterey Bay - aborted after M1 due to a 'frozen' battery, but good Gulps"
        " Use ctd2 per Monique - 29 July 2021"
        " QC note: Best CTD is ctd2"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2018.099.00"] = {
    "program": f"{DIAMOND}",
    "comment": (
        "MBTS mission - Overnight diamond in Monterey Bay"
        f" QC note: Best CTD is ctd1, ctd1 is bad in salt, ctd2 is worse. {ALLSALINITYBAD}"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2018.156.00"] = {
    "program": f"{CANONMAY2018} {DIAMOND}",
    "comment": (
        "CANON May 2018 - Overnight diamond run"
        f" QC note: Best CTD is ctd2?, marginal improvement, maybe just remove. {ALLSALINITYBAD}"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2018.164.00"] = {
    "program": f"{CANONMAY2018}",
    "comment": ("Criss-cross pattern in Monterey Bay for CANON May 2018 - ctdToUse = ctd1 "),
}
dorado_info["2018.170.00"] = {
    "program": f"{CANONMAY2018}",
    "comment": ("Criss-cross pattern in Monterey Bay for CANON May 2018 - ctdToUse = ctd1 "),
}
dorado_info["2018.191.00"] = {
    "program": f"{CANONMAY2018} {DIAMOND}",
    "comment": (
        "MBTS mission - Overnight diamond in Monterey Bay while mpm was in Cologne, Germany"
        " QC note: Best CTD is ctd1, not great but probably sufficiently OK"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2018.220.00"] = {
    "program": f"{DIAMOND}",
    "comment": (
        "MBTS mission - Overnight diamond in Monterey Bay"
        " QC note: Best CTD is ctd2"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2018.253.00"] = {
    "program": f"{DIAMOND}",
    "comment": (
        "MBTS mission - Overnight diamond in Monterey Bay"
        " QC note: Best CTD is ctd2"
        " - ctdToUse = ctd1 "
    ),
}

# ----------------------------- 2019 ---------------------------------------
dorado_info["2019.029.00"] = {
    "program": f"{DIAMOND}",
    "comment": (
        "MBTS mission - Overnight diamond in Monterey Bay"
        " QC note: Best CTD is ctd2"
        " - ctdToUse = ctd2 "
    ),
}
dorado_info["2019.042.00"] = {
    "program": f"{DIAMOND}",
    "comment": (
        "MBTS mission - Overnight diamond in Monterey Bay"
        " QC note: Best CTD is ctd2, marginal improvement but a bit better"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2019.066.02"] = {
    "program": f"{DIAMOND}",
    "comment": (
        "MBTS mission - Overnight diamond in Monterey Bay"
        " QC note: Best CTD is ctd2, marginal improvement but a bit better"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2019.093.01"] = {
    "program": f"{DIAMOND}",
    "comment": (
        "MBTS mission - Overnight diamond in Monterey Bay"
        " QC note: Best CTD is ctd2?, ctd2 temp & salt bad for the last 1/3 of survey, ctd1 salt bad at all times"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2019.176.00"] = {
    "program": f"{DIAMOND}",
    "comment": (
        "MBTS mission - Overnight diamond in Monterey Bay"
        " QC note: Best CTD is ctd2"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2019.196.04"] = {
    "program": f"{DIAMOND}",
    "comment": (
        "MBTS mission - Overnight diamond in Monterey Bay"
        " QC note: Best CTD is ctd2, some bad ctd2 data around profiles 250-300 but ctd1 salt is really bad"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2019.219.00"] = {
    "program": f"{DIAMOND}",
    "comment": (
        "MBTS mission - Overnight diamond in Monterey Bay"
        " QC note: Best CTD is ctd1, ctd1 better but still bad"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2019.276.02"] = {
    "program": f"{CANONFALL2019}",
    "comment": (
        "Mission for CANON Fall 2019 around DEIMOS - no water collected by Gulpers"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2019.303.00"] = {
    "program": f"{DIAMOND}",
    "comment": (
        "Monterey Bay Diamond Mission - aMBTS1909 - ISUS data incomplete due to full memory card"
        " QC note: Best CTD is ctd2?, remove at least last part (good until profile ~ 240) - ctd2 may be marginally better but not enough to reprocess"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2019.316.01"] = {
    "program": f"{DIAMOND}",
    "comment": (
        "Monterey Bay Diamond Mission - aMBTS1910 (Bad raw conductivity data - probably had some intense jelly action)"
        " QC note: Best CTD is ctd2, remove last profiles"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2019.350.01"] = {
    "program": f"{DIAMOND}",
    "comment": (
        "Aborted Monterey Bay Diamond Mission - 10 Gulpers taken, though - aMBTS1911*"
        " QC note: Best CTD is ctd1"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2019.351.00"] = {
    "program": f"{DIAMOND}",
    "comment": (
        "Monterey Bay Diamond Mission during LRAUV ESP test deployments - aMBTS1911"
        " QC note: Best CTD is ctd1"
        " - ctdToUse = ctd1 "
    ),
}

# ----------------------------- 2020 ---------------------------------------
dorado_info["2020.006.06"] = {
    "program": f"{DIAMOND}",
    "comment": (
        "Monterey Bay Diamond Mission - aMBTS2001 QC note: Best CTD is ctd1 - ctdToUse = ctd1 "
    ),
}
dorado_info["2020.035.00"] = {
    "program": f"{DIAMOND}",
    "comment": (
        "Monterey Bay Diamond Mission - aMBTS2002"
        " QC note: Best CTD is ctd1, remove last profiles"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2020.064.10"] = {
    "program": f"{DIAMOND}",
    "comment": (
        "Monterey Bay Diamond Mission - aMBTS2003 QC note: Best CTD is ctd1 - ctdToUse = ctd1 "
    ),
}
dorado_info["2020.218.03"] = {
    "program": f"{DIAMOND}",
    "comment": (
        "Monterey Bay Diamond Mission - looks like gulps were adaptive sampled - CN20S-2"
        " QC note: Best CTD is ctd1"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2020.231.01"] = {
    "program": f"{DIAMOND}",
    "comment": (
        "Monterey Bay Diamond Mission - 23120G"
        " QC note: Best CTD is ctd1, a few profiles should be removed (around 280-300)"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2020.233.14"] = {
    "program": f"{AUVCTD}",
    "comment": ("Overnight compass evaluation mission - ctdToUse = ctd1 "),
}
dorado_info["2020.245.00"] = {
    "program": f"{DIAMOND}",
    "comment": (
        "MBTS mission - Overnight diamond in Monterey Bay - 24520G - first one with ecopuck instrument: FLBBCD2K -"
        " had to restart mission, 2 missions (245.00 and 246.01) required to complete the diamond"
        " QC note: Best CTD is ctd1"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2020.246.01"] = {
    "program": f"{DIAMOND}",
    "comment": (
        "MBTS mission - Overnight diamond in Monterey Bay - 24520G - first one with ecopuck instrument: FLBBCD2K -"
        " had to restart mission, 2 missions (245.00 and 246.01) required to complete the diamond"
        " QC note: Best CTD is ctd1"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2020.282.01"] = {
    "program": f"{CANONOCT2020} {DIAMOND}",
    "comment": (
        "Monterey Bay Diamond Mission - 28220G QC note: Best CTD is ctd2 - ctdToUse = ctd1 "
    ),
}
dorado_info["2020.286.00"] = {
    "program": f"{CANONOCT2020} {DIAMOND}",
    "comment": (
        "CANON CN20F mission - Overnight diamond in Monterey Bay - The vehicle was stuck in the surface at the northern waypoint and at the end of the mission."
        " The port side CTD (ctd2 - ctdDriver2.log) had a lot of sand in the tube, also found sand in the LISST."
        " QC note: Best CTD is ctd1"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2020.301.03"] = {
    "program": f"{CANONOCT2020} {DIAMOND}",
    "comment": (
        "Post CANON CN20F mission - Overnight diamond in Monterey Bay - HS2 turned off - 30120G"
        " QC note: Best CTD is ?, a few screwy profiles in each"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2020.308.01"] = {
    "program": f"{DIAMOND}",
    "comment": (
        "Post post CANON CN20F mission - Overnight 48 hour diamond in Monterey Bay - 30820G"
        " QC note: Best CTD is ctd1, I think ctd1 is better?? not by much."
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2020.314.01"] = {
    "program": f"{DIAMOND}",
    "comment": (
        "Monterey Bay Diamond Mission - 31420G"
        " QC note: Best CTD is ctd1, likely screwy between 210-240 or so but ctd2 worse anyways"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2020.323.01"] = {
    "program": f"{AUVCTD}",
    "comment": (
        "Engineering test mssion out to M2 and back, with nightime profile at M1 and dayttime profiles at C1 and M2."
        " Gulper samples were discarded. No HS2 (this will be getting reinstalled before next run with new calibrations)."
        " We were carrying the FLBB instrument for fluorescence and backscatter. No water flow through LOPC or CDOM"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2020.335.00"] = {
    "program": f"{DIAMOND}",
    "comment": (
        "Monterey Bay Diamond Mission - 33520G. The hs2 instrument is returned from the vendor with new calibrations in the hs2Calibration.dat file."
        " The lisst-100x instrument was not running due to a cabling issue. Inlet tubes for both CTDs were clogged with sediment upon recovery."
        f" QC note: Best CTD is none, temp is bad, {ALLSALINITYBAD}"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2020.337.00"] = {
    "program": f"{MBTSLINE}",
    "comment": (
        "Monterey Bay MBTS Mission - 33720G. 45 hour mission to M2 and back. No lisst data."
        " Possible plumbing issue with the CTD2 chain, which includes CTD2, DO, and the ISUS."
        " The tube between the DO and the ISUS was poorly seated."
        " - ctdToUse = ctd1 "
    ),
}

# ----------------------------- 2021 ---------------------------------------
dorado_info["2021.102.02"] = {
    "program": f"{CANONAPR2021} {DIAMOND}",
    "comment": (
        "CANON CN21S mission - Monterey Bay Diamond Mission - 10221G - ISUS sampling at 10 seconds"
        " QC note: Best CTD is ctd2"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2021.104.00"] = {
    "program": f"{CANONAPR2021} {DIAMOND}",
    "comment": (
        "CANON CN21S mission - Monterey Bay Diamond Mission - 10421G"
        " - Visibly clogged port side intake, starboard CTD had a hose that came apart"
        " - ISUS sampling at 10 seconds"
        f" QC note: Best CTD is ctd2, iffy - bad after profile ~ 280 in T&S, not great in salt after ~ 190 and suspiscious around ~ 70-120, {ALLSALINITYBAD}"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2021.109.00"] = {
    "program": f"{CANONAPR2021} {DIAMOND}",
    "comment": (
        "CANON CN21S mission - Monterey Bay Diamond Mission - 10921G - ISUS sampling at 10 seconds"
        " QC note: Best CTD is ctd1, iffy, consider removing if bad results"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2021.111.00"] = {
    "program": f"{CANONAPR2021} {DIAMOND}",
    "comment": (
        "CANON CN21S mission - Monterey Bay Diamond Mission - 11121G - No isus data; instrument did not talk during checkout - it was turned off."
        " QC note: Best CTD is ctd2, bad salt, {ALLSALINITYBAD}"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2021.139.00"] = {
    "program": f"{DIAMOND}",
    "comment": (
        "Monterey Bay Diamond Mission - 13921G - No isus data"
        " QC note: Best CTD is ctd1, a bit iffy but may be OK in 2sec resolution"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2021.153.00"] = {
    "program": f"{DIAMOND}",
    "comment": (
        "Monterey Bay Diamond Mission - 15321G - ISUS sampling at 10 seconds"
        " QC note: Best CTD is ctd1"  # REMOVE color in Google sheet, but just a few bad profiles
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2021.181.01"] = {
    "program": f"{DIAMOND}",
    "comment": (
        "Monterey Bay Diamond Mission - 18121G - No isus data"
        " QC note: Best CTD is ctd1"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2021.278.00"] = {
    "program": f"{DIAMOND}",
    "comment": (
        "Monterey Bay Diamond Mission - 27821G - Some issues with the DVL, the vehicle hit the bottom several times, both CTD inlets clogged with mud upon recovery."
        " QC note: Best CTD is ctd1"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2021.301.00"] = {
    "program": f"{DIAMOND}",
    "comment": (
        "Monterey Bay Diamond Mission - 30121G - Vehicle running with 2 new CTDs, no evidence of mud in inlets"
        " QC note: Best CTD is none, temp is bad, according to Mike, {ALLSALINITYBAD}"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2021.334.00"] = {
    "program": f"{DIAMOND}",
    "comment": (
        "Monterey Bay Diamond Mission - 33421G - ctd1 data bad the entire survey, ctd2 data becomes bad 65 km into the survey"
        " - ctdToUse = ctd1 "
    ),
}

# ----------------------------- 2022 ---------------------------------------
dorado_info["2022.006.00"] = {
    "program": f"{DIAMOND}",
    "comment": ("Monterey Bay Diamond Mission - 00622G - ctdToUse = ctd1 "),
}
dorado_info["2022.054.01"] = {
    "program": f"{DIAMOND}",
    "comment": ("Monterey Bay Diamond Mission - 05422G - ctdToUse = ctd1 "),
}
dorado_info["2022.075.00"] = {
    "program": f"{DIAMOND}",
    "comment": ("Monterey Bay Diamond Mission - 07522G - ctdToUse = ctd1 "),
}
dorado_info["2022.110.13"] = {
    "program": f"{DIAMOND}",
    "comment": ("Monterey Bay Diamond Mission - 11022G - ctdToUse = ctd1 "),
}
dorado_info["2022.138.00"] = {
    "program": f"{DIAMOND}",
    "comment": ("Monterey Bay Diamond Mission - 13822G - ctdToUse = ctd1 "),
}
dorado_info["2022.158.00"] = {
    "program": f"{DIAMOND}",
    "comment": ("Monterey Bay Diamond Mission - 15822G - ctdToUse = ctd1 "),
}
dorado_info["2022.201.00"] = {
    "program": f"{DIAMOND}",
    "comment": ("Monterey Bay Diamond Mission - 20122G - ctdToUse = ctd1 "),
}
dorado_info["2022.243.00"] = {
    "program": f"{DIAMOND}",
    "comment": (
        "Monterey Bay Diamond Mission - 24322G - First time using integrated Dorado code"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2022.284.00"] = {
    "program": f"{CANONOCT2022} {DIAMOND}",
    "comment": (
        "Monterey Bay Diamond Mission - 28422G - During CANON October 2022 campaign"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2022.286.01"] = {
    "program": f"{CANONOCT2022}",
    "comment": (
        "Trip out to MARS location near where makai_ESPmv1 sampled - 28622G - During CANON October 2022 campaign"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2022.313.01"] = {
    "program": f"{DIAMOND}",
    "comment": ("Monterey Bay Diamond Mission - 31322G - ctdToUse = ctd1 "),
}
dorado_info["2022.348.00"] = {
    "program": f"{DIAMOND}",
    "comment": ("Monterey Bay Diamond Mission - 34822G - ctdToUse = ctd1 "),
}

# ----------------------------- 2023 ---------------------------------------
dorado_info["2023.018.03"] = {
    "program": f"{DIAMOND}",
    "comment": ("Monterey Bay Diamond Mission - 01823G - ctdToUse = ctd1 "),
}
dorado_info["2023.046.06"] = {
    "program": f"{DIAMOND}",
    "comment": ("Monterey Bay Diamond Mission - 04623G - ctdToUse = ctd1 "),
}
dorado_info["2023.123.00"] = {
    "program": f"{DIAMOND}",
    "comment": ("Monterey Bay Diamond Mission - 12323G - ctdToUse = ctd1 "),
}
dorado_info["2023.132.02"] = {
    # On May 12, 2023, at 1:01 PM, Erik Trauschke <etrauschke@mbari.org> wrote:
    #
    # Hi Mike,
    #
    # I just dropped a set of log files into your temp box (2023.132.02). Can you check if they pas cleanly through your script?
    #
    # Thanks
    # Erik
    "program": TEST,
    "comment": ("Test by Erik Trauschke of new config files - see email from Erik on 2023-05-12"),
}
dorado_info["2023.159.00"] = {
    "program": f"{DIAMOND}",
    "comment": ("Monterey Bay Diamond Mission - 15923G - ctdToUse = ctd1 "),
}
dorado_info["2023.192.01"] = {
    "program": f"{DIAMOND}",
    "comment": (
        "Monterey Bay Diamond Mission - 19223G"
        " Vehicle did no diving on second half of mission due to hardware issue"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2023.254.00"] = {
    "program": f"{DIAMOND}",
    "comment": ("Monterey Bay Diamond Mission - 25423G - ctdToUse = ctd1 "),
}
dorado_info["2023.285.01"] = {
    "program": f"{DIAMOND}",
    "comment": ("Monterey Bay Diamond Mission - 28523G - ctdToUse = ctd1 "),
}
dorado_info["2023.324.00"] = {
    "program": f"{DIAMOND}",
    "comment": ("Monterey Bay Diamond Mission - 32423G - ctdToUse = ctd1 "),
}
dorado_info["2023.346.00"] = {
    "program": f"{DIAMOND}",
    "comment": ("Monterey Bay Diamond Mission - 34623G - ctdToUse = ctd1 "),
}
dorado_info["2024.023.00"] = {
    "program": f"{DIAMOND}",
    "comment": (
        "Monterey Bay Diamond Mission - 02324G ISUS instrument back on board - ctdToUse = ctd1 "
    ),
}
dorado_info["2024.046.00"] = {
    "program": f"{DIAMOND}",
    "comment": (
        "Monterey Bay Diamond Mission - 04624G"
        " Disabled ISUS as it had issues during checkout"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2024.072.01"] = {
    "program": f"{DIAMOND}",
    "comment": (
        "Monterey Bay Diamond Mission - 07224G"
        " ISUS removed, Biolume flow sensor appears to have stopped working"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2024.107.02"] = {
    "program": f"{DIAMOND}",
    "comment": (
        "Monterey Bay Diamond Mission - 10724G"
        " No water collected due to software problem"
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2024.144.05"] = {
    "program": f"{DIAMOND}",
    "comment": ("Monterey Bay Diamond Mission - 14424G - ctdToUse = ctd1 "),
}
dorado_info["2024.170.00"] = {
    "program": f"{DIAMOND}",
    "comment": (
        "Monterey Bay Diamond Mission - 17024G"
        " The UBAT is still reporting 0 flow but collecting good data. The LISST is still deactivated. "
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2024.205.00"] = {
    "program": f"{DIAMOND}",
    "comment": (
        "Monterey Bay Diamond Mission - 20524G"
        " Biolume and LISST payloads removed, noticed the AUV has a slight roll/list post the reballast. "
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2024.226.00"] = {
    "program": f"{DIAMOND}",
    "comment": (
        "Monterey Bay Diamond Mission - 22624G"
        " Biolume, ISUS, and LISST payloads removed, speed test done on last yoyo, decreased from 1.5 to 1.4 m/s. "
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2024.255.01"] = {
    "program": f"{DIAMOND}",
    "comment": (
        "Monterey Bay Diamond Mission - 25524G"
        " Biolume, ISUS, and LISST payloads removed "
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2024.303.00"] = {
    "program": f"{DIAMOND}",
    "comment": (
        "Monterey Bay Diamond Mission - 30324G"
        " Biolume, ISUS, and LISST payloads removed "
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2024.317.01"] = {
    "program": f"{DIAMOND}",
    "comment": (
        "Monterey Bay Diamond Mission - 31724G"
        " Biolume, ISUS, and LISST payloads removed "
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2024.351.00"] = {
    "program": f"{MBTSLINE}",
    "comment": (
        "Monterey Bay MBTS Mission - 35124G"
        " Biolume, ISUS, and LISST payloads removed, "
        " Stations C1->M1->M2 and back. Gulper #7 did not fire. "
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2025.008.00"] = {
    "program": f"{MBTSLINE}",
    "comment": (
        "Monterey Bay MBTS Mission - 00825G"
        " Biolume, ISUS, and LISST payloads removed "
        " - ctdToUse = ctd1 "
    ),
}
dorado_info["2025.036.00"] = {
    "program": f"{MBTSLINE}",
    "comment": (
        "Monterey Bay MBTS Mission - 00365G"
        " Biolume, ISUS, and LISST payloads removed "
        " - ctdToUse = ctd1 "
    ),
}
