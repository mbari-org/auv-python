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
AUVCTD = "AUVCTD"
AOSN = "AOSN"
DIAMOND = "Monterey Bay Diamond"
CANON = "CANON"
CANONSEP2010 = "CANON September 2010"
CANONOCT2010 = "CANON October 2010"
CANONAPR2011 = "CANON April 2011"
CANONSEP2011 = "CANON September 2011"
CANONMAY2012 = "CANON May 2012"
CANONSEP2012 = "CANON September 2012"
CANONMAR2013 = "CANON March 2013"
CANONSEP2013 = "CANON September 2013"
CANONAPR2014 = "CANON April 2014"
CANONSEP2014 = "CANON September 2014"
CANONMAY2015 = "CANON May 2015"
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
CANONOCT2022 = "CANON October 2022"

# ----------------------------- 2003 ---------------------------------------
dorado_info = {}
for mission_number in range(2, 14):
    dorado_info[f"2003.106.{mission_number:02d}"] = {
        "program": OCCO,
        "comment": (
            "OCCO2003106: missions {02,03,04,05,06,07,08,09,10,11,12,13}"
            " - Very first Dorado missions in the archive - does not process with legacy Matlab code"
        ),
    }
for mission_number in range(3, 8):
    dorado_info[f"2003.111.{mission_number:02d}"] = {
        "program": OCCO,
        "comment": (
            "OCCO2003111: missions {03,04,05,06,07}"
            " - Second set of Dorado missions in the archive - does not process with legacy Matlab code"
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
        ),
    }
for mission_number in range(2, 5):
    dorado_info[f"2003.118.{mission_number:02d}"] = {
        "program": OCCO,
        "comment": (
            "OCCO2003118: missions {02,03,04}"
            " - 2003.118 Hydroscat cal changes - does not process with legacy Matlab code"
        ),
    }
# ------------ Lagacy Matlab processing works for these missions on... ---------------
dorado_info["2003.125.02"] = {
    "program": AUVCTD,
    "comment": (
        "Requested by Patrick McEnaney 5/20/05 (deviceIDs copied from later mission)"
        " - Needed to copy ctdDriver.cfg and ctdDriver2.cfg from 2003.207.07 too"
        " - calibrations may NOT be right"
    ),
}
for mission_number in range(2, 5):
    dorado_info[f"2003.164.{mission_number:02d}"] = {
        "program": OCCO,
        "comment": ("OCCO2003164: missions {02,03,04}"),
    }
for mission_number in range(2, 5):
    dorado_info[f"2003.167.{mission_number:02d}"] = {
        "program": OCCO,
        "comment": ("OCCO2003167: missions {02,03,04}"),
    }
for mission_number in range(4, 7):
    dorado_info[f"2003.169.{mission_number:02d}"] = {
        "program": OCCO,
        "comment": ("OCCO2003169: missions {04,05,06}"),
    }
for mission_number in range(3, 6):
    dorado_info[f"2003.174.{mission_number:02d}"] = {
        "program": OCCO,
        "comment": ("OCCO2003174: missions {03,04,05}"),
    }
for mission_number in range(2, 5):
    dorado_info[f"2003.176.{mission_number:02d}"] = {
        "program": OCCO,
        "comment": ("OCCO2003176: missions {02,03,04}"),
    }
dorado_info["2003.222.02"] = {
    "program": AUVCTD,
    "comment": (
        "Portaled to SSDS on 12 March 2014, but did not process for missing hydroscat cal file"
    ),
}
dorado_info["2003.223.02"] = {
    "program": AUVCTD,
    "comment": (
        "Re-portaled 7/1/05, added oxygen coefficients to ctdDriver2.cfg on 11 March 2015 - no change in oxygen values"
    ),
}
dorado_info["2003.224.05"] = {
    "program": BIOLUME,
    "comment": ("Re-portaled 7/1/05: 2 separate surveys close to the beach"),
}
dorado_info["2003.224.08"] = {
    "program": AUVCTD,
    "comment": ("Re-portaled 7/1/05: 2 separate surveys close to the beach"),
}
dorado_info["2003.225.02"] = {
    "program": BIOLUME,
    "comment": ("Re-portaled 7/7/05"),
}
dorado_info["2003.226.02"] = {
    "program": BIOLUME,
    "comment": ("Re-portaled 7/7/05"),
}
dorado_info["2003.226.05"] = {
    "program": AUVCTD,
    "comment": ("Re-portaled 7/7/05"),
}
dorado_info["2003.227.02"] = {
    "program": BIOLUME,
    "comment": ("Re-portaled 7/7/05"),
}
dorado_info["2003.228.02"] = {
    "program": BIOLUME,
    "comment": ("Re-portaled 7/7/05"),
}
dorado_info["2003.228.05"] = {
    "program": AOSN,
    "comment": ("Re-portaled 7/7/05"),
}
for mission_number in range(2, 6):
    dorado_info[f"2003.230.{mission_number:02d}"] = {
        "program": AUVCTD,
        "comment": ("Re-portaled 7/7/05: maybe these 4 missions are one survey ?"),
    }
dorado_info["2003.232.02"] = {
    "program": AUVCTD,
    "comment": ("Re-portaled 7/8/05"),
}
for mission_number in range(2, 7):
    dorado_info[f"2003.233.{mission_number:02d}"] = {
        "program": OCCO,
        "comment": ("Re-portaled 7/8/05.  Increased tDiffCrit to 800 on 8/3/06"),
    }
dorado_info["2003.234.02"] = {
    "program": AUVCTD,
    "comment": ("Re-portaled 7/8/05"),
}
for mission_number in range(3, 5):
    dorado_info[f"2003.237.{mission_number:02d}"] = {
        "program": OCCO,
        "comment": ("OCCO2003237: missions {03,04}: Re-portaled 7/11/05"),
    }
dorado_info["2003.238.02"] = {
    "program": AUVCTD,
    "comment": ("Re-portaled 7/11/05"),
}
dorado_info["2003.241.02"] = {
    "program": AUVCTD,
    "comment": ("Re-portaled 7/11/05"),
}
for mission_number in range(2, 5):
    dorado_info[f"2003.244.{mission_number:02d}"] = {
        "program": OCCO,
        "comment": ("OCCO2003245: missions {02,03,04}: Re-portaled 7/11/05"),
    }
dorado_info["2003.246.02"] = {
    "program": OCCO,
    "comment": ("OCCO2003246: missions {02}"),
}
for mission_number in range(2, 4):
    dorado_info[f"2003.280.{mission_number:02d}"] = {
        "program": AUVCTD,
        "comment": ("Portaled 7/11/05"),
    }
dorado_info["2003.281.02"] = {
    "program": AUVCTD,
    "comment": ("Portaled 7/11/05"),
}
for mission_number in range(2, 4):
    dorado_info[f"2003.308.{mission_number:02d}"] = {
        "program": AUVCTD,
        "comment": ("Portaled 7/12/05"),
    }
for mission_number in [0, 1, 2, 4]:
    dorado_info[f"2003.309.{mission_number:02d}"] = {
        "program": AUVCTD,
        "comment": ("Portaled 7/12/05"),
    }
for mission_number in range(2, 5):
    dorado_info[f"2003.317.{mission_number:02d}"] = {
        "program": "DMOSeaTrials",
        "comment": ("Portaled 7/12/05"),
    }
dorado_info["2003.336.02"] = {
    "program": BIOLUME,
    "comment": ("Portaled 7/12/05"),
}
dorado_info["2003.337.04"] = {
    "program": BIOLUME,
    "comment": ("Portaled 7/12/05"),
}
for mission_number in range(15, 23):
    dorado_info[f"2003.338.{mission_number:02d}"] = {
        "program": BIOLUME,
        "comment": ("Portaled 7/12/05"),
    }
dorado_info["2003.339.04"] = {
    "program": BIOLUME,
    "comment": ("Portaled 7/12/05"),
}
dorado_info["2003.340.02"] = {
    "program": BIOLUME,
    "comment": ("Portaled 7/12/05"),
}

# ----------------------------- 2004 ---------------------------------------
dorado_info["2004.028.05"] = {
    "program": AUVCTD,
    "comment": (
        "AUVCTD: Processed 10/4/04 - no metadata loaded anywhere, re-portaled & processed 7/26/05"
    ),
}
dorado_info["2004.029.03"] = {
    "program": AUVCTD,
    "comment": (
        "2004.029.03: parosci interp1 problem - multigenerateNetcdf failed to create navigation.nc file because of '#' problem"
        "# begin"
        "#<binary_data>"
        "Will need to rerun the portal with the fix to set byteOffset."
        "Re-portaled & processed 7/26/05"
        "According to GPS data AUV is in Mexico beginning: 29 Jan 2004 19:37:36 GMT	1.0754050561900759E9	0.4733113223544617"
        "*** Need to reprocess with just the .03 mission and not include the data equal to and after this time ***"
        "Had bad nav point - processed 8/10/05"
    ),
}
dorado_info["2004.029.05"] = {
    "program": AUVCTD,
    "comment": ("Re-portaled & processed 7/26/05"),
}


# ----------------------------- 2016 ---------------------------------------
dorado_info["2016.161.00"] = {
    "program": DIAMOND,
    "comment": ("Mini CANON - 1/4 mission completed: diamond between C1 and M1"),
}
dorado_info["2016.179.01"] = {
    "program": DIAMOND,
    "comment": (
        "Mini CANON - 1/4 mission completed: diamond between C1 and M1, recovered in Monterey"
    ),
}
dorado_info["2016.181.00"] = {
    "program": DIAMOND,
    "comment": (""),
}
dorado_info["2016.270.00"] = {
    "program": f"{CANONSEP2016} {DIAMOND}",
    "comment": ("CANON"),
}
dorado_info["2016.307.00"] = {
    "program": DIAMOND,
    "comment": ("Successful around-the-bay overnight tow-out mission"),
}
dorado_info["2016.348.00"] = {
    "program": DIAMOND,
    "comment": ("Around-the-bay overnight tow-out mission"),
}

# ----------------------------- 2017 ---------------------------------------
dorado_info["2017.044.00"] = {
    "program": DIAMOND,
    "comment": (
        "Around-the-bay overnight tow-out mission - messed up CTD data on ctd1 and ctd2 - flow problems?"
    ),
}
dorado_info["2017.068.00"] = {
    "program": DIAMOND,
    "comment": ("Around-the-bay overnight tow-out mission"),
}
dorado_info["2017.108.01"] = {
    "program": f"{CANONAPR2017} {DIAMOND}",
    "comment": ("Around-the-bay diamondovernight tow-out mission for CANON April 2017"),
}
dorado_info["2017.121.00"] = {
    "program": f"{CANONAPR2017} {DIAMOND}",
    "comment": ("Around-the-bay diamondovernight tow-out mission for CANON April 2017"),
}
dorado_info["2017.124.00"] = {
    "program": f"{CANONAPR2017} {DIAMOND}",
    "comment": ("Around-the-bay diamondovernight tow-out mission for CANON April 2017"),
}
dorado_info["2017.157.00"] = {
    "program": f"{CANONPS2017} {DIAMOND}",
    "comment": ("Around-the-bay diamondovernight tow-out mission for June 2017"),
}
dorado_info["2017.248.01"] = {
    "program": f"{CANONSEP2017} {DIAMOND}",
    "comment": ("Around-the-bay diamondovernight tow-out mission for September 2017"),
}
dorado_info["2017.269.01"] = {
    "program": f"{CANONSEP2017}",
    "comment": (
        "Overnight lawn-mower pattern during CPF deployment for CANON September 2017"
    ),
}
dorado_info["2017.275.01"] = {
    "program": f"{CANONSEP2017} {DIAMOND}",
    "comment": (
        "Overnight diamond pattern during CPF deployment for CANON September 2017"
    ),
}
dorado_info["2017.284.00"] = {
    "program": f"{CANONSEP2017} {DIAMOND}",
    "comment": (
        "Overnight diamond pattern for CANON September 2017"
        "*** TIME IS WRONG ON THE MVC FOR THE 284.00 mission !!! ***"
    ),
}
dorado_info["2017.304.00"] = {
    "program": f"{CANONSEP2017} {DIAMOND}",
    "comment": ("Overnight diamond pattern for CANON September 2017"),
}
dorado_info["2017.347.00"] = {
    "program": f"{CANONSEP2017} {DIAMOND}",
    "comment": ("December Dorado run with 60 cartridge ESP in the water"),
}

# ----------------------------- 2018 ---------------------------------------

# ----------------------------- 2019 ---------------------------------------
dorado_info["2019.276.02"] = {
    "program": CANONFALL2019,
    "comment": (
        "Mission for CANON Fall 2019 around DEIMOS - no water collected by Gulpers"
    ),
}
