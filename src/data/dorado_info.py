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
DIAMOND = "Monterey Bay Diamond"
CANON = "CANON"
CANONFALL2019 = "CANON Fall 2019"

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


dorado_info["2019.276.02"] = {
    "program": CANONFALL2019,
    "comment": (
        "Mission for CANON Fall 2019 around DEIMOS" "- no water collected by Gulpers"
    ),
}
