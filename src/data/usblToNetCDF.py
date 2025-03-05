#!/usr/bin/env python
__author__ = "Mike McCann"
__version__ = "$Revision: 1.2 $".split()[1]
__date__ = "$Date: 2010/08/24 18:58:19 $".split()[1]
__copyright__ = "2009"
__license__ = "GPL v3"
__contact__ = "mccann at mbari.org"
__idt__ = "$Id: usblToNetCDF.py,v 1.2 2010/08/24 18:58:19 ssdsadmin Exp $"

__doc__ = """

Convert usbl.dat file that Erich provided fromt he Gunter in the Gulf into
a NetCDF file that can be read by the doradosdp software.

@var __date__: Date of last cvs commit
@undocumented: __doc__ parser
@status: In development
@license: GPL
"""

import datetime
import logging
import sys
import time
from optparse import OptionParser
from pathlib import Path

import Nio

#
# Make a global logging object.
#
h = logging.StreamHandler()
f = logging.Formatter("%(levelname)s %(asctime)s %(message)s")
##f = logging.Formatter("%(levelname)s %(asctime)s %(funcName)s %(lineno)d %(message)s")	# Python 2.4 does not support funcName :-(
h.setFormatter(f)

logger = logging.getLogger("logger")
logger.addHandler(h)

##logging.getLogger("logger").setLevel(logging.DEBUG)
logging.getLogger("logger").setLevel(logging.INFO)


#
# Variables global to this module
#
verboseFlag = False
debugLevel = 0


def processRecords(aFile, ncFile):
    """Read record from ascii ubl file, parse it and write netCDF records"""

    count = 0
    for rec in aFile.readlines():
        logger.debug("processRecords(): rec = " + rec)
        items = rec.split(",")

        # Parse date/time
        logger.debug("processRecords(): date = %s" % items[1].split(" ")[0])
        mo = int(items[1].split(" ")[0].split("-")[0])
        da = int(items[1].split(" ")[0].split("-")[1])
        yr = int(items[1].split(" ")[0].split("-")[2]) + 2000
        logger.debug("processRecords(): time = %s" % items[1].split(" ")[1])
        hr = int(items[1].split(" ")[1].split(":")[0])
        mn = int(items[1].split(" ")[1].split(":")[1])
        se = int(items[1].split(" ")[1].split(":")[2].split(".")[0])
        us = int(items[1].split(" ")[1].split(":")[2].split(".")[1]) / 10 * 1000000  # microseconds

        t = datetime.datetime(yr, mo, da, hr, mn, se, us)
        t = t - datetime.timedelta(hours=5)  # Try subtracking 5 hours to make local time GMT
        esec = time.mktime(t.timetuple())
        logger.debug("processRecords(): esec = %s" % esec)

        # Parse Latitude (ship index = 2, auv index = 22)
        latD = float(items[22].split(" ")[0][1:])
        latFrac = float(items[22].split(" ")[1]) / 60.0
        lat = latD + latFrac
        if items[22].split(" ")[0][0] == "S":
            lat = -lat
        logger.debug("processRecords(): lat = %f" % lat)

        # Parse Longitude (ship index = 3, auv index = 23)
        lonD = float(items[23].split(" ")[0][1:])
        lonFrac = float(items[23].split(" ")[1]) / 60.0
        lon = lonD + lonFrac
        if items[23].split(" ")[0][0] == "W":
            lon = -lon
        logger.debug("processRecords(): lon = %f" % lon)

        # Write the record
        writeNetCDFRecord(ncFile, count, esec, lon, lat)
        count += 1


def openNetCDFFile(ncFileName):
    """Open netCDF file and write some global attributes and dimensions that
    we know at the beginning of processing.

    """

    logger.info("openNetCDFFile(): Will output NetCDF file to %s" % ncFileName)

    #
    # Set the PreFill option to False to improve writing performance
    #
    opt = Nio.options()
    opt.PreFill = False

    #
    # Set the history attribute
    #
    hatt = "Created by usblToNetCDF.py on " + time.ctime()

    #
    # Open the netCDF file and set some global attributes
    #
    ncFile = Nio.open_file(ncFileName, "c", opt, hatt)

    missionName = ""
    try:
        missionName = ncFileName.split("/")[-2]
    except IndexError:
        logger.warning(
            "openNetCDFFile(): Could not parse missionName from netCDF file name - probably not production processing."
        )

    logger.info("openNetCDFFile(): missionName = %s" % missionName)

    ncFile.title = "USBL data from AUV mission " + missionName
    ncFile.institution = "Monterey Bay Aquarium Research Institute"
    ncFile.summary = "These data have been processed from the original USBL.  The data in this file are to be considered as simple time series data only and are as close to the original data as possible. "
    ncFile.keywords = "Ultra Short Baseline, Navigation"
    ncFile.Conventions = "CF-1.1"
    ncFile.standard_name_vocabulary = "CF-1.1"

    # Time dimension is unlimited, we'll write variable records then write the time data at the end of processing
    ncFile.create_dimension("time", None)  # Unlimited dimension

    ncFile.create_variable("time", "d", ("time",))
    ncFile.variables["time"].units = "seconds since 1970-01-01 00:00:00"
    ncFile.variables["time"].long_name = "Time GMT"

    # --- Create the record variables ---
    logger.info("openNetCDFFile(): Creating variable latitude on axes time ")
    ncFile.create_variable("latitude", "f", ("time",))
    ncFile.variables["latitude"].units = "degrees_north"

    logger.info("openNetCDFFile(): Creating variable longitude on axes time ")
    ncFile.create_variable("longitude", "f", ("time",))
    ncFile.variables["longitude"].units = "degrees_east"

    return ncFile  # End openNetCDFFile()


def writeNetCDFRecord(ncFile, indx, esec, lon, lat):
    """Write records to the unlimited (time) dimension"""

    ncFile.variables["time"][indx] = esec
    ncFile.variables["longitude"][indx] = lon
    ncFile.variables["latitude"][indx] = lat


def closeNetCDFFile(ncFile):
    """Write the time axis data, additional global attributes and close the file"""

    ncFile.close()

    return ncFile  # End closeNetCDFFile()


def main():
    """Main routine: Parse command line options and call unpack and write functions.

    Example:

    >>> usblToNetCDF.py -i /mbari/AUVCTD/missionlogs/2010/2010151/2010.151.04/usbl.dat \
        -n /mbari/ssdsdata/ssds/generated/netcdf/files/ssds.shore.mbari.org/auvctd/missionlogs/2010/2010151/2010.151.04/usbl.nc

    """

    parser = OptionParser(
        usage="""\
Unpack data records from ascii USBL file and produce NetCDF file

Usage: %prog -i <ascii_fileName> -n <netCDF_fileName> [-d <level> -f]

Where:
	<ascii_fileName> is name of ascii usbl  file to process
	<netCDF_fileName> is the name of the NetCDF file to create

Options:
	-v:         verbose output
	-d <level>: debugging output (higher the numbe the more the output)
	-f:         Force removal of output files


Examples:
	usblToNetCDF.py -i /mbari/AUVCTD/missionlogs/2010/2010151/2010.151.04/usbl.dat -n /mbari/ssdsdata/ssds/generated/netcdf/files/ssds.shore.mbari.org/auvctd/missionlogs/2010/2010151/2010.151.04/usbl.nc

"""
    )
    parser.add_option(
        "-i",
        "--ascii_fileName",
        type="string",
        action="store",
        help="Specify an input binary data file name, e.g. /mbari/AUVCTD/missionlogs/2009/2009084/2009.084.00/lopc.bin.",
    )
    parser.add_option(
        "-n",
        "--netCDF_fileName",
        type="string",
        action="store",
        help="Specify a fully qualified output NetCDF file path, e.g. /mbari/tempbox/mccann/lopc/2009_084_00lopc.nc",
    )
    parser.add_option(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Turn on verbose output, which gives bit more processing feedback.",
    )
    parser.add_option(
        "-d",
        "--debugLevel",
        action="store",
        type="int",
        help="Turn on debugLevel [1..3].  The higher the number, the more the output.  E.g. us '-d 1' to understand framing errors, '-d 3' for all debug statements.",
    )
    parser.add_option(
        "-f",
        "--force",
        action="store_true",
        default=False,
        help="Force overwite of netCDF file.  Useful for using from a master reprocess script.",
    )
    opts, args = parser.parse_args()

    #
    # unpack data according to command line options
    #
    if opts.ascii_fileName and opts.netCDF_fileName:
        start = time.time()

        # Set global output flags
        global verboseFlag
        global hasMBARICframeData
        verboseFlag = opts.verbose
        if opts.debugLevel:
            logging.getLogger("logger").setLevel(logging.DEBUG)

        if Path(opts.netCDF_fileName).exists():
            if opts.force:
                if Path(opts.netCDF_fileName).exists():
                    Path(opts.netCDF_fileName).unlink()
            else:
                ans = input(
                    opts.netCDF_fileName
                    + " file exists.\nDo you want to remove it and continue processing? (y/[N]) "
                )
                if ans.upper() == "Y":
                    Path(opts.netCDF_fileName).unlink()
                else:
                    sys.exit(0)

        logger.info("main(): Processing begun: %s" % time.ctime())

        # Open input file
        asciiFile = open(opts.ascii_fileName)

        # Open output file
        ncFile = openNetCDFFile(opts.netCDF_fileName)

        # Read records and write netCDF records
        processRecords(asciiFile, ncFile)

        # Close the netCDF file writing the proper tsList data first
        closeNetCDFFile(ncFile)

        logger.info("main(): Created file: %s" % opts.netCDF_fileName)

        mark = time.time()
        logger.info(
            "main(): Processing finished: %s Elapsed processing time from start of processing = %d seconds"
            % (time.ctime(), (mark - start))
        )

    else:
        print("\nCannot process input.  Execute with -h for usage note.\n")


# Allow this file to be imported as a module
if __name__ == "__main__":
    main()
