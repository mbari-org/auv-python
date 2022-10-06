#!/usr/bin/env python
__author__ = "Mike McCann"
__version__ = "$Revision: 1.43 $".split()[1]
__date__ = "$Date: 2020/11/23 21:40:04 $".split()[1]
__copyright__ = "2009"
__license__ = "GPL v3"
__contact__ = "mccann at mbari.org"
__doc__ = """
Unpack Brooke Ocean Technology Laser Optical Plankton Counter binary data from
lopc.bin files logged on the vehicle.  These files are the BOT binary data stream
format as described in Section 6.2 of the LOPC Software Operation Manual. They
do not follow the same format as the control system software generated .log 
files that are processed by the SSDS-Java code (auvportal) therefore there is no
compulsion to use that same framework for processing these data.

This Python script is the shortest path to a file conversion.

Examples:

    1. A short mission useful for testing:
    lopcToNetCDF.py -i /mbari/AUVCTD/missionlogs/2009/2009084/2009.084.00/lopc.bin -n /mbari/ssdsdata/ssds/generated/netcdf/files/ssds.shore.mbari.org/auvctd/missionlogs/2009/2009084/2009.084.00/lopc.nc

    2. A complete around the bay survey:
    lopcToNetCDF.py -i /mbari/AUVCTD/missionlogs/2009/2009084/2009.084.02/lopc.bin -n /mbari/ssdsdata/ssds/generated/netcdf/files/ssds.shore.mbari.org/auvctd/missionlogs/2009/2009084/2009.084.02/lopc.nc
    
    3. A communications fixed short mission C Frame epoch seconds with output to ASCII text file:
    lopcToNetCDF.py -i /mbari/AUVCTD/missionlogs/2010/2010083/2010.083.08/lopc.bin -n /mbari/ssdsdata/ssds/generated/netcdf/files/ssds.shore.mbari.org/auvctd/missionlogs/2010/2010083/2010.083.08/lopc.nc -t lopc_2010_083_08.dat -v -f 

"""

import argparse
import logging
import math
import os
import re
import string
import struct
import sys
import time
from netCDF4 import Dataset

import numpy

import lopcMEP

#
# Exceptions
#
class BeginLFrameWithoutEndOfPreviousFrame(Exception):
    pass


class BeginMFrameWithoutEndOfPreviousFrame(Exception):
    pass


class EndOfFrameException(Exception):
    pass


class EndOfFileException(Exception):
    pass


class ShortLFrameError(Exception):
    pass


class GarbledLFrame(Exception):
    pass


class LOPC_Processor(object):
    #
    # Make a global logging object.
    #
    h = logging.StreamHandler()
    f = logging.Formatter(
        "%(levelname)s %(asctime)s %(filename)s "
        "%(funcName)s():%(lineno)d %(message)s"
    )
    ##f = logging.Formatter("%(levelname)s %(asctime)s %(funcName)s %(lineno)d %(message)s")    # Python 2.4 does not support funcName :-(
    h.setFormatter(f)

    logger = logging.getLogger("logger")
    logger.addHandler(h)
    _log_levels = (logging.WARN, logging.INFO, logging.DEBUG)

    ##logging.getLogger("logger").setLevel(logging.DEBUG)
    logging.getLogger("logger").setLevel(logging.INFO)

    #
    # Memebr Variables
    #
    debugLevel = 0
    hasMBARICframeData = False  # includes timing, depth, profile information after a certain data - flag used in netCDF functions
    WWbinningInterval = 1  # Test value
    binningInterval = 20  # 10 second binning interval for calculated MEP data output

    missing_value = -9999  # Used in netCDF file

    dataStructure = dict()

    dataStructureLongName = dict()
    dataStructureLongName[
        "countShortLFrameError"
    ] = "Accumulated count of L frames that did not have the full 128 bins"
    dataStructureLongName[
        "countBeginLFrameWithoutEndOfPreviousFrame"
    ] = "Accumulated count of L frames begun before an end of the previous frame"
    dataStructureLongName[
        "countBeginMFrameWithoutEndOfPreviousFrame"
    ] = "Accumulated count of M frames begun before an end of the previous frame"
    dataStructureLongName[
        "countUnknownFrameCharacter"
    ] = "Accumulated count of unknown frames"
    dataStructureLongName[
        "countGarbledLFrame"
    ] = "Accumulated count of garbled L frames"
    dataStructureLongName["flowSpeed"] = "Flow Speed"

    LframeScalarKeys = [
        "snapshot",
        "threshold",
        "sampleCount",
        "flowCount",
        "flowTime",
        "bufferStatus",
        "laserLevel",
        "counter",
        "counterPeriod",
        "laserControl",
    ]
    LframeScalarDict = {
        "snapshot": None,
        "threshold": None,
        "sampleCount": None,
        "flowCount": None,
        "flowTime": None,
        "bufferStatus": None,
        "laserLevel": None,
        "counter": None,
        "counterPeriod": None,
        "laserControl": None,
    }
    LframeScalarDictUnits = {
        "snapshot": None,
        "threshold": "microns",
        "sampleCount": None,
        "flowCount": None,
        "flowTime": "seconds",
        "bufferStatus": None,
        "laserLevel": None,
        "counter": None,
        "counterPeriod": "seconds",
        "laserControl": None,
    }
    LframeScalarDictLongName = {
        "snapshot": "1: snapshot in progress, 0: not in progress",
        "threshold": "lower limit on signal level detection",
        "sampleCount": "Number of sample",
        "flowCount": "Number of flow counts in .5 second period",
        "flowTime": "Accumulated time for all counts",
        "bufferStatus": "1: buffer overrun, 0: no overrun",
        "laserLevel": "Mean laser intensity",
        "counter": "Number of pulses detected on the counter input",
        "counterPeriod": "Period of the pulses detected on the counter input",
        "laserControl": "Configured set value for the Laser Control Algorithm",
    }

    MEPDataDictKeys = [
        "sepCountList",
        "mepCountList",
        "sepCountListSum",
        "mepCountListSum",
        "countList",
        "LCcount",
        "transCount",
        "nonTransCount",
        "ai_mean",
        "ai_min",
        "ai_max",
        "ai_std",
        "esd_mean",
        "esd_min",
        "esd_max",
        "esd_std",
    ]
    MEPDataDictUnits = {
        "sepCountList": "count",
        "mepCountList": "count",
        "sepCountListSum": "count",
        "mepCountListSum": "count",
        "countList": "count",
        "LCcount": "count",
        "transCount": "count",
        "nonTransCount": "count",
        "ai_mean": "",
        "ai_min": "",
        "ai_max": "",
        "ai_std": "",
        "esd_mean": "microns",
        "esd_min": "microns",
        "esd_max": "microns",
        "esd_std": "microns",
    }
    MEPDataDictLongName = {
        "sepCountList": "Single Element Particle counts by size class",
        "mepCountList": "Multiple Element Particle counts by size class",
        "sepCountListSum": "Sum of Single Element Particle counts",
        "mepCountListSum": "Sum of Multiple Element Particle counts",
        "countList": "Total Particle counts by size class",
        "LCcount": "Large Copepod count",
        "transCount": "Transparent particle count",
        "nonTransCount": "Non-Transparent particle count",
        "ai_mean": "Attenuation Index mean",
        "ai_min": "Attenuation Index minimum",
        "ai_max": "Attenuation Index maximum",
        "ai_std": "Attenuation Index Standard Deviation",
        "esd_mean": "Equivalent Spherical Diameter mean",
        "esd_min": "Equivalent Spherical Diameter minimum",
        "esd_max": "Equivalent Spherical Diameter ",
        "esd_std": "Equivalent Spherical Diameter Index Standard Deviation",
        "esd_mean": "microns",
        "esd_min": "microns",
        "esd_max": "microns",
        "esd_std": "microns",
    }

    frameText = ""

    countBeginLFrameWithoutEndOfPreviousFrame = 0
    countBeginMFrameWithoutEndOfPreviousFrame = 0
    countUnknownFrameCharacter = 0
    countGarbledLFrame = 0
    unknownFrameCharacters = []
    ncFile = None
    sampleCountList = []

    def checkForDLE(self, file, char):
        """Logic that is applied to each character read if a <DLE> ('~') is encounered"""
        nextChar = file.read(1)
        if self.args.debugLevel >= 3:
            self.logger.debug("nextChar = %s" % nextChar)
        if nextChar == b"*":
            if self.args.debugLevel >= 5:
                self.logger.debug("nextChar is *; this is the end of frame")
            self.frameText += nextChar.decode("utf-8")
            raise EndOfFrameException

        elif nextChar == b"~":
            self.frameText += nextChar.decode("utf-8")
            if self.args.debugLevel >= 5:
                self.logger.debug(
                    "nextChar is ~; skipping the second <DLE> that is added by the instrument"
                )

        elif nextChar == b"M":
            self.frameText += nextChar.decode("utf-8")
            if self.args.debugLevel >= 3:
                self.logger.warn(
                    "Detected beginning of new M frame without a detection of the end of the previous frame."
                )
            if self.args.debugLevel >= 3:
                self.logger.warn("self.frameText = \n" + self.frameText)
            raise BeginMFrameWithoutEndOfPreviousFrame

        elif nextChar == b"L":
            self.frameText += nextChar.decode("utf-8")
            if self.args.debugLevel >= 3:
                self.logger.warn(
                    "Detected beginning of new L frame without a detection of the end of the previous frame."
                )
            if self.args.debugLevel >= 3:
                self.logger.warn("self.frameText = \n" + self.frameText.decode("utf-8"))
            raise BeginLFrameWithoutEndOfPreviousFrame

        else:
            self.frameText += str(ord(nextChar))
            if self.args.debugLevel >= 3:
                self.logger.warn(
                    "this is not expected.  If we have a '~' then there should be either a '~' or  '*' following it."
                )
            if self.args.debugLevel >= 3:
                self.logger.warn("self.frameText = \n" + self.frameText)
            if self.args.debugLevel >= 3:
                self.logger.warn("nextChar = " + nextChar)
            self.countUnknownFrameCharacter += 1
            self.unknownFrameCharacters.append(str(ord(nextChar)))
            input(
                "Encountered unexpected area of code.  Please notify Mike McCann.  Press Enter to continue..."
            )

    def readBigEndianUShort(self, binFile):
        """Read next 16-bits from binFile and return byte-swapped unsiged short integer
        value.  If end of frame is detected ('~*') the raise an EndOfFrameException()

        """

        c1 = binFile.read(1)
        if len(c1) == 0:
            raise EndOfFileException

        if self.args.debugLevel >= 5:
            self.logger.debug("c1 = %s" % ord(c1))
        if c1 == b"~":
            self.frameText += c1.decode("utf-8")
            self.checkForDLE(binFile, c1.decode("utf-8"))
        else:
            self.frameText += str(c1[0])

        c2 = binFile.read(1)
        if len(c2) == 0:
            raise EndOfFileException

        if self.args.debugLevel >= 5:
            self.logger.debug("c2 = %s" % c2[0])
        if c2 == b"~":
            self.frameText += c2.decode("utf-8")
            self.checkForDLE(binFile, c2.decode("utf-8"))
        else:
            self.frameText += str(c2[0])

        value = struct.unpack("H", c2 + c1)[0]
        if self.args.debugLevel >= 5:
            self.logger.debug("value = %i" % value)

        return int(value)  # End readBigEndianUShort()

    def readChar(self, binFile):
        """Read next 8-bit character from binFile.  Handles checking for DLE.
        If end of frame is detected ('~*') the raise an EndOfFrameException()

        """
        c = binFile.read(1).decode("utf-8")
        if len(c) == 0:
            raise EndOfFileException
        if self.args.debugLevel >= 5:
            self.logger.debug("c = %s" % ord(c))
        if c == "~":
            self.frameText += c
            self.checkForDLE(binFile, c)
        else:
            self.frameText += str(ord(c))

        return c  # End readChar()

    def readLframeData(self, binFile, sampleCountList):
        """For L frame data read the integer count data from the 128 bins.
        Then read the scalar engineering values that follow.  Dorado must be
        running an LOPC that is v2.36 or greater as the test file I've tried
        (2009.084.00/lopc.bin) has 275 bytes in the L frame.
        Return: countList and dictionary of engineering values.

        """

        if self.args.debugLevel >= 2:
            self.logger.debug("-" * 81)
        if self.args.debugLevel >= 2:
            self.logger.debug("frameID = L: Read Binned Count Data")
        countList = []
        # Expecting to read 128 values.  The call to readBigEndianUShort() may raise an EndOfFrameException, in which case
        # is caught by the routine that calls this function and is an error.
        for i in range(128):
            try:
                value = self.readBigEndianUShort(binFile)
            except EndOfFrameException:
                if self.args.debugLevel >= 1:
                    self.logger.warn(
                        "Reached the end of this L frame before the end (# 127) at bin # %d."
                        % i
                    )
                if self.args.debugLevel >= 1:
                    self.logger.warn("L Frame text = \n" + self.frameText + "\n")
                if self.args.debugLevel >= 1:
                    self.logger.warn("countList = \n" + str(countList))
                if self.args.debugLevel >= 1:
                    self.logger.warn("Raising ShortLFrameError.")
                raise ShortLFrameError

            countList.append(value)

            if self.args.debugLevel >= 4:
                self.logger.debug("bin: %i count = %i" % (i, value))

        # Peek at last several values in countList for really bin values that indicates a garbled frame
        # Tests indicate the the scalar data are corrupted on frame that meet this criteria.  Let's just discard them.
        for c in countList[101:128]:
            if c > 1000:
                ##raw_input("Paused with too high a value at red-end of the countList, c = %d" % c)
                if self.args.debugLevel >= 1:
                    self.logger.warn(
                        "Detected garbled frame with count value = %d between indices [101:128] of countList"
                        % c
                    )
                raise GarbledLFrame

        # Detect partial frames and raise exception
        if len(countList) < 128:
            raise ShortLFrameError

        # Read the rest of the values from the L data frame - use a List to maintain order and a Dict to store the values
        ##logging.getLogger("self.logger").setLevel(logging.DEBUG)
        if self.args.debugLevel >= 2:
            self.logger.debug("-" * 80)
        try:
            if self.args.debugLevel >= 2:
                self.logger.debug("Reading scalar data:")
            for var in self.LframeScalarKeys:
                # All but counter are 2 bytes
                if var == "counter":
                    b1 = binFile.read(1)
                    try:
                        val = ord(b1.decode("utf-8"))
                    except UnicodeDecodeError:
                        val = ord(b1)
                    self.frameText += str(val)
                else:
                    val = self.readBigEndianUShort(binFile)

                if val == None:
                    self.logger.warn(
                        "L frame scalar: %s is None (unable to parse)" % var
                    )
                else:
                    try:
                        # Perform QC checks on threshold and bufferStatus, save sampleCount in List that is used for timeStamp
                        if self.args.debugLevel >= 2:
                            self.logger.debug("  %s = %d" % (var, val))
                        self.LframeScalarDict[var] = val
                        if var == "threshold" and val != 100:
                            if self.args.debugLevel >= 1:
                                self.logger.warn(
                                    "Detected garbled frame with threshold != 100 (%d)"
                                    % val
                                )
                            raise GarbledLFrame
                        if var == "sampleCount":
                            ##try:
                            ##  if abs(val - sampleCountList[-1]) > 100:
                            ##      raw_input("Paused at samplewCount excursion...")
                            ##except IndexError:
                            ##  pass

                            # Append to list that is passed in and returned in call argument list
                            sampleCountList.append(val)
                        if var == "bufferStatus" and (val != 0 and val != 1):
                            if self.args.debugLevel >= 1:
                                self.logger.warn(
                                    "Detected garbled frame with bufferStatus != 0|1 (%d)"
                                    % val
                                )
                            raise GarbledLFrame
                    except TypeError:
                        self.logger.error(
                            "TypeError in trying to print variable '%s' as an integer val = '%s'"
                            % (var, val)
                        )

        except EndOfFrameException:
            if self.args.debugLevel >= 1:
                self.logger.warn(
                    "Reached the end of this L frame while attempting to read the LframeScalar data."
                )
            if self.args.debugLevel >= 1:
                self.logger.warn(
                    "self.LframeScalarDict = " + str(self.LframeScalarDict)
                )
            return (countList, self.LframeScalarDict)

        # If we read one more character we should get an EndOfFrameException - make sure this happens
        endOfFrame = False
        while not endOfFrame:
            if self.args.debugLevel >= 2:
                self.logger.debug("Inside 'while not endOfFrame:' loop")
            try:
                # Read one byte at a time checking for EndOfFrame
                val = self.readBigEndianUShort(binFile)
            except EndOfFrameException:
                # Catch the EndOfFrameException and deal with it gracefully
                if self.args.debugLevel >= 2:
                    self.logger.debug("Reached the end of this L frame.")
                endOfFrame = True

        if self.args.debugLevel >= 2:
            self.logger.debug("L Frame text = \n" + self.frameText + "\n")

        if self.args.debugLevel >= 2:
            self.logger.debug("-" * 80)

        return (countList, self.LframeScalarDict)  # End readLframeData()

    def readMframeData(self, binFile):
        """For M frame data read the repeated 6-byte chunks of 3 16-bit blocks."""

        if self.args.debugLevel >= 2:
            self.logger.debug("-" * 80)
        if self.args.debugLevel >= 2:
            self.logger.debug(
                "frameID = M: Read Mulit-Element Particle Frame Format Data"
            )

        groupCount = 0
        mepCount = 0
        partMepCount = 0
        maxPartMepCount = 0

        # Lists local to this function whose items  get appended to lists maintained the function that calls this one
        nL = []
        pL = []
        eL = []
        lL = []
        sL = []

        try:
            while True:
                # Read packed MEP data as 3 short ints
                A = self.readBigEndianUShort(binFile)
                B = self.readBigEndianUShort(binFile)
                C = self.readBigEndianUShort(binFile)
                if self.args.debugLevel >= 4:
                    self.logger.debug(
                        "readMframeData():\n\tA = %d\n\tB = %d\n\tC = %d" % (A, B, C)
                    )

                # Use bit masks to read data MEP values
                n = (A & 0x00008000) >> 15  # bit to indicate new MEP or not
                p = (A & 0x00003FF8) >> 3  # peak intensity aka DS
                e = ((A & 0x00000007) << 3) + ((B & 0x00003800) >> 11)  # element number
                l = B & 0x000007FF  # length, aka time of flight
                s = C & 0x0000FFFF  # scan counter time reference
                if self.args.debugLevel >= 2:
                    self.logger.debug(
                        "readMframeData():n = %d, p = %5d, e = %2d, l = %5d, s = %5d"
                        % (n, p, e, l, s)
                    )

                nL.append(n)
                pL.append(p)
                eL.append(e)
                lL.append(l)
                sL.append(s)

                if n == 1:
                    mepCount += 1
                    partMepCount = 0
                elif n == 0:
                    partMepCount += 1

                groupCount = groupCount + 1
                if partMepCount > maxPartMepCount:
                    maxPartMepCount = partMepCount

        except EndOfFrameException:
            if self.args.debugLevel >= 4:
                self.logger.debug(
                    "Reached the end of this M frame after reading %d groups."
                    % groupCount
                )

        if self.args.debugLevel >= 4:
            self.logger.debug("M Frame text = \n" + self.frameText + "\n")

        return (nL, pL, eL, lL, sL, mepCount)  # End readMframeData()

    def flowSpeed(self, flowTime, flowCount):
        """Compute flow speed through LOPC cell following LOPC analysis methods (Herman 2009)
        Input: flowTime and flowCount scalar integer values
        Returns: flow speed in m/s or self.missing_value if speed cannot be calculated

        """

        # Prevent divide by 0
        if flowCount == 0:
            return self.missing_value

        # compute and screen flow count
        FC = float(flowTime) / float(flowCount)  # mean flow count

        if FC > 100:  # regression is not valid for FC>100 (flow speed likeley -> 0)
            return self.missing_value

        elif FC < 7:  # unlikeley that AUV can go faster than 2 m/s
            return self.missing_value

        elif FC <= 13:  # case 1 (FC<=13)
            idx = 0
            # 2nd element of regression coeffs to use

        elif FC > 13:  # case2 (FC>13)
            idx = 1
            # 2nd element of regression coeffs to use

        # regression coeffs
        a = numpy.array(
            [
                [23.10410966, 0.198996019],
                [-1.481499106, -2.603059062],
                [1.566460406, 0.892897609],
                [0.196311142, 0.006191239],
                [-0.05, -0.0013],
            ]
        )

        sqrtFC = math.sqrt(FC)
        if self.args.debugLevel >= 2:
            self.logger.debug(
                "sqrtFC = %f, idx = %d, a[1,idx] = %f " % (sqrtFC, idx, a[1, idx])
            )
        speed = a[0, idx] * math.exp(
            -(
                a[1, idx] * math.sqrt(sqrtFC)
                + a[2, idx] * sqrtFC
                + a[3, idx] * sqrtFC**2
                + a[4, idx] * sqrtFC**3
            )
        )

        if self.args.debugLevel >= 1:
            self.logger.debug("speed = %f " % speed)

        return speed

    def readCframeData(self, binFile):
        """For C frame data - Parses special MBARI C frame data record that Hans writes from the MVC"""

        if self.args.debugLevel >= 2:
            self.logger.debug("-" * 80)
        if self.args.debugLevel >= 2:
            self.logger.debug(
                "frameID = C: Read CTD Data (timestamp that the mvc writes)"
            )

        nChar = 0
        str = ""
        try:
            while True:
                # Read characters until end of frame
                char = self.readChar(binFile)
                str += char
                if self.args.debugLevel >= 2:
                    self.logger.debug("char = %s" % (char))
                nChar += 1

        except EndOfFrameException:
            if self.args.debugLevel >= 2:
                self.logger.debug(
                    "Reached the end of this C frame after reading %d characters. str = %s"
                    % (nChar, str)
                )
            # Parse for epoch seconds that Hans writes, starting after 15 March 2010 (day 074) - hasMBARICframeData flag set in main() and used by other functions
            if self.hasMBARICframeData:
                try:
                    esecs = float(str.split(" ")[0])
                except ValueError:
                    str = "".join(
                        s for s in str if s in string.printable
                    )  # Remove nonprintable characters
                    str = re.sub("\s+", "", str)  # REmove whitespace
                    self.logger.warn(
                        "Unable to parse a float from the string str[:70] = %s"
                        % str[:70]
                    )
                    esecs = self.missing_value

                if self.args.debugLevel >= 2:
                    self.logger.debug("Parse from C Frame esecs = %f" % esecs)
                return esecs
            else:
                self.logger.info(
                    "This mission is before esecs were written to the C frame.  No attempt made to parse."
                )
                return self.missing_value

    def writeTextLFrame(self, countList, textFile):
        """Write the L frame in the ASCII text format that is produced by the LOPC software
        and can be processed by Alex Herman's post processing software.

        """

        textFile.write("L1")
        for i in range(32):
            textFile.write(" %d" % countList[i])
        textFile.write("\n")

        textFile.write("L2")
        for i in range(32, 64):
            textFile.write(" %d" % countList[i])
        textFile.write("\n")

        textFile.write("L3")
        for i in range(64, 96):
            textFile.write(" %d" % countList[i])
        textFile.write("\n")

        textFile.write("L4")
        for i in range(96, 128):
            textFile.write(" %d" % countList[i])
        textFile.write("\n")

        textFile.write("L5")
        for k in self.LframeScalarKeys:
            textFile.write(" %d" % self.LframeScalarDict[k])
        textFile.write("\n")

        return  # End writeTextLFrame()

    def unpackLOPCbin(self, binFile, opts, textFile=None):
        """Loop though the .bin file reading all frames and unpacking as necessary.  Keep
        track of framing errors.  This function does not exit with a return, it reads
        records until an EndOfFileException is raised, which is then caught by the
        routine that calls this.  For this reason we need to assign to global variables
        in order to return values.
        Note that the binary data are written by the LOPC instrument in big-endian. At
        MBARI all of our Linux systems are little-endian.  We need to swap bytes on all
        the binary words.

        """

        self.logger.info(">>> Unpacking LOPC data from " + binFile.name)
        self.logger.info(">>> Writing NetCDF file " + self.args.netCDF_fileName)

        if textFile != None:
            self.logger.info(">>> Writing ASCII text file " + self.args.text_fileName)

        recCount = 0
        self.dataStructure["lFrameCount"] = 0
        self.dataStructure["mFrameCount"] = 0

        # Array for the size class dimension [108 microns to 1.5 cm] - for both SEP and MEP data
        self.dataStructure["binSizeList"] = numpy.array(
            list(range(108, 15015, 15)), dtype="float32"
        )

        # Instantiate MepData object to collect and compute MEP parameters
        mepData = lopcMEP.MepData()

        maxLenNList = 0

        countShortLFrameError = 0

        detectedLFrame = False
        detectedMFrame = False

        cFrameEsecs = self.missing_value
        lastCFrameEsecs = self.missing_value

        lastTime = time.time()  # For measuring elapsed time in verbose output in loop
        lastLframeCount = 0  # For binningInterval output

        sepCountArraySum = numpy.zeros(
            len(self.dataStructure["binSizeList"]), dtype="int32"
        )
        lastMEP = None  # For properly building the MEP list (for transferring partial frames between binning intervals)
        # The sampleCount from the instrument - used for getting accurate timeStamp
        sampleCountList = []
        # For keeping track of which frames (telling us the time) were written to the NetCDF file - this starts conting at 1
        lFrameCountWrittenList = []
        # For keeping track of which frames (telling us the time) were written to the NetCDF file - to better calculate time...
        sampleCountWrittenList = []
        # For recording 'ground-truth' time from the MVC
        cFrameEsecsList = []

        outRecNumFunc = (
            self.record_count().__next__
        )  # Generator of indices for netCDF output

        # write header for ASCII text file, if specified
        if textFile != None:
            textFile.write("# ___ PROCESSING INFORMATION ___\n")
            textFile.write(
                "# Date: %s\n" % time.strftime("%A, %B %d, %Y", time.localtime())
            )
            textFile.write("# Time: %s\n" % time.strftime("%H:%M:%S", time.localtime()))

        while True:
            """Outer read loop to read all records from the file. The loop is exited by an EndOfFileException which is handled
            by whatever calls this method."""
            if detectedLFrame:
                frameID = "L"
            if detectedMFrame:
                frameID = "M"
            else:
                while True:
                    "Read characters until we reach the first <DLE> ('~'), then read another char to get the frameID"
                    char = binFile.read(1)
                    # char = struct.unpack("B", binFile.read(1))[0]
                    if self.args.debugLevel >= 3:
                        self.logger.debug(
                            "char = %s, len(char) = %d" % (char, len(char))
                        )
                    if len(char) == 0:
                        raise EndOfFileException
                    if char == b"~":
                        frameID = binFile.read(1).decode("utf-8")
                        if self.args.debugLevel >= 2:
                            self.logger.debug("-" * 80)
                        if self.args.debugLevel >= 2:
                            self.logger.debug(
                                "frameID = %s (lFrameCount = %d, mFrameCount = %d)"
                                % (
                                    frameID,
                                    self.dataStructure["lFrameCount"],
                                    self.dataStructure["mFrameCount"],
                                )
                            )
                        break  # Exit the enclosing 'while True:' loop

            if frameID == "L":
                # Initialize global debugging text string representation of the data frame
                self.frameText = str(char) + str(frameID)
                try:
                    (countList, self.LframeScalarDict) = self.readLframeData(
                        binFile, sampleCountList
                    )
                except BeginLFrameWithoutEndOfPreviousFrame:
                    detectedLFrame = True
                    if self.args.debugLevel >= 1:
                        self.logger.error(
                            "Begin of L Frame (around # %d) witout encountering end of previous frame."
                            % self.dataStructure["lFrameCount"]
                        )
                    self.countBeginLFrameWithoutEndOfPreviousFrame += 1
                    continue  # exit to outer while True loop
                except BeginMFrameWithoutEndOfPreviousFrame:
                    self.countBeginMFrameWithoutEndOfPreviousFrame += 1
                    if self.args.debugLevel >= 1:
                        self.logger.error(
                            "Begin of M Frame (around # %d) witout encountering end of previous frame."
                            % self.dataStructure["lFrameCount"]
                        )
                    detectedMFrame = True
                    continue  # exit to outer while True loop
                except ShortLFrameError:
                    if self.args.debugLevel >= 1:
                        self.logger.warn(
                            "Reached the end of this L frame (around sample # %d) before expected."
                            % self.dataStructure["lFrameCount"]
                        )
                    countShortLFrameError += 1
                    continue

                except GarbledLFrame:
                    self.countGarbledLFrame += 1
                    continue
                else:
                    detectedLFrame = False

                # Save values specific to this L frame
                self.dataStructure["countShortLFrameError"] = countShortLFrameError
                self.dataStructure[
                    "countBeginLFrameWithoutEndOfPreviousFrame"
                ] = self.countBeginLFrameWithoutEndOfPreviousFrame
                self.dataStructure[
                    "countBeginMFrameWithoutEndOfPreviousFrame"
                ] = self.countBeginMFrameWithoutEndOfPreviousFrame
                self.dataStructure[
                    "countUnknownFrameCharacter"
                ] = self.countUnknownFrameCharacter
                self.dataStructure[
                    "unknownFrameCharacters"
                ] = self.unknownFrameCharacters
                self.dataStructure["countGarbledLFrame"] = self.countGarbledLFrame
                self.dataStructure["sampleCountList"] = sampleCountList
                self.dataStructure["cFrameEsecsList"] = cFrameEsecsList

                # Compute flow speed
                if self.args.debugLevel >= 2:
                    self.logger.debug("Calling self.flowSpeed()")
                self.dataStructure["flowSpeed"] = self.flowSpeed(
                    self.LframeScalarDict["flowTime"],
                    self.LframeScalarDict["flowCount"],
                )
                if self.args.debugLevel >= 2:
                    self.logger.debug(
                        "flowSpeed = %f" % self.dataStructure["flowSpeed"]
                    )

                # Save values specifc to the C frame
                if self.hasMBARICframeData:
                    if cFrameEsecs != lastCFrameEsecs:
                        self.dataStructure["cFrameEsecs"] = cFrameEsecs
                        cFrameEsecsList.append(cFrameEsecs)
                    else:
                        self.dataStructure["cFrameEsecs"] = self.missing_value
                        cFrameEsecsList.append(self.missing_value)

                # Open netCDF file on first Frame
                if self.dataStructure["lFrameCount"] == 0:
                    """Open netCDF file on first read where we get a full count"""
                    self.openNetCDFFile(opts)

                # Confirm contents of Dictionary
                if self.args.debugLevel >= 2:
                    self.logger.debug(
                        "self.LframeScalarDict contents after return from readLframeData():"
                    )
                for k in self.LframeScalarKeys:
                    if self.args.debugLevel >= 2:
                        self.logger.debug("  %s = %d" % (k, self.LframeScalarDict[k]))

                # Write ASCII L frame data, if requested
                if self.args.text_fileName:
                    self.writeTextLFrame(countList, self.LframeScalarDict, textFile)

                # Pad out the list of counts returned by readLframeData so that it's the same as what we write to the netCDF file
                sepCountArray = numpy.zeros(
                    len(self.dataStructure["binSizeList"]), dtype="int32"
                )
                i = 0
                for c in countList:
                    sepCountArray[i] = c
                    i += 1

                # Append these data to the numpy array that collects the SEP counts for each binning interval
                sepCountArraySum += sepCountArray

                # Count each L frame, at 2 hz this is our best estimate of elapsed time
                self.dataStructure["lFrameCount"] += 1
                lastCFrameEsecs = cFrameEsecs

                # Verify the collection: loop through all the saved data structures so far.  This debug output was added to discover
                # the fact that Python appends a reference to the LframeScalarDict dictionary resulting in every item in the list to
                # point to the last LframeScalarDict dictionary returned by readLframeData().  Appending a shallow copy() fixes this.
                if self.args.debugLevel >= 3:
                    self.logger.debug(
                        "Reading data from DataStructure dictionary after appending:"
                    )
                for countList, lframeScalarDict in zip(
                    countList, self.LframeScalarDict
                ):
                    if self.args.debugLevel >= 3:
                        self.logger.debug("\tlen(countList) = %d" % len(countList))
                    if self.args.debugLevel >= 3:
                        self.logger.debug(
                            "\tnumpy.sum(countList) = %d" % numpy.sum(countList)
                        )
                    for item in self.LframeScalarKeys:
                        if self.args.debugLevel >= 3:
                            self.logger.debug(
                                "\t%s = %d" % (item, lframeScalarDict[item])
                            )

                # Verbose output table column header wait until after first frame so that openNetCDFFile ouput is printed first
                if self.args.verbose and self.dataStructure["lFrameCount"] == 100:
                    self.logger.info(
                        ""
                        + "".join(
                            [
                                s.center(12)
                                for s in (
                                    "L Frames",
                                    "sample",
                                    "M Frames",
                                    "ShortLFrame",
                                    "LBeforeMEnd",
                                    "MBeforeLEnd",
                                    "UnknownFrame",
                                    "Garbled L",
                                )
                            ]
                        )
                    )
                    self.logger.info(
                        ""
                        + "".join(
                            [
                                s.center(12)
                                for s in (
                                    "Count",
                                    "Count",
                                    "Count",
                                    "Error",
                                    "Error",
                                    "Error",
                                    "Error",
                                    "Error",
                                )
                            ]
                        )
                    )
                    self.logger.info(
                        ""
                        + "".join(
                            [
                                s.center(12)
                                for s in (
                                    "--------",
                                    "--------",
                                    "--------",
                                    "-----------",
                                    "-----------",
                                    "-----------",
                                    "-----------",
                                    "------------",
                                )
                            ]
                        )
                    )

                # Give a little feedback during this long read process
                # Value of 100 is a lot, 10000 gives about 15 lines for a Diamond mission
                lframe_interval = 10000
                if (
                    self.args.verbose
                    and not self.dataStructure["lFrameCount"] % lframe_interval
                ):
                    lastLFrame = 2 * int(
                        self.dataStructure["tsList"][-1]
                        - self.dataStructure["tsList"][0]
                    )
                    self.logger.info(
                        "%s %.1f seconds, last L Frame: %d"
                        % (
                            "".join(
                                [
                                    str(s).center(12)
                                    for s in (
                                        self.dataStructure["lFrameCount"],
                                        self.LframeScalarDict["sampleCount"],
                                        self.dataStructure["mFrameCount"],
                                        countShortLFrameError,
                                        self.countBeginLFrameWithoutEndOfPreviousFrame,
                                        self.countBeginMFrameWithoutEndOfPreviousFrame,
                                        self.countUnknownFrameCharacter,
                                        self.countGarbledLFrame,
                                    )
                                ]
                            ),
                            time.time() - lastTime,
                            lastLFrame,
                        )
                    )
                    lastTime = time.time()

            elif frameID == "M":
                self.frameText = str(char) + str(
                    frameID
                )  # Initialize global debugging text string representation of the data frame
                try:
                    (nL, pL, eL, lL, sL, mepCount) = self.readMframeData(binFile)
                    ##print "pL = " + str(pL)
                except BeginMFrameWithoutEndOfPreviousFrame:
                    detectedMFrame = True
                    continue  # exit to outer while True loop
                except BeginLFrameWithoutEndOfPreviousFrame:
                    stillHaveLFrame = True
                    continue  # exit to outer while True loop
                else:
                    detectedMFrame = False

                self.dataStructure["mFrameCount"] += 1

                # Collect MEP data in the instance of the lopcMEP.MepData class
                mepData.extend(nL, pL, eL, lL, sL)
                ##if len(mepData.mepList) > 0:
                ##  mepData.build(mepData.mepList[-1])
                ##else:
                ##  mepData.build()
                ##print "mepData = %s\n" % (mepData)

                ##foo = raw_input('Got M frame')

                # Write ASCII M frame data - for just this frame, if requested
                if self.args.text_fileName:
                    textFile.write(mepData.frameToASCII(nL, pL, eL, lL, sL))

            elif frameID == "C":
                self.frameText = str(char) + str(
                    frameID
                )  # Initialize global debugging text string representation of the data frame
                if self.args.debugLevel >= 2:
                    self.logger.debug("C Frame text = \n" + self.frameText + "\n")

                # Note that before 2010074 a C frame is probably garbled data and should not be parsed for esecs.  This logic is in readCframeData().
                try:
                    esecs = self.readCframeData(binFile)
                except BeginLFrameWithoutEndOfPreviousFrame:
                    detectedLFrame = True
                    if self.args.debugLevel >= 1:
                        self.logger.error(
                            "Begin of L Frame (around # %d) witout encountering end of previous frame."
                            % self.dataStructure["lFrameCount"]
                        )
                    self.countBeginLFrameWithoutEndOfPreviousFrame += 1
                    continue  # exit to outer while True loop
                except BeginMFrameWithoutEndOfPreviousFrame:
                    self.countBeginMFrameWithoutEndOfPreviousFrame += 1
                    if self.args.debugLevel >= 1:
                        self.logger.error(
                            "Begin of M Frame (around # %d) witout encountering end of previous frame."
                            % self.dataStructure["lFrameCount"]
                        )
                    detectedMFrame = True
                    continue  # exit to outer while True loop

                if self.args.debugLevel >= 1:
                    self.logger.debug(
                        "C frame after L Frame # %d esecs = %f"
                        % (self.dataStructure["lFrameCount"], esecs)
                    )
                cFrameEsecs = esecs

            elif frameID == "~":
                pass  # Just skip this and look for the next valid frameID

            else:
                if self.args.debugLevel >= 2:
                    self.logger.debug(
                        "Encountered frameID = %s at L frame # %d"
                        % (frameID, self.dataStructure["lFrameCount"])
                    )

            #
            # At binning interval save calculated ESD and other derived products from the MEP data
            #
            if (
                self.dataStructure["lFrameCount"]
                > lastLframeCount + self.binningInterval
            ):
                ##raw_input('Paused at count = %d with binningInterval = %d' % (self.dataStructure['lFrameCount'], binningInterval))

                ##logging.getLogger("self.logger").setLevel(logging.DEBUG)
                ##self.args.debugLevel = 1

                # Build the mepList in mepData from all that data that have been collected so far, this allows .count() to work.
                mepData.build(lastMEP)

                if self.args.debugLevel >= 1:
                    self.logger.debug(
                        "len(mepData.mepList) = %d" % len(mepData.mepList)
                    )

                if self.args.debugLevel >= 2:
                    self.logger.debug("SEP-only counts")
                if self.args.debugLevel >= 2:
                    self.logger.debug("===============")
                if self.args.debugLevel >= 1:
                    self.logger.debug(
                        "SEP-only counts: sepCountArraySum.sum() = %d"
                        % sepCountArraySum.sum()
                    )
                if self.args.debugLevel >= 2:
                    self.logger.debug("sepCountArraySum = %s" % sepCountArraySum)

                # Add the MEP counts to the sepCountArray that has been summed up
                ##logging.getLogger("MEP").setLevel(logging.DEBUG)
                ##self.logger.info("\n\nmepData = \n%s" % (mepData))

                ##countArray = mepData.count(self.dataStructure['binSizeList'], sepCountArraySum)

                mepCountArray = mepData.count(self.dataStructure["binSizeList"])
                lcAIcrit = 0.4
                lcESDmin = 800
                lcESDmax = 1200
                LCcount = mepData.countLC(
                    self.args.LargeCopepod_AIcrit,
                    self.args.LargeCopepod_ESDmin,
                    self.args.LargeCopepod_ESDmax,
                )

                trAIcrit = 0.4
                (transCount, nonTransCount) = mepData.countTrans(trAIcrit)

                countArray = (
                    sepCountArraySum + mepCountArray
                )  # With numpy Arrays we can do element-by-element addition with '+'

                if self.args.debugLevel >= 1:
                    self.logger.debug(
                        "MEP-only counts: mepCountArray.sum() = %d"
                        % mepCountArray.sum()
                    )

                if self.args.debugLevel >= 2:
                    self.logger.debug("After adding MEP counts to SEP counts")
                if self.args.debugLevel >= 2:
                    self.logger.debug("=====================================")
                if self.args.debugLevel >= 1:
                    self.logger.debug(
                        "SEP+MEP counts: countArray.sum() = %d" % countArray.sum()
                    )
                if self.args.debugLevel >= 2:
                    self.logger.debug("countArray = %s" % countArray)
                if self.args.debugLevel >= 1:
                    self.logger.debug("LCcount = %d" % LCcount)
                if self.args.debugLevel >= 1:
                    self.logger.debug(
                        "transCount = %d, nonTransCount = %d"
                        % (transCount, nonTransCount)
                    )
                if mepCountArray.sum() > 0:
                    if self.args.debugLevel >= 1:
                        self.logger.debug(
                            "ai:   mean = %f, min = %f, max = %f, std = %f"
                            % (
                                mepData.aiArray.mean(),
                                mepData.aiArray.min(),
                                mepData.aiArray.max(),
                                mepData.aiArray.std(),
                            )
                        )
                    if self.args.debugLevel >= 1:
                        self.logger.debug(
                            "esd:   mean = %f, min = %f, max = %f, std = %f"
                            % (
                                mepData.esdArray.mean(),
                                mepData.esdArray.min(),
                                mepData.esdArray.max(),
                                mepData.esdArray.std(),
                            )
                        )

                ##logging.getLogger("self.logger").setLevel(logging.INFO)
                ##self.args.debugLevel = 0
                ##raw_input('Paused ')
                ##if nonTransCount > 0:
                ##  raw_input('Paused ')

                # Collect all of the MEP related data and stats into a dictionary to make passing to writeNetCDFRecord() a little easier
                # The corresonding MEPDataDictKeys list needs to be in sync with the keys in this list
                MEPDataDict = {}
                MEPDataDict["sepCountList"] = list(sepCountArraySum)
                MEPDataDict["mepCountList"] = list(mepCountArray)
                MEPDataDict["sepCountListSum"] = numpy.sum(list(sepCountArraySum))
                MEPDataDict["mepCountListSum"] = numpy.sum(list(mepCountArray))
                MEPDataDict["countList"] = list(countArray)

                MEPDataDict["LCcount"] = LCcount
                MEPDataDict["transCount"] = transCount
                MEPDataDict["nonTransCount"] = nonTransCount

                if mepCountArray.sum() != 0:
                    MEPDataDict["ai_mean"] = mepData.aiArray.mean()
                    MEPDataDict["ai_min"] = mepData.aiArray.min()
                    MEPDataDict["ai_max"] = mepData.aiArray.max()
                    MEPDataDict["ai_std"] = mepData.aiArray.std()
                    MEPDataDict["esd_mean"] = mepData.esdArray.mean()
                    MEPDataDict["esd_min"] = mepData.esdArray.min()
                    MEPDataDict["esd_max"] = mepData.esdArray.max()
                    MEPDataDict["esd_std"] = mepData.esdArray.std()

                # Write accumulated counts to the NetCDF file
                # Other data (e.g. C frame esecs) get written too - from the self.dataStructure hash list - missing values filling in the gaps
                self.writeNetCDFRecord(MEPDataDict, outRecNumFunc)

                # Save the LframeCount to use for writing the time axis data when we close the file - it's a python array index, so subtract 1
                ##self.logger.info("appending lFrameCount = %d" % self.dataStructure['lFrameCount'])
                lFrameCountWrittenList.append(self.dataStructure["lFrameCount"] - 1)
                sampleCountWrittenList.append(self.LframeScalarDict["sampleCount"])

                # Save the last MEP (in case it's the beginning of the next one)
                if len(mepData.mepList) > 0:
                    lastMEP = mepData.mepList[-1]

                # Clear MEP lists by creating a new mepData object and zero the SEP count sum list
                mepData = lopcMEP.MepData()
                sepCountArraySum = numpy.zeros(
                    len(self.dataStructure["binSizeList"]), dtype="int32"
                )

                lastLframeCount = self.dataStructure["lFrameCount"]

                self.dataStructure["lFrameCountWrittenList"] = lFrameCountWrittenList
                self.dataStructure["sampleCountWrittenList"] = sampleCountWrittenList
                ##self.logger.info("self.dataStructure['lFrameCountWrittenList'] = " + str(self.dataStructure['lFrameCountWrittenList']))

            # NOTE: This function cannot return any values as it reads records from the input file until
            # the end is trapped with an exception.  That is now this function ends.  This is why assignments
            # are made to the self.dataStructure dictionary - that's how we pass information back.

        return  # End unpackLOPCbin()

    def constructTimestampList(
        self, binFile, recCount=None, sampleCountList=None, cFrameEsecsList=None
    ):
        """Given an absolute path to an lopc.bin file (`binFile`) lookup
        the start and and time from the associated parosci.nc file
        and generate a list of Unix epoch second time values.  If
        `recCount` and `sampleCountList` are provided then construct a
        more accurate time value list based on an index to a 2 Hz time
        base provided by the `sampleCountList`.

        """

        # Get time information from ctd file associated with this mission.  Translate .bin file name to a missionlog netcdf file:
        # Legacy locations:
        #   /mbari/AUVCTD/missionlogs/2009/2009084/2009.084.00/lopc.bin ->
        #   /mbari/ssdsdata/ssds/generated/netcdf/files/ssds.shore.mbari.org/auvctd/missionlogs/2009/2009084/2009.084.00/parosci.nc
        parosci_nc = binFile.name.replace("lopc.bin", "parosci.nc").replace(
            "missionlogs", "missionnetcdfs"
        )
        self.logger.info("parosci_nc = %s" % parosci_nc)
        self.logger.info(
            "Using NetCDF4 to get start and end epoch seconds for this mission from this URL:"
        )
        self.logger.info("%s" % parosci_nc)
        sensor_on_time = 0  # Bogus initial starting time
        sensor_off_time = 10000  # Bogus initial ending time
        try:
            parosci = Dataset(parosci_nc)
        except FileNotFoundError:
            self.logger.error(
                "Could not open %s.  Cannot process until the netCDF file exists.  Exiting."
                % parosci_nc
            )
            sys.exit(-1)
        else:
            # sensor_on_time = parosci.time[0]
            # sensor_off_time = parosci.time[-1]
            sensor_on_time = parosci["time"][0]
            sensor_off_time = parosci["time"][-1]

        self.logger.info(
            "From associated paroscinc file: sensor_on_time = %.1f, sensor_off_time = %.1f"
            % (sensor_on_time, sensor_off_time)
        )
        self.logger.info(
            "Duration is %d seconds.  Expecting to read %d L frames from the lopc.bin file."
            % (
                int(sensor_off_time - sensor_on_time),
                2 * int(sensor_off_time - sensor_on_time),
            )
        )

        if recCount == None:
            recCount = 2 * int(sensor_off_time - sensor_on_time)
            self.logger.info(
                "recCount not passed in, assuming we'll have %d records from the lopc.bin file."
                % (recCount)
            )

        timestampList = []
        self.logger.debug("sampleCountList = %s" % (sampleCountList))
        if sampleCountList != None:
            # Analyze the sampleCountList - correct for 16-bit overflow, and interpolate over 0-values, stuck values, and spikes
            self.logger.info(
                "Calling correctSampleCountList() with sampleCountList[0] = %d and len(sampleCountList) = %d"
                % (sampleCountList[0], len(sampleCountList))
            )
            correctedSampleCountList = self.correctSampleCountList(sampleCountList)
            self.logger.info(
                "correctSampleCountList() returned correctedSampleCountList = %s with len(correctedSampleCountList) = %d"
                % (correctedSampleCountList, len(correctedSampleCountList))
            )

            # SampleCount does not always begin at 0 so subtract the first element's value so that we can use it as a time base
            # Use the instrument concept of sample count to construct the best possible time stamp list
            self.logger.info(
                "Subtracting %d from all values of correctedSampleCountList"
                % correctedSampleCountList[0]
            )
            deltaT = 0.5
            self.logger.info(
                "Constructing timestampArray from instrument corrected sampleCount and constant deltaT = %f"
                % deltaT
            )
            self.logger.info(
                "New re-zeroed correctedSampleCountList = %s"
                % ((correctedSampleCountList - correctedSampleCountList[0]) * deltaT)
            )
            timestampArray = (
                numpy.array(
                    ((correctedSampleCountList - correctedSampleCountList[0]) * deltaT),
                    dtype="float64",
                )
                + sensor_on_time
            )
            self.logger.info("timestampArray = %s" % timestampArray)
            self.logger.info(
                "timestampArray[:2] = [%.1f %.1f, ..., timestampArray[-2:] = %.1f %.1f]"
                % (
                    timestampArray[0],
                    timestampArray[1],
                    timestampArray[-2],
                    timestampArray[-1],
                )
            )

            timestampList = list(timestampArray)
            self.logger.info(
                "timestampList[:2] = %s, ..., timestampList[-2:] = %s]"
                % (timestampList[:2], timestampList[-2:])
            )
            self.logger.info(
                "Subsampling correctedSampleCountList (len = %d) according to what got written to the netCDF file by the binning interval"
                % len(correctedSampleCountList)
            )
            self.logger.info(
                "lFrameCountWrittenList[:2] = %s, ... lFrameCountWrittenList[-2:] = %s"
                % (
                    self.dataStructure["lFrameCountWrittenList"][:2],
                    self.dataStructure["lFrameCountWrittenList"][-2:],
                )
            )

            if self.dataStructure["lFrameCountWrittenList"][-1] > len(
                correctedSampleCountList
            ):
                self.logger.info(
                    "lFrameCountWrittenList[-1] > len(correctedSampleCountList). Truncating the list before subsampling."
                )
                self.dataStructure["lFrameCountWrittenList"] = self.dataStructure[
                    "lFrameCountWrittenList"
                ][:-2]

            subSampledCorrectedSampleCountList = correctedSampleCountList[
                self.dataStructure["lFrameCountWrittenList"]
            ]
            self.dataStructure[
                "correctedSampleCountList"
            ] = subSampledCorrectedSampleCountList
            self.logger.info(
                "len(subSampledCorrectedSampleCountList) = %d"
                % len(subSampledCorrectedSampleCountList)
            )

        else:
            # Create timestamp list that corresponds best to the frame times - uses the parosci on & off times to construct an evenly spaced timestampList
            # Used at the beginning of processing to estimate time to completion - DO NOT USE THE RESULTS OF THIS IN THE NETCDF FILE!
            deltaT = float(sensor_off_time - sensor_on_time) / float(recCount - 1)
            if self.args.debugLevel >= 1:
                self.logger.debug("deltaT = %f [Should be 0.5 seconds]" % (deltaT))
            for i in range(recCount):
                timestamp = sensor_on_time + i * deltaT
                timestampList.append(timestamp)

            return timestampList

        self.logger.info("len(timestampList) = %i" % (len(timestampList)))
        self.logger.info(
            "timestampList[:2] = %s, ..., timestampList[-2:] = %s]"
            % (timestampList[:2], timestampList[-2:])
        )

        self.logger.info(
            "Subsampling timestampList (len = %d) according to what got written to the netCDF file by the binning interval"
            % len(timestampList)
        )
        writtenSampleCounts = list(
            numpy.array(self.dataStructure["sampleCountWrittenList"], dtype="int32") - 1
        )  # Subtract 1 for Python's 0-based list indexing
        writtenLFrameCounts = list(
            numpy.array(self.dataStructure["lFrameCountWrittenList"], dtype="int32") - 1
        )  # Subtract 1 for Python's 0-based list indexing
        self.logger.info(
            "Taking indices [%s ... %s] from timestampList to create subSampledTimestampList"
            % (writtenLFrameCounts[:2], writtenLFrameCounts[-2:])
        )
        subSampledTimestampList = list(numpy.array(timestampList)[writtenLFrameCounts])
        self.logger.info(
            "len(subSampledTimestampList) = %d" % len(subSampledTimestampList)
        )
        self.logger.debug(
            "subSampledTimestampList[:2] = %s, ..., subSampledTimestampList[-2:] = %s"
            % (subSampledTimestampList[:2], subSampledTimestampList[-2:])
        )

        timestampList = subSampledTimestampList

        if cFrameEsecsList != None:
            # First replace all NaNs with an interpolation between the non-NaNed values
            self.logger.info(
                "Finding elements of cFrameEsecsList that != %d" % self.missing_value
            )
            ##self.logger.info("cFrameEsecsList = " + str(cFrameEsecsList))
            hasValue_indices = list(
                numpy.nonzero(
                    numpy.array(cFrameEsecsList, dtype="float64") != self.missing_value
                )[0]
            )
            self.logger.debug("len(cFrameEsecsList) = %d" % len(cFrameEsecsList))
            self.logger.debug("len(hasValue_indices) = %d" % len(hasValue_indices))
            self.logger.debug(
                "hasValue_indices[:2] = %s, ..., hasValue_indices[-2:] = %s"
                % (hasValue_indices[:2], hasValue_indices[-2:])
            )

            # Interpolate cFrameEsecs (sent every 5 secsonds or so) to the same sample interval as timestampList
            try:
                cFrameEsecsListInterpolated = numpy.interp(
                    writtenSampleCounts,
                    hasValue_indices,
                    numpy.array(cFrameEsecsList)[hasValue_indices],
                )
            except ValueError as e:
                self.logger.error(
                    "Cannot interpolate to cFrameEsecsListInterpolated: %s", str(e)
                )
            else:
                self.logger.debug(
                    "cFrameEsecsListInterpolated[:2] = %s, ..., cFrameEsecsListInterpolated[-2:] = %s"
                    % (
                        cFrameEsecsListInterpolated[:2],
                        cFrameEsecsListInterpolated[-2:],
                    )
                )
                cFrameEsecsList = cFrameEsecsListInterpolated

        return timestampList, cFrameEsecsList  # End constructTimestampList()

    def getMedian(numericValues):
        """Return the median of the numericValues"""

        theValues = sorted(numericValues)

        if len(theValues) % 2 == 1:
            return theValues[(len(theValues) + 1) / 2 - 1]
        else:
            lower = theValues[len(theValues) / 2 - 1]
            upper = theValues[len(theValues) / 2]

        return (float(lower + upper)) / 2

    def interpolate(self, darray, i, npts, goodValueFcn, spikeValue):
        """Replace `darray[i]` with linearly interpolated value found within `npts`
        of `i` that is not `spikeValue` based on function in `goodValueFcn`

        """

        self.logger.debug("iReplacing value %d at index %d" % (darray[i], i))

        # locate start point for interpolation
        for si in range(i - 1, i - npts, -1):
            self.logger.debug(
                "Looking up npts = %d to index %d for starting goodValue at index = %d, value = %d"
                % (npts, i - npts, si, darray[si])
            )
            s_indx = si
            if goodValueFcn(darray[si], spikeValue):
                break

        # locate end point for interpolation
        for ei in range(i, i + npts):
            self.logger.debug(
                "Looking down npts = %d to index %d for ending goodValue at index = %d, value = %d"
                % (npts, i + npts, ei, darray[ei])
            )
            e_indx = ei
            if goodValueFcn(darray[ei], spikeValue):
                break

        self.logger.debug(
            "Replacing value with interpolation over values %d and %d at indices %d and %d"
            % (darray[s_indx], darray[e_indx], s_indx, e_indx)
        )
        value = (
            int((i - s_indx) * (darray[e_indx] - darray[s_indx]) / (e_indx - s_indx))
            + darray[s_indx]
        )
        self.logger.debug("interpolated value = %d" % value)

        return value  # End interpolate()

    def deSpike(self, sc_array, crit):
        """Remove both single point and multiple point spikes from sc_list based on the crit value.
        The algorithm computes the diff of the values and identifies the nature of the spikes (diff
        values > crit) and interpolates over them returning a numpy array of monotonically increasing
        de-spiked sample count values.

        """

        # Make copy and get first difference of sample count lust
        sampleCountList = sc_array.copy()
        d_sampleCountList = numpy.diff(sampleCountList)
        spike_crit = 300  # equivalent to 2.5 minutes
        self.logger.info("d_sampleCountList = %s" % d_sampleCountList)
        spike_indx = numpy.nonzero(abs(d_sampleCountList) > spike_crit)[0] + 1
        self.logger.info(
            "\nFound %d spike indicators at indices: %s" % (len(spike_indx), spike_indx)
        )
        i = 0
        lastIndx = 0
        endOfMultipleSpikeIndx = 0
        for indx in spike_indx:
            self.logger.info(
                "Identifying nature of spike value = %d at index = %d"
                % (sampleCountList[indx], indx)
            )

            # Assign look back and look ahead indices, making sure we don't IndexError
            if indx > 2:
                minIndx = indx - 2
            else:
                minIndx = 0
            if indx < len(sampleCountList) - 4:
                maxIndx = indx + 3
            else:
                maxIndx = len(sampleCountList) - 1

            # Check if single point spike (next index is in the list)
            if indx + 1 in spike_indx:
                self.logger.info("Single point spike, interpolating over this point")

                # Test for interpolation
                def notSpikeValue(value, spikeValue):
                    return value != spikeValue

                self.logger.info(
                    "minIndx = %d, indx = %d, maxIndx = %d" % (minIndx, indx, maxIndx)
                )

                self.logger.info(
                    "Before interpolation: %s" % sampleCountList[minIndx:maxIndx]
                )
                sampleCountList[indx] = self.interpolate(
                    sampleCountList, indx, 5, notSpikeValue, sampleCountList[indx]
                )
                self.logger.info(
                    "After  interpolation: %s" % sampleCountList[minIndx:maxIndx]
                )

            elif indx == lastIndx + 1:
                self.logger.debug("Second part of single point spike, skipping.")

            elif indx == endOfMultipleSpikeIndx:
                self.logger.debug("Second part of multiple point spike, skipping.")

            else:
                if i + 1 < len(spike_indx):
                    self.logger.info(
                        "\nMultiple point spike over indices [%d:%d], interpolating over these points"
                        % (indx, spike_indx[i + 1])
                    )

                    # See if spike values are less than or greater than what they need to be and set up the test for interpolation
                    if sampleCountList[indx] > sampleCountList[indx - 1]:
                        spikeValue = numpy.min(
                            sampleCountList[indx : spike_indx[i + 1]]
                        )
                        self.logger.debug(
                            "Spike values are greater than the true values, good values are less than %d"
                            % spikeValue
                        )

                        def notSpikeValues(value, spikeValue):
                            self.logger.debug(
                                "Testing if value = %d is less than %d"
                                % (value, spikeValue)
                            )
                            return value < spikeValue

                    else:
                        spikeValue = numpy.max(
                            sampleCountList[indx : spike_indx[i + 1]]
                        )
                        self.logger.debug(
                            "Spike values are less than the true values, good values are greater than %d"
                            % spikeValue
                        )

                        def notSpikeValues(value, spikeValue):
                            self.logger.debug(
                                "Testing if value = %d is greater than %d"
                                % (value, spikeValue)
                            )
                            return value > spikeValue

                    self.logger.info(
                        "Before interpolation: %s"
                        % sampleCountList[minIndx : spike_indx[i + 1] + 3]
                    )
                    for ix in range(indx, spike_indx[i + 1]):
                        lookingRange = spike_indx[i + 1] - indx + 2
                        self.logger.debug(
                            "Calling interpolate with ix = %d, lookingRange = %d"
                            % (ix, lookingRange)
                        )
                        if (
                            lookingRange > 20
                        ):  # Not sure why I have this here, make it just a WARN.  -MPM 23 June 2011
                            self.logger.warn(
                                "lookingRange = %d is too big, returning for you to examine..."
                                % lookingRange
                            )
                            input("Paused")
                            ##self.logger.error("lookingRange = %d is too big, returning for you to examine..." % lookingRange)
                            return sampleCountList

                        sampleCountList[ix] = self.interpolate(
                            sampleCountList,
                            ix,
                            lookingRange,
                            notSpikeValues,
                            spikeValue,
                        )

                    self.logger.info(
                        "After  interpolation: %s"
                        % sampleCountList[minIndx : spike_indx[i + 1] + 3]
                    )
                    endOfMultipleSpikeIndx = spike_indx[i + 1]

                else:
                    self.logger.info(
                        "\nThis is the last element in spike_indx.  No need to deSpike()."
                    )

            lastIndx = indx
            i += 1

        return sampleCountList  # End deSpike()

    def correctSampleCountList(self, orig_sc):
        """Find indices where overflow happens and assign offset-corrected slices to new sampleCountList.
        Use numpy arrays from original passed in list.  Replace zero values, stuck values, and spikes
        with interpolations.

        """

        sampleCountList = numpy.array(orig_sc).copy()
        d_orig_sc = numpy.diff(numpy.array(orig_sc))
        overflow_indx = numpy.nonzero((d_orig_sc < -65530) & (d_orig_sc > -65540))[
            0
        ]  # Pull off 0 index of tuple of array
        overflow_indx += 1  # Start slices one index more
        self.logger.info("Found overflows at indices: %s" % overflow_indx)
        self.logger.debug("len(sampleCountList) = %d" % len(sampleCountList))
        i = 0
        for indx in overflow_indx:
            i += 1
            self.logger.info(
                "Assigning values from slice starting at index %d, i = %d" % (indx, i)
            )
            sampleCountList[indx:] = numpy.array(orig_sc[indx:]).copy() + (i * 65537)

        # Find and interpolate over 0 values
        npts = 5  # Number of points to look back & forward for non-zero values
        zero_indx = numpy.nonzero(sampleCountList == 0)[0]
        self.logger.info("Original sampleCountList =  %s" % (sampleCountList))
        self.logger.info(
            "Found %d 0 values at indices: %s" % (len(zero_indx), zero_indx)
        )

        # Test function for interpolate to find good value to use in interpolation
        def nonZeroValue(value, badValue):
            return value != 0

        for indx in zero_indx:
            # Assign look back and look ahead indices, making sure we don't IndexError
            if indx > 2:
                minIndx = indx - 2
            else:
                minIndx = 0
            if indx < len(sampleCountList) - 4:
                maxIndx = indx + 3
            else:
                maxIndx = len(sampleCountList) - 1

            self.logger.debug(
                "Interpolating over value = %d at index = %d"
                % (sampleCountList[indx], indx)
            )
            self.logger.debug(
                "Before interpolation: %s" % sampleCountList[minIndx:maxIndx]
            )
            sampleCountList[indx] = self.interpolate(
                sampleCountList, indx, 5, nonZeroValue, 0
            )
            self.logger.debug(
                "After  interpolation: %s" % sampleCountList[minIndx:maxIndx]
            )

        # Find stuck values and assign to one more than the previous value
        d_sampleCountList = numpy.diff(sampleCountList)
        stuck_indx = numpy.nonzero(d_sampleCountList == 0)[0] + 1
        self.logger.info(
            "Found %d stuck values at indices: %s" % (len(stuck_indx), stuck_indx)
        )
        for indx in stuck_indx:
            # Assign look back and look ahead indices, making sure we don't IndexError
            if indx > 2:
                minIndx = indx - 2
            else:
                minIndx = 0
            if indx < len(sampleCountList) - 4:
                maxIndx = indx + 3
            else:
                maxIndx = len(sampleCountList) - 1

            self.logger.debug(
                "Replacing stuck value = %d at index = %d"
                % (sampleCountList[indx], indx)
            )
            self.logger.debug(
                "Before incrementing stuck value: %s" % sampleCountList[minIndx:maxIndx]
            )
            sampleCountList[indx] = sampleCountList[indx - 1] + 1
            self.logger.debug(
                "After  incrementing stuck value: %s" % sampleCountList[minIndx:maxIndx]
            )

        # Find and interpolate over spikes using crit=300 (appropriate for 2009.084.02)
        sampleCountList = self.deSpike(sampleCountList, 300)

        self.logger.info("After despike()  sampleCountList =  %s" % (sampleCountList))
        return sampleCountList  # End correctSampleCountList()

    def testCSCL(
        self,
        lopcNC="http://dods.mbari.org/cgi-bin/nph-nc/data/ssdsdata/ssds/generated/netcdf/files/ssds.shore.mbari.org/auvctd/missionlogs/2009/2009084/2009.084.02/lopc.nc",
    ):
        """
            Test the correctSampleCountList() function by reading (via NetCDF4) a previously saved netCDF file's original sampleCount.
            Runs the code with debugging turned on.

            ipyhton test procedure:

        import lopcToNetCDF
        url='http://dods.mbari.org/cgi-bin/nph-nc/data/ssdsdata/ssds/generated/netcdf/files/ssds.shore.mbari.org/auvctd/missionlogs/2008/2008154/2008.154.01/lopc.nc'
        (scList, correctedSC) = lopcToNetCDF.testCSCL(url)
        plot(scList)
        plot(correctedSC)

        """

        logging.getLogger("self.logger").setLevel(logging.DEBUG)

        self.logger.info("Opening URL: %s" % lopcNC)
        ds = Dataset(lopcNC, "w")

        self.logger.info("Extracting sampleCount[:] from the NetCDF4 proxy.")
        scList = ds["sampleCount"].sampleCount[:]

        self.logger.info("Calling correctSampleCountList()")
        correctedSC = self.correctSampleCountList(scList)

        self.logger.info("Returning scList and correctedSC")
        self.logger.info(
            "From ipython you may plot with\nplot(scList)\nplot(correctedSC)\n"
        )

        return scList, correctedSC  # End testCSCL()

    def testFlowSpeed(self):
        """Test using values provided in Herman 2009"""
        self.logger.info(
            "flowSpeed = %f for flowCount = %d and flowTime = %d"
            % (self.flowSpeed(1026, 143), 143, 1026)
        )

        return

    def openNetCDFFile(self, opts):
        """Open netCDF file and write some global attributes and dimensions that
        we know at the beginning of processing.

        """

        ncFileName = self.args.netCDF_fileName
        self.logger.info("Will output NetCDF file to %s" % ncFileName)

        # Improve long names of MEP counts based on passed in arguments
        self.MEPDataDictLongName[
            "LCcount"
        ] += " with aiCrit = %.2f, esdMinCrit = %.0f, esdMaxCrit = %.0f" % (
            self.args.LargeCopepod_AIcrit,
            self.args.LargeCopepod_ESDmin,
            self.args.LargeCopepod_ESDmax,
        )
        self.MEPDataDictLongName["transCount"] += (
            " with ai < %.2f" % self.args.trans_AIcrit
        )
        self.MEPDataDictLongName["transCount"] += (
            " with ai > %.2f" % self.args.trans_AIcrit
        )

        #
        # Open the netCDF file and set some global attributes
        #
        self.ncFile = Dataset(ncFileName, "w")

        missionName = ""
        try:
            missionName = ncFileName.split("/")[-2]
        except IndexError:
            self.logger.warn(
                "Could not parse missionName from netCDF file name - probably not production processing."
            )

        self.logger.info("missionName = %s" % missionName)

        self.ncFile.title = (
            "Laser Optical Plankton Counter measurements from AUV mission "
            + missionName
        )
        self.ncFile.institution = "Monterey Bay Aquarium Research Institute"
        self.ncFile.summary = "These data have been processed from the original lopc.bin file produced by the LOPC instrument.  The data in this file are to be considered as simple time series data only and are as close to the original data as possible.  Further processing is required to turn the data into a time series of profiles."
        self.ncFile.keywords = (
            "plankton, particles, detritus, marine snow, particle counter"
        )
        self.ncFile.Conventions = "CF-1.6"
        self.ncFile.standard_name_vocabulary = "CF-1.6"

        # Time dimension is unlimited, we'll write variable records then write the time data at the end of processing
        self.ncFile.createDimension("time", None)  # Unlimited dimension

        # Create bin size dimension for the counts variable
        self.ncFile.createDimension("bin", len(self.dataStructure["binSizeList"]))
        self.logger.info(
            "Writing bin axis for len(self.dataStructure['binSizeList']) = %d"
            % len(self.dataStructure["binSizeList"])
        )
        self.ncFile.createVariable("bin", "f", ("bin",))
        self.ncFile.variables["bin"].units = "microns"
        self.ncFile.variables["bin"].long_name = "Equivalent Spherical Diameter"
        self.ncFile.variables["bin"][:] = self.dataStructure["binSizeList"]

        # --- Create the record variables ---
        for key in self.MEPDataDictKeys:
            if key.endswith("List"):
                self.logger.info("Creating variable %s on axes time and bin" % key)
                self.ncFile.createVariable(key, "i", ("time", "bin"))
            else:
                self.logger.info("Creating variable %s on axes time" % key)
                self.ncFile.createVariable(key, "f", ("time",))

            self.ncFile.variables[key].units = self.MEPDataDictUnits[key]
            self.ncFile.variables[key].long_name = self.MEPDataDictLongName[key]

        self.logger.info("Creating variable countListSum on axis time")
        self.ncFile.createVariable("countListSum", "i", ("time",))
        self.ncFile.variables["countListSum"].units = "count"
        self.ncFile.variables[
            "countListSum"
        ].long_name = f"Sum of Total Particle counts"

        # Create scalar list item variable
        self.logger.info(
            "Creating variables from engineering data collected in LframeScalarDict:"
        )
        for var in self.LframeScalarKeys:
            self.logger.info("Creating variable %s on axis time" % var)
            if var == "snapshot" or var == "bufferStatus":
                self.ncFile.createVariable(var, "b", ("time",))
            else:
                self.ncFile.createVariable(var, "i", ("time",))

            if self.args.debugLevel >= 2:
                self.logger.debug("  units = %s " % self.LframeScalarDictUnits[var])
            if self.LframeScalarDictUnits[var] == None:
                self.ncFile.variables[
                    var
                ].units = ""  # For unitless variables we do need a '' units attribute
            else:
                self.ncFile.variables[var].units = self.LframeScalarDictUnits[var]

            if self.args.debugLevel >= 2:
                self.logger.debug(
                    "  long_name = %s " % self.LframeScalarDictLongName[var]
                )
            if self.LframeScalarDictLongName[var] != None:
                self.ncFile.variables[var].long_name = self.LframeScalarDictLongName[
                    var
                ]

        # Create other (mainly Error count) variables from the self.dataStructure
        self.logger.info(
            "Creating variables from processing information collected in the self.dataStructure dictionary:"
        )
        for var in list(self.dataStructure.keys()):
            self.logger.info("Checking var = %s" % var)
            if var.startswith("count"):
                self.logger.info("Creating variable %s on axis time" % var)
                self.ncFile.createVariable(var, "i", ("time",))
                self.ncFile.variables[var].units = "count"
                self.ncFile.variables[var].long_name = self.dataStructureLongName[var]
            if var.startswith("flowSpeed"):
                self.logger.info("Creating variable %s on axis time" % var)
                self.ncFile.createVariable(
                    var,
                    "f",
                    ("time",),
                    fill_value=self.missing_value,
                )
                self.ncFile.variables[var].units = "m/s"
                self.ncFile.variables[var].missing_value = self.missing_value
                self.ncFile.variables[var].long_name = self.dataStructureLongName[var]

        # Create cFrameEsecs variable for missions after March 2010 when Hans added it
        # There is a record for each L frame record, but most are fill values as Hans sends an updated Esecs every 5 seconds or so
        if self.hasMBARICframeData:
            self.ncFile.createVariable(
                "cFrameEsecs",
                "f",
                ("time",),
                fill_value=self.missing_value,
            )
            # Xarray wants only one time variable with udunits, so we use epoch seconds
            self.ncFile.variables["cFrameEsecs"].units = "epoch seconds"
            self.ncFile.variables["cFrameEsecs"].missing_value = self.missing_value
            self.ncFile.variables[
                "cFrameEsecs"
            ].long_name = "Epoch seconds from the main vehicle computer fed into the CTD port of the LOPC at 0.2 Hz"

        # End openNetCDFFile()

    def record_count(self):
        """Generator function to return the next integer for an incrementing index."""
        k = -1
        while True:
            k += 1
            yield k

    def writeNetCDFRecord(self, MEPDataDict, outRecNumFunc):
        """Write records to the unlimited (time) dimension"""

        indx = outRecNumFunc()

        if self.args.debugLevel >= 1:
            self.logger.debug("Appending variables to time axis at index # %d:" % indx)
        ##if self.args.debugLevel >= 1: self.logger.debug("appending countList variable for len(countList) = %d , len(self.dataStructure['binSizeList']) = %d" % (len(countList),  len(dataStructure['binSizeList'])))
        ##if self.args.debugLevel >= 2: self.logger.debug("countList = %s" % countList)

        # Make sure that the countList is the right length
        ##lenInitialbinSizeList = 128   # Hard coded for now, but if this should change we need to assign this variable when counts is created in openNetCDFFile()
        lenInitialbinSizeList = 994
        if len(MEPDataDict["countList"]) != lenInitialbinSizeList:
            self.logger.warn(
                "len(countList) of %d != %d"
                % (len(MEPDataDict["countList"]), lenInitialbinSizeList)
            )
            if self.args.debugLevel >= 1:
                self.logger.warn(
                    "appending the last good record counts as I doubt we can trust the values that were parsed in this incomplete record."
                )
            ##countList = [0] * lenInitialbinSizeList   # A list of zero counts

        # Go through all the MEP data items, those that end in 'List' are 2D and were properly dimensioned in openNetCDFFile()
        for key in list(MEPDataDict.keys()):
            self.ncFile.variables[key][indx] = MEPDataDict[key]

        ##self.ncFile.variables['SEPcounts'][indx] = sepCountList
        ##self.ncFile.variables['MEPcounts'][indx] = mepCountList
        ##self.ncFile.variables['counts'][indx] = countList
        ##self.ncFile.variables['LCcount'][indx] = LCcount
        ##self.ncFile.variables['transCount'][indx] = transCount
        ##self.ncFile.variables['nonTransCount'][indx] = nonTransCount

        # Write the countListSum - making it easier for the user of the data
        countListSum = numpy.sum(MEPDataDict["countList"])
        if self.args.debugLevel >= 2:
            self.logger.debug("appending countListSum = %d" % countListSum)
        self.ncFile.variables["countListSum"][indx] = countListSum

        # Write scalar list items
        for var in self.LframeScalarKeys:
            if self.args.debugLevel >= 2:
                self.logger.debug(
                    "appending %s = %d" % (var, self.LframeScalarDict[var])
                )
            self.ncFile.variables[var][indx] = self.LframeScalarDict[var]

        # Write accumulated error counts
        for var in list(self.dataStructure.keys()):
            if var.startswith("count"):
                if self.args.debugLevel >= 2:
                    self.logger.debug(
                        "appending %s = %d" % (var, self.dataStructure[var])
                    )
                self.ncFile.variables[var][indx] = self.dataStructure[var]
            if var.startswith("flow"):
                if self.args.debugLevel >= 2:
                    self.logger.debug(
                        "appending %s = %d" % (var, self.dataStructure[var])
                    )
                self.ncFile.variables[var][indx] = self.dataStructure[var]

        # Write mvc epoch seconds if we have it
        if "cFrameEsecs" in list(self.dataStructure.keys()):
            if self.args.debugLevel >= 2:
                self.logger.debug(
                    "appending %s = %f"
                    % ("cFrameEsecs", self.dataStructure["cFrameEsecs"])
                )
            self.ncFile.variables["cFrameEsecs"][indx] = self.dataStructure[
                "cFrameEsecs"
            ]

        return  # End writeNetCDFRecord()

    def closeNetCDFFile(self, tsList, cFrameEsecsList):
        """Write the time axis data, additional global attributes and close the file"""

        # Save corrected sample count list
        self.ncFile.createVariable("correctedSampleCountList", "i", ("time",))
        self.ncFile.variables[
            "correctedSampleCountList"
        ].long_name = "Corrected instrumenet sample count"
        self.ncFile.variables["correctedSampleCountList"].units = "count"
        self.ncFile.variables["correctedSampleCountList"][:] = self.dataStructure[
            "correctedSampleCountList"
        ]

        # Write time axis
        self.logger.info("Writing time axis for len(tsList) = " + str(len(tsList)))
        if self.args.debugLevel >= 2:
            self.logger.info("tsList = %s" % tsList)
        self.logger.info(
            "tsList[:1] = %s, ..., tsList[-2:] = %s" % (tsList[:1], tsList[-2:])
        )
        self.logger.info(
            "Begin time = %s"
            % (time.strftime("%Y-%m-%d %H:%M:%S Z", time.gmtime(tsList[0])))
        )
        self.logger.info(
            "End time   = %s"
            % (time.strftime("%Y-%m-%d %H:%M:%S Z", time.gmtime(tsList[-1])))
        )

        self.logger.info(
            "Writing time axis for len(cFrameEsecsList) = " + str(len(cFrameEsecsList))
        )
        self.logger.info(
            "cFrameEsecsList[:1] = %s, ..., cFrameEsecsList[-2:] = %s"
            % (cFrameEsecsList[:1], cFrameEsecsList[-2:])
        )
        self.logger.info(
            "Begin time = %s"
            % (time.strftime("%Y-%m-%d %H:%M:%S Z", time.gmtime(cFrameEsecsList[0])))
        )
        self.logger.info(
            "End time   = %s"
            % (time.strftime("%Y-%m-%d %H:%M:%S Z", time.gmtime(cFrameEsecsList[-1])))
        )

        # Write time axis that is used for all the time dependent variables
        self.ncFile.createVariable("time", "d", ("time",))
        self.ncFile.variables["time"].units = "seconds since 1970-01-01 00:00:00Z"
        self.ncFile.variables["time"].standard_name = "time"
        self.ncFile.variables["time"].long_name = "Time GMT"
        self.ncFile.variables["time"][:] = tsList

        # Write main vehicle time variable as received via C Frame
        self.ncFile.createVariable("mvctime", "f", ("time",))
        # Xarray wants only one time variable with udunits, so we use epoch seconds
        self.ncFile.variables["mvctime"].units = "epoch seconds"
        self.ncFile.variables["mvctime"].long_name = "Main Vehicle Computer Time GMT"
        self.ncFile.variables["mvctime"] = cFrameEsecsList

        return self.ncFile  # End closeNetCDFFile()

    def process_command_line(self):
        """Main routine: Parse command line options and call unpack and write functions."""

        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter,
            description=__doc__,
        )

        parser.add_argument(
            "-i",
            "--bin_fileName",
            action="store",
            help="Specify an input binary data file name, e.g. /mbari/AUVCTD/missionlogs/2009/2009084/2009.084.00/lopc.bin.",
        )
        parser.add_argument(
            "-n",
            "--netCDF_fileName",
            action="store",
            help="Specify a fully qualified output NetCDF file path, e.g. /mbari/tempbox/mccann/lopc/2009_084_00lopc.nc",
        )
        parser.add_argument(
            "-t",
            "--text_fileName",
            action="store",
            help="Specify a fully qualified output ASCII text file path, e.g. /mbari/tempbox/mccann/lopc/2009_084_00lopc.dat",
        )
        parser.add_argument(
            "-v",
            "--verbose",
            type=int,
            choices=range(3),
            action="store",
            default=0,
            const=1,
            nargs="?",
            help="verbosity level: "
            + ", ".join(
                [f"{i}: {v}" for i, v, in enumerate(("WARN", "INFO", "DEBUG"))]
            ),
        )
        parser.add_argument(
            "-d",
            "--debugLevel",
            action="store",
            type=int,
            help="Turn on debugLevel [1..3].  The higher the number, the more the output.  E.g. us '-d 1' to understand framing errors, '-d 3' for all debug statements.",
        )
        parser.add_argument(
            "-f",
            "--force",
            action="store_true",
            default=False,
            help="Force overwite of netCDF file.  Useful for using from a master reprocess script.",
        )

        # Additional criteria setting options
        parser.add_argument(
            "--trans_AIcrit",
            type=float,
            action="store",
            default=0.4,
            help="Criteria for Attenuation Index to separate transparent from non-transparent particles (0..1), default = 0.4",
        )
        parser.add_argument(
            "--LargeCopepod_AIcrit",
            type=float,
            action="store",
            default=0.6,
            help="Criteria for Attenuation Index to identify Large Copepod particles (0..1), default = 0.6",
        )
        parser.add_argument(
            "--LargeCopepod_ESDmin",
            type=float,
            action="store",
            default=1100,
            help="Criteria for minimum Equivalent Spherical Diameter to identify Large Copepod particles (microns), default = 1100",
        )
        parser.add_argument(
            "--LargeCopepod_ESDmax",
            type=float,
            action="store",
            default=1700.0,
            help="Criteria for maximum Equivalent Spherical Diameter to identify Large Copepod particles (microns), default = 1700",
        )
        self.args = parser.parse_args()
        self.logger.setLevel(self._log_levels[self.args.verbose])
        self.commandline = " ".join(sys.argv)

    def main(self):
        #
        # unpack data according to command line options
        #
        if self.args.bin_fileName and self.args.netCDF_fileName:

            start = time.time()

            if self.args.debugLevel:
                logging.getLogger("self.logger").setLevel(logging.DEBUG)

            # Check for output file and offer to overwrite
            if os.path.exists(self.args.netCDF_fileName):
                if self.args.force:
                    if os.path.exists(self.args.netCDF_fileName):
                        os.remove(self.args.netCDF_fileName)
                else:
                    ans = input(
                        self.args.netCDF_fileName
                        + " file exists.\nDo you want to remove it and continue processing? (y/[N]) "
                    )
                    if ans.upper() == "Y":
                        os.remove(self.args.netCDF_fileName)
                    else:
                        sys.exit(0)

            textFile = None
            if self.args.text_fileName:
                if os.path.exists(self.args.text_fileName):
                    if self.args.force:
                        if os.path.exists(self.args.text_fileName):
                            os.remove(self.args.text_fileName)
                    else:
                        ans = input(
                            self.args.text_fileName
                            + " file exists.\nDo you want to remove it and continue processing? (y/[N]) "
                        )
                        if ans.upper() == "Y":
                            os.remove(self.args.text_fileName)
                        else:
                            sys.exit(0)

                textFile = open(self.args.text_fileName, "w")

            self.logger.info("Processing begun: %s" % time.ctime())
            # Open input file
            binFile = open(self.args.bin_fileName, "rb")

            # Set flag for whether we need to look for data that Hans writes to the C Frame - implemented after 15 March 2010 (day 2010074)
            # Assume binFile.name is like: '/mbari/AUVCTD/missionlogs/2009/2009084/2009.084.02/lopc.bin'
            try:
                if int(binFile.name.split("/")[-3]) > 2010074:
                    self.hasMBARICframeData = True
            except ValueError:
                self.hasMBARICframeData = (
                    True  # For testing assume we have CFrame time data
                )

            # Add time axis data to the data structure and estimate the number of records and time to parse
            tsList = self.constructTimestampList(binFile)
            self.dataStructure["tsList"] = tsList
            self.logger.info(
                "Examined sibling parosci.nc file to find startTime = %s and endTime = %s with %d records expected to be read from lopc.bin"
                % (
                    time.strftime("%Y-%m-%d %H:%M:%S Z", time.gmtime(tsList[0])),
                    time.strftime("%Y-%m-%d %H:%M:%S Z", time.gmtime(tsList[-1])),
                    len(tsList),
                )
            )

            # Unpack LOPC binary data into a data structure that we can later write to a NetCDF file (with the time information)
            # unpackLOPCbin() blocks until end of file is encountered, then we close the output NetCDF file and finish up.
            try:
                # The major workhorse function: populates global self.dataStructure dictionary
                # On first L frame read this function opens the netCDF file for appending
                # then calls writeNetCDFRecord at the binningInterval. self.dataStructure[] is
                # populated with lots of information by unpackLOPCbin.
                self.unpackLOPCbin(binFile, self.args, textFile)
            except EndOfFileException:
                self.logger.info(">>> Done reading file.")
                self.logger.info(
                    "lFrameCount = %d, mFrameCount = %d"
                    % (
                        self.dataStructure["lFrameCount"],
                        self.dataStructure["mFrameCount"],
                    )
                )
                if self.countBeginMFrameWithoutEndOfPreviousFrame:
                    self.logger.info(
                        "countBeginMFrameWithoutEndOfPreviousFrame = %d, countBeginLFrameWithoutEndOfPreviousFrame = %d"
                        % (
                            self.dataStructure[
                                "countBeginMFrameWithoutEndOfPreviousFrame"
                            ],
                            self.dataStructure[
                                "countBeginLFrameWithoutEndOfPreviousFrame"
                            ],
                        )
                    )

            # Close the ASCII text file, if it exists
            if textFile != None:
                self.logger.info("Closing test file")
                textFile.close()

            # Make sure that tsList from the parosci.nc file and the self.dataStructure parsed from lopc.bin are the same lengths
            (tsList, cFrameEsecsList) = self.constructTimestampList(
                binFile,
                sampleCountList=self.dataStructure["sampleCountList"],
                cFrameEsecsList=self.dataStructure["cFrameEsecsList"],
            )

            # Close the netCDF file writing the proper tsList data first
            self.closeNetCDFFile(tsList, cFrameEsecsList)

            self.logger.info("Created file: %s" % self.args.netCDF_fileName)

            mark = time.time()
            self.logger.info(
                "Processing finished: %s Elapsed processing time from start of processing = %d seconds"
                % (time.ctime(), (mark - start))
            )

        else:
            print("\nCannot process input.  Execute with -h for usage note.\n")

        return  # End main()


# Allow this file to be imported as a module
if __name__ == "__main__":
    lp = LOPC_Processor()
    lp.process_command_line()
    try:
        if sys.argv[1].startswith("test"):
            """Run a function that begins with 'test' using the locals() dictionary of functions in this module."""
            lp.logger.info("Running test function: " + sys.argv[1] + "()...")
            locals()[sys.argv[1]]()
    except IndexError:
        lp.main()
    else:
        """Will print usage note if improper arguments are given."""
        lp.main()
