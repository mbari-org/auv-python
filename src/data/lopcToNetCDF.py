#!/usr/bin/env python
__author__    = 'Mike McCann'
__version__ = '$Revision: 1.43 $'.split()[1]
__date__ = '$Date: 2020/11/23 21:40:04 $'.split()[1]
__copyright__ = '2009'
__license__   = 'GPL v3'
__contact__   = 'mccann at mbari.org'
__idt__   = '$Id: lopcToNetCDF.py,v 1.43 2020/11/23 21:40:04 ssdsadmin Exp $'

__doc__ = '''

Unpack Brooke Ocean Technology Laser Optical Plankton Counter binary data from
lopc.bin files logged on the vehicle.  These files are the BOT binary data stream
format as described in Section 6.2 of the LOPC Software Operation Manual. They
do not follow the same format as the control system software generated .log 
files that are processed by the SSDS-Java code (auvportal) therefore there is no
compulsion to use that same framework for processing these data.

This Python script is the shortest path to a file conversion.  Once an SSDS
interface is available for Python we'll integrate ProcessRun recording - but
for now we just need to produce netCDF files from the .bin files.

@var __date__: Date of last cvs commit
@undocumented: __doc__ parser
@status: In development
@license: GPL
'''

import sys
import struct
import logging
from optparse import OptionParser
import os
import datetime
import urllib.request, urllib.parse, urllib.error
import zipfile
import numpy
import time
import Nio
import pydap.client
import pydap.exceptions
import lopcMEP
import math
import string
import re


#
# Make a global logging object.
#
h = logging.StreamHandler()
f = logging.Formatter("%(levelname)s %(asctime)s %(message)s")
##f = logging.Formatter("%(levelname)s %(asctime)s %(funcName)s %(lineno)d %(message)s")    # Python 2.4 does not support funcName :-(
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
hasMBARICframeData = False  # includes timing, depth, profile information after a certain data - flag used in netCDF functions
WWbinningInterval = 1       # Test value
binningInterval = 20        # 10 second binning interval for calculated MEP data output

missing_value = -9999       # Used in netCDF file

dataStructure = dict()

dataStructureLongName = dict()
dataStructureLongName['countShortLFrameError'] = 'Accumulated count of L frames that did not have the full 128 bins'
dataStructureLongName['countBeginLFrameWithoutEndOfPreviousFrame'] = 'Accumulated count of L frames begun before an end of the previous frame'
dataStructureLongName['countBeginMFrameWithoutEndOfPreviousFrame'] = 'Accumulated count of M frames begun before an end of the previous frame'
dataStructureLongName['countUnknownFrameCharacter'] = 'Accumulated count of unknown frames'
dataStructureLongName['countGarbledLFrame'] = 'Accumulated count of garbled L frames'
dataStructureLongName['flowSpeed'] = 'Flow Speed'

LframeScalarKeys = [ 'snapshot', 'threshold', 'sampleCount', 'flowCount', 
            'flowTime', 'bufferStatus', 'laserLevel', 
            'counter', 'counterPeriod', 'laserControl' ]
LframeScalarDict = { 'snapshot': None, 'threshold': None, 'sampleCount': None, 'flowCount': None, 
            'flowTime': None, 'bufferStatus': None, 'laserLevel': None, 
            'counter': None, 'counterPeriod': None, 'laserControl': None }
LframeScalarDictUnits = { 'snapshot': None, 'threshold': 'microns', 'sampleCount': None, 'flowCount': None, 
            'flowTime': 'seconds', 'bufferStatus': None, 'laserLevel': None, 
            'counter': None, 'counterPeriod': 'seconds', 'laserControl': None }
LframeScalarDictLongName = { 'snapshot': '1: snapshot in progress, 0: not in progress', 'threshold': 'lower limit on signal level detection', 
            'sampleCount': 'Number of sample', 'flowCount': 'Number of flow counts in .5 second period', 
            'flowTime': 'Accumulated time for all counts', 'bufferStatus': '1: buffer overrun, 0: no overrun', 
            'laserLevel': 'Mean laser intensity', 'counter': 'Number of pulses detected on the counter input', 
            'counterPeriod': 'Period of the pulses detected on the counter input', 'laserControl': 'Configured set value for the Laser Control Algorithm'}
            
MEPDataDictKeys = [ 'sepCountList', 'mepCountList', 'countList', 'LCcount', 'transCount', 'nonTransCount', 
            'ai_mean', 'ai_min', 'ai_max', 'ai_std', 'esd_mean', 'esd_min', 'esd_max', 'esd_std' ]
MEPDataDictUnits = { 'sepCountList': 'count' , 'mepCountList': 'count', 'countList': 'count', 'LCcount': 'count', 'transCount': 'count', 'nonTransCount': 'count', 
            'ai_mean': '', 'ai_min': '', 'ai_max': '', 'ai_std': '', 'esd_mean': 'microns', 'esd_min': 'microns', 'esd_max': 'microns', 'esd_std': 'microns' }
MEPDataDictLongName = { 'sepCountList': 'Single Element Plankton count' , 
            'mepCountList': 'Multiple Element Plankton count', 
            'countList': 'Total Plankton count', 
            'LCcount': 'Large Copepod count', 
            'transCount': 'Transparent particle count', 
            'nonTransCount': 'Non-Transparent particle count', 
            'ai_mean': 'Attenuation Index mean', 'ai_min': 'Attenuation Index minimum', 'ai_max': 'Attenuation Index maximum', 
            'ai_std': 'Attenuation Index Standard Deviation', 
            'esd_mean': 'Equivalent Spherical Diameter mean', 'esd_min': 'Equivalent Spherical Diameter minimum', 
            'esd_max': 'Equivalent Spherical Diameter ', 'esd_std': 'Equivalent Spherical Diameter Index Standard Deviation', 
            'esd_mean': 'microns', 'esd_min': 'microns', 'esd_max': 'microns', 'esd_std': 'microns' }

frameText = ''

countBeginLFrameWithoutEndOfPreviousFrame = 0
countBeginMFrameWithoutEndOfPreviousFrame = 0
countUnknownFrameCharacter = 0
countGarbledLFrame = 0
unknownFrameCharacters = []
ncFile = None
sampleCountList = []

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


def checkForDLE(file, char):
    """ Logic that is applied to each character read if a <DLE> ('~') is encounered
    """
    global frameText, countUnknownFrameCharacter, unknownFrameCharacters
    nextChar = file.read(1)
    if debugLevel >= 3: logger.debug('checkForDLE(): nextChar = %s' % nextChar)
    if nextChar == '*':
        if debugLevel >= 5: logger.debug('checkForDLE(): nextChar is *; this is the end of frame')
        frameText += nextChar
        raise EndOfFrameException

    elif nextChar == '~':
        frameText += nextChar
        if debugLevel >= 5: logger.debug('checkForDLE(): nextChar is ~; skipping the second <DLE> that is added by the instrument')

    elif nextChar == 'M':
        frameText += nextChar
        if debugLevel >= 3: logger.warn("checkForDLE(): Detected beginning of new M frame without a detection of the end of the previous frame.")
        if debugLevel >= 3: logger.warn("checkForDLE(): frameText = \n" + frameText)
        raise BeginMFrameWithoutEndOfPreviousFrame

    elif nextChar == 'L':
        frameText += nextChar
        if debugLevel >= 3: logger.warn("checkForDLE(): Detected beginning of new L frame without a detection of the end of the previous frame.")
        if debugLevel >= 3: logger.warn("checkForDLE(): frameText = \n" + frameText)
        raise BeginLFrameWithoutEndOfPreviousFrame

    else:
        frameText += str(ord(nextChar))
        if debugLevel >= 3: logger.warn("checkForDLE(): this is not expected.  If we have a '~' then there should be either a '~' or  '*' following it.")
        if debugLevel >= 3: logger.warn("checkForDLE(): frameText = \n" + frameText)
        if debugLevel >= 3: logger.warn("checkForDLE(): nextChar = " + nextChar)
        countUnknownFrameCharacter += 1
        unknownFrameCharacters.append(str(ord(nextChar)))
        ##raw_input("Encountered unexpected area of code.  Please notify Mike McCann.  Press Enter to continue...")



def readBigEndianUShort(binFile):
    """Read next 16-bits from binFile and return byte-swapped unsiged short integer
    value.  If end of frame is detected ('~*') the raise an EndOfFrameException()

    """
    global frameText
    c1 = binFile.read(1)
    if len(c1) == 0:
        raise EndOfFileException

    if debugLevel >= 5: logger.debug('readBigEndianUShort(): c1 = %s' % ord(c1))
    if c1 == '~':
        frameText += c1
        checkForDLE(binFile, c1)
    else:
        frameText += str(ord(c1))

    c2 = binFile.read(1)
    if len(c2) == 0:
        raise EndOfFileException

    if debugLevel >= 5: logger.debug('readBigEndianUShort(): c2 = %s' % ord(c2))
    if c2 == '~':
        frameText += c2
        checkForDLE(binFile, c2)
    else:
        frameText += str(ord(c2))

    value = struct.unpack('H', c2 + c1)
    if debugLevel >= 5: logger.debug("readBigEndianUShort(): value = %i" % value)

    return int(value[0]) # End readBigEndianUShort()


def readChar(binFile):
    """Read next 8-bit character from binFile.  Handles checking for DLE.
    If end of frame is detected ('~*') the raise an EndOfFrameException()

    """
    global frameText
    c = binFile.read(1)
    if len(c) == 0:
        raise EndOfFileException
    if debugLevel >= 5: logger.debug('readChar(): c = %s' % ord(c))
    if c == '~':
        frameText += c
        checkForDLE(binFile, c)
    else:
        frameText += str(ord(c))

    return c # End readChar()


def readLframeData(binFile, sampleCountList):
    """For L frame data read the integer count data from the 128 bins.
    Then read the scalar engineering values that follow.  Dorado must be
    running an LOPC that is v2.36 or greater as the test file I've tried
    (2009.084.00/lopc.bin) has 275 bytes in the L frame.
    Return: countList and dictionary of engineering values.

    """
    global frameText
    global LframeScalarDict

    if debugLevel >= 2: logger.debug('-' * 81)
    if debugLevel >= 2: logger.debug("readLframeData(): frameID = L: Read Binned Count Data")
    countList = []
    # Expecting to read 128 values.  The call to readBigEndianUShort() may raise an EndOfFrameException, in which case 
    # is caught by the routine that calls this function and is an error.
    for i in range(128):
        try:
            value = readBigEndianUShort(binFile)
        except EndOfFrameException:
            if debugLevel >= 1: logger.warn("readLframeData(): Reached the end of this L frame before the end (# 127) at bin # %d." % i)
            if debugLevel >= 1: logger.warn("readLframeData(): L Frame text = \n" + frameText + "\n")
            if debugLevel >= 1: logger.warn("readLframeData(): countList = \n" + str(countList))
            if debugLevel >= 1: logger.warn("readLframeData(): Raising ShortLFrameError.")
            raise ShortLFrameError

        countList.append(value)

        if debugLevel >= 4: logger.debug('readLframeData(): bin: %i count = %i' % (i, value))

    # Peek at last several values in countList for really bin values that indicates a garbled frame
    # Tests indicate the the scalar data are corrupted on frame that meet this criteria.  Let's just discard them.
    for c in countList[101:128]:
        if c > 1000:
            ##raw_input("Paused with too high a value at red-end of the countList, c = %d" % c)
            if debugLevel >= 1: logger.warn("readLframeData(): Detected garbled frame with count value = %d between indices [101:128] of countList"% c)
            raise GarbledLFrame

    # Detect partial frames and raise exception
    if len(countList) < 128:
        raise ShortLFrameError

    # Read the rest of the values from the L data frame - use a List to maintain order and a Dict to store the values
    ##logging.getLogger("logger").setLevel(logging.DEBUG)
    if debugLevel >= 2: logger.debug('-' * 80)
    try:
        if debugLevel >= 2: logger.debug('readLframeData(): Reading scalar data:')
        for var in LframeScalarKeys:
            # All but counter are 2 bytes
            if var == 'counter':
                val = ord(binFile.read(1))
                frameText += str(val)
            else:
                val = readBigEndianUShort(binFile)

            if val == None:
                logger.warn("L frame scalar: %s is None (unable to parse)" % var)
            else:
                try:
                    # Perform QC checkes on threshold and bufferStatus, save sampleCount in List that is used for timeStamp
                    if debugLevel >= 2: logger.debug("readLframeData():   %s = %d" % (var, val))
                    LframeScalarDict[var] = val
                    if var == 'threshold' and val != 100:
                        if debugLevel >= 1: logger.warn("readLframeData(): Detected garbled frame with threshold != 100 (%d)"% val)
                        raise GarbledLFrame
                    if var == 'sampleCount':
                        ##try:
                        ##  if abs(val - sampleCountList[-1]) > 100:
                        ##      raw_input("Paused at samplewCount excursion...")
                        ##except IndexError:
                        ##  pass
                        sampleCountList.append(val) # Appends to list that is passed in and returned in call argument list
                    if var == 'bufferStatus' and (val != 0 and val != 1):
                        if debugLevel >= 1: logger.warn("readLframeData(): Detected garbled frame with bufferStatus != 0|1 (%d)"% val)
                        raise GarbledLFrame
                except TypeError:
                    logger.error("TypeError in trying to print variable '%s' as an integer val = '%s'" % (var, val))

    except EndOfFrameException:
        if debugLevel >= 1: logger.warn("readLframeData(): Reached the end of this L frame while attempting to read the LframeScalar data.")
        if debugLevel >= 1: logger.warn("readLframeData(): LframeScalarDict = " + str(LframeScalarDict))
        return(countList, LframeScalarDict)
                
    # If we read one more character we should get an EndOfFrameException - make sure this happens
    endOfFrame = False
    while not endOfFrame:
        if debugLevel >= 2: logger.debug("readLframeData(): Inside 'while not endOfFrame:' loop")
        try:
            val = readBigEndianUShort(binFile)  # Reads one byte at a time checking for EndOfFrame

        # Catch the EndOfFrameException and deal with it gracefully
        except EndOfFrameException:
            if debugLevel >= 2: logger.debug("readLframeData(): Reached the end of this L frame.")
            endOfFrame = True

    if debugLevel >= 2: logger.debug("readLframeData(): L Frame text = \n" + frameText + "\n")

    if debugLevel >= 2: logger.debug('-' * 80)
        
    return (countList, LframeScalarDict) # End readLframeData()


def readMframeData(binFile):
    """For M frame data read the repeated 6-byte chunks of 3 16-bit blocks.
    """
    global frameText
    if debugLevel >= 2: logger.debug('-' * 80)
    if debugLevel >= 2: logger.debug("readMframeData(): frameID = M: Read Mulit-Element Plankton Frame Format Data")

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
            A = readBigEndianUShort(binFile)
            B = readBigEndianUShort(binFile)
            C = readBigEndianUShort(binFile)
            if debugLevel >= 4: logger.debug("readMframeData():\n\tA = %d\n\tB = %d\n\tC = %d" %(A, B, C))

            # Use bit masks to read data MEP values
            n = (A & 0x00008000) >> 15                  # bit to indicate new MEP or not
            p = (A & 0x00003ff8) >> 3                   # peak intensity aka DS
            e = ((A & 0x00000007) << 3) + ((B & 0x00003800) >> 11)      # element number
            l = B & 0x000007ff                      # length, aka time of flight
            s = C & 0x0000ffff                      # scan counter time reference
            if debugLevel >= 2: logger.debug("readMframeData():n = %d, p = %5d, e = %2d, l = %5d, s = %5d" %(n, p, e, l, s))

            nL.append(n)
            pL.append(p)
            eL.append(e)
            lL.append(l)
            sL.append(s)

            if n == 1:
                mepCount += 1 
                partMepCount =0
            elif n == 0:
                partMepCount += 1

            groupCount = groupCount + 1
            if partMepCount > maxPartMepCount:
                maxPartMepCount = partMepCount

    except EndOfFrameException:
        if debugLevel >= 4: logger.debug("Reached the end of this M frame after reading %d groups." % groupCount)

    if debugLevel >= 4: logger.debug("readMframeData(): M Frame text = \n" + frameText + "\n")

    return (nL, pL, eL, lL, sL, mepCount)  # End readMframeData()


def flowSpeed(flowTime, flowCount):
    '''Compute flow speed through LOPC cell following LOPC analysis methods (Herman 2009)
    Input: flowTime and flowCount scalar integer values
    Returns: flow speed in m/s or missing_value if speed cannot be calculated

    '''

    # Prevent divide by 0
    if flowCount == 0:
        return(missing_value)

    # compute and screen flow count 
    FC = float(flowTime) / float(flowCount)                 # mean flow count 

    if FC > 100:                                # regression is not valid for FC>100 (flow speed likeley -> 0)
        return(missing_value)

    elif FC < 7:                                # unlikeley that AUV can go faster than 2 m/s
        return(missing_value)   

    elif FC <= 13:                  # case 1 (FC<=13)
        idx = 0;                # 2nd element of regression coeffs to use

    elif FC > 13:                   # case2 (FC>13)
        idx = 1;                # 2nd element of regression coeffs to use

    # regression coeffs
    a = numpy.array([[23.10410966, 0.198996019], 
              [-1.481499106, -2.603059062],
              [1.566460406, 0.892897609],
              [0.196311142, 0.006191239],
                  [-0.05, -0.0013]])

    sqrtFC = math.sqrt(FC);
    if debugLevel >= 2: logger.debug("flowSpeed(): sqrtFC = %f, idx = %d, a[1,idx] = %f " % (sqrtFC, idx, a[1,idx]))
    speed = a[0,idx] * math.exp(-(a[1,idx] * math.sqrt(sqrtFC) + a[2,idx] * sqrtFC + 
            a[3,idx] * sqrtFC ** 2 + a[4,idx] * sqrtFC ** 3))

    if debugLevel >= 1: logger.debug("flowSpeed(): speed = %f " % speed)

    return(speed)


def readCframeData(binFile):
    """For C frame data - Parses special MBARI C frame data record that Hans writes from the MVC
    """
    global frameText
    if debugLevel >= 2: logger.debug('-' * 80)
    if debugLevel >= 2: logger.debug("readCframeData(): frameID = C: Read CTD Data (timestamp that the mvc writes)")

    nChar = 0
    str = ''
    try:
        while True:
            # Read characters until end of frame
            char = readChar(binFile)
            str += char
            if debugLevel >= 2: logger.debug("readCframeData(): char = %s" % (char))
            nChar += 1

    except EndOfFrameException:
        if debugLevel >= 2: logger.debug("Reached the end of this C frame after reading %d characters. str = %s" % (nChar, str))
        # Parse for epoch seconds that Hans writes, starting after 15 March 2010 (day 074) - hasMBARICframeData flag set in main() and used by other functions
        if hasMBARICframeData:
            try:
                esecs = float(str.split(' ')[0])
            except ValueError:
                str = ''.join(s for s in str if s in string.printable)      # Remove nonprintable characters
                str = re.sub('\s+', '', str)                    # REmove whitespace
                logger.warn("readCframeData(): Unable to parse a float from the string str[:70] = %s" % str[:70])
                esecs = missing_value   

            if debugLevel >= 2: logger.debug("Parse from C Frame esecs = %f" % esecs)
            return esecs
        else:
            logger.info("readCframeData(): This mission is before esecs were written to the C frame.  No attempt made to parse.")
            return missing_value
            
def writeTextLFrame(countList, LframeScalarDict, textFile):
    """Write the L frame in the ASCII text format that is produced by the LOPC software
    and can be processed by Alex Herman's post processing software.

    """

    textFile.write("L1")
    for i in range(32):
        textFile.write(" %d" % countList[i])
    textFile.write("\n")

    textFile.write("L2")
    for i in range(32,64):
        textFile.write(" %d" % countList[i])
    textFile.write("\n")

    textFile.write("L3")
    for i in range(64,96):
        textFile.write(" %d" % countList[i])
    textFile.write("\n")

    textFile.write("L4")
    for i in range(96,128):
        textFile.write(" %d" % countList[i])
    textFile.write("\n")

    textFile.write("L5")
    for k in LframeScalarKeys:
        textFile.write(" %d" % LframeScalarDict[k])
    textFile.write("\n")

    return # End writeTextLFrame()


def unpackLOPCbin(binFile, opts, textFile = None):
    """Loop though the .bin file reading all frames and unpacking as necessary.  Keep
    track of framing errors.  This function does not exit with a return, it reads
    records until an EndOfFileException is raised, which is then caught by the
    routine that calls this.  For this reason we need to assign to global variables
    in order to return values.
    Note that the binary data are written by the LOPC instrument in big-endian. At
    MBARI all of our Linux systems are little-endian.  We need to swap bytes on all 
    the binary words.

    """

    global countBeginLFrameWithoutEndOfPreviousFrame
    global countBeginMFrameWithoutEndOfPreviousFrame
    global countGarbledLFrame
    global ncFile
    global LframeScalarDict
    global debugLevel


    logger.info("unpackLOPCbin(): >>> Unpacking LOPC data from " + binFile.name)
    logger.info("unpackLOPCbin(): >>> Writing NetCDF file " + opts.netCDF_fileName)
    
    if textFile != None:
        logger.info("unpackLOPCbin(): >>> Writing ASCII text file " + opts.text_fileName)
    

    recCount = 0
    dataStructure['lFrameCount'] = 0
    dataStructure['mFrameCount'] = 0

    # Array for the size class dimension [108 microns to 1.5 cm] - for both SEP and MEP data
    dataStructure['binSizeList'] = numpy.array(list(range(108,15015,15)),dtype='float32')

    # Instantiate MepData object to collect and compute MEP parameters
    mepData = lopcMEP.MepData()

    maxLenNList = 0

    countShortLFrameError = 0
    global frameText
    detectedLFrame = False
    detectedMFrame = False

    cFrameEsecs = missing_value
    lastCFrameEsecs = missing_value

    lastTime = time.time()  # For measuring elapsed time in verbose output in loop
    lastLframeCount = 0 # For binningInterval output

    sepCountArraySum = numpy.zeros(len(dataStructure['binSizeList']), dtype='int32')
    lastMEP = None      # For properly building the MEP list (for transferring partial frames between binning intervals)
    sampleCountList = []        # The sampleCount from the instrument - used for getting accurate timeStamp
    lFrameCountWrittenList = [] # For keeping track of which frames (telling us the time) were written to the NetCDF file - this starts conting at 1
    sampleCountWrittenList = [] # For keeping track of which frames (telling us the time) were written to the NetCDF file - to better calculate time...
    cFrameEsecsList = []        # For recording 'ground-truth' time from the MVC 

    outRecNumFunc = record_count().__next__ # Generator of indices for netCDF output

    # write header for ASCII text file, if specified
    if textFile != None:
        textFile.write("# ___ PROCESSING INFORMATION ___\n")
        textFile.write("# Date: %s\n" % time.strftime("%A, %B %d, %Y", time.localtime()))
        textFile.write("# Time: %s\n" % time.strftime("%H:%M:%S", time.localtime()))

    while True:
        """Outer read loop to read all records from the file. The loop is exited by an EndOfFileException which is handled 
        by whatever calls this method."""
        if detectedLFrame:
            frameID = 'L'
        if detectedMFrame:
            frameID = 'M'
        else:
            while True:
                "Read characters until we reach the first <DLE> ('~'), then read another char to get the frameID"
                char = binFile.read(1)
                if debugLevel >= 3: logger.debug("unpackLOPCbin(): char = %s, len(char) = %d" % (char, len(char)))
                if len(char) == 0:
                    raise EndOfFileException
                if char == '~':
                    frameID = binFile.read(1)
                    if debugLevel >= 2: logger.debug('-' * 80)
                    if debugLevel >= 2: logger.debug('unpackLOPCbin(): frameID = %s (lFrameCount = %d, mFrameCount = %d)' % (frameID, dataStructure['lFrameCount'], dataStructure['mFrameCount']))
                    break # Exit the enclosing 'while True:' loop

        if frameID == 'L':
            frameText = str(char) + str(frameID)        # Initialize global debugging text string representation of the data frame
            try:
                (countList, LframeScalarDict) = readLframeData(binFile, sampleCountList)
            except BeginLFrameWithoutEndOfPreviousFrame:
                detectedLFrame = True
                if debugLevel >= 1: logger.error("unpackLOPCbin(): Begin of L Frame (around # %d) witout encountering end of previous frame." % dataStructure['lFrameCount'])
                countBeginLFrameWithoutEndOfPreviousFrame += 1
                continue    # exit to outer while True loop
            except BeginMFrameWithoutEndOfPreviousFrame:
                countBeginMFrameWithoutEndOfPreviousFrame += 1
                if debugLevel >= 1: logger.error("unpackLOPCbin(): Begin of M Frame (around # %d) witout encountering end of previous frame." % dataStructure['lFrameCount'])
                detectedMFrame = True
                continue    # exit to outer while True loop
            except ShortLFrameError:
                if debugLevel >= 1: logger.warn("unpackLOPCbin(): Reached the end of this L frame (around sample # %d) before expected." % dataStructure['lFrameCount'])
                countShortLFrameError += 1
                continue

            except GarbledLFrame:
                countGarbledLFrame += 1
                continue
            else:
                detectedLFrame = False

            # Save values specific to this L frame  
            dataStructure['countShortLFrameError'] = countShortLFrameError
            dataStructure['countBeginLFrameWithoutEndOfPreviousFrame'] = countBeginLFrameWithoutEndOfPreviousFrame
            dataStructure['countBeginMFrameWithoutEndOfPreviousFrame'] = countBeginMFrameWithoutEndOfPreviousFrame
            dataStructure['countUnknownFrameCharacter'] = countUnknownFrameCharacter
            dataStructure['unknownFrameCharacters'] = unknownFrameCharacters
            dataStructure['countGarbledLFrame'] = countGarbledLFrame
            dataStructure['sampleCountList'] = sampleCountList
            dataStructure['cFrameEsecsList'] = cFrameEsecsList

            # Compute flow speed 
            if debugLevel >= 2: logger.debug("unpackLOPCbin(): Calling flowSpeed()")
            dataStructure['flowSpeed'] = flowSpeed(LframeScalarDict['flowTime'], LframeScalarDict['flowCount'])
            if debugLevel >= 2: logger.debug("unpackLOPCbin(): flowSpeed = %f" % dataStructure['flowSpeed'])

            # Save values specifc to the C frame
            if hasMBARICframeData:
                if cFrameEsecs != lastCFrameEsecs:
                    dataStructure['cFrameEsecs'] = cFrameEsecs
                    cFrameEsecsList.append(cFrameEsecs)
                else:
                    dataStructure['cFrameEsecs'] = missing_value
                    cFrameEsecsList.append(missing_value)


            # Open netCDF file on first Frame
            if dataStructure['lFrameCount'] == 0:
                '''Open netCDF file on first read where we get a full count'''
                ncFile = openNetCDFFile(opts)

            # Confirm contents of Dictionary
            if debugLevel >= 2: logger.debug("unpackLOPCbin(): LframeScalarDict contents after return from readLframeData():")
            for k in LframeScalarKeys:
                if debugLevel >= 2: logger.debug("unpackLOPCbin():   %s = %d" % (k, LframeScalarDict[k]))

            # Write ASCII L frame data, if requested
            if opts.text_fileName:
                writeTextLFrame(countList, LframeScalarDict, textFile) 



            # Pad out the list of counts returned by readLframeData so that it's the same as what we write to the netCDF file
            sepCountArray = numpy.zeros(len(dataStructure['binSizeList']), dtype='int32')
            i = 0
            for c in countList:
                sepCountArray[i] = c
                i += 1

            # Append these data to the numpy array that collects the SEP counts for each binning interval
            sepCountArraySum += sepCountArray

            # Count each L frame, at 2 hz this is our best estimate of elapsed time
            dataStructure['lFrameCount'] += 1
            lastCFrameEsecs = cFrameEsecs

            # Verify the collection: loop through all the saved data structures so far.  This debug output was added to discover
            # the fact that Python appends a reference to the LframeScalarDict dictionary resulting in every item in the list to
            # point to the last LframeScalarDict dictionary returned by readLframeData().  Appending a shallow copy() fixes this.
            if debugLevel >= 3: logger.debug("unpackLOPCbin(): Reading data from DataStructure dictionary after appending:")
            for countList, lframeScalarDict in zip(countList, LframeScalarDict):
                if debugLevel >= 3: logger.debug("unpackLOPCbin(): \tlen(countList) = %d" % len(countList))
                if debugLevel >= 3: logger.debug("unpackLOPCbin(): \tnumpy.sum(countList) = %d" % numpy.sum(countList))
                for item in LframeScalarKeys:
                    if debugLevel >= 3: logger.debug("unpackLOPCbin(): \t%s = %d" % (item, lframeScalarDict[item]))

            # Verbose output table column header wait until after first frame so that openNetCDFFile ouput is printed first
            if verboseFlag and dataStructure['lFrameCount'] == 100:
                logger.info("unpackLOPCbin(): " + 
                    ''.join([s.center(12) for s in ('L Frames', 'sample', 'M Frames', 'ShortLFrame', 'LBeforeMEnd', 'MBeforeLEnd', 'UnknownFrame', 'Garbled L')]))
                logger.info("unpackLOPCbin(): " + 
                    ''.join([s.center(12) for s in ('Count', 'Count', 'Count', 'Error', 'Error', 'Error', 'Error', 'Error')]))
                logger.info("unpackLOPCbin(): " + 
                    ''.join([s.center(12) for s in ('--------', '--------', '--------', '-----------', '-----------', '-----------', '-----------', '------------')]))

            # Give a little feedback during this long read process
            if verboseFlag and not dataStructure['lFrameCount'] % 100:
                lastLFrame = 2 * int(dataStructure['tsList'][-1] - dataStructure['tsList'][0])
                logger.info("unpackLOPCbin(): %s %.1f seconds, last L Frame: %d" %
                    (''.join([str(s).center(12) for s in (dataStructure['lFrameCount'], LframeScalarDict['sampleCount'], dataStructure['mFrameCount'], countShortLFrameError,
                                                countBeginLFrameWithoutEndOfPreviousFrame, countBeginMFrameWithoutEndOfPreviousFrame,
                                                countUnknownFrameCharacter, countGarbledLFrame)]),  time.time() - lastTime, lastLFrame ))
                lastTime = time.time()  

        elif frameID == 'M':
            frameText = str(char) + str(frameID)        # Initialize global debugging text string representation of the data frame
            try:
                (nL, pL, eL, lL, sL, mepCount) = readMframeData(binFile)
                ##print "pL = " + str(pL)
            except BeginMFrameWithoutEndOfPreviousFrame:
                detectedMFrame = True
                continue    # exit to outer while True loop
            except BeginLFrameWithoutEndOfPreviousFrame:
                stillHaveLFrame = True
                continue    # exit to outer while True loop
            else:
                detectedMFrame = False
            
            dataStructure['mFrameCount'] += 1

            # Collect MEP data in the instance of the lopcMEP.MepData class 
            mepData.extend(nL, pL, eL, lL, sL) 
            ##if len(mepData.mepList) > 0:  
            ##  mepData.build(mepData.mepList[-1])
            ##else:
            ##  mepData.build()
            ##print "mepData = %s\n" % (mepData)

            ##foo = raw_input('Got M frame')

            # Write ASCII M frame data - for just this frame, if requested
            if opts.text_fileName:
                textFile.write(mepData.frameToASCII(nL, pL, eL, lL, sL))


        elif frameID == 'C':
            frameText = str(char) + str(frameID)        # Initialize global debugging text string representation of the data frame
            if debugLevel >= 2: logger.debug("unpackLOPCbin(): C Frame text = \n" + frameText + "\n")

            # Note that before 2010074 a C frame is probably garbled data and should not be parsed for esecs.  This logic is in readCframeData().
            try:
                esecs  = readCframeData(binFile)
            except BeginLFrameWithoutEndOfPreviousFrame:
                detectedLFrame = True
                if debugLevel >= 1: logger.error("unpackLOPCbin(): Begin of L Frame (around # %d) witout encountering end of previous frame." % dataStructure['lFrameCount'])
                countBeginLFrameWithoutEndOfPreviousFrame += 1
                continue    # exit to outer while True loop
            except BeginMFrameWithoutEndOfPreviousFrame:
                countBeginMFrameWithoutEndOfPreviousFrame += 1
                if debugLevel >= 1: logger.error("unpackLOPCbin(): Begin of M Frame (around # %d) witout encountering end of previous frame." % dataStructure['lFrameCount'])
                detectedMFrame = True
                continue    # exit to outer while True loop

            if debugLevel >= 1: logger.debug("C frame after L Frame # %d esecs = %f" % (dataStructure['lFrameCount'], esecs))
            cFrameEsecs = esecs
            

        elif frameID == '~':
            pass # Just skip this and look for the next valid frameID

        else:
            if debugLevel >= 2: logger.debug("unpackLOPCbin(): Encountered frameID = %s at L frame # %d" % (frameID, dataStructure['lFrameCount']))

        #
        # At binning interval save calculated ESD and other derived products from the MEP data
        #
        if dataStructure['lFrameCount'] > lastLframeCount + binningInterval:
            ##raw_input('Paused at count = %d with binningInterval = %d' % (dataStructure['lFrameCount'], binningInterval))

            ##logging.getLogger("logger").setLevel(logging.DEBUG)
            ##debugLevel = 1
            
            # Build the mepList in mepData from all that data that have been collected so far, this allows .count() to work.
            mepData.build(lastMEP)

            if debugLevel >= 1: logger.debug("unpackLOPCbin(): len(mepData.mepList) = %d" % len(mepData.mepList))

            if debugLevel >= 2: logger.debug("unpackLOPCbin(): SEP-only counts")
            if debugLevel >= 2: logger.debug("unpackLOPCbin(): ===============")
            if debugLevel >= 1: logger.debug("unpackLOPCbin(): SEP-only counts: sepCountArraySum.sum() = %d" % sepCountArraySum.sum())
            if debugLevel >= 2: logger.debug("unpackLOPCbin(): sepCountArraySum = %s" % sepCountArraySum)

            # Add the MEP counts to the sepCountArray that has been summed up
            ##logging.getLogger("MEP").setLevel(logging.DEBUG)
            ##logger.info("\n\nmepData = \n%s" % (mepData))

            ##countArray = mepData.count(dataStructure['binSizeList'], sepCountArraySum)

            mepCountArray = mepData.count(dataStructure['binSizeList'])
            lcAIcrit = 0.4
            lcESDmin = 800
            lcESDmax = 1200
            LCcount = mepData.countLC(opts.LargeCopepod_AIcrit, opts.LargeCopepod_ESDmin, opts.LargeCopepod_ESDmax)

            trAIcrit = 0.4
            (transCount, nonTransCount) = mepData.countTrans(trAIcrit)

            countArray = sepCountArraySum + mepCountArray       # With numpy Arrays we can do element-by-element addition with '+'

            if debugLevel >= 1: logger.debug("unpackLOPCbin(): MEP-only counts: mepCountArray.sum() = %d" % mepCountArray.sum())

            if debugLevel >= 2: logger.debug("unpackLOPCbin(): After adding MEP counts to SEP counts")
            if debugLevel >= 2: logger.debug("unpackLOPCbin(): =====================================")
            if debugLevel >= 1: logger.debug("unpackLOPCbin(): SEP+MEP counts: countArray.sum() = %d" % countArray.sum())
            if debugLevel >= 2: logger.debug("unpackLOPCbin(): countArray = %s" % countArray)
            if debugLevel >= 1: logger.debug("unpackLOPCbin(): LCcount = %d" % LCcount)
            if debugLevel >= 1: logger.debug("unpackLOPCbin(): transCount = %d, nonTransCount = %d" % (transCount, nonTransCount))
            if mepCountArray.sum() > 0:
                if debugLevel >= 1: logger.debug("unpackLOPCbin(): ai:   mean = %f, min = %f, max = %f, std = %f" % 
                        (mepData.aiArray.mean(), mepData.aiArray.min(), mepData.aiArray.max(), mepData.aiArray.std()))
                if debugLevel >= 1: logger.debug("unpackLOPCbin(): esd:   mean = %f, min = %f, max = %f, std = %f" % 
                        (mepData.esdArray.mean(), mepData.esdArray.min(), mepData.esdArray.max(), mepData.esdArray.std()))


            ##logging.getLogger("logger").setLevel(logging.INFO)
            ##debugLevel = 0
            ##raw_input('Paused ')
            ##if nonTransCount > 0:
            ##  raw_input('Paused ')

            # Collect all of the MEP related data and stats into a dictionary to make passing to writeNetCDFRecord() a little easier
            # The corresonding MEPDataDictKeys list needs to be in sync with the keys in this list
            MEPDataDict = {}
            MEPDataDict['sepCountList'] = list(sepCountArraySum)
            MEPDataDict['mepCountList'] = list(mepCountArray)
            MEPDataDict['countList'] = list(countArray)

            MEPDataDict['LCcount'] = LCcount
            MEPDataDict['transCount'] = transCount
            MEPDataDict['nonTransCount'] = nonTransCount

            if mepCountArray.sum() != 0:
                MEPDataDict['ai_mean'] = mepData.aiArray.mean()
                MEPDataDict['ai_min'] = mepData.aiArray.min()
                MEPDataDict['ai_max'] = mepData.aiArray.max()
                MEPDataDict['ai_std'] = mepData.aiArray.std()
                MEPDataDict['esd_mean'] = mepData.esdArray.mean()
                MEPDataDict['esd_min'] = mepData.esdArray.min()
                MEPDataDict['esd_max'] = mepData.esdArray.max()
                MEPDataDict['esd_std'] = mepData.esdArray.std()
            
            # Write accumulated counts to the NetCDF file
            # Other data (e.g. C frame esecs) get written too - from the dataStructure hash list - missing values filling in the gaps
            writeNetCDFRecord(ncFile, MEPDataDict, LframeScalarDict, outRecNumFunc)

            # Save the LframeCount to use for writing the time axis data when we close the file - it's a python array index, so subtract 1
            ##logger.info("unpackLOPCbin(): appending lFrameCount = %d" % dataStructure['lFrameCount'])
            lFrameCountWrittenList.append(dataStructure['lFrameCount'] - 1)
            sampleCountWrittenList.append(LframeScalarDict['sampleCount'])

            # Save the last MEP (in case it's the beginning of the next one) 
            if len(mepData.mepList) > 0:
                lastMEP = mepData.mepList[-1]

            # Clear MEP lists by creating a new mepData object and zero the SEP count sum list
            mepData = lopcMEP.MepData()
            sepCountArraySum = numpy.zeros(len(dataStructure['binSizeList']), dtype='int32')

            lastLframeCount = dataStructure['lFrameCount']

            dataStructure['lFrameCountWrittenList'] = lFrameCountWrittenList
            dataStructure['sampleCountWrittenList'] = sampleCountWrittenList
            ##logger.info("unpackLOPCbin(): dataStructure['lFrameCountWrittenList'] = " + str(dataStructure['lFrameCountWrittenList']))


        # NOTE: This function cannot return any values as it reads records from the input file until
        # the end is trapped with an exception.  That is now this function ends.  This is why assignments
        # are made to the global dataStructure dictionary - that's how we pass information back.

    return # End unpackLOPCbin()


def constructTimestampList(binFile, recCount = None, sampleCountList = None, cFrameEsecsList = None):
    """Given an absolute path to an lopc.bin file (`binFile`) lookup 
    the start and and time from the associated parosci.nc file
    and generate a list of Unix epoch second time values.  If 
    `recCount` and `sampleCountList` are provided then construct a
    more accurate time value list based on an index to a 2 Hz time
    base provided by the `sampleCountList`.

    """

    global dataStructure
    
    # Get time information from ctd file associated with this mission.  Translate .bin file name to a missionlog netcdf file:
    #   /mbari/AUVCTD/missionlogs/2009/2009084/2009.084.00/lopc.bin -> 
    #   /mbari/ssdsdata/ssds/generated/netcdf/files/ssds.shore.mbari.org/auvctd/missionlogs/2009/2009084/2009.084.00/parosci.nc
    parosciURL = 'http://dods.mbari.org/cgi-bin/nph-nc/data/ssdsdata/ssds/generated/netcdf/files/ssds.shore.mbari.org/auvctd/'
    parosciURL += '/'.join(binFile.name.split('/')[-5:-1]) + '/parosci.nc'
    logger.info("constructTimestampList(): parosciURL = %s" % parosciURL)
    logger.info("constructTimestampList(): Using pydap to get start and end epoch seconds for this mission from this URL:")
    logger.info("constructTimestampList(): %s" % parosciURL)
    sensor_on_time = 0  # Bogus initial starting time
    sensor_off_time = 10000 # Bogus initial ending time
    try:
        parosci = pydap.client.open_url(parosciURL)
    except pydap.exceptions.ClientError:
        logger.error("constructTimestampList(): Could not open %s.  Cannot process until the netCDF file exists.  Exiting." % parosciURL)
        sys.exit(-1)
    else: 
        sensor_on_time = parosci.time[0]
        sensor_off_time = parosci.time[-1]

    logger.info("constructTimestampList(): From associated paroscinc file: sensor_on_time = %.1f, sensor_off_time = %.1f" %(sensor_on_time, sensor_off_time))
    logger.info("constructTimestampList(): Duration is %d seconds.  Expecting to read %d L frames from the lopc.bin file." % (int(sensor_off_time - sensor_on_time), 2 * int(sensor_off_time - sensor_on_time)))

    if recCount == None:
        recCount =  2 * int(sensor_off_time - sensor_on_time)
        logger.info("constructTimestampList(): recCount not passed in, assuming we'll have %d records from the lopc.bin file." % (recCount))

    timestampList = []
    logger.debug("constructTimestampList(): sampleCountList = %s" % (sampleCountList))
    if sampleCountList != None:
        # Analyze the sampleCountList - correct for 16-bit overflow, and interpolate over 0-values, stuck values, and spikes
        logger.info("constructTimestampList(): Calling correctSampleCountList() with sampleCountList[0] = %d and len(sampleCountList) = %d" % 
            (sampleCountList[0], len(sampleCountList)))
        correctedSampleCountList = correctSampleCountList(sampleCountList)  
        logger.info("constructTimestampList(): correctSampleCountList() returned correctedSampleCountList = %s with len(correctedSampleCountList) = %d" % 
            (correctedSampleCountList, len(correctedSampleCountList)))

        # SampleCount does not always begin at 0 so subtract the first element's value so that we can use it as a time base
        # Use the instrument concept of sample count to construct the best possible time stamp list
        logger.info("constructTimestampList(): Subtracting %d from all values of correctedSampleCountList" % correctedSampleCountList[0])
        deltaT = 0.5
        logger.info("constructTimestampList(): Constructing timestampArray from instrument corrected sampleCount and constant deltaT = %f" % deltaT)
        logger.info("constructTimestampList(): New re-zeroed correctedSampleCountList = %s" %  ((correctedSampleCountList - correctedSampleCountList[0]) * deltaT))
        timestampArray = numpy.array(((correctedSampleCountList - correctedSampleCountList[0]) * deltaT), dtype='float64') + sensor_on_time
        logger.info("constructTimestampList(): timestampArray = %s" % timestampArray)
        logger.info("constructTimestampList(): timestampArray[:2] = [%.1f %.1f, ..., timestampArray[-2:] = %.1f %.1f]" % 
            (timestampArray[0], timestampArray[1], timestampArray[-2], timestampArray[-1]))
        
        timestampList = list(timestampArray)
        logger.info("constructTimestampList(): timestampList[:2] = %s, ..., timestampList[-2:] = %s]" % (timestampList[:2], timestampList[-2:]))
        logger.info("constructTimestampList(): Subsampling correctedSampleCountList (len = %d) according to what got written to the netCDF file by the binning interval" % len(correctedSampleCountList))
        logger.info("constructTimestampList(): lFrameCountWrittenList[:2] = %s, ... lFrameCountWrittenList[-2:] = %s" % 
            (dataStructure['lFrameCountWrittenList'][:2], dataStructure['lFrameCountWrittenList'][-2:]) )

        if dataStructure['lFrameCountWrittenList'][-1] > len(correctedSampleCountList):
            logger.info("constructTimestampList(): lFrameCountWrittenList[-1] > len(correctedSampleCountList). Truncating the list before subsampling.")
            dataStructure['lFrameCountWrittenList'] = dataStructure['lFrameCountWrittenList'][:-2]

        subSampledCorrectedSampleCountList = correctedSampleCountList[dataStructure['lFrameCountWrittenList']]
        dataStructure['correctedSampleCountList'] = subSampledCorrectedSampleCountList
        logger.info("constructTimestampList(): len(subSampledCorrectedSampleCountList) = %d" % len(subSampledCorrectedSampleCountList))

    else:   
        # Create timestamp list that corresponds best to the frame times - uses the parosci on & off times to construct an evenly spaced timestampList
        # Used at the beginning of processing to estimate time to completion - DO NOT USE THE RESULTS OF THIS IN THE NETCDF FILE!
        deltaT = float(sensor_off_time - sensor_on_time) / float(recCount - 1)
        if debugLevel >= 1: logger.debug("constructTimestampList(): deltaT = %f [Should be 0.5 seconds]" % (deltaT) )
        for i in range(recCount):
            timestamp = sensor_on_time + i * deltaT
            timestampList.append(timestamp) 

        return timestampList

    logger.info("constructTimestampList(): len(timestampList) = %i" % (len(timestampList)) )
    logger.info("constructTimestampList(): timestampList[:2] = %s, ..., timestampList[-2:] = %s]" % (timestampList[:2], timestampList[-2:]))

    logger.info("constructTimestampList(): Subsampling timestampList (len = %d) according to what got written to the netCDF file by the binning interval" % len(timestampList))
    writtenSampleCounts = list(numpy.array(dataStructure['sampleCountWrittenList'], dtype='int32') - 1) # Subtract 1 for Python's 0-based list indexing
    writtenLFrameCounts = list(numpy.array(dataStructure['lFrameCountWrittenList'], dtype='int32') - 1) # Subtract 1 for Python's 0-based list indexing
    logger.info("constructTimestampList(): Taking indices [%s ... %s] from timestampList to create subSampledTimestampList" % 
        (writtenLFrameCounts[:2], writtenLFrameCounts[-2:]))
    subSampledTimestampList = list(numpy.array(timestampList)[writtenLFrameCounts])
    logger.info("constructTimestampList(): len(subSampledTimestampList) = %d" % len(subSampledTimestampList))
    logger.debug("constructTimestampList(): subSampledTimestampList[:2] = %s, ..., subSampledTimestampList[-2:] = %s" % (subSampledTimestampList[:2], subSampledTimestampList[-2:]))

    timestampList = subSampledTimestampList

    if cFrameEsecsList != None:
        # First replace all NaNs with an interpolation between the non-NaNed values
        logger.info("constructTimestampList(): Finding elements of cFrameEsecsList that != %d" % missing_value)
        ##logger.info("constructTimestampList(): cFrameEsecsList = " + str(cFrameEsecsList))
        hasValue_indices = list(numpy.nonzero(numpy.array(cFrameEsecsList, dtype='float64') != missing_value)[0])
        logger.debug("constructTimestampList(): len(cFrameEsecsList) = %d" % len(cFrameEsecsList))
        logger.debug("constructTimestampList(): len(hasValue_indices) = %d" % len(hasValue_indices))
        logger.debug("constructTimestampList(): hasValue_indices[:2] = %s, ..., hasValue_indices[-2:] = %s" % (hasValue_indices[:2],  hasValue_indices[-2:]))

        # Interpolate cFrameEsecs (sent every 5 secsonds or so) to the same sample interval as timestampList
        try:
            cFrameEsecsListInterpolated = numpy.interp(writtenSampleCounts, hasValue_indices, numpy.array(cFrameEsecsList)[hasValue_indices])
        except ValueError as e:
            logger.error('Cannot interpolate to cFrameEsecsListInterpolated: %s', str(e))
        else:
            logger.debug("constructTimestampList(): cFrameEsecsListInterpolated[:2] = %s, ..., cFrameEsecsListInterpolated[-2:] = %s" % (cFrameEsecsListInterpolated[:2], cFrameEsecsListInterpolated[-2:]))
            cFrameEsecsList = cFrameEsecsListInterpolated

    return timestampList, cFrameEsecsList # End constructTimestampList()



def getMedian(numericValues):
    '''Return the median of the numericValues'''

    theValues = sorted(numericValues)

    if len(theValues) % 2 == 1:
        return theValues[(len(theValues)+1)/2-1]
    else:
        lower = theValues[len(theValues)/2-1]
        upper = theValues[len(theValues)/2]

    return (float(lower + upper)) / 2  


def interpolate(darray, i, npts, goodValueFcn, spikeValue):
    """Replace `darray[i]` with linearly interpolated value found within `npts` 
    of `i` that is not `spikeValue` based on function in `goodValueFcn`

    """
    
    logger.debug("interpolate(): Replacing value %d at index %d" % (darray[i], i))

    # locate start point for interpolation
    for si in range(i - 1, i - npts, -1):
        logger.debug("interpolate(): Looking up npts = %d to index %d for starting goodValue at index = %d, value = %d" % (npts, i - npts, si, darray[si]))
        s_indx = si
        if goodValueFcn(darray[si], spikeValue):
            break

    # locate end point for interpolation
    for ei in range(i, i + npts):
        logger.debug("interpolate(): Looking down npts = %d to index %d for ending goodValue at index = %d, value = %d" % (npts, i + npts, ei, darray[ei]))
        e_indx = ei
        if goodValueFcn(darray[ei], spikeValue):
            break
             
    logger.debug("interpolate(): Replacing value with interpolation over values %d and %d at indices %d and %d" % (darray[s_indx], darray[e_indx], s_indx, e_indx)          )
    value = int( (i - s_indx) * (darray[e_indx] - darray[s_indx]) / (e_indx - s_indx) ) + darray[s_indx]
    logger.debug("interpolate(): interpolated value = %d" % value)
    
    return value # End interpolate()

def deSpike(sc_array, crit):
    """Remove both single point and multiple point spikes from sc_list based on the crit value.
    The algorithm computes the diff of the values and identifies the nature of the spikes (diff
    values > crit) and interpolates over them returning a numpy array of monotonically increasing
    de-spiked sample count values.

    """

    # Make copy and get first difference of sample count lust
    sampleCountList = sc_array.copy()
    d_sampleCountList = numpy.diff(sampleCountList)
    spike_crit = 300    # equivalent to 2.5 minutes
    logger.info("deSpike(): d_sampleCountList = %s" % d_sampleCountList)
    spike_indx = numpy.nonzero(abs(d_sampleCountList) > spike_crit)[0] + 1
    logger.info("\ndeSpike(): Found %d spike indicators at indices: %s" % (len(spike_indx), spike_indx))
    i = 0
    lastIndx = 0
    endOfMultipleSpikeIndx = 0
    for indx in spike_indx: 
        logger.info("deSpike(): Identifying nature of spike value = %d at index = %d" % (sampleCountList[indx], indx))

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
            logger.info("deSpike(): Single point spike, interpolating over this point")

            # Test for interpolation
            def notSpikeValue(value, spikeValue):
                return value != spikeValue

            logger.info("deSpike(): minIndx = %d, indx = %d, maxIndx = %d" % (minIndx, indx, maxIndx))

            logger.info("deSpike(): Before interpolation: %s" % sampleCountList[minIndx:maxIndx] )
            sampleCountList[indx] = interpolate(sampleCountList, indx, 5, notSpikeValue, sampleCountList[indx])
            logger.info("deSpike(): After  interpolation: %s" % sampleCountList[minIndx:maxIndx] )
        
        elif indx == lastIndx + 1:
            logger.debug("deSpike(): Second part of single point spike, skipping.")

        elif indx == endOfMultipleSpikeIndx:
            logger.debug("deSpike(): Second part of multiple point spike, skipping.")

        else:
            if i + 1 < len(spike_indx):
                logger.info("\ndeSpike(): Multiple point spike over indices [%d:%d], interpolating over these points" % (indx, spike_indx[i+1]))

                # See if spike values are less than or greater than what they need to be and set up the test for interpolation
                if sampleCountList[indx] > sampleCountList[indx-1]:
                    spikeValue = numpy.min(sampleCountList[indx:spike_indx[i+1]])
                    logger.debug("deSpike(): Spike values are greater than the true values, good values are less than %d" % spikeValue)
                    def notSpikeValues(value, spikeValue):
                        logger.debug("deSpike(): notSpikeValues(): Testing if value = %d is less than %d" % (value, spikeValue))
                        return value < spikeValue
                else:
                    spikeValue = numpy.max(sampleCountList[indx:spike_indx[i+1]])
                    logger.debug("deSpike(): Spike values are less than the true values, good values are greater than %d" % spikeValue)
                    def notSpikeValues(value, spikeValue):
                        logger.debug("deSpike(): notSpikeValues(): Testing if value = %d is greater than %d" % (value, spikeValue))
                        return value > spikeValue

                logger.info("deSpike(): Before interpolation: %s" % sampleCountList[minIndx:spike_indx[i+1]+3] )
                for ix in range(indx, spike_indx[i+1]):
                    lookingRange = spike_indx[i+1] - indx + 2
                    logger.debug("deSpike(): Calling interpolate with ix = %d, lookingRange = %d" % (ix, lookingRange))
                    if lookingRange > 20:       # Not sure why I have this here, make it just a WARN.  -MPM 23 June 2011
                        logger.warn("deSpike(): lookingRange = %d is too big, returning for you to examine..." % lookingRange)
                        input('Paused')
                        ##logger.error("deSpike(): lookingRange = %d is too big, returning for you to examine..." % lookingRange)
                        return sampleCountList

                    sampleCountList[ix] = interpolate(sampleCountList, ix, lookingRange, notSpikeValues, spikeValue)

                logger.info("deSpike(): After  interpolation: %s" % sampleCountList[minIndx:spike_indx[i+1]+3])
                endOfMultipleSpikeIndx = spike_indx[i+1]

            else:
                logger.info("\ndeSpike(): This is the last element in spike_indx.  No need to deSpike().")

        lastIndx = indx
        i += 1

    return sampleCountList # End deSpike()


def correctSampleCountList(orig_sc):
    """Find indices where overflow happens and assign offset-corrected slices to new sampleCountList.
    Use numpy arrays from original passed in list.  Replace zero values, stuck values, and spikes
    with interpolations.

    """

    global dataStructure

    sampleCountList = numpy.array(orig_sc).copy()
    d_orig_sc = numpy.diff(numpy.array(orig_sc))
    overflow_indx = numpy.nonzero((d_orig_sc < -65530) & (d_orig_sc > -65540))[0]   # Pull off 0 index of tuple of array
    overflow_indx += 1      # Start slices one index more 
    logger.info("correctSampleCountList(): Found overflows at indices: %s" % overflow_indx)
    logger.debug("correctSampleCountList(): len(sampleCountList) = %d" % len(sampleCountList))
    i = 0
    for indx in overflow_indx:
        i += 1
        logger.info("correctSampleCountList(): Assigning values from slice starting at index %d, i = %d" % (indx, i))
        sampleCountList[indx:] = numpy.array(orig_sc[indx:]).copy() + (i * 65537)

    # Find and interpolate over 0 values
    npts = 5    # Number of points to look back & forward for non-zero values
    zero_indx = numpy.nonzero(sampleCountList == 0)[0]
    logger.info("correctSampleCountList(): Original sampleCountList =  %s" % (sampleCountList))
    logger.info("correctSampleCountList(): Found %d 0 values at indices: %s" % (len(zero_indx), zero_indx))

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

        logger.debug("correctSampleCountList(): Interpolating over value = %d at index = %d" % (sampleCountList[indx], indx))
        logger.debug("correctSampleCountList(): Before interpolation: %s" % sampleCountList[minIndx:maxIndx] )
        sampleCountList[indx] = interpolate(sampleCountList, indx, 5, nonZeroValue, 0)
        logger.debug("correctSampleCountList(): After  interpolation: %s" % sampleCountList[minIndx:maxIndx] )

    # Find stuck values and assign to one more than the previous value
    d_sampleCountList = numpy.diff(sampleCountList)
    stuck_indx = numpy.nonzero(d_sampleCountList == 0)[0] + 1
    logger.info("correctSampleCountList(): Found %d stuck values at indices: %s" % (len(stuck_indx), stuck_indx))
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

        logger.debug("correctSampleCountList(): Replacing stuck value = %d at index = %d" % (sampleCountList[indx], indx))
        logger.debug("correctSampleCountList(): Before incrementing stuck value: %s" % sampleCountList[minIndx:maxIndx] )
        sampleCountList[indx] = sampleCountList[indx-1] + 1
        logger.debug("correctSampleCountList(): After  incrementing stuck value: %s" % sampleCountList[minIndx:maxIndx] )

    # Find and interpolate over spikes using crit=300 (appropriate for 2009.084.02)
    sampleCountList = deSpike(sampleCountList, 300)
        
    logger.info("correctSampleCountList(): After despike()  sampleCountList =  %s" % (sampleCountList))
    return sampleCountList # End correctSampleCountList()


def testCSCL(lopcNC = 'http://dods.mbari.org/cgi-bin/nph-nc/data/ssdsdata/ssds/generated/netcdf/files/ssds.shore.mbari.org/auvctd/missionlogs/2009/2009084/2009.084.02/lopc.nc'):
    """
    Test the correctSampleCountList() function by reading (via pydap) a previously saved netCDF file's original sampleCount.
    Runs the code with debugging turned on.

    ipyhton test procedure:

import lopcToNetCDF
url='http://dods.mbari.org/cgi-bin/nph-nc/data/ssdsdata/ssds/generated/netcdf/files/ssds.shore.mbari.org/auvctd/missionlogs/2008/2008154/2008.154.01/lopc.nc'
(scList, correctedSC) = lopcToNetCDF.testCSCL(url)
plot(scList)
plot(correctedSC)

    """

    logging.getLogger("logger").setLevel(logging.DEBUG)

    logger.info("testCSCL(): Opening URL: %s" % lopcNC) 
    ds = pydap.client.open_url(lopcNC)

    logger.info("testCSCL(): Extracting sampleCount[:] from the pydap proxy.")
    scList = ds['sampleCount'].sampleCount[:]

    logger.info("testCSCL(): Calling correctSampleCountList()")
    correctedSC = correctSampleCountList(scList)

    logger.info("testCSCL(): Returning scList and correctedSC")
    logger.info("testCSCL(): From ipython you may plot with\nplot(scList)\nplot(correctedSC)\n")

    return scList, correctedSC # End testCSCL()


def testFlowSpeed():
    '''Test using values provided in Herman 2009'''
    logger.info("testFlowSpeed(): flowSpeed = %f for flowCount = %d and flowTime = %d" % (flowSpeed(1026, 143), 143, 1026))

    return


def openNetCDFFile(opts):
    """Open netCDF file and write some global attributes and dimensions that
    we know at the beginning of processing.

    """

    ncFileName = opts.netCDF_fileName
    logger.info("openNetCDFFile(): Will output NetCDF file to %s" % ncFileName)

    # Improve long names of MEP counts based on passed in arguements
    MEPDataDictLongName['LCcount'] += " with aiCrit = %.2f, esdMinCrit = %.0f, esdMaxCrit = %.0f" % (opts.LargeCopepod_AIcrit, opts.LargeCopepod_ESDmin, opts.LargeCopepod_ESDmax)
    MEPDataDictLongName['transCount'] += " with ai < %.2f" % opts.trans_AIcrit
    MEPDataDictLongName['transCount'] += " with ai > %.2f" % opts.trans_AIcrit

    #
    # Set the PreFill option to False to improve writing performance
    #
    opt = Nio.options()
    opt.PreFill = False

    #
    # Set the history attribute
    #
    hatt = "Created by lopcToNetCDF.py on " + time.ctime()

    #
    # Open the netCDF file and set some global attributes
    #
    ncFile = Nio.open_file(ncFileName, "c", opt, hatt)

    missionName = ''
    try:
        missionName = ncFileName.split('/')[-2]
    except IndexError:
        logger.warn("openNetCDFFile(): Could not parse missionName from netCDF file name - probably not production processing.")

    logger.info("openNetCDFFile(): missionName = %s" % missionName)

    ncFile.title = "Laser Optical Plankton Counter measurements from AUV mission " + missionName
    ncFile.institution = "Monterey Bay Aquarium Research Institute"
    ncFile.summary = "These data have been processed from the original lopc.bin file produced by the LOPC instrument.  The data in this file are to be considered as simple time series data only and are as close to the original data as possible.  Further processing is required to turn the data into a time series of profiles."
    ncFile.keywords = "plankton, particles, detritus, marine snow, particle counter"
    ncFile.Conventions = "CF-1.1"
    ncFile.standard_name_vocabulary = "CF-1.1"

    # Time dimension is unlimited, we'll write variable records then write the time data at the end of processing
    ncFile.create_dimension('time', None)       # Unlimited dimension

    # Create bin size dimension for the counts variable 
    ncFile.create_dimension('bin', len(dataStructure['binSizeList']))
    logger.info("openNetCDFFile(): Writing bin axis for len(dataStructure['binSizeList']) = %d" % len(dataStructure['binSizeList']))
    ncFile.create_variable("bin", 'f', ('bin',))
    ncFile.variables['bin'].units = "microns"
    ncFile.variables['bin'].long_name = "Equivalent Spherical Diameter" 
    ncFile.variables['bin'][:] = dataStructure['binSizeList']

    # --- Create the record variables ---
    for key in MEPDataDictKeys:
        if key.endswith('List'):
            logger.info("openNetCDFFile(): Creating variable %s on axes time and bin" % key)
            ncFile.create_variable(key, 'i', ('time', 'bin'))
        else:
            logger.info("openNetCDFFile(): Creating variable %s on axes time" % key)
            ncFile.create_variable(key, 'f', ('time',))

        ncFile.variables[key].units = MEPDataDictUnits[key]
        ncFile.variables[key].long_name = MEPDataDictLongName[key]


    logger.info("openNetCDFFile(): Creating variable countSum on axis time")
    ncFile.create_variable('countSum', 'i', ('time', ))
    ncFile.variables['countSum'].units = 'count'

    # Create scalar list item variable
    logger.info("openNetCDFFile(): Creating variables from engineering data collected in LframeScalarDict:")
    for var in LframeScalarKeys:
        logger.info("openNetCDFFile(): Creating variable %s on axis time" % var)
        if var == 'snapshot' or var == 'bufferStatus':
            ncFile.create_variable(var, 'b', ('time', ))
        else:
            ncFile.create_variable(var, 'i', ('time', ))

        if debugLevel >= 2: logger.debug("openNetCDFFile():   units = %s " % LframeScalarDictUnits[var])
        if LframeScalarDictUnits[var] == None:
            ncFile.variables[var].units = ''    # For unitless variables we do need a '' units attribute
        else:
            ncFile.variables[var].units = LframeScalarDictUnits[var]

        if debugLevel >= 2: logger.debug("openNetCDFFile():   long_name = %s " % LframeScalarDictLongName[var])
        if LframeScalarDictLongName[var] != None:
            ncFile.variables[var].long_name = LframeScalarDictLongName[var]

    # Create other (mainly Error count) variables from the dataStructure 
    logger.info("openNetCDFFile(): Creating variables from processing information collected in the dataStructure dictionary:")
    for var in list(dataStructure.keys()):
        logger.info("openNetCDFFile(): Checking var = %s" % var)
        if var.startswith('count'):
            logger.info("openNetCDFFile(): Creating variable %s on axis time" % var)
            ncFile.create_variable(var, 'i', ('time', ))
            ncFile.variables[var].units = 'count'
            ncFile.variables[var].long_name = dataStructureLongName[var]
        if var.startswith('flowSpeed'):
            logger.info("openNetCDFFile(): Creating variable %s on axis time" % var)
            ncFile.create_variable(var, 'f', ('time', ))
            ncFile.variables[var].units = 'm/s'
            ncFile.variables[var]._FillValue = missing_value
            ncFile.variables[var].missing_value = missing_value
            ncFile.variables[var].long_name = dataStructureLongName[var]

    # Create cFrameEsecs variable for missions after March 2010 when Hans added it
    # There is a record for each L frame record, but most are fill values as Hans sends an updated Esecs every 5 seconds or so
    if hasMBARICframeData:
        ncFile.create_variable('cFrameEsecs', 'f', ('time', ))
        ncFile.variables['cFrameEsecs'].units = 'seconds since 1970-01-01 00:00:00'
        ncFile.variables['cFrameEsecs']._FillValue = missing_value
        ncFile.variables['cFrameEsecs'].missing_value = missing_value
        ncFile.variables['cFrameEsecs'].long_name = 'Epoch seconds from the main vehicle computer fed into the CTD port of the LOPC at 0.2 Hz'

    return ncFile # End openNetCDFFile()


def record_count():
    """Generator function to return the next integer for an incrementing index. 
    """
    k = -1
    while True:
        k += 1
        yield k


def writeNetCDFRecord(ncFile, MEPDataDict, LframeScalarDict, outRecNumFunc):
    """Write records to the unlimited (time) dimension
    """


    indx = outRecNumFunc()
    

    if debugLevel >= 1: logger.debug("writeNetCDFRecord(): Appending variables to time axis at index # %d:" % indx)
    ##if debugLevel >= 1: logger.debug("writeNetCDFRecord(): appending countList variable for len(countList) = %d , len(dataStructure['binSizeList']) = %d" % (len(countList),  len(dataStructure['binSizeList'])))
    ##if debugLevel >= 2: logger.debug("writeNetCDFRecord(): countList = %s" % countList)

    # Make sure that the countList is the right length
    ##lenInitialbinSizeList = 128   # Hard coded for now, but if this should change we need to assign this variable when counts is created in openNetCDFFile()  
    lenInitialbinSizeList = 994
    if len(MEPDataDict['countList']) != lenInitialbinSizeList:
        logger.warn("writeNetCDFRecord(): len(countList) of %d != %d" % (len(MEPDataDict['countList']), lenInitialbinSizeList))
        if debugLevel >= 1: logger.warn("writeNetCDFRecord(): appending the last good record counts as I doubt we can trust the values that were parsed in this incomplete record.")
        ##countList = [0] * lenInitialbinSizeList   # A list of zero counts
        
    # Go through all the MEP data items, those that end in 'List' are 2D and were properly dimensioned in openNetCDFFile()
    for key in list(MEPDataDict.keys()):
        ncFile.variables[key][indx] = MEPDataDict[key]

    ##ncFile.variables['SEPcounts'][indx] = sepCountList
    ##ncFile.variables['MEPcounts'][indx] = mepCountList
    ##ncFile.variables['counts'][indx] = countList
    ##ncFile.variables['LCcount'][indx] = LCcount
    ##ncFile.variables['transCount'][indx] = transCount
    ##ncFile.variables['nonTransCount'][indx] = nonTransCount

    # Write the countSum - making it easier for the user of the data
    countSum = numpy.sum(MEPDataDict['countList'])
    if debugLevel >=2: logger.debug ("writeNetCDFRecord(): appending countSum = %d" % countSum)
    ncFile.variables['countSum'][indx] = countSum

    # Write scalar list items
    for var in LframeScalarKeys:
        if debugLevel >= 2: logger.debug("writeNetCDFRecord(): appending %s = %d" % (var, LframeScalarDict[var]))
        ncFile.variables[var][indx] = LframeScalarDict[var]

    # Write accumulated error counts
    for var in list(dataStructure.keys()):
        if var.startswith('count'):
            if debugLevel >= 2: logger.debug("writeNetCDFRecord(): appending %s = %d" % (var, dataStructure[var]))
            ncFile.variables[var][indx] = dataStructure[var]
        if var.startswith('flow'):
            if debugLevel >= 2: logger.debug("writeNetCDFRecord(): appending %s = %d" % (var, dataStructure[var]))
            ncFile.variables[var][indx] = dataStructure[var]

    # Write mvc epoch seconds if we have it
    if 'cFrameEsecs' in list(dataStructure.keys()):
        if debugLevel >= 2: logger.debug("writeNetCDFRecord(): appending %s = %f" % ('cFrameEsecs', dataStructure['cFrameEsecs']))
        ncFile.variables['cFrameEsecs'][indx] = dataStructure['cFrameEsecs']

    return  # End writeNetCDFRecord()


def closeNetCDFFile(ncFile, tsList, cFrameEsecsList):
    """Write the time axis data, additional global attributes and close the file
    """

    global dataStructure

    # Save corrected sample count list
    ncFile.create_variable("correctedSampleCountList", 'i', ('time',))
    ncFile.variables['correctedSampleCountList'].long_name = "Corrected instrumenet sample count"
    ncFile.variables['correctedSampleCountList'].units = "count"
    ncFile.variables['correctedSampleCountList'][:] = dataStructure['correctedSampleCountList']

    # Write time axis
    logger.info("closeNetCDFFile(): Writing time axis for len(tsList) = " + str(len(tsList)))
    if debugLevel >= 2: logger.info("closeNetCDFFile(): tsList = %s" % tsList)
    logger.info("closeNetCDFFile(): tsList[:1] = %s, ..., tsList[-2:] = %s" % (tsList[:1], tsList[-2:]))
    logger.info("closeNetCDFFile(): Begin time = %s" % (time.strftime("%Y-%m-%d %H:%M:%S Z",time.gmtime(tsList[0]))))
    logger.info("closeNetCDFFile(): End time   = %s" % (time.strftime("%Y-%m-%d %H:%M:%S Z",time.gmtime(tsList[-1]))))

    logger.info("closeNetCDFFile(): Writing time axis for len(cFrameEsecsList) = " + str(len(cFrameEsecsList)))
    logger.info("closeNetCDFFile(): cFrameEsecsList[:1] = %s, ..., cFrameEsecsList[-2:] = %s" % (cFrameEsecsList[:1], cFrameEsecsList[-2:]))
    logger.info("closeNetCDFFile(): Begin time = %s" % (time.strftime("%Y-%m-%d %H:%M:%S Z",time.gmtime(cFrameEsecsList[0]))))
    logger.info("closeNetCDFFile(): End time   = %s" % (time.strftime("%Y-%m-%d %H:%M:%S Z",time.gmtime(cFrameEsecsList[-1]))))

    # Write time axis that is used for all the time dependent variables
    ncFile.create_variable("time", 'd', ('time',))
    ncFile.variables['time'].units = "seconds since 1970-01-01 00:00:00"
    ncFile.variables['time'].long_name = "Time GMT" 
    ncFile.variables['time'][:] = tsList
    
    # Write main vehicle time variable as received via C Frame
    ncFile.create_variable("mvctime", 'f', ('time',))
    ncFile.variables['mvctime'].units = "seconds since 1970-01-01 00:00:00"
    ncFile.variables['mvctime'].long_name = "Main Vehicle Computer Time GMT" 
    ncFile.variables['mvctime'] = cFrameEsecsList

    return ncFile # End closeNetCDFFile()


def main():
    """Main routine: Parse command line options and call unpack and write functions.

    """

    global ncFile
    global dataStructure

    parser = OptionParser(usage="""\
Unpack binary data records from binary LOPC instrument and produce NetCDF file and optional text file for import to A. Herman processing routines.

Synopsis: %prog -i <bin_fileName> -n <netCDF_fileName> [-t <text_fileName> -v -d <level> -f]

Where: 
    <bin_fileName> is name of lopc binary file to process
    <netCDF_fileName> is the name of the NetCDF file to create

Options:
    -t:         ASCII text output file name (the .dat file produced by BOT s/w)
    -v:         verbose output
    -d <level>: debugging output (higher the number the more detailed the output)
    -f:         Force removal of output files
    (Additional options for setting count criteria listed below.)

Examples:

1. A short mission useful for testing:
   lopcToNetCDF.py -i /mbari/AUVCTD/missionlogs/2009/2009084/2009.084.00/lopc.bin -n /mbari/ssdsdata/ssds/generated/netcdf/files/ssds.shore.mbari.org/auvctd/missionlogs/2009/2009084/2009.084.00/lopc.nc

2. A complete around the bay survey:
   lopcToNetCDF.py -i /mbari/AUVCTD/missionlogs/2009/2009084/2009.084.02/lopc.bin -n /mbari/ssdsdata/ssds/generated/netcdf/files/ssds.shore.mbari.org/auvctd/missionlogs/2009/2009084/2009.084.02/lopc.nc
 
3. A communications fixed short mission C Frame epoch seconds with output to ASCII text file:
   lopcToNetCDF.py -i /mbari/AUVCTD/missionlogs/2010/2010083/2010.083.08/lopc.bin -n /mbari/ssdsdata/ssds/generated/netcdf/files/ssds.shore.mbari.org/auvctd/missionlogs/2010/2010083/2010.083.08/lopc.nc -t lopc_2010_083_08.dat -v -f 

""")
    parser.add_option('-i', '--bin_fileName',
                      type='string', action='store',
                      help="Specify an input binary data file name, e.g. /mbari/AUVCTD/missionlogs/2009/2009084/2009.084.00/lopc.bin.")
    parser.add_option('-n', '--netCDF_fileName',
                      type='string', action='store',
                      help="Specify a fully qualified output NetCDF file path, e.g. /mbari/tempbox/mccann/lopc/2009_084_00lopc.nc")
    parser.add_option('-t', '--text_fileName',
                      type='string', action='store',
                      help="Specify a fully qualified output ASCII text file path, e.g. /mbari/tempbox/mccann/lopc/2009_084_00lopc.dat")
    parser.add_option('-v', '--verbose',
                      action='store_true', default=False,
                      help="Turn on verbose output, which gives bit more processing feedback.")
    parser.add_option('-d', '--debugLevel',
                      action='store', type='int',
                      help="Turn on debugLevel [1..3].  The higher the number, the more the output.  E.g. us '-d 1' to understand framing errors, '-d 3' for all debug statements.")
    parser.add_option('-f', '--force',
                      action='store_true', default=False,
                      help="Force overwite of netCDF file.  Useful for using from a master reprocess script.")

    # Additional criteria setting options
    parser.add_option('', '--trans_AIcrit',
                      type='float', action='store', default = 0.4,
                      help="Criteria for Attenuation Index to separate transparent from non-transparent particles (0..1), default = 0.4")
    parser.add_option('', '--LargeCopepod_AIcrit',
                      type='float', action='store', default = 0.6,
                      help="Criteria for Attenuation Index to identify Large Copepod particles (0..1), default = 0.6")
    parser.add_option('', '--LargeCopepod_ESDmin',
                      type='float', action='store', default = 1100,
                      help="Criteria for minimum Equivalent Spherical Diameter to identify Large Copepod particles (microns), default = 1100")
    parser.add_option('', '--LargeCopepod_ESDmax',
                      type='float', action='store', default = 1700.0,
                      help="Criteria for maximum Equivalent Spherical Diameter to identify Large Copepod particles (microns), default = 1700")
    opts, args = parser.parse_args()

    #
    # unpack data according to command line options 
    #
    if opts.bin_fileName and opts.netCDF_fileName:

        start = time.time()

        # Set global output flags
        global verboseFlag
        global debugLevel
        global hasMBARICframeData
        verboseFlag = opts.verbose
        if opts.debugLevel: 
            logging.getLogger("logger").setLevel(logging.DEBUG)
            debugLevel = opts.debugLevel

        # Check for output file and offer to overwrite
        if os.path.exists(opts.netCDF_fileName):
            if opts.force:
                if os.path.exists(opts.netCDF_fileName):
                    os.remove(opts.netCDF_fileName)
            else:
                ans = input(opts.netCDF_fileName + " file exists.\nDo you want to remove it and continue processing? (y/[N]) ")
                if ans.upper() == 'Y':
                    os.remove(opts.netCDF_fileName)
                else:
                    sys.exit(0)

        textFile = None
        if opts.text_fileName:
            if os.path.exists(opts.text_fileName):
                if opts.force:
                    if os.path.exists(opts.text_fileName):
                        os.remove(opts.text_fileName)
                else:
                    ans = input(opts.text_fileName + " file exists.\nDo you want to remove it and continue processing? (y/[N]) ")
                    if ans.upper() == 'Y':
                        os.remove(opts.text_fileName)
                    else:
                        sys.exit(0)

            textFile = open(opts.text_fileName, 'w')


        logger.info("main(): Processing begun: %s" % time.ctime())
        # Open input file
        binFile=open(opts.bin_fileName,'rb')

        # Set flag for whether we need to look for data that Hans writes to the C Frame - implemented after 15 March 2010 (day 2010074)
        # Assume binFile.name is like: '/mbari/AUVCTD/missionlogs/2009/2009084/2009.084.02/lopc.bin'
        try:
            if int(binFile.name.split('/')[-3]) > 2010074:
                hasMBARICframeData = True
        except ValueError:
            hasMBARICframeData = True       # For testing assume we have CFrame time data

        # Add time axis data to the data structure and estimate the number of records and time to parse
        tsList = constructTimestampList(binFile)
        dataStructure['tsList'] = tsList
        logger.info("main(): Examined sibling parosci.nc file to find startTime = %s and endTime = %s with %d records expected to be read from lopc.bin" %
                (time.strftime("%Y-%m-%d %H:%M:%S Z",time.gmtime(tsList[0])), time.strftime("%Y-%m-%d %H:%M:%S Z",time.gmtime(tsList[-1])), len(tsList)))
        

        # Unpack LOPC binary data into a data structure that we can later write to a NetCDF file (with the time information)
        # unpackLOPCbin() blocks until end of file is encountered, then we close the output NetCDF file and finish up.
        try:
            unpackLOPCbin(binFile, opts, textFile)          # The major workhorse function: populates global dataStructure dictionary
                                        # On first L frame read this function opens the netCDF file for appending
                                        # then calls writeNetCDFRecord at the binningInterval. dataStructure[] is
                                        # populated with lots of information by unpackLOPCbin.
        except EndOfFileException:
            logger.info("main(): >>> Done reading file.")
            logger.info("main(): lFrameCount = %d, mFrameCount = %d" % (dataStructure['lFrameCount'], dataStructure['mFrameCount']))
            if countBeginMFrameWithoutEndOfPreviousFrame:
                logger.info("main(): countBeginMFrameWithoutEndOfPreviousFrame = %d, countBeginLFrameWithoutEndOfPreviousFrame = %d" % 
                        (dataStructure['countBeginMFrameWithoutEndOfPreviousFrame'], dataStructure['countBeginLFrameWithoutEndOfPreviousFrame']))

        # Close the ASCII text file, if it exists
        if textFile != None:
            logger.info("main(): Closing test file")
            textFile.close()

        # Make sure that tsList from the parosci.nc file and the dataStructure parsed from lopc.bin are the same lengths
        (tsList, cFrameEsecsList) = constructTimestampList(binFile, sampleCountList = dataStructure['sampleCountList'], cFrameEsecsList = dataStructure['cFrameEsecsList'])

        # Close the netCDF file writing the proper tsList data first
        closeNetCDFFile(ncFile, tsList, cFrameEsecsList)

        logger.info("main(): Created file: %s" % opts.netCDF_fileName)

        mark = time.time()  
        logger.info("main(): Processing finished: %s Elapsed processing time from start of processing = %d seconds" % (time.ctime(), (mark - start)))

    else:
        print("\nCannot process input.  Execute with -h for usage note.\n")  

    return # End main()


# Allow this file to be imported as a module
if __name__ == '__main__':

    try:
        if sys.argv[1].startswith('test'):
            '''Run a function that begins with 'test' using the locals() dictionary of functions in this module.'''
            logger.info("Running test function: " + sys.argv[1] + "()...")
            locals()[sys.argv[1]]()
    except IndexError:
        main()
    else:
        '''Will print usage note if improper arguments are given.'''
        main()


