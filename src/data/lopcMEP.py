#!/usr/bin/env python
__author__ = "Mike McCann"
__version__ = "$Revision: 1.8 $".split()[1]
__date__ = "$Date: 2010/08/30 23:24:40 $".split()[1]
__copyright__ = "2010"
__license__ = "GPL v3"
__contact__ = "mccann at mbari.org"
__idt__ = "$Id: lopcMEP.py,v 1.8 2010/08/30 23:24:40 ssdsadmin Exp $"

__doc__ = """

Module for processing MEP data from the Brooke Ocean Technology Laser 
Optical Plankton Counter.  The functions here work in concert with the
.bin file processing done in the lopcToNetCDF.py module.  That module
unpacks MEP binary data and passes them as python objects to functions
here that collect the MEPs into time/depth bins and compute ESD, AI,
and other products - maybe even large copepod concentrations and marine
snow estimates.

@var __date__: Date of last cvs commit
@undocumented: __doc__ parser
@status: In development
@license: GPL
"""

import logging

import numpy
from logs2netcdfs import AUV_NetCDF

#
# Global logger object, name it MEP to avoid clashes with calling module logger output
#
h = logging.StreamHandler()
h.setFormatter(AUV_NetCDF._formatter)
logger = logging.getLogger("MEP")
logger.addHandler(h)

# Change to DEBUG for debug output from this module
logging.getLogger("MEP").setLevel(logging.INFO)


class MEP:
    """Class for an MEP.  Instances are a single MEP."""

    def __init__(self, p, e, l, s):  # noqa: E741

        self.p = [p]
        self.e = [e]
        self.l = [l]
        self.s = [s]

    def addPartial(self, p, e, l, s):  # noqa: E741
        """Add data from a partial MEP"""
        self.p.extend([p])
        self.e.extend([e])
        self.l.extend([l])
        self.s.extend([s])

    def build(self):
        """Calculate the attributes that are specific for each MEP.
        This follows the method described in Checkly et al. (2008).
        See also: http://www.alexherman.com/lopc_post.php.

        After all the partial data are added this method is called
        to compute attributes for the MEP.

        """

        # Attenuance index - use numpy arrays for methods we get for free
        DS = numpy.array(self.p)
        DSmax = 3800  # Herman June 2009 p.6 - and Checkley pers comm
        self.ai = DS.mean() / DSmax

        # Optical Density - count the number of 1-mm elements that are occluded
        eDict = {}
        for e in self.e:
            try:
                eDict[e] += 1
            except KeyError:
                eDict[e] = 0

        eSum = 0
        for k in list(eDict.keys()):
            eSum += 1

        self.od = eSum

        # Equivalent Spherical Diameter calculation
        # - from http://www.alexherman.com/LOPC_AnalysesMethods_V104.pdf
        a1 = 0.1806059
        a2 = 0.00025459
        a3 = -1.0988e-09
        a4 = 9.54e-15

        DS = numpy.array(self.p).sum()
        self.esd = 1000 * (
            a1 + a2 * DS + a3 * DS**2 + a4 * DS**3
        )  # in units of microns

    def __repr__(self):
        """Return a string that is reasonable to print"""

        ##str = "p = %s\ne = %s\nl = %s\ns = %s\n" % (self.p, self.e, self.l, self.s)

        # Format used in Alex Herman papers
        str = ""
        n = 1
        for p, e, l, s in zip(self.p, self.e, self.l, self.s):  # noqa: E741
            if n == 1:
                p += 32768  # Add the bit back

            str += "M %2d %5d %4d %5d\n" % (e, s, l, p)
            n += 1

        try:
            str += "ai = %f\n" % self.ai
            str += "od = %f\n" % self.od
            str += "esd = %.1f microns\n" % self.esd
        except AttributeError:
            ##logger.debug("__repr__(): Attribute 'ai' does not exist.  Call build() to create it.")
            pass

        return str


class MepData:
    """Class the holds MEP data and operates on them"""

    def __init__(self):
        """Initialze member variables"""
        self.n = list()
        self.p = list()
        self.e = list()
        self.l = list()
        self.s = list()

        self.mepList = []

    def extend(self, nL, pL, eL, lL, sL):
        """Extend the member list items with passed in lists"""
        self.n.extend(nL)
        self.p.extend(pL)
        self.e.extend(eL)
        self.l.extend(lL)
        self.s.extend(sL)

    def build(self, previousMEP=None):
        """Loop through collected lists and assemble data for each unique MEP.  The
        `previousMEP` is passed in for the case when the binning interval splits an
        MEP. In this case the beginning of the last MEP from the last binning interval
        is used to stitch the first partial MEP from this binning interval on to.
        """

        # Build MEP list from raw data parsed from M frames
        for n, p, e, l, s in zip(self.n, self.p, self.e, self.l, self.s):  # noqa: E741
            logger.debug("n = %s, p = %s, e = %s, l = %s, s = %s" % (n, p, e, l, s))
            if n == 1:
                # Instantiate MEP object (n == 1 indicates a new MEP detected by the instrument)
                mep = MEP(p, e, l, s)
                self.mepList.append(mep)

                ##logger.debug("Beginning new mep")
            else:
                # Not the first part of an MEP, add any partial data to the previous [-1] MEP
                try:
                    self.mepList[-1].addPartial(p, e, l, s)
                except IndexError:
                    # No MEPs in list yet, append the previous MEP passed in then add partial
                    ##logger.debug("previousMEP = " + str(previousMEP))
                    if previousMEP is None:
                        return

                    self.mepList.append(previousMEP)
                    self.mepList[-1].addPartial(p, e, l, s)

        # Append the last MEP
        ##try:
        ##	self.mepList.append(mep)
        ##except NameError:
        ##logger.error("No mep assembled at end of loop.")
        ##	pass

        # For each MEP calculate its specific attributes
        i = 0
        aiList = []
        esdList = []
        for mep in self.mepList:
            i += 1
            mep.build()
            aiList.append(mep.ai)
            esdList.append(mep.esd)
            ##logger.debug("built mep # %d = \n%s" % (i, mep))

        # Compute statistics for this set of MEPs
        self.aiArray = numpy.array(aiList, dtype="float32")
        self.esdArray = numpy.array(esdList, dtype="float32")

        return  # End build(self):

    def count(self, binSizeList, countList=None):
        """Gived a sizeclass numpy.array `binSizeList` loop through the MEPs and increment the
        count for the ESD. Return the countList for this set of MEP data. Add to the countList
        that is passed in if it is provided.

        """

        if countList is None:
            countList = numpy.zeros(binSizeList.shape)

        for mep in self.mepList:
            """Find the index of the closest bin to the mep's esd and increment the count"""

            logger.debug("Finding index for mep.esd = %f" % (mep.esd))
            diffBin = abs(binSizeList - mep.esd)
            indx = numpy.nonzero(diffBin == diffBin.min())[0][0]
            logger.debug("increment index = %d for mep.esd = %f" % (indx, mep.esd))
            countList[indx] += 1

        return countList

    def countLC(self, aiCrit=0.6, esdLowCrit=1100.0, esdHiCrit=1700.0):
        """Using a criteria based on attenuation index (ai) and Equivalent Spherical Diameter (esd)
        create a count of large copepods (LC) for all the MEPs in the mepList.
        The default criteria values are from Checkley et. al 2008:
                ai > 0.6 and 1.1 < esd < 1.7 mm		is a LC

        Return the scalar count value.

        """

        lcCount = 0
        logger.debug(
            "aiCrit = %f,  esdLowCrit %f, esdHiCrit = %f"
            % (aiCrit, esdLowCrit, esdHiCrit)
        )
        for mep in self.mepList:
            logger.debug("mep.ai = %f, mep.esd = %f" % (mep.ai, mep.esd))
            if mep.ai > aiCrit and mep.esd > esdLowCrit and mep.esd < esdHiCrit:
                lcCount += 1

        return lcCount

    def countTrans(self, aiCrit=0.6):
        """Divide all the MEPs based on an attenuation index criteria and return counts of those
        below and above that value.

        """

        transCount = 0
        nonTransCount = 0
        logger.debug("aiCrit = %f" % aiCrit)
        for mep in self.mepList:
            logger.debug("mep.ai = %f, mep.esd = %f" % (mep.ai, mep.esd))
            if mep.ai > aiCrit:
                nonTransCount += 1
            else:
                transCount += 1

        return (transCount, nonTransCount)

    def toASCII(self):
        """Convert member lists to ASCII string"""

        # Format used in Alex Herman papers
        str = ""
        for n, p, e, l, s in zip(self.n, self.p, self.e, self.l, self.s):  # noqa: E741
            if n == 1:
                p += 32768  # Add the bit back

            str += "M %2d %5d %4d %5d\n" % (e, s, l, p)

        return str

    def frameToASCII(self, nL, pL, eL, lL, sL):
        """Convert just the passed in lists to ASCII string"""

        # Format used in Alex Herman papers
        str = ""
        for n, p, e, l, s in zip(nL, pL, eL, lL, sL):  # noqa: E741
            if n == 1:
                p += 32768  # Add the bit back

            str += "M %d %d %d %d\n" % (e, s, l, p)

        return str

    def __repr__(self):
        """Return a string that is reasonable to print"""

        # Raw input data
        ##str =  "n = %s\np = %s\ne = %s\nl = %s\ns = %s\n" % (self.n, self.p, self.e, self.l, self.s)

        # Format used in Alex Herman papers
        str = self.toASCII()

        # Loop trough the MEPs built and print some attributes
        try:
            i = 0
            for mep in self.mepList:
                i += 1
                str += "%d. mep: ai = %f, od = %d, esd = %.1f microns\n" % (
                    i,
                    mep.ai,
                    mep.od,
                    mep.esd,
                )
        except AttributeError:
            ##logger.debug("__repr__(): Attribute 'mepList' does not exist.  Call build() to create it.")
            pass

        # Output some stats on ai and esd
        str += "\nNumber of MEPs = %d\n" % len(self.aiArray)
        str += "ai:   mean = %f, min = %f, max = %f, std = %f\n" % (
            self.aiArray.mean(),
            self.aiArray.min(),
            self.aiArray.max(),
            self.aiArray.std(),
        )
        str += "esd:  mean = %f, min = %f, max = %f, std = %f\n" % (
            self.esdArray.mean(),
            self.esdArray.min(),
            self.esdArray.max(),
            self.esdArray.std(),
        )

        return str


# Allow this file to be imported as a module
if __name__ == "__main__":
    """Run tests if executed at command line.  This module is designed to be imported."""

    logging.getLogger("MEP").setLevel(
        logging.DEBUG
    )  # Set debug for additional output from the methods
    ##logging.getLogger("MEP").setLevel(logging.INFO)

    # Get a MepData object
    mepDataFake = MepData()

    # Accunmulate some bogus test data
    mepDataFake.extend([1, 0], [1234, 1235], [11, 12], [1432, 1433], [1456, 1457])
    mepDataFake.extend([0], [2234], [12], [2432], [2456])
    mepDataFake.extend([0], [3234], [13], [3432], [3456])
    mepDataFake.extend([1], [1330], [14], [4432], [4456])
    mepDataFake.extend([0], [2340], [15], [5432], [5456])
    mepDataFake.extend([1], [2300], [14], [4432], [4456])
    mepDataFake.extend([0], [2500], [15], [5432], [5456])

    # Build the mepList - with DEBUG turned on we'll get some output
    mepDataFake.build()

    # Print out the input data and the built values
    logger.info("Input sample (really fake) MEP data = \n%s" % (mepDataFake))

    # Run the count() method
    counts = mepDataFake.count(
        numpy.array(list(range(108, 15015, 15)), dtype="float32")
    )
    logger.info("counts = %s" % counts)
    logger.info("len(mepDataFake.mepList)  = %d" % len(mepDataFake.mepList))
    LCcount = mepDataFake.countLC()
    logger.info("LCcount = %d" % LCcount)
    (transCount, nonTransCount) = mepDataFake.countTrans()
    logger.info("transCount = %d, nonTransCount = %d" % (transCount, nonTransCount))

    # Loop trough the MEPs built and print some attributes
    i = 0
    for mep in mepDataFake.mepList:
        i += 1
        logger.info(
            "%d. mep: ai = %f, od = %f, esd = %f" % (i, mep.ai, mep.od, mep.esd)
        )

    #
    # Now operate on some test data where we can verify the answer
    #

    # Get a new MepData object
    mepData = MepData()

    # Example from Alex Herman data analysis paper of June 2009
    # M 8 50987 12 33814
    # M 7 50986 4 143
    # M 6 50996 7 219

    mepData.extend([1], [1046], [8], [12], [50987])
    mepData.extend([0], [143], [7], [4], [50986])
    mepData.extend([0], [219], [6], [7], [50996])

    # Build the mepList - with DEBUG turned on we'll get some output
    mepData.build()

    # Print out the input data and the built values
    logger.info("\n\nInput sample from Herman June 2009 MEP data = \n%s" % (mepData))

    # Run the count() method
    counts = mepData.count(numpy.array(list(range(108, 15015, 15)), dtype="float32"))
    logger.info("counts = %s" % counts)
    logger.info("len(mepData.mepList)  = %d" % len(mepData.mepList))

    # Loop trough the MEPs built and print some attributes
    i = 0
    for mep in mepData.mepList:
        i += 1
        logger.info(
            "%d. mep: ai = %f, od = %f, esd = %f" % (i, mep.ai, mep.od, mep.esd)
        )
