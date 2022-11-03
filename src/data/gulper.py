#!/usr/bin/env python
"""
Parse auvctd syslog file for gulper times and bottle numbers
This is a utility script for pulling out Gulper information from 
the auvctd syslog files.  A copy of it will be used by the STOQS
loader for adding dorado_Gulper Activities to the Campaign.  This 
will achieve better harmony with the way other Samples (Sipper, ESP)
are loaded and accessible in STOQS. 
"""

import argparse
import logging
import os
import re
import sys

from logs2netcdfs import BASE_PATH, MISSIONLOGS
from resample import FREQ

LOG_NAME = "processing.log"
AUVCTD_VOL = "/Volumes/AUVCTD"
VEHICLE = "dorado"


class Gulper:
    logger = logging.getLogger(__name__)
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        "%(levelname)s %(asctime)s %(filename)s "
        "%(funcName)s():%(lineno)d %(message)s"
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    _log_levels = (logging.WARN, logging.INFO, logging.DEBUG)

    def parse_gulpers(self):
        "Parse the Gulper times and bottle numbers from the auvctd syslog file"
        mission_dir = os.path.join(BASE_PATH, VEHICLE, MISSIONLOGS, self.args.mission)
        syslog_file = os.path.join(mission_dir, "syslog")
        if not os.path.exists(syslog_file):
            self.logger.error(f"{syslog_file} not found")
            return

        self.logger.info(f"Reading {syslog_file}")
        with open(syslog_file, mode="r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        #
        # Get potential mission elapsed time for the gulper firing.  When followed by a bottle
        # number match then this time will be used
        #
        # if ( /.+t =\s+([\d\.]+)\).+Behavior FireTheGulper/ ) {
        # 	print if $debug;
        # 	$etime = $1;
        # 	print "etime = $etime\n" if $debug;
        # }
        #
        # Get potential Gulper number, needed for when we see "...FireTheGulper:-1 has changed to state Finished"
        # if ( /GulperServer - firing gulper (\d+)/ ) {
        #    $number = $1;
        # }
        #
        #
        # Get line with the bottle number.  Staring 15 Oct 2008 with 20008.289.08 we have epoch
        # seconds on this line.  Parse that record with the 2nd if block
        #
        # if ( /: (\d+) Gulper::fireGulper - cmd is \$(\d\d)1Fff/ ) {
        # 	print if $debug;
        # 	$esecs = $1;
        # 	$number = sprintf("%d",$2);             # Gets rid of leading 0.
        # 	print "esecs = $esecs, number = $number\n" if $debug;
        #
        # 	#
        # 	# After first instance of bottle number undef $etime so we don't re-set it
        # 	#
        # 	if ( $etime ) {
        # 		print "Saving this bottle time $etime for bottle number $number\n" if $debug;
        # 		print "Adding $etime to $startEsecs to get " if $debug;
        # 		$bottles{$number} = $etime + $startEsecs;
        # 		print "$bottles{$number}\n" if $debug;
        # 		print "Epoch seconds from syslog file for bottle = $esecs\n" if $debug;
        # 		print "Difference = ", $bottles{$number} - $esecs, "\n" if $debug;
        # 		undef $etime;
        # 	}
        #
        # 	$bottles{$number} = $esecs if $esecs;
        # }
        # elsif ( defined $number && /\(t = ([\d\.]+)\) Behavior FireTheGulper:-1 has changed to state Finished/ ) {
        #    $etime = $1;
        #
        # After first instance of bottle number undef $etime so we don't re-set it
        #
        # 	if ( $etime ) {
        # 		print "Saving this bottle time $etime for bottle number $number\n" if $debug;
        # 		print "Adding $etime to $startEsecs to get " if $debug;
        # 		$bottles{$number} = $etime + $startEsecs;
        # 		print "$bottles{$number}\n" if $debug;
        # 		print "Epoch seconds from syslog file for bottle = $esecs\n" if $debug;
        # 		print "Difference = ", $bottles{$number} - $esecs, "\n" if $debug;
        # 		undef $etime;
        # 	}
        # }
        #
        # elsif ( /: Gulper::fireGulper - cmd is \$(\d\d)1Fff/ ) {
        # 	print if $debug;
        # 	$number = sprintf("%d",$1);		# Gets rid of leading 0.
        # 	print "number = $number\n" if $debug;
        #
        # 	#
        # 	# After first instance of bottle number undef $etime so we don't re-set it
        # 	#
        # 	if ( $etime ) {
        # 		print "Saving this bottle time $etime for bottle number $number\n" if $debug;
        # 		print "Adding $etime to $startEsecs\n" if $debug;
        # 		$bottles{$number} = $etime + $startEsecs;
        # 		undef $etime;
        # 	}
        # }

        fire_the_gulper_re = re.compile(".+t =\s+([\d\.]+)\).+Behavior FireTheGulper")
        gulper_number_re = re.compile("GulperServer - firing gulper (\d+)")
        adaptive_gulper_re = re.compile(
            "Adaptive Sampler has fired gulper (\d+) at t =\s+([\d\.]+)"
        )
        gulper_times = []
        gulper_bottles = []
        mission_started = False
        for line in lines:
            if "t = 0.000000" in line:
                mission_started = True
            if match := fire_the_gulper_re.search(line):
                etime = match.group(1)
                self.logger.debug(f"etime = {etime}")
            if match := gulper_number_re.search(line):
                number = match.group(1)
                self.logger.debug(f"number = {number}")
            if match := adaptive_gulper_re.search(line):
                number = match.group(1)
                etime = match.group(2)
                self.logger.debug(f"number = {number}, etime = {etime}")

    def process_command_line(self):
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter,
            description=__doc__,
        )
        parser.add_argument("--mission", help="Mission directory, e.g.: 2020.064.10")
        parser.add_argument(
            "--start_esecs", help="Start time of mission in epoch seconds"
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

        self.args = parser.parse_args()

        self.logger.setLevel(self._log_levels[self.args.verbose])
        self.commandline = " ".join(sys.argv)


if __name__ == "__main__":
    VEHICLE_DIR = "/Volumes/AUVCTD/missionlogs"
    MOUNT_DIR = "cifs://atlas.shore.mbari.org/AUVCTD"
    START_YEAR = 2007  # First Gulper was on 2007.120.01

    gulper = Gulper()
    gulper.process_command_line()
    gulper.parse_gulpers()
