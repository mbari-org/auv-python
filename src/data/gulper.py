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

    def parse_gulpers(self) -> dict:
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

        # Starting with 2005.299.12 and continuing through 2022.286.01 and later
        # Used to get mission elapsed time (etime) - matches 'changed state' messages too
        fire_the_gulper_re = re.compile(".+t =\s+([\d\.]+)\).+Behavior FireTheGulper")

        # Starting with 2008.281.03 and continuing through 2021.111.00 and later
        adaptive_gulper_re = re.compile(
            "Adaptive Sampler has fired gulper (\d+) at t =\s+([\d\.]+)"
        )

        # Starting with 2008.289.03 and continuing through 2014.212.00
        num_fire_gulper_cmd_re = re.compile(
            ": (\d+) Gulper::fireGulper - cmd is \$(\d\d)1Fff"
        )

        # Starting with 2007.120.00 and continuing through 2022.286.01 and later
        gulper_state_finished_re = re.compile(
            "\(t = ([\d\.]+)\) Behavior FireTheGulper:-1 has changed to state Finished"
        )

        # Starting with 2007.093.12 and continuing through 2009.342.04
        fire_gulper_cmd_re = re.compile(": Gulper::fireGulper - cmd is \$(\d\d)1Fff")

        # Starting with 2014.266.04 and continuing through 2022.286.01 and later
        gulper_number_re = re.compile("GulperServer - firing gulper (\d+)")

        # Logic translated to here from parseGulperLog.pl Perl script
        bottles = {}
        for line in lines:
            if "t = 0.000000" in line:
                # The navigation.nc file has the best match to mission start time.
                # Use that to match to this zero elapsed mission time.
                self.logger.debug(
                    f"Mission {self.args.mission} started at {self.args.start_esecs}"
                )
            if match := fire_the_gulper_re.search(line):
                # .+t =\s+([\d\.]+)\).+Behavior FireTheGulper
                etime = float(match.group(1))
                self.logger.debug(f"etime = {etime}")
            if match := gulper_number_re.search(line):
                # GulperServer - firing gulper (\d+)
                number = int(match.group(1))
                self.logger.debug(f"number = {number}")
            if match := adaptive_gulper_re.search(line):
                # Adaptive Sampler has fired gulper (\d+) at t =\s+([\d\.]+)
                number = int(match.group(1))
                esecs = float(match.group(2))
                self.logger.debug(f"number = {number}, esecs = {esecs}")

            if match := num_fire_gulper_cmd_re.search(line):
                # ": (\d+) Gulper::fireGulper - cmd is \$(\d\d)1Fff
                esecs = float(match.group(1))
                number = int(match.group(2))
                self.logger.debug(f"eseconds = {eses}, number = {number}")
                if etime:
                    # After first instance of bottle number undef $etime so we don't re-set it
                    bottles[number] = etime + self.args.start_esecs
                    self.logger.debug(
                        f"Saving time {etime + self.args.start_esecs} for bottle number {number}"
                    )
                    etime = None
                bottles[number] = esecs
            elif match := gulper_state_finished_re.search(line):
                # t = ([\d\.]+)\) Behavior FireTheGulper:-1 has changed to state Finished
                if number:
                    etime = float(match.group(1))
                    # After first instance of bottle number undef $etime so we don't re-set it
                    bottles[number] = etime + self.args.start_esecs
                    self.logger.debug(
                        f"Saving time {etime + self.args.start_esecs} for bottle number {number}"
                    )
                    etime = None
            elif match := fire_gulper_cmd_re.search(line):
                # : Gulper::fireGulper - cmd is \$(\d\d)1Fff
                number = int(match.group(1))
                if etime:
                    # After first instance of bottle number undef $etime so we don't re-set it
                    bottles[number] = etime + self.args.start_esecs
                    self.logger.debug(
                        f"Saving time {etime + self.args.start_esecs} for bottle number {number}"
                    )
                    etime = None
        return bottles

    def process_command_line(self) -> None:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter,
            description=__doc__,
        )
        parser.add_argument("--mission", help="Mission directory, e.g.: 2020.064.10")
        parser.add_argument(
            "--start_esecs", help="Start time of mission in epoch seconds", type=float
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
    # First Gulper was on 2007.120.01
    gulper = Gulper()
    gulper.process_command_line()
    gulper_times = gulper.parse_gulpers()
    gulper.logger.info(f"number, epoch seconds")
    for number, esecs in gulper_times.items():
        gulper.logger.info(f"{number}, {esecs}")
