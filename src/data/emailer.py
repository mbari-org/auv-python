#!/usr/bin/env python
"""
Send email to notify of new data.

"""

__author__ = "Mike McCann"
__copyright__ = "Copyright 2023, Monterey Bay Aquarium Research Institute"

import argparse
import logging
import platform
import sys
import time
from pathlib import Path

from logs2netcdfs import BASE_PATH, MISSIONNETCDFS, AUV_NetCDF

NOTIFICATION_EMAIL = "auvctd@listserver.mbari.org"
TEMPLATE = """
Hello,

AUV survey {SURVEY_NAME} beginning on DATE has been processed.
Below are the science data products now available for review:

Quick look plots:
{BIOLUME_SECTION}
{_2COLUMN_SECTION}
{LOPC_SECTION}
{NAV_ADJUST}
{HIST_STATS}
{PROF_STATS}


Full resolution matlab data file:
FULL_MAT

its NetCDF counterpart (via OPeNDAP URL):
{FULL_NC}

Decimated NetCDF file (sampled at the ISUS or 2 sec sampling frequency) with all variables interpolated to the same time axis:
{DECIM_NC}

Decimated Ocean Data View import data file:
ODV

Gulper bottle numbers and data at sample collection (ODV tab-delimited) file:
{GULPER}

Google Earth sensor point tracks
KML

The above web links are a mapping from the \\atlas\\AUVCTD\\surveys directory.
You may get the data from this network share or from the above URLs.

The processing log output:
{LOG}

- AUVprocess_main run by {USER} on {HOSTNAME} at {DATE}
"""


class Emailer:
    logger = logging.getLogger(__name__)
    _handler = logging.StreamHandler()
    _handler.setFormatter(AUV_NetCDF._formatter)
    logger.addHandler(_handler)
    _log_levels = (logging.WARN, logging.INFO, logging.DEBUG)

    def compose_message(self) -> str:
        msg = TEMPLATE.format(
            SURVEY_NAME=self.args.mission,
            BIOLUME_SECTION="",
            _2COLUMN_SECTION="",
            LOPC_SECTION="",
            NAV_ADJUST="",
            HIST_STATS="",
            PROF_STATS="",
            FULL_MAT="",
            FULL_NC="",
            DECIM_NC="",
            ODV="",
            GULPER="",
            KML="",
            LOG="",
            USER=self.args.email_to,
            HOSTNAME=platform.node(),
            DATE=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        return msg

    def process_command_line(self):
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter,
            description=__doc__,
        )
        (
            parser.add_argument(
                "--base_path",
                action="store",
                default=BASE_PATH,
                help="Base directory for missionlogs and missionnetcdfs, default: auv_data",
            ),
        )
        parser.add_argument(
            "--auv_name",
            action="store",
            default="Dorado389",
            help="Dorado389 (default), i2map, or Multibeam",
        )
        (
            parser.add_argument(
                "--mission",
                action="store",
                help="Mission directory, e.g.: 2020.064.10",
            ),
        )
        parser.add_argument(
            "--email_to",
            action="store",
            default=NOTIFICATION_EMAIL,
            help=f"Send email to this address when processing is complete, default: {NOTIFICATION_EMAIL}",
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
                [f"{i}: {v}" for i, v in enumerate(("WARN", "INFO", "DEBUG"))],
            ),
        )
        self.args = parser.parse_args()
        self.logger.setLevel(self._log_levels[self.args.verbose])
        self.commandline = " ".join(sys.argv)


if __name__ == "__main__":
    email = Emailer()
    email.process_command_line()
    file_name_base = f"{email.args.auv_name}_{email.args.mission}"
    nc_file_base = Path(
        BASE_PATH,
        email.args.auv_name,
        MISSIONNETCDFS,
        email.args.mission,
        file_name_base,
    )
    p_start = time.time()
    msg = email.compose_message()
    email.send_email(msg)
    email.logger.info(f"Time to process: {(time.time() - p_start):.2f} seconds")
