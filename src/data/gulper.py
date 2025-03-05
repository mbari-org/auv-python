#!/usr/bin/env python
"""
Parse auvctd syslog file for gulper times and bottle numbers
This is a utility script for pulling out Gulper information from
the auvctd syslog files. Developed for the auv-python project.

A copy of it will be used by the STOQS loader for adding dorado_Gulper
Activities to the Campaign.  This will achieve better harmony with the
way other Samples (Sipper, ESP) are loaded and accessible in STOQS.
"""

import argparse
import logging
import os
import re
import sys
from http import HTTPStatus
from pathlib import Path

import requests
import xarray as xr
from logs2netcdfs import TIMEOUT


class Gulper:
    logger = logging.getLogger(__name__)
    _log_levels = (logging.WARN, logging.INFO, logging.DEBUG)

    def mission_start_esecs(self) -> float:
        "Return the start time of the mission in epoch seconds"
        if self.args.start_esecs:
            return self.args.start_esecs

        # Get the first time record from mission's navigation.nc file
        if self.args.local:
            base_path = Path(__file__).parent.joinpath("../../data/auv_data").resolve()
            url = Path(
                base_path,
                "dorado",
                "missionnetcdfs",
                self.args.mission,
                "navigation.nc",
            )
        else:
            # Relies on auv-python having processed the mission
            url = Path(
                "http://dods.mbari.org/opendap/data/auvctd/",
                "missionnetcdfs",
                self.args.mission.split(".")[0],
                self.args.mission.split(".")[0] + self.args.mission.split(".")[1],
                self.args.mission,
                "navigation.nc",
            )
        self.logger.info("Reading mission start time from %s", url)
        ds = xr.open_dataset(url)
        return ds.time[0].values.astype("float64") / 1e9

    def parse_gulpers(self, sec_delay: int = 1) -> dict:  # noqa: C901, PLR0912, PLR0915
        """Parse the Gulper times and bottle numbers from the auvctd syslog file.
        Subtract bottle times by sec_delay seconds to account for the time it takes
        the gulper midsection to reach the water that's measured by the nosecone
        sensors."""

        bottles = {}
        if self.args.local:
            # Read from local file - useful for testing in auv-python
            base_path = Path(__file__).parent.joinpath("../../data/auv_data").resolve()
            mission_dir = Path(
                base_path,
                "dorado",
                "missionlogs",
                self.args.mission,
            )
            syslog_file = Path(mission_dir, "syslog")
            self.logger.info("Reading local file %s", syslog_file)
            if not syslog_file.exists():
                self.logger.error("%s not found", syslog_file)
                raise FileNotFoundError(syslog_file)
            with syslog_file.open(encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
        else:
            syslog_url = os.path.join(  # noqa: PTH118
                "http://dods.mbari.org/data/auvctd/",
                "missionlogs",
                self.args.mission.split(".")[0],
                self.args.mission.split(".")[0] + self.args.mission.split(".")[1],
                self.args.mission,
                "syslog",
            )
            self.logger.info("Reading %s", syslog_url)
            with requests.get(str(syslog_url), stream=True, timeout=TIMEOUT) as resp:
                if resp.status_code != HTTPStatus.OK:
                    self.logger.error(
                        "Cannot read %s, resp.status_code = %d", syslog_url, resp.status_code,
                    )
                    if self.args.mission in (
                        "2012.256.00",
                        "2012.257.01",
                        "2012.258.00",
                    ):
                        # Hans created tarballs for offshore missions do not include syslogs
                        # per email thread on 12 September 2012 - Mike McCann
                        self.logger.info(
                            "Known missing syslog for mission %s", self.args.mission,
                        )
                        return bottles
                    error_message = f"Cannot read {syslog_url}"
                    raise FileNotFoundError(error_message)
                lines = [line.decode(errors="ignore") for line in resp.iter_lines()]

        mission_start_esecs = self.mission_start_esecs()

        # Starting with 2005.299.12 and continuing through 2022.286.01 and later
        # Used to get mission elapsed time (etime) - matches 'changed state' messages too
        fire_the_gulper_re = re.compile(r".+t =\s+([\d\.]+)\).+Behavior FireTheGulper")

        # Starting with 2008.281.03 and continuing through 2021.111.00 and later
        adaptive_gulper_re = re.compile(
            r"Adaptive Sampler has fired gulper (\d+) at t =\s+([\d\.]+)",
        )

        # Starting with 2008.289.03 and continuing through 2014.212.00
        num_fire_gulper_cmd_re = re.compile(
            r": (\d+) Gulper::fireGulper - cmd is \$(\d\d)1Fff",
        )

        # Starting with 2007.120.00 and continuing through 2022.286.01 and later
        gulper_state_finished_re = re.compile(
            r"\(t = ([\d\.]+)\) Behavior FireTheGulper:-1 has changed to state Finished",
        )

        # Starting with 2007.093.12 and continuing through 2009.342.04
        fire_gulper_cmd_re = re.compile(r": Gulper::fireGulper - cmd is \$(\d\d)1Fff")

        # Starting with 2014.266.04 and continuing through 2022.286.01 and later
        gulper_number_re = re.compile(r"GulperServer - firing gulper (\d+)")

        # Logic translated to here from parseGulperLog.pl Perl script
        etime = None
        number = None
        for line in lines:
            if "t = 0.000000" in line:
                # The navigation.nc file has the best match to mission start time.
                # Use that to match to this zero elapsed mission time.
                self.logger.debug(
                    "Mission %s started at %s", self.args.mission, mission_start_esecs,
                )
            if match := fire_the_gulper_re.search(line):
                # .+t =\s+([\d\.]+)\).+Behavior FireTheGulper
                etime = float(match.group(1))
                self.logger.debug("etime = %s", etime)
            if match := gulper_number_re.search(line):
                # GulperServer - firing gulper (\d+)
                number = int(match.group(1))
                self.logger.debug("number = %s", number)
            if match := adaptive_gulper_re.search(line):
                # Adaptive Sampler has fired gulper (\d+) at t =\s+([\d\.]+)
                number = int(match.group(1))
                esecs = float(match.group(2))
                self.logger.debug("number = %s, esecs = %s", number, esecs)
                bottles[number] = esecs

            if match := num_fire_gulper_cmd_re.search(line):
                # ": (\d+) Gulper::fireGulper - cmd is \$(\d\d)1Fff
                esecs = float(match.group(1))
                number = int(match.group(2))
                self.logger.debug("eseconds = %s, number = %s", esecs, number)
                if etime:
                    # After first instance of bottle number undef $etime so we don't re-set it
                    bottles[number] = etime + mission_start_esecs
                    self.logger.debug(
                        "Saving time %s for bottle number %s", etime + mission_start_esecs, number,
                    )
                    etime = None
                bottles[number] = esecs
            elif match := gulper_state_finished_re.search(line):
                # t = ([\d\.]+)\) Behavior FireTheGulper:-1 has changed to state Finished
                if number is not None:
                    etime = float(match.group(1))
                    # After first instance of bottle number undef $etime so we don't re-set it
                    bottles[number] = etime + mission_start_esecs
                    self.logger.debug(
                        "Saving time %s for bottle number %s", etime + mission_start_esecs, number,
                    )
                    etime = None
            elif match := fire_gulper_cmd_re.search(line):
                # : Gulper::fireGulper - cmd is \$(\d\d)1Fff
                number = int(match.group(1))
                if etime:
                    # After first instance of bottle number undef $etime so we don't re-set it
                    bottles[number] = etime + mission_start_esecs
                    self.logger.debug(
                        "Saving time %s for bottle number %s", etime + mission_start_esecs, number,
                    )
                    etime = None
        self.logger.debug("Subtracting %s second(s) from bottle times", sec_delay)
        for number, esecs in bottles.items():
            self.logger.debug(
                "number = %s, esecs = %s, new esecs = %s",
                number, esecs, esecs - sec_delay
            )
            bottles[number] = esecs - sec_delay
        if not bottles:
            self.logger.debug("No gulper times found in syslog")
        else:
            self.logger.info("Found %d gulper times in syslog", len(bottles))
        return bottles

    def process_command_line(self) -> None:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter,
            description=__doc__,
        )
        parser.add_argument(
            "--mission",
            help="Mission directory, e.g.: 2020.064.10",
            required=True,
        )
        parser.add_argument(
            "--start_esecs",
            help="Start time of mission in epoch seconds",
            type=float,
        )
        parser.add_argument("--local", help="Read local files", action="store_true")

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
        _handler = logging.StreamHandler()
        _formatter = logging.Formatter(
            "%(levelname)s %(asctime)s %(filename)s %(funcName)s():%(lineno)d %(message)s",
        )
        _handler.setFormatter(_formatter)
        self.logger.addHandler(_handler)

        self.logger.setLevel(self._log_levels[self.args.verbose])
        self.commandline = " ".join(sys.argv)


if __name__ == "__main__":
    # First Gulper was on 2007.120.01
    gulper = Gulper()
    gulper.process_command_line()
    gulper_times = gulper.parse_gulpers()
    gulper.logger.info("number, epoch seconds")
    for number, esecs in gulper_times.items():
        gulper.logger.info("%s, %s", number, esecs)
