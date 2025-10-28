#!/usr/bin/env python
"""
Special one-off program to correct the logged times in a set of log files.
Used (initially) for 2017.284.00.  Files must first be downloaded by running:
./logs2netcdfs.py --auv_name Dorado389 --mission 2017.284.00 -v

See https://bitbucket.org/mbari/auv-python/issues/6/dorado_2017_284_00-clock-is-wrong
"""

__author__ = "Mike McCann"
__copyright__ = "Copyright 2020, Monterey Bay Aquarium Research Institute"

import argparse
import logging
import struct
import sys
from datetime import datetime, timedelta
from pathlib import Path
from shutil import copyfile

from logs2netcdfs import AUV_NetCDF
from readauvlog import log_record

LOG_FILES = (
    "ctdDriver.log",
    "ctdDriver2.log",
    "gps.log",
    "hydroscatlog.log",
    "navigation.log",
    "isuslog.log",
    "parosci.log",
    "seabird25p.log",
)
BASE_PATH = "auv_data"

MISSIONLOGS = "missionlogs"
MISSIONNETCDFS = "missionnetcdfs"
PORTAL_BASE = "http://portal.shore.mbari.org:8080/auvdata/v1"
DEPLOYMENTS_URL = Path(PORTAL_BASE, "deployments")
TIME = "time"


class TimeCorrect:
    logger = logging.getLogger(__name__)
    _handler = logging.StreamHandler()
    _handler.setFormatter(AUV_NetCDF._formatter)
    logger.addHandler(_handler)
    _log_levels = (logging.WARN, logging.INFO, logging.DEBUG)

    def read(self, file: str) -> list[log_record]:
        """Reads and parses an AUV log and returns a list of `log_records`"""
        byte_offset = 0
        records = []
        (byte_offset, records, header_text) = self._read_header(file)
        self._read_data(file, records, byte_offset)

        return records, header_text

    def _read_header(self, file: str):
        """Parses the ASCII header of the log file"""
        with Path(file).open(encoding="ISO-8859-15") as f:
            byte_offset = 0
            records = []
            instrument_name = Path(f.name).name

            # Yes, read 2 lines here.
            line = f.readline()
            text = line
            line = f.readline()
            while line:
                text += line
                if "begin" in line:
                    break
                # parse line
                ssv = line.split(" ")
                data_type = ssv[1]
                short_name = ssv[2]

                csv = line.split(",")
                long_name = csv[1].strip()
                units = csv[2].strip()
                if short_name == TIME:
                    units = "seconds since 1970-01-01 00:00:00Z"
                r = log_record(
                    data_type,
                    short_name,
                    long_name,
                    units,
                    instrument_name,
                    [],
                )
                records.append(r)

                line = f.readline()
                byte_offset = f.tell()

            return (byte_offset, records, text)

    def _read_data(self, file: str, records: list[log_record], byte_offset: int):
        """Parse the binary section of the log file"""
        if byte_offset == 0:
            error_message = f"{file}: 0 sized file"
            raise EOFError(error_message)
        file_size = Path(file).stat().st_size

        ok = True
        rec_count = 0
        len_sum = 0
        with Path(file).open("rb") as f:
            f.seek(byte_offset)
            while ok:
                for r in records:
                    b = f.read(r.length())
                    len_sum += r.length()
                    if not b:
                        ok = False
                        len_sum -= r.length()
                        break
                    s = "<d"
                    if r.data_type == "float":
                        s = "<f"
                    elif r.data_type == "integer":
                        s = "<i"
                    elif r.data_type == "short":
                        s = "<h"
                    try:
                        v = struct.unpack(s, b)[0]
                    except struct.error as e:
                        self.logger.warning(
                            "%s, b = %s at record %d, for %s in file %s",
                            e,
                            b,
                            rec_count,
                            r.short_name,
                            file,
                        )
                        self.logger.info(
                            "bytes read = %d file size = %d",
                            byte_offset + len_sum,
                            file_size,
                        )
                        self.logger.info(
                            "Tried to read %d bytes, but only %d bytes remaining",
                            r.length(),
                            byte_offset + len_sum - file_size,
                        )
                        raise
                    r.data.append(v)
                rec_count += 1

        self.logger.debug(
            "bytes read = %d file size = %d",
            byte_offset + len_sum,
            file_size,
        )

    def _correct_dup_short_names(self, log_data):
        short_names = [v.short_name for v in log_data]
        dupes = {x for n, x in enumerate(short_names) if x in short_names[:n]}
        if len(dupes) > 1:
            error_message = f"Found more than one duplicate: {dupes}"
            raise ValueError(error_message)
        if len(dupes) == 1:
            count = 0
            for i, variable in enumerate(log_data):
                if variable.short_name in dupes:
                    count += 1
                    log_data[i].short_name = f"{variable.short_name}{count}"

        return log_data

    def _new_base_filename(self):
        mission_date = "".join(self.args.mission.split(".")[:2])
        ndt = datetime.strptime(mission_date, "%Y%j").replace(tzinfo=datetime.timezone.utc)
        ndt += timedelta(seconds=self.args.add_seconds)
        return f"{ndt.strftime('%Y.%j')}.{self.args.mission.split('.')[-1]}"

    def _add_and_write(self, log_data, header_text, new_logs_dir, filename):
        log_data = self._correct_dup_short_names(log_data)
        log_filename = Path(new_logs_dir, filename)
        self.logger.debug("Writing log file %s", log_filename)
        self.logger.info(
            "Adding %s seconds to variable %s",
            self.args.add_seconds,
            TIME,
        )
        with Path(log_filename).open("wb") as fh:
            fh.write(bytes(header_text, encoding="utf8"))
            ok = True
            while ok:
                sdata = b""
                for var in log_data:
                    try:
                        datum = var.data.pop(0)
                    except IndexError:
                        ok = False
                        break
                    if var.short_name == TIME:
                        datum += self.args.add_seconds
                    sf = "<d"
                    if var.data_type == "float":
                        sf = "<f"
                    elif var.data_type == "short":
                        sf = "<h"
                    elif var.data_type == "integer":
                        sf = "<i"
                    sdata += struct.pack(sf, datum)

                fh.write(sdata)

        fh.close()
        self.logger.info("Wrote log file %s", log_filename)

    def _verify(self, log_filename):
        self.logger.info("verifying file %s", log_filename)
        log_data, header_text = self.read(log_filename)

    def correct_times(self):
        vehicle = self.args.auv_name
        name = self.args.mission
        logs_dir = Path(self.args.base_path, vehicle, MISSIONLOGS, name)
        new_basename = self._new_base_filename()
        new_logs_dir = Path(
            self.args.base_path,
            vehicle,
            MISSIONLOGS,
            new_basename,
        )
        Path(new_logs_dir).mkdir(parents=True, exist_ok=True)
        for log_filename in Path(logs_dir).glob("*"):
            nlfn = Path(new_logs_dir, Path(log_filename).name)
            if Path(log_filename).stat().st_size == 0 or not str(log_filename).endswith(".log"):
                self.logger.info("Copying file %s", log_filename)
                copyfile(log_filename, nlfn)
            else:
                try:
                    self.logger.info("Reading file %s", log_filename)
                    log_data, header_text = self.read(log_filename)
                except (FileNotFoundError, EOFError, struct.error) as e:
                    self.logger.debug("%s", e)

                self._add_and_write(
                    log_data,
                    header_text,
                    new_logs_dir,
                    Path(log_filename).name,
                )

            # Uncomment to verify correct writing - use with debugger
            ##self._verify(nlfn)

    def process_command_line(self):
        examples = "Example:" + "\n\n"
        examples += "  Write new original log files with time correction:\n"
        examples += f"    {sys.argv[0]} --auv_name Dorado389 --mission 2017.284.00"
        examples += " --add_seconds 1146649.348504"

        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter,
            description="Convert AUV log files to NetCDF files",
            epilog=examples,
        )

        parser.add_argument(
            "--base_path",
            action="store",
            default=BASE_PATH,
            help="Base directory for missionlogs and missionnetcdfs, default: auv_data",
        )
        parser.add_argument(
            "--auv_name",
            action="store",
            default="Dorado389",
            help="Dorado389, i2map, or multibeam",
        )
        parser.add_argument(
            "--mission",
            action="store",
            default="2017.284.00",
            help="Mission directory, e.g.: 2020.064.10",
        )
        parser.add_argument(
            "--add_seconds",
            help="Seconds to add to the data",
            type=float,
            default=1146649.348504,
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
    tc = TimeCorrect()
    tc.process_command_line()
    tc.correct_times()
