#!/usr/bin/env python
'''
Special one-off program to correct the logged times in a set of log files.
Used (initially) for 2017.284.00.  Files must first be downloaded by running:
./logs2netcdfs.py --auv_name Dorado389 --mission 2017.284.00 -v

See https://bitbucket.org/mbari/auv-python/issues/6/dorado_2017_284_00-clock-is-wrong
'''

__author__ = "Mike McCann"
__copyright__ = "Copyright 2020, Monterey Bay Aquarium Research Institute"

import argparse
import asyncio
import concurrent
import os
import sys
import logging
import struct
import time
from array import array
from AUV import AUV
from dataclasses import dataclass
from datetime import datetime, timedelta
from glob import glob
from pathlib import Path
from readauvlog import log_record
from typing import List

LOG_FILES = ('ctdDriver.log', 'ctdDriver2.log', 'gps.log', 'hydroscatlog.log', 
             'navigation.log', 'isuslog.log', 'parosci.log', 'seabird25p.log')
BASE_PATH = 'auv_data'

MISSIONLOGS = 'missionlogs'
MISSIONNETCDFS = 'missionnetcdfs'
PORTAL_BASE = 'http://portal.shore.mbari.org:8080/auvdata/v1'
DEPLOYMENTS_URL = os.path.join(PORTAL_BASE, 'deployments')
TIME = 'time'

class TimeCorrect(AUV):

    logger = logging.getLogger(__name__)
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter('%(levelname)s %(asctime)s %(filename)s '
                                   '%(funcName)s():%(lineno)d %(message)s')
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    _log_levels = (logging.WARN, logging.INFO, logging.DEBUG)

    def read(self, file: str) -> List[log_record]:
        """Reads and parses an AUV log and returns a list of `log_records`
        """
        byte_offset = 0
        records = []
        (byte_offset, records, header_text) = self._read_header(file)
        self._read_data(file, records, byte_offset)

        return records, header_text

    def _read_header(self, file: str):
        """Parses the ASCII header of the log file
        """
        with open(file, 'r', encoding="ISO-8859-15") as f:
            byte_offset = 0
            records = []
            instrument_name = os.path.basename(f.name)

            # Yes, read 2 lines here.
            line = f.readline()
            text = line
            line = f.readline()
            while line:
                text += line
                if "begin" in line:
                    break
                else:
                    # parse line
                    ssv = line.split(" ")
                    data_type = ssv[1]
                    short_name = ssv[2]

                    csv = line.split(",")
                    long_name = csv[1].strip()
                    units = csv[2].strip()
                    if short_name == TIME:
                        units = 'seconds since 1970-01-01 00:00:00Z'
                    r = log_record(data_type, short_name, long_name,
                                   units, instrument_name, [])
                    records.append(r)

                line = f.readline()
                byte_offset = f.tell()

            return (byte_offset, records, text)

    def _read_data(self, file: str, records: List[log_record], byte_offset: int):
        """Parse the binary section of the log file
        """
        if byte_offset == 0:
            raise EOFError(f"{file}: 0 sized file")
        file_size = os.path.getsize(file)

        ok = True
        rec_count = 0
        len_sum = 0
        with open(file, 'rb') as f:
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
                        self.logger.warning(f"{e}, b = {b} at record {rec_count},"
                                            f" for {r.short_name} in file {file}")
                        self.logger.info(f"bytes read = {byte_offset + len_sum}"
                                         f" file size = {file_size}")
                        self.logger.info(f"Tried to read {r.length()} bytes, but"
                                         f" only {byte_offset+len_sum-file_size}"
                                         f" bytes remaining")
                        raise
                    r.data.append(v)
                rec_count += 1

        self.logger.debug(f"bytes read = {byte_offset + len_sum}"
                         f" file size = {file_size}")

    def _correct_dup_short_names(self, log_data):
        short_names = [v.short_name for v in log_data]
        dupes = set([x for n, x in enumerate(short_names) if x in short_names[:n]])
        if len(dupes) > 1:
            raise ValueError(f"Found more than one duplicate: {dupes}")
        elif len(dupes) == 1:
            count = 0
            for i, variable in enumerate(log_data):
                if variable.short_name in dupes:
                    count += 1
                    log_data[i].short_name = f"{log_data[i].short_name}{count}"
                
        return log_data

    def _new_base_filename(self):
        ndt = datetime.strptime(''.join(self.args.mission.split('.')[:2]), '%Y%j')
        ndt += timedelta(seconds=self.args.add_seconds)
        nbf = f"{ndt.strftime('%Y.%j')}.{self.args.mission.split('.')[-1]}"

        return nbf

    def _write_log(self, log_filename, log_data, header_text):
        self.logger.debug(f"Writing log file {log_filename}")
        with open(log_filename, 'wb') as fh:
            fh.write(bytes(header_text, encoding='utf8'))
            for var in log_data:
                self.logger.debug(f"Writing variable {var.short_name}")
                if (var.data_type == 'double' or var.data_type == 'angle'
                    or var.data_type == 'timeTag'):
                    fh.write(array('d', var.data))
                elif var.data_type == 'float':
                    fh.write(array('f', var.data))
                elif var.data_type == 'short':
                    fh.write(array('h', var.data))
                elif var.data_type == 'integer':
                    fh.write(array('i', var.data))
                else:
                    raise(ValueError('No handler for data_type {var.data_type'))

    def _add_and_write(self, log_data, header_text, new_logs_dir, filename):
        log_data = self._correct_dup_short_names(log_data)
        for variable in log_data:
            if variable.short_name == TIME:
                self.logger.debug(f"Adding {self.args.add_seconds} seconds to"
                                  f" Variable {variable.short_name}:"
                                  f" {variable.long_name} ({variable.units})")
                new_times = []
                for datum in variable.data:
                    datum += self.args.add_seconds
                    new_times.append(datum)
                variable.data = new_times
                    
        self._write_log(os.path.join(new_logs_dir, filename),
                        log_data, header_text)

    def correct_times(self):
        vehicle = self.args.auv_name
        name = self.args.mission
        logs_dir = os.path.join(self.args.base_path, vehicle, MISSIONLOGS, name)
        new_basename = self._new_base_filename()
        new_logs_dir = os.path.join(self.args.base_path, vehicle, MISSIONLOGS,
                                    new_basename)
        Path(new_logs_dir).mkdir(parents=True, exist_ok=True)
        for log_filename in glob(os.path.join(logs_dir, '*.log')):
            try:
                self.logger.info(f"Reading file {log_filename}")
                log_data, header_text = self.read(log_filename)
                self._add_and_write(log_data, header_text, new_logs_dir,
                                   os.path.basename(log_filename))
            except (FileNotFoundError, EOFError, struct.error) as e:
                self.logger.debug(f"{e}")

    def process_command_line(self):
        examples = 'Example:' + '\n\n'
        examples += '  Write new original log files with time correction:\n'
        examples += f'    {sys.argv[0]} --auv_name Dorado389 --mission 2017.284.00'
        examples += f' --add_seconds 1146649.348504'

        parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                         description='Convert AUV log files to NetCDF files',
                                         epilog=examples)

        parser.add_argument('--base_path', action='store', default=BASE_PATH, 
                            help="Base directory for missionlogs and"
                                 " missionnetcdfs, default: auv_data")
        parser.add_argument('--auv_name', action='store', default='Dorado389',
                            help="Dorado389, i2map, or multibeam")
        parser.add_argument('--mission', action='store', default='2017.284.00',
                            help="Mission directory, e.g.: 2020.064.10")
        parser.add_argument('--add_seconds', help='Seconds to add to the data',
                            type=float, default=1146649.348504)
        parser.add_argument('-v', '--verbose', type=int, choices=range(3), 
                            action='store', default=0, const=1, nargs='?',
                            help="verbosity level: " + ', '.join(
                                [f"{i}: {v}" for i, v, in enumerate(('WARN', 'INFO', 'DEBUG'))]))

        self.args = parser.parse_args()
        self.logger.setLevel(self._log_levels[self.args.verbose])

        self.commandline = ' '.join(sys.argv)


if __name__ == '__main__':
    tc = TimeCorrect()
    tc.process_command_line()
    tc.correct_times()
