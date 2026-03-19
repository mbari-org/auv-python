#!/usr/bin/env python
"""
Parse LRAUV syslog file for sipper sample times and numbers.

The LRAUV water sampler (Sipper) records sample events in the syslog as
CANONSampler component messages.  Two message types are needed to identify
each sample event:

  Start message  (INFO level):
      "ELMO will go to position ..."

  Sample-number/error message  (IMPORTANT level):
      "Sample <n>, err_code=<m>"   or   "Sample <n>, (err_code=<m>)"

Messages are paired in arrival order following the convention established in
STOQS SampleLoaders.py (_sippers_from_json / _match_sippers).

Example syslog line format:
    2019-08-16T19:51:31.135Z,1565985091.135 [CANONSampler](INFO): ELMO will go...
"""

import argparse
import logging
import re
import sys
from http import HTTPStatus
from pathlib import Path

import requests
from logs2netcdfs import TIMEOUT
from nc42netcdfs import BASE_LRAUV_PATH, BASE_LRAUV_WEB

# ---------------------------------------------------------------------------
# Module-level constants (derived from STOQS SampleLoaders.py conventions)
# ---------------------------------------------------------------------------

# Substring that marks the start of a sip
SIPPERING = "ELMO will go to position"

# Pattern to extract the sample number from the confirmation message.
# Handles both "Sample 1, err_code=0" and "Sample 1, (err_code=0)"
_SIPPER_NUM_ERR_RE = re.compile(r"Sample (?P<sipper_num>\d+), \(?err_code=(?P<sipper_err>\d+)\)?")

# Pattern for a well-formed LRAUV syslog line:
#   <ISO-timestamp>,<epoch_seconds> [<Component>](<LEVEL>): <message>
_SYSLOG_LINE_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?"  # ISO timestamp
    r",(?P<log_esec>\d+\.\d+)\s"  # epoch seconds
    r"\[(?P<log_component>[^\]]+)\]"  # [Component]
    r"\((?P<log_level>[^\)]+)\):\s"  # (LEVEL):
    r"(?P<log_message>.+)"  # message text
)


class Sipper:
    logger = logging.getLogger(__name__)
    _log_levels = (logging.WARN, logging.INFO, logging.DEBUG)

    def parse_sippers(self) -> dict:
        """Parse sipper sample times from the LRAUV syslog.

        Reads the syslog that lives alongside ``self.args.log_file``.
        Extracts CANONSampler start events and sample-number/error events,
        pairs them in arrival order, and returns a mapping of
        ``{sipper_number: start_epoch_seconds}``.

        Returns:
            dict mapping integer sample number to float epoch seconds.

        Raises:
            FileNotFoundError: if the syslog cannot be located (local mode) or
                the HTTP request fails (remote mode).
        """
        lines = self._read_syslog()
        return self._parse_lines(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_syslog(self) -> list:
        """Return syslog lines from a local file or HTTP request."""
        log_dir = Path(self.args.log_file).parent
        if self.args.local:
            syslog_path = Path(BASE_LRAUV_PATH, log_dir, "syslog")
            self.logger.info("Reading local syslog %s", syslog_path)
            if not syslog_path.exists():
                msg = f"Syslog not found: {syslog_path}"
                raise FileNotFoundError(msg)
            with syslog_path.open(encoding="utf-8", errors="ignore") as fh:
                return fh.readlines()
        else:
            # Construct URL: BASE_LRAUV_WEB already ends with "/"
            syslog_url = BASE_LRAUV_WEB + log_dir.as_posix() + "/syslog"
            self.logger.info("Reading remote syslog %s", syslog_url)
            with requests.get(syslog_url, stream=True, timeout=TIMEOUT) as resp:
                if resp.status_code != HTTPStatus.OK:
                    msg = f"Cannot read {syslog_url}, " f"HTTP status={resp.status_code}"
                    raise FileNotFoundError(msg)
                return [line.decode(errors="ignore") for line in resp.iter_lines()]

    def _parse_lines(self, lines: list) -> dict:
        """Scan syslog lines for CANONSampler events and pair them.

        Args:
            lines: Raw syslog lines (strings, with or without trailing newline).

        Returns:
            dict mapping sipper_number → start_epoch_seconds.
        """
        samplings_at_info: list = []  # [(start_esec, message), ...]
        samplings_at_other: list = []  # [(start_esec, message), ...]
        sample_num_errs_info: list = []  # [(esec, sipper_num), ...]
        sample_num_errs_important: list = []  # [(esec, sipper_num), ...]

        for line in lines:
            m = _SYSLOG_LINE_RE.match(line.rstrip("\n"))
            if not m:
                continue
            if m.group("log_component") != "CANONSampler":
                continue

            esec = float(m.group("log_esec"))
            log_level = m.group("log_level").upper()
            message = m.group("log_message")

            if SIPPERING in message:
                if log_level == "INFO":
                    samplings_at_info.append((esec, message))
                else:
                    samplings_at_other.append((esec, message))
                self.logger.debug("Sipper start at %.3f: %s", esec, message)
            elif num_m := _SIPPER_NUM_ERR_RE.search(message):
                sipper_num = int(num_m.group("sipper_num"))
                if log_level == "INFO":
                    sample_num_errs_info.append((esec, sipper_num))
                elif log_level == "IMPORTANT":
                    sample_num_errs_important.append((esec, sipper_num))
                self.logger.debug("Sample-number event at %.3f: sipper=%d", esec, sipper_num)

        # Mirror STOQS behavior: prefer INFO lines for both SIPPERING and
        # Sample/err_code parsing. Use fallbacks only when INFO is unavailable.
        samplings_at = samplings_at_info or samplings_at_other
        sample_num_errs = sample_num_errs_info or sample_num_errs_important

        if sample_num_errs_info and sample_num_errs_important:
            self.logger.debug("Using INFO Sample/err_code lines and ignoring IMPORTANT duplicates")

        return self._match_sippers(samplings_at, sample_num_errs)

    def _match_sippers(
        self,
        samplings_at: list,
        sample_num_errs: list,
    ) -> dict:
        """Pair start events with sample-number events in arrival order.

        Following STOQS convention the two lists are zipped positionally.
        If the lengths differ a warning is logged and the shorter list
        determines how many pairs are returned.

        Args:
            samplings_at:   [(start_esec, message), ...]
            sample_num_errs: [(esec, sipper_num), ...]

        Returns:
            dict mapping sipper_number (int) → start_epoch_seconds (float).
        """
        if len(samplings_at) != len(sample_num_errs):
            self.logger.warning(
                "Unmatched sipper events: %d start message(s) vs "
                "%d sample-number message(s). Truncating to shorter list.",
                len(samplings_at),
                len(sample_num_errs),
            )

        samples = {}
        for (start_esec, _), (_, sipper_num) in zip(samplings_at, sample_num_errs, strict=False):
            samples[sipper_num] = start_esec
            self.logger.info("Sipper sample %d at %.3f epoch seconds", sipper_num, start_esec)

        if not samples:
            self.logger.debug("No sipper events found in syslog")
        else:
            self.logger.info("Found %d sipper event(s) in syslog", len(samples))

        return samples

    # ------------------------------------------------------------------
    # CLI
    # ------------------------------------------------------------------

    def process_command_line(self) -> None:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter,
            description=__doc__,
        )
        parser.add_argument(
            "--log_file",
            help=(
                "Relative path (from BASE_LRAUV_PATH) to the resampled log file,\n"
                "e.g.: tethys/missionlogs/2019/.../mission.nc4"
            ),
            required=True,
        )
        parser.add_argument(
            "--local",
            help="Read from local filesystem instead of HTTP",
            action="store_true",
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

        _handler = logging.StreamHandler()
        _formatter = logging.Formatter(
            "%(levelname)s %(asctime)s %(filename)s %(funcName)s():%(lineno)d %(message)s",
        )
        _handler.setFormatter(_formatter)
        self.logger.addHandler(_handler)
        self.logger.setLevel(self._log_levels[self.args.verbose])
        self.commandline = " ".join(sys.argv)


if __name__ == "__main__":
    sipper = Sipper()
    sipper.process_command_line()
    sipper_times = sipper.parse_sippers()
    sipper.logger.info("sipper_number, epoch_seconds")
    for number, esecs in sipper_times.items():
        sipper.logger.info("%s, %s", number, esecs)
