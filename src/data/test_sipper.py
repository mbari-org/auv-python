# noqa: INP001
"""Tests for sipper.py — LRAUV CANONSampler syslog parser."""

import argparse
import logging

import pytest
from sipper import SIPPERING, Sipper

# ---------------------------------------------------------------------------
# Synthetic syslog fragments
# ---------------------------------------------------------------------------

_ISO = "2019-08-16T19:51:31.000Z"


def _line(esec: float, component: str, level: str, message: str) -> str:
    """Build a single well-formed syslog line (with trailing newline)."""
    return f"{_ISO},{esec:.3f} [{component}]({level}): {message}\n"


# Normal case: 2 start events, 2 sample-number events, one unrelated component
NORMAL_SYSLOG = [
    _line(1565985091.135, "CANONSampler", "INFO", f"{SIPPERING} at 37.0N -121.9W"),
    _line(1565985120.000, "CANONSampler", "IMPORTANT", "Sample 1, err_code=0"),
    _line(1565987000.500, "CANONSampler", "INFO", f"{SIPPERING} at 37.1N -121.8W"),
    _line(1565987050.200, "CANONSampler", "IMPORTANT", "Sample 2, err_code=0"),
    # Non-CANONSampler line with SIPPERING text — must be ignored
    _line(1565987100.000, "OtherComponent", "INFO", f"{SIPPERING} irrelevant"),
]

EMPTY_SYSLOG: list = []

NO_CANONSAMPLER_SYSLOG = [
    _line(1565985091.135, "Navigation", "INFO", "Fix acquired"),
    _line(1565985120.000, "CTD", "DEBUG", "Temperature 12.3"),
]

# 2 start events but only 1 sample-number message
MISMATCH_SYSLOG = [
    _line(1565985091.135, "CANONSampler", "INFO", f"{SIPPERING} at 37.0N"),
    _line(1565987000.500, "CANONSampler", "INFO", f"{SIPPERING} at 37.1N"),
    _line(1565987050.200, "CANONSampler", "IMPORTANT", "Sample 1, err_code=0"),
]

# err_code wrapped in parentheses: "Sample 3, (err_code=2)"
PAREN_ERR_SYSLOG = [
    _line(1565985091.135, "CANONSampler", "INFO", f"{SIPPERING} at 37.0N"),
    _line(1565985120.000, "CANONSampler", "IMPORTANT", "Sample 3, (err_code=2)"),
]

# Mix of malformed lines (no syslog format) and valid lines
MALFORMED_MIXED_SYSLOG = [
    "this line does not match the syslog format at all\n",
    "   \n",
    _line(1565985091.135, "CANONSampler", "INFO", f"{SIPPERING} at 37.0N"),
    _line(1565985120.000, "CANONSampler", "IMPORTANT", "Sample 1, err_code=0"),
]

# Lines without trailing newlines (as returned by requests.iter_lines)
STRIPPED_SYSLOG = [line.rstrip("\n") for line in NORMAL_SYSLOG]

# Sample-number lines appear at both INFO and IMPORTANT levels (duplicate text).
# Parser should prefer INFO and avoid double-counting/zip truncation artifacts.
DUPLICATE_LEVEL_SAMPLE_NUM_SYSLOG = [
    _line(1565985091.135, "CANONSampler", "INFO", f"{SIPPERING} at 37.0N"),
    _line(1565985120.000, "CANONSampler", "INFO", "Sample 1, (err_code=0)"),
    _line(1565985120.000, "CANONSampler", "IMPORTANT", "Sample 1, (err_code=0)"),
    _line(1565987000.500, "CANONSampler", "INFO", f"{SIPPERING} at 37.1N"),
    _line(1565987050.200, "CANONSampler", "INFO", "Sample 2, (err_code=0)"),
    _line(1565987050.200, "CANONSampler", "IMPORTANT", "Sample 2, (err_code=0)"),
]


# ---------------------------------------------------------------------------
# _parse_lines tests
# ---------------------------------------------------------------------------


class TestParseLines:
    """Unit tests for Sipper._parse_lines()."""

    def _sipper(self) -> Sipper:
        s = Sipper()
        s.logger.setLevel(logging.WARNING)
        return s

    def test_normal_two_samples(self):
        result = self._sipper()._parse_lines(NORMAL_SYSLOG)
        assert result == {1: pytest.approx(1565985091.135), 2: pytest.approx(1565987000.500)}  # noqa: S101

    def test_returns_dict(self):
        result = self._sipper()._parse_lines(NORMAL_SYSLOG)
        assert isinstance(result, dict)  # noqa: S101

    def test_empty_syslog(self):
        result = self._sipper()._parse_lines(EMPTY_SYSLOG)
        assert result == {}  # noqa: S101

    def test_no_canonsampler_lines(self):
        result = self._sipper()._parse_lines(NO_CANONSAMPLER_SYSLOG)
        assert result == {}  # noqa: S101

    def test_mismatch_truncates_to_shorter(self, caplog):
        s = self._sipper()
        with caplog.at_level(logging.WARNING, logger=s.logger.name):
            result = s._parse_lines(MISMATCH_SYSLOG)
        assert len(result) == 1  # noqa: S101
        assert 1 in result  # noqa: S101
        assert result[1] == pytest.approx(1565985091.135)  # noqa: S101
        assert "Unmatched sipper" in caplog.text  # noqa: S101

    def test_err_code_parentheses_variant(self):
        result = self._sipper()._parse_lines(PAREN_ERR_SYSLOG)
        assert result == {3: pytest.approx(1565985091.135)}  # noqa: S101

    def test_malformed_lines_are_skipped(self):
        result = self._sipper()._parse_lines(MALFORMED_MIXED_SYSLOG)
        assert result == {1: pytest.approx(1565985091.135)}  # noqa: S101

    def test_unrelated_component_ignored(self):
        """SIPPERING text in a non-CANONSampler line must not be counted."""
        result = self._sipper()._parse_lines(NORMAL_SYSLOG)
        # NORMAL_SYSLOG has one OtherComponent SIPPERING line → still only 2 samples
        assert len(result) == 2  # noqa: S101, PLR2004

    def test_lines_without_trailing_newline(self):
        """iter_lines() from requests strips newlines; parser must handle both."""
        result = self._sipper()._parse_lines(STRIPPED_SYSLOG)
        assert result == {1: pytest.approx(1565985091.135), 2: pytest.approx(1565987000.500)}  # noqa: S101

    def test_duplicate_info_and_important_sample_lines(self):
        """When Sample lines exist in INFO and IMPORTANT, parse INFO once per sample."""
        result = self._sipper()._parse_lines(DUPLICATE_LEVEL_SAMPLE_NUM_SYSLOG)
        assert result == {1: pytest.approx(1565985091.135), 2: pytest.approx(1565987000.500)}  # noqa: S101

    def test_sample_numbers_are_int_keys(self):
        result = self._sipper()._parse_lines(NORMAL_SYSLOG)
        for key in result:
            assert isinstance(key, int)  # noqa: S101

    def test_epoch_seconds_are_float_values(self):
        result = self._sipper()._parse_lines(NORMAL_SYSLOG)
        for val in result.values():
            assert isinstance(val, float)  # noqa: S101


# ---------------------------------------------------------------------------
# _match_sippers tests
# ---------------------------------------------------------------------------


class TestMatchSippers:
    """Unit tests for Sipper._match_sippers()."""

    def _sipper(self) -> Sipper:
        s = Sipper()
        s.logger.setLevel(logging.WARNING)
        return s

    def test_equal_length_paired_in_order(self):
        starts = [(100.0, "msg"), (200.0, "msg")]
        nums = [(110.0, 1), (210.0, 2)]
        result = self._sipper()._match_sippers(starts, nums)
        assert result == {1: 100.0, 2: 200.0}  # noqa: S101

    def test_empty_both_lists(self):
        result = self._sipper()._match_sippers([], [])
        assert result == {}  # noqa: S101

    def test_more_starts_than_nums(self, caplog):
        s = self._sipper()
        starts = [(100.0, "msg"), (200.0, "msg")]
        nums = [(110.0, 1)]
        with caplog.at_level(logging.WARNING, logger=s.logger.name):
            result = s._match_sippers(starts, nums)
        assert len(result) == 1  # noqa: S101
        assert "Unmatched" in caplog.text  # noqa: S101

    def test_more_nums_than_starts(self, caplog):
        s = self._sipper()
        starts = [(100.0, "msg")]
        nums = [(110.0, 1), (210.0, 2)]
        with caplog.at_level(logging.WARNING, logger=s.logger.name):
            result = s._match_sippers(starts, nums)
        assert len(result) == 1  # noqa: S101
        assert "Unmatched" in caplog.text  # noqa: S101

    def test_uses_start_esec_not_num_esec(self):
        """The returned epoch seconds must come from the start message, not the num message."""
        starts = [(100.0, "msg")]
        nums = [(999.0, 5)]
        result = self._sipper()._match_sippers(starts, nums)
        assert result == {5: 100.0}  # noqa: S101


# ---------------------------------------------------------------------------
# _read_syslog filesystem tests
# ---------------------------------------------------------------------------


class TestReadSyslogLocal:
    """Tests for Sipper._read_syslog() using the local filesystem path."""

    def _make_sipper(self, log_file: str) -> Sipper:
        s = Sipper()
        s.args = argparse.Namespace(log_file=log_file, local=True)
        s.logger.setLevel(logging.WARNING)
        return s

    def test_reads_local_syslog(self, tmp_path, monkeypatch):
        import sipper as sipper_mod

        monkeypatch.setattr(sipper_mod, "BASE_LRAUV_PATH", tmp_path)

        log_file = "tethys/missionlogs/2019/20190816/mission.nc4"
        syslog_dir = tmp_path / "tethys/missionlogs/2019/20190816"
        syslog_dir.mkdir(parents=True)
        (syslog_dir / "syslog").write_text("".join(NORMAL_SYSLOG), encoding="utf-8")

        s = self._make_sipper(log_file)
        lines = s._read_syslog()
        assert len(lines) == len(NORMAL_SYSLOG)  # noqa: S101

    def test_missing_syslog_raises_file_not_found(self, tmp_path, monkeypatch):
        import sipper as sipper_mod

        monkeypatch.setattr(sipper_mod, "BASE_LRAUV_PATH", tmp_path)

        s = self._make_sipper("tethys/missionlogs/2019/nodir/mission.nc4")
        with pytest.raises(FileNotFoundError, match="syslog"):
            s._read_syslog()

    def test_parse_sippers_end_to_end(self, tmp_path, monkeypatch):
        import sipper as sipper_mod

        monkeypatch.setattr(sipper_mod, "BASE_LRAUV_PATH", tmp_path)

        log_file = "makai/missionlogs/2021/20210101/mission.nc4"
        syslog_dir = tmp_path / "makai/missionlogs/2021/20210101"
        syslog_dir.mkdir(parents=True)
        (syslog_dir / "syslog").write_text("".join(NORMAL_SYSLOG), encoding="utf-8")

        s = Sipper()
        s.args = argparse.Namespace(log_file=log_file, local=True)
        s.logger.setLevel(logging.WARNING)

        result = s.parse_sippers()
        assert len(result) == 2  # noqa: S101, PLR2004
        assert set(result.keys()) == {1, 2}  # noqa: S101

    def test_parse_sippers_no_events_returns_empty(self, tmp_path, monkeypatch):
        import sipper as sipper_mod

        monkeypatch.setattr(sipper_mod, "BASE_LRAUV_PATH", tmp_path)

        log_file = "pontus/missionlogs/2020/20200601/mission.nc4"
        syslog_dir = tmp_path / "pontus/missionlogs/2020/20200601"
        syslog_dir.mkdir(parents=True)
        (syslog_dir / "syslog").write_text("".join(NO_CANONSAMPLER_SYSLOG), encoding="utf-8")

        s = Sipper()
        s.args = argparse.Namespace(log_file=log_file, local=True)
        s.logger.setLevel(logging.WARNING)

        result = s.parse_sippers()
        assert result == {}  # noqa: S101
