"""Tests for DeploymentPlotter — per-PNG HTML generation and helpers."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import lzstring
import numpy as np
import pandas as pd
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))

from lrauv_deployment_plots import DeploymentPlotter
from make_permalink import stoqs_url_from_ds

# Representative OPeNDAP URL that matches the shape produced by the real code
_OPENDAP_BASE = "http://dods.mbari.org/opendap/data/lrauv"
_NC_URL = f"{_OPENDAP_BASE}/ahi/missionlogs/2025/20250414_20250418/20250414T120000/ahi_1S.nc"
_STOQS_URL = "https://tethysviz.shore.mbari.org/stoqs_lrauv_apr2025/query/?permalink_id=abc123"
_PNG_URL = (
    "https://dods.mbari.org/opendap/data/lrauv/ahi/missionlogs/2025/20250414_20250418/depl.png"
)

_DLIST_CONTENT = """\
# Deployment name: CANON April 2025
20250414T120000
"""
_DLIST = "ahi/missionlogs/2025/20250414_20250418.dlist"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_ds(start="2025-04-14", periods=3, freq="1D", min_depth=0.0, max_depth=200.0):
    """Return a tiny xarray Dataset with CF-compliant time and depth."""
    times = pd.date_range(start, periods=periods, freq=freq)
    depths = np.linspace(min_depth, max_depth, periods)
    ds = xr.Dataset(
        {"depth": ("time", depths)},
        coords={"time": times},
    )
    ds["depth"].attrs["standard_name"] = "depth"
    ds["time"].attrs["standard_name"] = "time"
    return ds


def _mock_session_get(platform_id="42"):
    """Return a mock requests.Session().get() that responds with a CSV platform row."""
    csv_body = f"id,name\n{platform_id},ahi\n".encode()
    mock_resp = MagicMock()
    mock_resp.content = csv_body
    return MagicMock(return_value=mock_resp)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=False)
def dp():
    plotter = DeploymentPlotter()
    plotter.logger.setLevel("DEBUG")
    return plotter


@pytest.fixture(autouse=True)
def _clear_notify_env(monkeypatch):
    """Prevent tests from accidentally sending real Slack/email notifications."""
    monkeypatch.delenv("LRAUV_NOTIFY", raising=False)
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)


# ---------------------------------------------------------------------------
# Tests for _write_per_png_html()
# ---------------------------------------------------------------------------


class TestWritePerPngHtml:
    def test_basic_structure(self, dp, tmp_path):
        html_path = tmp_path / "deployment_2column_cmocean.html"
        with (
            patch.object(dp, "_url_exists", return_value=False),
            patch.object(dp, "_stoqs_url_for_nc_url", return_value=None),
        ):
            dp._write_per_png_html(
                html_path,
                "Test Deployment\nApril 2025",
                "deployment_2column_cmocean.png",
                _PNG_URL,
                None,
                [_NC_URL],
            )
        html = html_path.read_text()
        assert "<!DOCTYPE html>" in html  # noqa: S101
        assert "Test Deployment" in html  # noqa: S101
        assert "April 2025" in html  # noqa: S101

    def test_png_embedded_as_img(self, dp, tmp_path):
        html_path = tmp_path / "depl_cmocean.html"
        with (
            patch.object(dp, "_url_exists", return_value=False),
            patch.object(dp, "_stoqs_url_for_nc_url", return_value=None),
        ):
            dp._write_per_png_html(
                html_path,
                "Title",
                "depl_cmocean.png",
                _PNG_URL,
                None,
                [_NC_URL],
            )
        html = html_path.read_text()
        assert '<img src="' + _PNG_URL + '"' in html  # noqa: S101

    def test_stoqs_link_present_when_url_given(self, dp, tmp_path):
        html_path = tmp_path / "depl.html"
        with (
            patch.object(dp, "_url_exists", return_value=False),
            patch.object(dp, "_stoqs_url_for_nc_url", return_value=None),
        ):
            dp._write_per_png_html(
                html_path,
                "Title",
                "depl.png",
                _PNG_URL,
                _STOQS_URL,
                [_NC_URL],
            )
        html = html_path.read_text()
        assert _STOQS_URL in html  # noqa: S101
        assert "stoqs_lrauv_apr2025" in html  # noqa: S101

    def test_stoqs_link_absent_when_no_url(self, dp, tmp_path):
        html_path = tmp_path / "depl.html"
        with (
            patch.object(dp, "_url_exists", return_value=False),
            patch.object(dp, "_stoqs_url_for_nc_url", return_value=None),
        ):
            dp._write_per_png_html(
                html_path,
                "Title",
                "depl.png",
                _PNG_URL,
                None,
                [_NC_URL],
            )
        assert "STOQS" not in html_path.read_text()  # noqa: S101

    def test_opendap_link_present(self, dp, tmp_path):
        html_path = tmp_path / "depl.html"
        with (
            patch.object(dp, "_url_exists", return_value=False),
            patch.object(dp, "_stoqs_url_for_nc_url", return_value=None),
        ):
            dp._write_per_png_html(
                html_path,
                "Title",
                "depl.png",
                _PNG_URL,
                None,
                [_NC_URL],
            )
        assert _NC_URL + ".html" in html_path.read_text()  # noqa: S101

    def test_log_directory_listed(self, dp, tmp_path):
        html_path = tmp_path / "depl.html"
        with (
            patch.object(dp, "_url_exists", return_value=False),
            patch.object(dp, "_stoqs_url_for_nc_url", return_value=None),
        ):
            dp._write_per_png_html(
                html_path,
                "Title",
                "depl.png",
                _PNG_URL,
                None,
                [_NC_URL],
            )
        assert "20250414T120000" in html_path.read_text()  # noqa: S101

    def test_multiple_log_dirs_listed(self, dp, tmp_path):
        html_path = tmp_path / "depl.html"
        nc_url_b = _NC_URL.replace("20250414T120000", "20250415T080000")
        with (
            patch.object(dp, "_url_exists", return_value=False),
            patch.object(dp, "_stoqs_url_for_nc_url", return_value=None),
        ):
            dp._write_per_png_html(
                html_path,
                "Title",
                "depl.png",
                _PNG_URL,
                None,
                [_NC_URL, nc_url_b],
            )
        html = html_path.read_text()
        assert "20250414T120000" in html  # noqa: S101
        assert "20250415T080000" in html  # noqa: S101

    def test_per_log_html_link_when_url_exists(self, dp, tmp_path):
        html_path = tmp_path / "depl.html"
        with (
            patch.object(dp, "_url_exists", return_value=True),
            patch.object(dp, "_stoqs_url_for_nc_url", return_value=None),
        ):
            dp._write_per_png_html(
                html_path,
                "Title",
                "depl.png",
                _PNG_URL,
                None,
                [_NC_URL],
            )
        assert "2column_cmocean" in html_path.read_text()  # noqa: S101

    def test_per_log_png_links_included(self, dp, tmp_path):
        html_path = tmp_path / "depl.html"
        fake_png_url = _NC_URL.replace("_1S.nc", "_1S_2column_cmocean.png")
        with (
            patch.object(dp, "_url_exists", return_value=False),
            patch.object(
                dp,
                "_per_log_png_links",
                return_value=[(fake_png_url, "ahi_1S_2column_cmocean.png")],
            ),
            patch.object(dp, "_stoqs_url_for_nc_url", return_value=None),
        ):
            dp._write_per_png_html(
                html_path,
                "Title",
                "depl.png",
                _PNG_URL,
                None,
                [_NC_URL],
            )
        assert "ahi_1S_2column_cmocean.png" in html_path.read_text()  # noqa: S101

    def test_per_log_stoqs_url_used_per_log(self, dp, tmp_path):
        html_path = tmp_path / "depl.html"
        per_log_url = _STOQS_URL.replace("abc123", "xyzlog")
        with (
            patch.object(dp, "_url_exists", return_value=False),
            patch.object(dp, "_per_log_stoqs_url", return_value=per_log_url),
        ):
            dp._write_per_png_html(
                html_path,
                "Title",
                "depl.png",
                _PNG_URL,
                _STOQS_URL,
                [_NC_URL],
                auv_name="ahi",
            )
        html = html_path.read_text()
        assert "xyzlog" in html  # noqa: S101

    def test_footer_contains_script_link_and_timestamp(self, dp, tmp_path):
        html_path = tmp_path / "depl_cmocean.html"
        with (
            patch.object(dp, "_url_exists", return_value=False),
            patch.object(dp, "_stoqs_url_for_nc_url", return_value=None),
        ):
            dp._write_per_png_html(
                html_path,
                "Title",
                "depl_cmocean.png",
                _PNG_URL,
                None,
                [_NC_URL],
            )
        html = html_path.read_text()
        assert "<hr>" in html  # noqa: S101
        assert "lrauv_deployment_plots.py" in html  # noqa: S101
        assert "github.com/mbari-org/auv-python" in html  # noqa: S101
        assert "Created by" in html  # noqa: S101


# ---------------------------------------------------------------------------
# Tests for helper methods
# ---------------------------------------------------------------------------


class TestPerLogStoqsUrl:
    def test_returns_nc_scoped_url_when_available(self, dp):
        scoped = _STOQS_URL.replace("abc123", "scoped")
        with patch.object(dp, "_stoqs_url_for_nc_url", return_value=scoped):
            result = dp._per_log_stoqs_url([_NC_URL], "ahi", _STOQS_URL)
        assert result == scoped  # noqa: S101

    def test_falls_back_to_deployment_url(self, dp):
        with patch.object(dp, "_stoqs_url_for_nc_url", return_value=None):
            result = dp._per_log_stoqs_url([_NC_URL], "ahi", _STOQS_URL)
        assert result == _STOQS_URL  # noqa: S101

    def test_returns_fallback_when_no_auv_name(self, dp):
        result = dp._per_log_stoqs_url([_NC_URL], "", _STOQS_URL)
        assert result == _STOQS_URL  # noqa: S101

    def test_returns_none_when_no_url_and_no_auv_name(self, dp):
        assert dp._per_log_stoqs_url([_NC_URL], "", None) is None  # noqa: S101


class TestPerLogPngLinks:
    def test_returns_links_for_existing_pngs(self, dp):
        with patch.object(dp, "_url_exists", return_value=True):
            result = dp._per_log_png_links([_NC_URL])
        assert any("2column_cmocean" in lbl for _, lbl in result)  # noqa: S101

    def test_returns_empty_when_no_pngs_exist(self, dp):
        with patch.object(dp, "_url_exists", return_value=False):
            assert dp._per_log_png_links([_NC_URL]) == []  # noqa: S101

    def test_returns_multiple_tuples_for_multiple_kinds(self, dp):
        with patch.object(dp, "_url_exists", return_value=True):
            result = dp._per_log_png_links([_NC_URL])
        assert len(result) > 1  # noqa: S101


# ---------------------------------------------------------------------------
# Tests for _build_and_write_html()
# ---------------------------------------------------------------------------


class TestBuildAndWriteHtml:
    _DLIST = "ahi/missionlogs/2025/20250414_20250418.dlist"

    def _call(self, dp, tmp_path, *, session_side_effect=None):
        png = tmp_path / "CANON_April_2025_2column_cmocean.png"
        png.touch()
        with (
            patch("make_permalink.requests.Session") as mock_session_cls,
            patch.object(dp, "_url_exists", return_value=False),
            patch.object(dp, "_stoqs_url_for_nc_url", return_value=None),
        ):
            if session_side_effect:
                mock_session_cls.return_value.__enter__.return_value.get.side_effect = (
                    session_side_effect
                )
            else:
                mock_session_cls.return_value.__enter__.return_value.get = _mock_session_get("7")
            dp._build_and_write_html(
                tmp_path,
                self._DLIST,
                "CANON_April_2025",
                "CANON April 2025",
                _make_ds("2025-04-14"),
                [str(png)],
                [_NC_URL],
            )

    def test_per_png_html_written(self, dp, tmp_path):
        self._call(dp, tmp_path)
        assert (tmp_path / "CANON_April_2025_2column_cmocean.html").exists()  # noqa: S101

    def test_no_index_html_written(self, dp, tmp_path):
        self._call(dp, tmp_path)
        assert not (tmp_path / "CANON_April_2025.html").exists()  # noqa: S101

    def test_stoqs_url_in_per_png_html(self, dp, tmp_path):
        self._call(dp, tmp_path)
        html = (tmp_path / "CANON_April_2025_2column_cmocean.html").read_text()
        assert "stoqs_lrauv_apr2025" in html  # noqa: S101
        assert "/query/?permalink_id=" in html  # noqa: S101

    def test_title_in_per_png_html(self, dp, tmp_path):
        self._call(dp, tmp_path)
        html = (tmp_path / "CANON_April_2025_2column_cmocean.html").read_text()
        assert "CANON April 2025" in html  # noqa: S101

    def test_stoqs_failure_still_writes_html(self, dp, tmp_path):
        """An error inside stoqs_url_from_ds must not prevent HTML generation."""
        png = tmp_path / "CANON_April_2025_2column_cmocean.png"
        png.touch()
        with (
            patch("lrauv_deployment_plots.stoqs_url_from_ds", side_effect=OSError("network down")),
            patch.object(dp, "_url_exists", return_value=False),
            patch.object(dp, "_stoqs_url_for_nc_url", return_value=None),
        ):
            dp._build_and_write_html(
                tmp_path,
                self._DLIST,
                "CANON_April_2025",
                "CANON April 2025",
                _make_ds("2025-04-14"),
                [str(png)],
                [_NC_URL],
            )
        html = (tmp_path / "CANON_April_2025_2column_cmocean.html").read_text()
        assert "CANON April 2025" in html  # noqa: S101
        assert "STOQS" not in html  # noqa: S101

    def test_missing_png_produces_no_html(self, dp, tmp_path):
        """A PNG path that doesn't exist on disk must not produce an HTML file."""
        with (
            patch("make_permalink.requests.Session") as mock_session_cls,
            patch.object(dp, "_url_exists", return_value=False),
            patch.object(dp, "_stoqs_url_for_nc_url", return_value=None),
        ):
            mock_session_cls.return_value.__enter__.return_value.get = _mock_session_get("7")
            dp._build_and_write_html(
                tmp_path,
                self._DLIST,
                "CANON_April_2025",
                "CANON April 2025",
                _make_ds("2025-04-14"),
                [str(tmp_path / "ghost.png")],
                [_NC_URL],
            )
        assert not (tmp_path / "ghost.html").exists()  # noqa: S101


class TestPendingLogDirs:
    """_pending_log_dirs() identifies .dlist log directories still missing data,
    i.e. still being transferred from the vehicle to the archive filesystem."""

    _DLIST_CONTENT = "# Deployment name: CANON April 2025\n20250414T120000\n"

    def test_empty_when_all_dirs_present(self, dp):
        assert dp._pending_log_dirs(self._DLIST_CONTENT, [_NC_URL]) == []  # noqa: S101

    def test_lists_dir_with_no_nc_file(self, dp):
        dlist_content = self._DLIST_CONTENT + "20250415T080000\n"
        assert dp._pending_log_dirs(dlist_content, [_NC_URL]) == [  # noqa: S101
            "20250415T080000"
        ]

    def test_ignores_commented_out_dirs(self, dp):
        dlist_content = self._DLIST_CONTENT + "# 20250415T080000\n"
        assert dp._pending_log_dirs(dlist_content, [_NC_URL]) == []  # noqa: S101


class TestNotificationGating:
    """_build_and_write_html() must withhold --notify until every .dlist log
    directory has produced data, logging a warning instead in the meantime."""

    _DLIST = "ahi/missionlogs/2025/20250414_20250418.dlist"

    def _call(self, dp, tmp_path, *, pending_log_dirs, notify=None):
        png = tmp_path / "CANON_April_2025_2column_cmocean.png"
        png.touch()
        with (
            patch("make_permalink.requests.Session") as mock_session_cls,
            patch.object(dp, "_url_exists", return_value=False),
            patch.object(dp, "_stoqs_url_for_nc_url", return_value=None),
            patch.object(dp, "_notify") as mock_notify,
        ):
            mock_session_cls.return_value.__enter__.return_value.get = _mock_session_get("7")
            dp._build_and_write_html(
                tmp_path,
                self._DLIST,
                "CANON_April_2025",
                "CANON April 2025",
                _make_ds("2025-04-14"),
                [str(png)],
                [_NC_URL],
                notify=notify,
                pending_log_dirs=pending_log_dirs,
            )
        return mock_notify

    def test_notifies_when_fully_transferred(self, dp, tmp_path):
        mock_notify = self._call(dp, tmp_path, pending_log_dirs=[], notify=["a@b.com"])
        mock_notify.assert_called_once()

    def test_withholds_notification_when_pending(self, dp, tmp_path, caplog):
        caplog.set_level("WARNING")
        mock_notify = self._call(
            dp, tmp_path, pending_log_dirs=["20250415T080000"], notify=["a@b.com"]
        )
        mock_notify.assert_not_called()
        assert "still waiting" in caplog.text  # noqa: S101
        assert "20250415T080000" in caplog.text  # noqa: S101

    def test_no_warning_when_notify_not_requested(self, dp, tmp_path, caplog):
        """Nothing was ever going to be sent, so withholding it isn't warning-worthy."""
        caplog.set_level("WARNING")
        mock_notify = self._call(dp, tmp_path, pending_log_dirs=["20250415T080000"], notify=None)
        mock_notify.assert_not_called()
        assert "still waiting" not in caplog.text  # noqa: S101


class TestUpdateIndexHtml:
    """Regression tests for quick_look_plots.html accumulating multiple deployments.

    a3f4b45 ("Drop <auv_name> suffix from all the file names.") added a filter
    intended to purge leftover old auv_name-prefixed hrefs when re-reading an
    existing index, but it compared the href's filename against the date-range
    dlist_dir instead of the old auv_name prefix. Since filenames are built from
    the deployment name (not the dlist_dir), the filter matched nothing on
    re-read and silently dropped every previously recorded deployment's entries.
    """

    def test_preserves_prior_deployments_across_calls(self, dp, tmp_path):
        """A second deployment's _update_index_html() call must not erase the
        first deployment's entries."""
        year_dir = tmp_path / "2025"
        depl_dir_1 = year_dir / "20250414_20250418"
        depl_dir_2 = year_dir / "20250501_20250505"
        depl_dir_1.mkdir(parents=True)
        depl_dir_2.mkdir(parents=True)

        dp._update_index_html(
            depl_dir_1,
            [depl_dir_1 / "CANON_April_2025_2column_cmocean.html"],
            deployment_name="CANON April 2025",
        )
        dp._update_index_html(
            depl_dir_2,
            [depl_dir_2 / "CANON_May_2025_2column_cmocean.html"],
            deployment_name="CANON May 2025",
        )

        content = (year_dir / "quick_look_plots.html").read_text()
        assert (  # noqa: S101
            'href="20250414_20250418/CANON_April_2025_2column_cmocean.html"' in content
        )
        assert (  # noqa: S101
            'href="20250501_20250505/CANON_May_2025_2column_cmocean.html"' in content
        )
        assert "CANON April 2025" in content  # noqa: S101
        assert "CANON May 2025" in content  # noqa: S101


# ---------------------------------------------------------------------------
# Tests for plot_deployment() — exercises the stoqs_url_from_ds call path
# ---------------------------------------------------------------------------


class TestPlotDeploymentStoqsUrl:
    """Drive plot_deployment() to the stoqs_url_from_ds call without hitting
    the network or running any real plot generation."""

    def _make_deployment_dir(self, tmp_path: Path) -> Path:
        """Create the deployment directory that plot_deployment() requires."""
        depl_dir = tmp_path / "ahi" / "missionlogs" / "2025" / "20250414_20250418"
        depl_dir.mkdir(parents=True)
        return depl_dir

    def test_per_png_html_written_by_plot_deployment(self, dp, tmp_path):
        """plot_deployment() must produce per-PNG HTML files, not an index."""
        depl_dir = self._make_deployment_dir(tmp_path)
        fake_png = depl_dir / "CANON_April_2025_2column_cmocean.png"
        fake_png.touch()

        mock_cp = MagicMock()
        mock_cp.plot_2column.return_value = str(fake_png)
        mock_cp.plot_biolume_2column.return_value = None
        mock_cp.plot_planktivore_2column.return_value = None

        with (
            patch("lrauv_deployment_plots.BASE_LRAUV_PATH", tmp_path),
            patch.object(dp, "_read_dlist_content", return_value=_DLIST_CONTENT),
            patch.object(dp, "_collect_nc_files", return_value=[_NC_URL]),
            patch.object(dp, "_concat_datasets", return_value=_make_ds("2025-04-14")),
            patch("lrauv_deployment_plots.CreateProducts", return_value=mock_cp),
            patch("make_permalink.requests.Session") as mock_session_cls,
            patch.object(dp, "_url_exists", return_value=False),
            patch.object(dp, "_stoqs_url_for_nc_url", return_value=None),
        ):
            mock_session_cls.return_value.__enter__.return_value.get = _mock_session_get("7")
            dp.plot_deployment(_DLIST, verbose=1, force=True)

        assert (depl_dir / "CANON_April_2025_2column_cmocean.html").exists()  # noqa: S101
        assert not (depl_dir / "CANON_April_2025.html").exists()  # noqa: S101

    def test_stoqs_url_in_html(self, dp, tmp_path):
        """Real stoqs_url_from_ds() runs; only the HTTP call inside it is mocked."""
        depl_dir = self._make_deployment_dir(tmp_path)

        fake_png = depl_dir / "CANON_April_2025_2column_cmocean.png"
        fake_png.touch()

        real_ds = _make_ds("2025-04-14")

        mock_cp = MagicMock()
        mock_cp.plot_2column.return_value = str(fake_png)
        mock_cp.plot_biolume_2column.return_value = None
        mock_cp.plot_planktivore_2column.return_value = None

        with (
            patch("lrauv_deployment_plots.BASE_LRAUV_PATH", tmp_path),
            patch.object(dp, "_read_dlist_content", return_value=_DLIST_CONTENT),
            patch.object(dp, "_collect_nc_files", return_value=[_NC_URL]),
            patch.object(dp, "_concat_datasets", return_value=real_ds),
            patch("lrauv_deployment_plots.CreateProducts", return_value=mock_cp),
            patch("make_permalink.requests.Session") as mock_session_cls,
            patch.object(dp, "_url_exists", return_value=False),
            patch.object(dp, "_stoqs_url_for_nc_url", return_value=None),
        ):
            mock_session_cls.return_value.__enter__.return_value.get = _mock_session_get("7")
            dp.plot_deployment(_DLIST, verbose=1, force=True)

        html = (depl_dir / "CANON_April_2025_2column_cmocean.html").read_text()
        assert "stoqs_lrauv_apr2025" in html  # noqa: S101
        assert "/query/?permalink_id=" in html  # noqa: S101

    def test_no_html_when_no_pngs(self, dp, tmp_path):
        """When all plot methods return None the HTML file must not be written."""
        depl_dir = self._make_deployment_dir(tmp_path)

        real_ds = _make_ds("2025-04-14")

        mock_cp = MagicMock()
        mock_cp.plot_2column.return_value = None
        mock_cp.plot_biolume_2column.return_value = None
        mock_cp.plot_planktivore_2column.return_value = None

        with (
            patch("lrauv_deployment_plots.BASE_LRAUV_PATH", tmp_path),
            patch.object(dp, "_read_dlist_content", return_value=_DLIST_CONTENT),
            patch.object(dp, "_collect_nc_files", return_value=[_NC_URL]),
            patch.object(dp, "_concat_datasets", return_value=real_ds),
            patch("lrauv_deployment_plots.CreateProducts", return_value=mock_cp),
        ):
            dp.plot_deployment(_DLIST, verbose=1)

        assert not (depl_dir / "CANON_April_2025.html").exists()  # noqa: S101


# ---------------------------------------------------------------------------
# Tests for --force / _deployment_has_outputs()
# ---------------------------------------------------------------------------


class TestForceFlag:
    """Verify that existing outputs are skipped by default and reprocessed with force=True."""

    def _make_deployment_dir(self, tmp_path: Path) -> Path:
        depl_dir = tmp_path / "ahi" / "missionlogs" / "2025" / "20250414_20250418"
        depl_dir.mkdir(parents=True)
        return depl_dir

    def test_skips_when_outputs_exist(self, dp, tmp_path):
        """plot_deployment() must return without calling CreateProducts when
        outputs exist and are newer than every source nc file."""
        depl_dir = self._make_deployment_dir(tmp_path)
        # Pre-create an output PNG (+ HTML sibling) that _deployment_has_outputs() will find
        png = depl_dir / "CANON_April_2025_2column_cmocean.png"
        png.touch()
        (depl_dir / "CANON_April_2025_2column_cmocean.html").touch()

        with (
            patch("lrauv_deployment_plots.BASE_LRAUV_PATH", tmp_path),
            patch.object(dp, "_read_dlist_content", return_value=_DLIST_CONTENT),
            patch.object(dp, "_collect_nc_files", return_value=[_NC_URL]),
            patch.object(dp, "_nc_file_mtime", return_value=png.stat().st_mtime - 100),
            patch("lrauv_deployment_plots.CreateProducts") as mock_cp_cls,
        ):
            dp.plot_deployment(_DLIST, verbose=1)  # force=False by default

        mock_cp_cls.assert_not_called()  # noqa: S101

    def test_reprocesses_when_nc_file_newer_than_outputs(self, dp, tmp_path):
        """plot_deployment() must reprocess when a source nc file is newer than
        the existing outputs, even without --force (the staleness case)."""
        depl_dir = self._make_deployment_dir(tmp_path)
        png = depl_dir / "CANON_April_2025_2column_cmocean.png"
        png.touch()
        (depl_dir / "CANON_April_2025_2column_cmocean.html").touch()

        mock_cp = MagicMock()
        mock_cp.plot_2column.return_value = None
        mock_cp.plot_biolume_2column.return_value = None
        mock_cp.plot_planktivore_2column.return_value = None

        with (
            patch("lrauv_deployment_plots.BASE_LRAUV_PATH", tmp_path),
            patch.object(dp, "_read_dlist_content", return_value=_DLIST_CONTENT),
            patch.object(dp, "_collect_nc_files", return_value=[_NC_URL]),
            patch.object(dp, "_nc_file_mtime", return_value=png.stat().st_mtime + 100),
            patch.object(dp, "_concat_datasets", return_value=_make_ds("2025-04-14")),
            patch("lrauv_deployment_plots.CreateProducts", return_value=mock_cp),
        ):
            dp.plot_deployment(_DLIST, verbose=1)  # force=False by default

        mock_cp.plot_2column.assert_called_once()  # noqa: S101

    def test_force_reprocesses_when_outputs_exist(self, dp, tmp_path):
        """plot_deployment(force=True) must proceed even when a PNG already exists."""
        depl_dir = self._make_deployment_dir(tmp_path)
        fake_png = depl_dir / "CANON_April_2025_2column_cmocean.png"
        fake_png.touch()

        mock_cp = MagicMock()
        mock_cp.plot_2column.return_value = str(fake_png)
        mock_cp.plot_biolume_2column.return_value = None
        mock_cp.plot_planktivore_2column.return_value = None

        with (
            patch("lrauv_deployment_plots.BASE_LRAUV_PATH", tmp_path),
            patch.object(dp, "_read_dlist_content", return_value=_DLIST_CONTENT),
            patch.object(dp, "_collect_nc_files", return_value=[_NC_URL]),
            patch.object(dp, "_concat_datasets", return_value=_make_ds("2025-04-14")),
            patch("lrauv_deployment_plots.CreateProducts", return_value=mock_cp),
            patch.object(dp, "_url_exists", return_value=False),
            patch.object(dp, "_stoqs_url_for_nc_url", return_value=None),
            patch("make_permalink.requests.Session") as mock_session_cls,
        ):
            mock_session_cls.return_value.__enter__.return_value.get = _mock_session_get("7")
            dp.plot_deployment(_DLIST, verbose=1, force=True)

        mock_cp.plot_2column.assert_called_once()  # noqa: S101

    def test_deployment_has_outputs_false_when_empty(self, dp, tmp_path):
        assert not dp._deployment_has_outputs(tmp_path, "CANON_April_2025", [])  # noqa: S101

    def test_deployment_has_outputs_true_when_png_and_html_present_and_fresh(self, dp, tmp_path):
        png = tmp_path / "CANON_April_2025_2column_cmocean.png"
        png.touch()
        (tmp_path / "CANON_April_2025_2column_cmocean.html").touch()
        with patch.object(dp, "_nc_file_mtime", return_value=png.stat().st_mtime - 100):
            assert dp._deployment_has_outputs(  # noqa: S101
                tmp_path, "CANON_April_2025", [_NC_URL]
            )

    def test_deployment_has_outputs_false_when_html_missing(self, dp, tmp_path):
        """A PNG without its HTML sibling must be treated as incomplete, not up to date."""
        (tmp_path / "CANON_April_2025_2column_cmocean.png").touch()
        assert not dp._deployment_has_outputs(  # noqa: S101
            tmp_path, "CANON_April_2025", [_NC_URL]
        )

    def test_deployment_has_outputs_false_when_nc_file_newer(self, dp, tmp_path):
        """A source nc file modified after the existing outputs must force a reprocess."""
        png = tmp_path / "CANON_April_2025_2column_cmocean.png"
        png.touch()
        (tmp_path / "CANON_April_2025_2column_cmocean.html").touch()
        with patch.object(dp, "_nc_file_mtime", return_value=png.stat().st_mtime + 100):
            assert not dp._deployment_has_outputs(  # noqa: S101
                tmp_path, "CANON_April_2025", [_NC_URL]
            )

    def test_deployment_has_outputs_false_when_mtime_unknown(self, dp, tmp_path):
        """An nc file whose modification time can't be determined must be treated as stale."""
        png = tmp_path / "CANON_April_2025_2column_cmocean.png"
        png.touch()
        (tmp_path / "CANON_April_2025_2column_cmocean.html").touch()
        with patch.object(dp, "_nc_file_mtime", return_value=None):
            assert not dp._deployment_has_outputs(  # noqa: S101
                tmp_path, "CANON_April_2025", [_NC_URL]
            )

    def test_deployment_has_outputs_false_when_expected_kind_missing(self, dp, tmp_path):
        """A per-log biolume PNG with no combined-deployment biolume PNG must be stale,
        even though the cmocean PNG+HTML pair is present and fresh."""
        png = tmp_path / "CANON_April_2025_2column_cmocean.png"
        png.touch()
        (tmp_path / "CANON_April_2025_2column_cmocean.html").touch()
        with (
            patch.object(dp, "_nc_file_mtime", return_value=png.stat().st_mtime - 100),
            patch.object(dp, "_expected_deployment_plot_kinds", return_value={"2column_biolume"}),
        ):
            assert not dp._deployment_has_outputs(  # noqa: S101
                tmp_path, "CANON_April_2025", [_NC_URL]
            )

    def test_deployment_has_outputs_true_when_expected_kind_present(self, dp, tmp_path):
        """A per-log biolume PNG is satisfied when the matching combined-deployment
        biolume PNG (+ HTML sibling) already exists."""
        cmocean_png = tmp_path / "CANON_April_2025_2column_cmocean.png"
        cmocean_png.touch()
        (tmp_path / "CANON_April_2025_2column_cmocean.html").touch()
        biolume_png = tmp_path / "CANON_April_2025_2column_biolume.png"
        biolume_png.touch()
        (tmp_path / "CANON_April_2025_2column_biolume.html").touch()
        oldest_mtime = min(cmocean_png.stat().st_mtime, biolume_png.stat().st_mtime)
        with (
            patch.object(dp, "_nc_file_mtime", return_value=oldest_mtime - 100),
            patch.object(dp, "_expected_deployment_plot_kinds", return_value={"2column_biolume"}),
        ):
            assert dp._deployment_has_outputs(  # noqa: S101
                tmp_path, "CANON_April_2025", [_NC_URL]
            )

    def test_expected_deployment_plot_kinds_excludes_cbit(self, dp):
        """cbit is sbd-only and must never be reported as an expected combined-deployment kind."""
        with patch.object(dp, "_url_exists", return_value=True):
            expected = dp._expected_deployment_plot_kinds([_NC_URL])
        assert "2column_cbit" not in expected  # noqa: S101
        assert expected == set(dp._DEPLOYMENT_PLOT_KINDS)  # noqa: S101


class TestStoqsUrlFromDs:
    """Tests for stoqs_url_from_ds() — real logic, only requests.Session mocked."""

    _BASE_URL = "https://tethysviz.shore.mbari.org/stoqs_lrauv_apr2025"

    def test_base_url_auto_derived_from_ds(self):
        """When base_url is omitted, lrauv_stoqs_base_url() is used."""
        ds = _make_ds("2025-04-14")
        with patch("make_permalink.requests.Session") as mock_session_cls:
            mock_session_cls.return_value.__enter__.return_value.get = _mock_session_get("7")
            url = stoqs_url_from_ds(ds, auv_name="ahi")

        assert "stoqs_lrauv_apr2025" in url  # noqa: S101

    def test_explicit_base_url_used(self):
        ds = _make_ds("2025-04-14")
        with patch("make_permalink.requests.Session") as mock_session_cls:
            mock_session_cls.return_value.__enter__.return_value.get = _mock_session_get("7")
            url = stoqs_url_from_ds(ds, base_url=self._BASE_URL, auv_name="ahi")

        assert url.startswith(self._BASE_URL)  # noqa: S101
        assert "/query/?permalink_id=" in url  # noqa: S101

    def test_platform_id_in_permalink(self):
        """The platform name must appear in the platform_clicks entry of the
        compressed permalink payload (after decompression)."""

        ds = _make_ds("2025-04-14")
        url = stoqs_url_from_ds(ds, base_url=self._BASE_URL, auv_name="ahi")

        compressed = url.split("permalink_id=", 1)[1]
        decompressed = lzstring.LZString().decompressFromEncodedURIComponent(compressed)
        payload = json.loads(decompressed)
        # The platform name appears in one of the 'platform_clicks' selector strings
        assert any("ahi" in str(v) for v in payload.get("platform_clicks", []))  # noqa: S101

    def test_time_window_in_permalink(self):
        """start-ems and end-ems must bracket the dataset's time range."""

        ds = _make_ds("2025-04-14", periods=10)
        with patch("make_permalink.requests.Session") as mock_session_cls:
            mock_session_cls.return_value.__enter__.return_value.get = _mock_session_get("1")
            url = stoqs_url_from_ds(ds, base_url=self._BASE_URL, auv_name="ahi")

        compressed = url.split("permalink_id=", 1)[1]
        payload = json.loads(lzstring.LZString().decompressFromEncodedURIComponent(compressed))
        assert payload["start-ems"] < payload["end-ems"]  # noqa: S101

    def test_depth_range_in_permalink(self):
        """start-depth and end-depth must reflect the dataset's depth range."""

        ds = _make_ds("2025-04-14", min_depth=10.0, max_depth=50.0)
        with patch("make_permalink.requests.Session") as mock_session_cls:
            mock_session_cls.return_value.__enter__.return_value.get = _mock_session_get("1")
            url = stoqs_url_from_ds(ds, base_url=self._BASE_URL, auv_name="ahi")

        compressed = url.split("permalink_id=", 1)[1]
        payload = json.loads(lzstring.LZString().decompressFromEncodedURIComponent(compressed))
        assert payload["start-depth"] < payload["end-depth"]  # noqa: S101
        # padding of 1 m applied each side
        assert payload["start-depth"] == pytest.approx(9.0)  # noqa: S101
        assert payload["end-depth"] == pytest.approx(51.0)  # noqa: S101


# ===========================================================================
# Tests for _notify() and _send_slack_file_upload()
# ===========================================================================


class TestNotify:
    """Unit tests for DeploymentPlotter._notify()."""

    def test_email_sent(self, dp, tmp_path):
        """_notify() with an email target should call smtplib.SMTP.send_message."""
        html_file = tmp_path / "test.html"
        html_file.touch()

        with patch("smtplib.SMTP") as mock_smtp_cls:
            mock_smtp = MagicMock()
            mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_smtp)
            mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)
            dp._notify(
                ["team@mbari.org"],
                "CANON_April_2025",
                [html_file],
            )

        mock_smtp.send_message.assert_called_once()  # noqa: S101
        msg = mock_smtp.send_message.call_args[0][0]
        assert "team@mbari.org" in msg["To"]  # noqa: S101
        assert "CANON_April_2025" in msg["Subject"]  # noqa: S101

    def test_slack_webhook_posts(self, dp, tmp_path):
        """_notify() with a Slack webhook URL should POST blocks to that URL."""
        html_file = tmp_path / "test_2column_cmocean.html"
        html_file.touch()
        webhook_url = "https://hooks.slack.com/services/T0000/B0000/xxxx"

        with patch("lrauv_deployment_plots.requests.post") as mock_post:
            mock_post.return_value.raise_for_status = MagicMock()
            dp._notify([webhook_url], "CANON_April_2025", [html_file])

        mock_post.assert_called_once()  # noqa: S101
        call_kwargs = mock_post.call_args
        assert call_kwargs[0][0] == webhook_url  # noqa: S101
        assert "blocks" in call_kwargs[1]["json"]  # noqa: S101

    def test_noop_when_no_target_and_no_env(self, dp, monkeypatch):
        """_notify() must not call anything when targets is empty and env var unset."""
        monkeypatch.delenv("LRAUV_NOTIFY", raising=False)
        monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)

        with (
            patch("smtplib.SMTP") as mock_smtp_cls,
            patch("lrauv_deployment_plots.requests.post") as mock_post,
        ):
            dp._notify([], "CANON_April_2025", [])

        mock_smtp_cls.assert_not_called()  # noqa: S101
        mock_post.assert_not_called()  # noqa: S101

    def test_env_var_fallback_used(self, dp, tmp_path, monkeypatch):
        """When --notify is given with no value (empty list), LRAUV_NOTIFY is used."""
        monkeypatch.setenv("LRAUV_NOTIFY", "fallback@mbari.org")
        html_file = tmp_path / "test.html"
        html_file.touch()

        with patch("smtplib.SMTP") as mock_smtp_cls:
            mock_smtp = MagicMock()
            mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_smtp)
            mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)
            dp._notify([], "CANON_April_2025", [html_file])

        mock_smtp.send_message.assert_called_once()  # noqa: S101
        msg = mock_smtp.send_message.call_args[0][0]
        assert "fallback@mbari.org" in msg["To"]  # noqa: S101

    def test_none_targets_skips_notification(self, dp, monkeypatch):
        """When targets is None (--notify not given), no notification is sent."""
        monkeypatch.setenv("LRAUV_NOTIFY", "fallback@mbari.org")

        with (
            patch("smtplib.SMTP") as mock_smtp_cls,
            patch("lrauv_deployment_plots.requests.post") as mock_post,
        ):
            dp._notify(None, "CANON_April_2025", [])

        mock_smtp_cls.assert_not_called()  # noqa: S101
        mock_post.assert_not_called()  # noqa: S101


class TestSendSlackFileUpload:
    """Unit tests for DeploymentPlotter._send_slack_file_upload()."""

    def _make_responses(self, upload_url="https://files.slack.com/upload/v1/abc"):
        """Return side_effect list for three requests.post calls in the upload flow."""
        get_url_resp = MagicMock()
        get_url_resp.raise_for_status = MagicMock()
        get_url_resp.json.return_value = {
            "ok": True,
            "upload_url": upload_url,
            "file_id": "F0TEST123",
        }
        upload_resp = MagicMock()
        upload_resp.raise_for_status = MagicMock()
        complete_resp = MagicMock()
        complete_resp.raise_for_status = MagicMock()
        complete_resp.json.return_value = {"ok": True}
        return [get_url_resp, upload_resp, complete_resp]

    def test_three_posts_made_when_png_present(self, dp, tmp_path, monkeypatch):
        """Upload flow must make exactly three POST requests when a PNG exists."""
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test-token")
        png = tmp_path / "depl_2column_cmocean.png"
        png.write_bytes(b"fakepng")
        html = tmp_path / "depl_2column_cmocean.html"
        html.touch()

        _EXPECTED_POST_COUNT = 3
        with patch(
            "lrauv_deployment_plots.requests.post", side_effect=self._make_responses()
        ) as mock_post:
            dp._send_slack_file_upload("C0TEST", "CANON April 2025", [html])

        assert mock_post.call_count == _EXPECTED_POST_COUNT  # noqa: S101

    def test_channel_id_sent_in_complete_call(self, dp, tmp_path, monkeypatch):
        """completeUploadExternal must include the channel_id in its payload."""
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test-token")
        png = tmp_path / "depl_2column_cmocean.png"
        png.write_bytes(b"fakepng")
        html = tmp_path / "depl_2column_cmocean.html"
        html.touch()

        with patch(
            "lrauv_deployment_plots.requests.post", side_effect=self._make_responses()
        ) as mock_post:
            dp._send_slack_file_upload("C0MYCHANNEL", "CANON April 2025", [html])

        complete_call = mock_post.call_args_list[2]
        assert complete_call[1]["json"]["channel_id"] == "C0MYCHANNEL"  # noqa: S101

    def test_missing_token_skips_upload(self, dp, tmp_path, monkeypatch):
        """When SLACK_BOT_TOKEN is unset, no requests should be made."""
        monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
        html = tmp_path / "depl_2column_cmocean.html"
        html.touch()

        with patch("lrauv_deployment_plots.requests.post") as mock_post:
            dp._send_slack_file_upload("C0TEST", "CANON April 2025", [html])

        mock_post.assert_not_called()  # noqa: S101

    def test_no_png_falls_back_to_chat_post_message(self, dp, tmp_path, monkeypatch):
        """When no PNG exists, a single chat.postMessage call must be made instead."""
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test-token")
        html = tmp_path / "depl_2column_cmocean.html"
        html.touch()
        # no PNG created — std_png will be None

        chat_resp = MagicMock()
        chat_resp.raise_for_status = MagicMock()
        chat_resp.json.return_value = {"ok": True}

        with patch("lrauv_deployment_plots.requests.post", return_value=chat_resp) as mock_post:
            dp._send_slack_file_upload("C0TEST", "CANON April 2025", [html])

        assert mock_post.call_count == 1  # noqa: S101
        assert "chat.postMessage" in mock_post.call_args[0][0]  # noqa: S101
