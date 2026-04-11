"""Tests for DeploymentPlotter._write_html() and plot_deployment() HTML path."""

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


@pytest.fixture(scope="session", autouse=False)
def dp():
    plotter = DeploymentPlotter()
    plotter.logger.setLevel("DEBUG")
    return plotter


class TestWriteHtml:
    def test_basic_structure(self, dp, tmp_path):
        html_path = tmp_path / "deployment.html"
        title = "Test Deployment\nApril 2025"

        with patch.object(dp, "_url_exists", return_value=False):
            dp._write_html(html_path, title, [], [_NC_URL])

        html = html_path.read_text()
        assert "<!DOCTYPE html>" in html  # noqa: S101
        assert "Test Deployment" in html  # noqa: S101
        assert "April 2025" in html  # noqa: S101

    def test_deployment_png_linked(self, dp, tmp_path):
        html_path = tmp_path / "deployment.html"
        # Touch a fake PNG so Path(p).exists() returns True
        png = tmp_path / "deployment_2column_cmocean.png"
        png.touch()

        with patch.object(dp, "_url_exists", return_value=False):
            dp._write_html(html_path, "Title", [str(png)], [_NC_URL])

        assert "deployment_2column_cmocean.png" in html_path.read_text()  # noqa: S101

    def test_deployment_png_missing_not_linked(self, dp, tmp_path):
        html_path = tmp_path / "deployment.html"
        nonexistent = str(tmp_path / "ghost.png")

        with patch.object(dp, "_url_exists", return_value=False):
            dp._write_html(html_path, "Title", [nonexistent], [_NC_URL])

        assert "ghost.png" not in html_path.read_text()  # noqa: S101

    def test_opendap_link_appears(self, dp, tmp_path):
        html_path = tmp_path / "deployment.html"

        with patch.object(dp, "_url_exists", return_value=False):
            dp._write_html(html_path, "Title", [], [_NC_URL])

        # OPeNDAP data-access-form URL (.nc.html)
        assert _NC_URL + ".html" in html_path.read_text()  # noqa: S101

    def test_per_log_png_linked_when_url_exists(self, dp, tmp_path):
        html_path = tmp_path / "deployment.html"

        with patch.object(dp, "_url_exists", return_value=True):
            dp._write_html(html_path, "Title", [], [_NC_URL])

        html = html_path.read_text()
        assert any(kind in html for kind in dp._PLOT_KINDS)  # noqa: S101

    def test_per_log_png_absent_when_url_missing(self, dp, tmp_path):
        html_path = tmp_path / "deployment.html"

        with patch.object(dp, "_url_exists", return_value=False):
            dp._write_html(html_path, "Title", [], [_NC_URL])

        html = html_path.read_text()
        assert not any(kind in html for kind in dp._PLOT_KINDS)  # noqa: S101

    def test_stoqs_section_included(self, dp, tmp_path):
        html_path = tmp_path / "deployment.html"
        stoqs_url = (
            "https://tethysviz.shore.mbari.org/stoqs_lrauv_apr2025/query/?permalink_id=abc123"
        )

        with patch.object(dp, "_url_exists", return_value=False):
            dp._write_html(html_path, "Title", [], [_NC_URL], stoqs_url=stoqs_url)

        html = html_path.read_text()
        assert stoqs_url in html  # noqa: S101
        assert "STOQS" in html  # noqa: S101

    def test_stoqs_section_absent_when_no_url(self, dp, tmp_path):
        html_path = tmp_path / "deployment.html"

        with patch.object(dp, "_url_exists", return_value=False):
            dp._write_html(html_path, "Title", [], [_NC_URL], stoqs_url=None)

        assert "STOQS" not in html_path.read_text()  # noqa: S101

    def test_log_directory_grouping(self, dp, tmp_path):
        html_path = tmp_path / "deployment.html"
        nc_url_a = (
            f"{_OPENDAP_BASE}/ahi/missionlogs/2025/20250414_20250418/20250414T120000/ahi_1S.nc"
        )
        nc_url_b = (
            f"{_OPENDAP_BASE}/ahi/missionlogs/2025/20250414_20250418/20250415T080000/ahi_1S.nc"
        )

        with patch.object(dp, "_url_exists", return_value=False):
            dp._write_html(html_path, "Title", [], [nc_url_a, nc_url_b])

        html = html_path.read_text()
        # Both log-directory h3 headings should appear
        assert "20250414T120000" in html  # noqa: S101
        assert "20250415T080000" in html  # noqa: S101


class TestBuildAndWriteHtml:
    """Tests for _build_and_write_html() directly — real stoqs_url_from_ds().

    Only network mocked.
    """

    _DLIST = "ahi/missionlogs/2025/20250414_20250418.dlist"

    def _call(self, dp, tmp_path, *, platform_id="7", session_side_effect=None):
        png = tmp_path / "depl_2column_cmocean.png"
        png.touch()
        with (
            patch("make_permalink.requests.Session") as mock_session_cls,
            patch.object(dp, "_url_exists", return_value=False),
        ):
            if session_side_effect:
                mock_session_cls.return_value.__enter__.return_value.get.side_effect = (
                    session_side_effect
                )
            else:
                mock_session_cls.return_value.__enter__.return_value.get = _mock_session_get(
                    platform_id
                )
            dp._build_and_write_html(
                tmp_path,
                self._DLIST,
                "CANON_April_2025",
                "CANON April 2025",
                _make_ds("2025-04-14"),
                [str(png)],
                [_NC_URL],
            )

    def test_stoqs_url_in_html(self, dp, tmp_path):
        self._call(dp, tmp_path)
        html = (tmp_path / "CANON_April_2025.html").read_text()
        assert "stoqs_lrauv_apr2025" in html  # noqa: S101
        assert "/query/?permalink_id=" in html  # noqa: S101

    def test_html_file_written_with_correct_name(self, dp, tmp_path):
        self._call(dp, tmp_path)
        assert (tmp_path / "CANON_April_2025.html").exists()  # noqa: S101

    def test_title_contains_deployment_name(self, dp, tmp_path):
        self._call(dp, tmp_path)
        html = (tmp_path / "CANON_April_2025.html").read_text()
        assert "CANON April 2025" in html  # noqa: S101

    def test_stoqs_failure_still_writes_html(self, dp, tmp_path):
        """An error inside stoqs_url_from_ds must not propagate."""
        png = tmp_path / "depl_2column_cmocean.png"
        png.touch()
        with (
            patch("lrauv_deployment_plots.stoqs_url_from_ds", side_effect=OSError("network down")),
            patch.object(dp, "_url_exists", return_value=False),
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
        html = (tmp_path / "CANON_April_2025.html").read_text()
        assert "CANON April 2025" in html  # noqa: S101
        assert "STOQS" not in html  # noqa: S101


# ---------------------------------------------------------------------------
# Tests for plot_deployment() — exercises the stoqs_url_from_ds call path
# ---------------------------------------------------------------------------

_DLIST_CONTENT = """\
# Deployment name: CANON April 2025
20250414T120000
"""
_DLIST = "ahi/missionlogs/2025/20250414_20250418.dlist"


class TestPlotDeploymentStoqsUrl:
    """Drive plot_deployment() to the stoqs_url_from_ds call without hitting
    the network or running any real plot generation."""

    def _make_deployment_dir(self, tmp_path: Path) -> Path:
        """Create the deployment directory that plot_deployment() requires."""
        depl_dir = tmp_path / "ahi" / "missionlogs" / "2025" / "20250414_20250418"
        depl_dir.mkdir(parents=True)
        return depl_dir

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
        ):
            mock_session_cls.return_value.__enter__.return_value.get = _mock_session_get("7")
            dp.plot_deployment(_DLIST, verbose=1)

        html = (depl_dir / "CANON_April_2025.html").read_text()
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
# Helper shared by TestStoqsUrlFromDs
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
