#!/usr/bin/env python
"""
Create plots for an entire LRAUV deployment by combining all processed log_file
datasets into a single xarray Dataset and delegating to CreateProducts.

The .dlist file identifies the deployment (its first line holds the deployment
name) and its path encodes both the vehicle and the YYYY/YYYYMMDD_YYYYMMDD
deployment directory layout.

Usage:
    python lrauv_deployment_plots.py \\
        --dlist tethys/missionlogs/2012/20120908_20120920.dlist -v 1
"""

__author__ = "Mike McCann"
__copyright__ = "Copyright 2026, Monterey Bay Aquarium Research Institute"

import argparse  # noqa: I001
import http
import logging
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path

import xarray as xr

from archive import LRAUV_VOL, Archiver
from create_products import CreateProducts
from logs2netcdfs import AUV_NetCDF
from make_permalink import stoqs_url_from_ds
from nc42netcdfs import BASE_LRAUV_PATH, BASE_LRAUV_WEB
from provenance import get_dods_url, submit_process_run
from resample import FREQ, LRAUV_OPENDAP_BASE


class DeploymentPlotter:
    logger = logging.getLogger(__name__)
    _handler = logging.StreamHandler()
    _handler.setFormatter(AUV_NetCDF._formatter)
    logger.addHandler(_handler)
    _log_levels = (logging.WARN, logging.INFO, logging.DEBUG)

    def _read_dlist_content(self, dlist: str) -> str | None:
        """Return the text content of the .dlist file from the first available source.

        Search order (relative paths only):
        1. ``/Volumes/LRAUV/{dlist}``  — network file share
        2. ``BASE_LRAUV_PATH/{dlist}`` — local working copy
        3. ``BASE_LRAUV_WEB/{dlist}``  — DODS/OPeNDAP HTTP server

        Absolute paths are used directly without fallback.
        """
        dlist_path = Path(dlist)
        if dlist_path.is_absolute():
            candidates: list[Path | str] = [dlist_path]
        else:
            candidates = [
                Path("/Volumes/LRAUV", dlist),
                Path(BASE_LRAUV_PATH, dlist),
                BASE_LRAUV_WEB.rstrip("/") + "/" + dlist,
            ]

        for candidate in candidates:
            if isinstance(candidate, Path):
                if candidate.exists():
                    self.logger.info("Reading dlist from %s", candidate)
                    try:
                        return candidate.read_text()
                    except OSError as e:
                        self.logger.warning("Could not read %s: %s", candidate, e)
            else:
                # HTTP URL
                self.logger.info("Trying dlist URL: %s", candidate)
                try:
                    with urllib.request.urlopen(candidate, timeout=10) as resp:  # noqa: S310
                        return resp.read().decode()
                except (urllib.error.URLError, OSError) as e:
                    self.logger.debug("URL fetch failed for %s: %s", candidate, e)

        self.logger.error("dlist not found in any location for: %s", dlist)
        return None

    def _parse_deployment_name(self, dlist_content: str) -> str | None:
        """Return the deployment name from the first line of .dlist text content.

        Expected format: ``# Deployment name: CANON September 2012``
        Returns the name with spaces preserved (caller converts to filename stem).
        """
        try:
            first_line = dlist_content.splitlines()[0].strip()
            if first_line.lower().startswith("# deployment name:"):
                return first_line.split(":", 1)[1].strip()
        except (IndexError, AttributeError):
            pass
        return None

    def _nc_files_for_dir(self, deployment_dir: Path, rel_dir: str) -> list[str]:
        """Return OPeNDAP URLs for *_{FREQ}.nc files in one log subdirectory.

        Fetches the HTTP directory listing from BASE_LRAUV_WEB and returns
        OPeNDAP URL strings (which xr.open_dataset can open directly).
        """
        rel_deployment = deployment_dir.relative_to(BASE_LRAUV_PATH)
        rel_path = str(rel_deployment).replace("\\", "/") + f"/{rel_dir}/"

        dir_url = BASE_LRAUV_WEB.rstrip("/") + "/" + rel_path
        self.logger.info("Fetching HTTP listing: %s", dir_url)
        try:
            with urllib.request.urlopen(dir_url, timeout=10) as resp:  # noqa: S310
                html = resp.read().decode()
        except (urllib.error.URLError, OSError) as e:
            self.logger.warning("HTTP listing failed for %s: %s", dir_url, e)
            return []

        nc_names = sorted(m.group(1) for m in re.finditer(rf'href="([^"]+_{FREQ}\.nc)"', html))
        if not nc_names:
            self.logger.warning("No *_%s.nc links in HTTP listing %s, skipping", FREQ, dir_url)
            return []

        opendap_base = LRAUV_OPENDAP_BASE.rstrip("/") + "/" + rel_path
        return [opendap_base + name for name in nc_names]

    def _collect_nc_files(self, deployment_dir: Path, dlist_content: str) -> list[Path | str]:
        """Return *_{FREQ}.nc files (local Paths or OPeNDAP URL strings) for
        each log directory listed in the .dlist.

        Non-comment, non-empty lines in the .dlist are timestamp subdirectory
        names (e.g. ``20230213T183535``).  Lines starting with ``#`` are
        skipped — including commented-out directories for short/excluded runs.
        Missing or empty subdirectories generate a warning and are skipped.
        """
        log_dirs = [
            line.strip()
            for line in dlist_content.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        if not log_dirs:
            self.logger.warning("No log directories found in dlist content")
            return []

        nc_files: list[Path | str] = []
        for dir_name in log_dirs:
            nc_files.extend(self._nc_files_for_dir(deployment_dir, dir_name))

        return nc_files

    def _concat_datasets(self, nc_files: list[Path | str]) -> xr.Dataset | None:
        """Concatenate per-log datasets into a single deployment-wide Dataset.

        Accepts both local Paths and OPeNDAP URL strings — xr.open_dataset
        handles both transparently.
        Tries xr.open_mfdataset first; falls back to manual concat on failure.
        Uses join='outer' so every variable present in any log_file is retained
        (absent values filled with NaN).
        """
        if not nc_files:
            return None

        paths = [str(p) for p in nc_files]
        try:
            self.logger.info("Concatenating %d files via open_mfdataset", len(paths))
            return xr.open_mfdataset(
                paths,
                combine="by_coords",
                join="outer",
                parallel=True,
                chunks="auto",
            )
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("open_mfdataset failed (%s), falling back to xr.concat", exc)

        datasets = []
        for p in nc_files:
            try:
                datasets.append(xr.open_dataset(p))
            except OSError as e:
                self.logger.warning("Skipping %s: %s", p, e)

        if not datasets:
            return None

        return xr.concat(datasets, dim="time", join="outer")

    def plot_deployment(  # noqa: C901, PLR0912, PLR0915
        self,
        dlist: str,
        verbose: int = 0,
        update_ssds_provenance: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Main entry point: generate deployment-level plots from a .dlist path.

        Args:
            dlist: Path to .dlist file (relative to BASE_LRAUV_PATH or absolute).
            verbose: Verbosity level (0-2).
            update_ssds_provenance: Submit provenance records to SSDS_Metadata.
        """
        self.logger.setLevel(self._log_levels[min(verbose, 2)])

        # Relative dlist path (normalised, never absolute unless passed as absolute)
        dlist_rel = Path(dlist)

        # Output goes to BASE_LRAUV_PATH; validate existence against LRAUV_VOL too.
        if dlist_rel.is_absolute():
            deployment_dir = dlist_rel.parent / dlist_rel.stem
        else:
            deployment_dir = Path(BASE_LRAUV_PATH, dlist_rel.parent, dlist_rel.stem)

        vol_dir = Path(LRAUV_VOL, dlist_rel.parent, dlist_rel.stem)
        if not deployment_dir.is_dir() and not vol_dir.is_dir():
            self.logger.error(
                "Expected deployment directory not found in %s or %s",
                deployment_dir,
                vol_dir,
            )
            return
        deployment_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("Deployment directory: %s", deployment_dir)

        # Fetch .dlist content from network share, local copy, or DODS web server
        dlist_content = self._read_dlist_content(dlist)
        if dlist_content is None:
            self.logger.warning(
                "Could not read .dlist; plot_name_stem will fall back to %s",
                dlist_rel.stem,
            )

        # Deployment name (spaces → underscores for filenames)
        raw_name = self._parse_deployment_name(dlist_content) if dlist_content else None
        if raw_name:
            plot_name_stem = raw_name.replace(" ", "_")
            self.logger.info("Deployment name: %s", raw_name)
        else:
            plot_name_stem = dlist_rel.stem
            self.logger.warning(
                "Could not parse deployment name from dlist; using %s",
                plot_name_stem,
            )

        # Gather and concatenate per-log resampled files
        if dlist_content is None:
            self.logger.error("Cannot collect nc files without dlist content")
            return
        nc_files = self._collect_nc_files(deployment_dir, dlist_content)
        if not nc_files:
            return

        self.logger.info("Found %d *_%s.nc file(s)", len(nc_files), FREQ)
        for f in nc_files:
            self.logger.info("  %s", f)

        combined_ds = self._concat_datasets(nc_files)
        if combined_ds is None:
            self.logger.error("No data to plot after concatenation")
            return

        # Use first nc_file's corresponding .nc4 log_file for _is_lrauv()
        first_nc = nc_files[0]
        if isinstance(first_nc, Path):
            nc4_candidate = first_nc.parent / first_nc.name.replace(f"_{FREQ}.nc", ".nc4")
            if nc4_candidate.exists():
                first_log_file = str(nc4_candidate.relative_to(BASE_LRAUV_PATH))
            else:
                first_log_file = str(first_nc.relative_to(BASE_LRAUV_PATH))
        else:
            # OPeNDAP URL — strip base and swap suffix
            rel = first_nc.replace(LRAUV_OPENDAP_BASE.rstrip("/") + "/", "")
            first_log_file = re.sub(rf"_{FREQ}\.nc$", ".nc4", rel)

        self.logger.info("Using log_file for CreateProducts: %s", first_log_file)

        cp = CreateProducts(
            log_file=first_log_file,
            ds=combined_ds,
            output_dir=deployment_dir,
            plot_name_stem=plot_name_stem,
            verbose=verbose,
            nc_files=[str(f) for f in nc_files],
        )

        p_start = time.time()
        png_paths = [
            p
            for p in (
                cp.plot_2column(),
                cp.plot_biolume_2column(),
                cp.plot_planktivore_2column(),
            )
            if p is not None
        ]
        self.logger.info("Deployment plots completed in %.1f s", time.time() - p_start)

        if png_paths:
            self._build_and_write_html(
                deployment_dir,
                dlist,
                plot_name_stem,
                raw_name,
                combined_ds,
                png_paths,
                nc_files,
                verbose=verbose,
                update_ssds_provenance=update_ssds_provenance,
            )

    def _build_and_write_html(  # noqa: PLR0913
        self,
        deployment_dir: Path,
        dlist: str,
        plot_name_stem: str,
        raw_name: str | None,
        combined_ds: xr.Dataset,
        png_paths: list[str],
        nc_files: list[str],
        verbose: int = 0,
        update_ssds_provenance: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Fetch STOQS permalink and write the deployment HTML index file."""
        html_path = deployment_dir / f"{plot_name_stem}.html"
        dlist_no_ext = str(Path(dlist).with_suffix(""))
        html_title = (
            "Combined, Aligned, and Resampled LRAUV instrument data from "
            f"Deployment\n{raw_name or plot_name_stem}\n{dlist_no_ext}"
        )
        stoqs_url = None
        try:
            stoqs_url = stoqs_url_from_ds(combined_ds, auv_name=dlist.split("/")[0])
            self.logger.info("STOQS permalink: %s", stoqs_url)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Could not generate STOQS permalink: %s", exc)
        self._write_html(html_path, html_title, png_paths, nc_files, stoqs_url)
        self.logger.info("HTML index written to %s", html_path)
        for png_path in png_paths:
            if Path(png_path).exists():
                per_png_html = Path(png_path).with_suffix(".html")
                self._write_per_png_html(
                    per_png_html,
                    html_title,
                    Path(png_path).name,
                    stoqs_url,
                    nc_files,
                    auv_name=dlist.split("/")[0],
                )
                self.logger.info("Per-PNG HTML written to %s", per_png_html)
        archiver = Archiver(add_handlers=True, clobber=True)
        archiver.logger.setLevel(self._log_levels[min(verbose, 2)])
        archiver.copy_lrauv_deployment(deployment_dir, plot_name_stem)
        if update_ssds_provenance:
            self._submit_provenance(
                deployment_dir=deployment_dir,
                dlist=dlist,
                plot_name_stem=plot_name_stem,
                raw_name=raw_name,
                png_paths=png_paths,
                nc_files=nc_files,
            )

    def _submit_provenance(  # noqa: PLR0913
        self,
        deployment_dir: Path,
        dlist: str,
        plot_name_stem: str,
        raw_name: str | None,
        png_paths: list[str],
        nc_files: list[str],
    ) -> None:
        """Submit one ProcessRun record per deployment PNG to SSDS_Metadata.

        Each PNG is recorded as the output; the input nc_files are its sources.
        The producer name and description are derived from the HTML title text.
        """
        from datetime import UTC, datetime  # noqa: PLC0415

        dlist_no_ext = str(Path(dlist).with_suffix(""))
        producer_name = (
            "auv-python - lrauv_deployment_plots.py producing "
            f"deployment plots for {raw_name or plot_name_stem}"
        )
        producer_description = (
            "Combined, Aligned, and Resampled LRAUV instrument data from "
            f"Deployment: {raw_name or plot_name_stem} — {dlist_no_ext}"
        )
        cmd_line = " ".join(sys.argv)
        input_uris = list(nc_files)  # already OPeNDAP URLs
        now = datetime.now(tz=UTC).isoformat()

        html_path = deployment_dir / f"{plot_name_stem}.html"
        additional_resources = []
        if html_path.exists():
            additional_resources.append(
                {
                    "name": "deployment_html_index",
                    "uristring": get_dods_url(str(html_path)),
                    "description": f"HTML index page for {plot_name_stem}",
                    "resourcetype_name": "html",
                }
            )

        for png_path in png_paths:
            if not Path(png_path).exists():
                self.logger.debug("PNG not found, skipping provenance: %s", png_path)
                continue
            png_resources = additional_resources + [
                {
                    "name": Path(png_path).name,
                    "uristring": get_dods_url(png_path),
                    "description": f"Deployment quick look plot: {Path(png_path).name}",
                    "resourcetype_name": "Quick Look Plot",
                }
            ]
            try:
                submit_process_run(
                    nc_file_path=png_path,
                    input_uris=input_uris,
                    producer_name=producer_name,
                    producer_description=producer_description,
                    pr_start=now,
                    pr_end=now,
                    script_name="src/data/lrauv_deployment_plots.py",
                    cmd_line_args=cmd_line,
                    additional_resources=png_resources,
                    log=self.logger,
                )
            except Exception:  # noqa: BLE001
                self.logger.warning("Provenance submission failed for %s", png_path, exc_info=True)

    def _stoqs_url_for_nc_url(self, nc_url: str, auv_name: str) -> str | None:
        """Return a STOQS permalink scoped to the time range of one nc file, or None."""
        try:
            ds = xr.open_dataset(nc_url)
            return stoqs_url_from_ds(ds, auv_name=auv_name)
        except Exception as exc:  # noqa: BLE001
            self.logger.debug("Could not generate per-log STOQS URL for %s: %s", nc_url, exc)
            return None

    def _per_log_html_url(self, nc_urls: list[str]) -> str:
        """Return the first reachable per-log HTML URL for a list of nc URLs, or empty string."""
        for nc_url in nc_urls:
            candidate = nc_url.replace(
                LRAUV_OPENDAP_BASE.rstrip("/"), BASE_LRAUV_WEB.rstrip("/")
            ).replace(f"_{FREQ}.nc", f"_{FREQ}.html")
            if self._url_exists(candidate):
                return candidate
        return ""

    def _per_log_stoqs_url(
        self, nc_urls: list[str], auv_name: str, fallback: str | None
    ) -> str | None:
        """Return a STOQS URL scoped to the first nc file's time range, or *fallback*."""
        if auv_name:
            for nc_url in nc_urls:
                url = self._stoqs_url_for_nc_url(nc_url, auv_name)
                if url:
                    return url
        return fallback

    def _per_log_png_links(self, nc_urls: list[str]) -> str:
        """Return HTML anchor tags for each existing per-log PNG, pipe-separated."""
        parts: list[str] = []
        for nc_url in nc_urls:
            for png_url in self._png_urls_for_nc(nc_url):
                if self._url_exists(png_url):
                    pname = png_url.rsplit("/", 1)[1]
                    parts.append(f'<a href="{png_url}">{pname}</a>')
        return " | ".join(parts)

    def _write_per_png_html(  # noqa: C901, PLR0913
        self,
        html_path: Path,
        title: str,
        png_name: str,
        stoqs_url: str | None,
        nc_files: list[str],
        auv_name: str = "",
    ) -> None:
        """Write a plain HTML page for one deployment PNG.

        Embeds the full-size PNG, links to the STOQS database, then lists
        per-log image / data / STOQS links — no CSS.
        """
        # Group nc_files by log directory (second-to-last path component)
        grouped: dict[str, list[str]] = {}
        for url in nc_files:
            log_dir = url.rsplit("/", 2)[1]
            grouped.setdefault(log_dir, []).append(url)

        log_items = ""
        for log_dir in sorted(grouped):
            nc_urls = grouped[log_dir]
            per_log_html_url = self._per_log_html_url(nc_urls)
            log_stoqs_url = self._per_log_stoqs_url(nc_urls, auv_name, stoqs_url)

            links = ""
            if per_log_html_url:
                links += f'      <a href="{per_log_html_url}">image</a>'
            png_links = self._per_log_png_links(nc_urls)
            if png_links:
                if links:
                    links += " | "
                links += png_links
            for nc_url in nc_urls:
                dap_form_url = nc_url + ".html"
                if links:
                    links += " | "
                links += f'<a href="{dap_form_url}">OPeNDAP Data Access Form</a>'
            if log_stoqs_url:
                if links:
                    links += " | "
                links += f'<a href="{log_stoqs_url}">STOQS</a>'

            log_items += f"  <li>{log_dir} — {links}</li>\n"

        stoqs_line = ""
        if stoqs_url:
            after_scheme = stoqs_url.split("//", 1)[-1] if "//" in stoqs_url else stoqs_url
            db_label = after_scheme.split("/")[1] if "/" in after_scheme else after_scheme
            stoqs_line = f'<p><a href="{stoqs_url}">View these data in {db_label}</a></p>\n'

        html_title_single = title.replace("\n", " \u2014 ")
        html = (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n'
            "<head>\n"
            '  <meta charset="utf-8">\n'
            f"  <title>{html_title_single}</title>\n"
            "</head>\n"
            "<body>\n"
            f"  <h1>{html_title_single}</h1>\n"
            f'  <img src="{png_name}" alt="{png_name}">\n'
            f"  {stoqs_line}"
            "  <h2>Log files</h2>\n"
            "  <ul>\n"
            f"{log_items}"
            "  </ul>\n"
            "</body>\n"
            "</html>\n"
        )
        html_path.write_text(html, encoding="utf-8")

    _PLOT_KINDS = ("2column_cmocean", "2column_biolume", "2column_planktivore")

    def _png_urls_for_nc(self, nc_url: str) -> list[str]:
        """Return web-accessible PNG URLs for all plot kinds for one OPeNDAP nc URL."""
        rel = nc_url.replace(LRAUV_OPENDAP_BASE.rstrip("/") + "/", "")
        base = BASE_LRAUV_WEB.rstrip("/") + "/" + rel[: -len(".nc")]
        return [f"{base}_{kind}.png" for kind in self._PLOT_KINDS]

    def _url_exists(self, url: str) -> bool:
        """Return True if the URL responds with HTTP 200 to a HEAD request."""
        try:
            req = urllib.request.Request(url, method="HEAD")  # noqa: S310
            with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310
                return resp.status == http.client.OK
        except (urllib.error.URLError, OSError):
            return False

    _CSS = """
    :root {
      --navy: #0d1b2a;
      --teal: #00b4d8;
      --teal-light: #90e0ef;
      --bg: #f4f6f9;
      --card-bg: #ffffff;
      --text: #1a1a2e;
      --muted: #6c757d;
      --radius: 8px;
      --shadow: 0 2px 8px rgba(0,0,0,0.10);
    }
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.6;
    }
    header {
      background: var(--bg);
      color: var(--text);
      padding: 2rem 2rem 1.5rem;
      border-bottom: 1px solid #d0d7de;
      text-align: center;
    }
    header h1 { font-size: 1.4rem; font-weight: 600; line-height: 1.4; color: var(--navy); }
    main { max-width: 1400px; margin: 0 auto; padding: 2rem; }
    section { margin-bottom: 2.5rem; }
    h2 {
      font-size: 1.1rem;
      font-weight: 600;
      color: var(--navy);
      border-left: 4px solid var(--teal);
      padding-left: 0.75rem;
      margin-bottom: 1rem;
    }
    .plots-grid { display: flex; flex-wrap: wrap; gap: 1rem; }
    .plot-card {
      background: var(--card-bg);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      overflow: hidden;
      display: flex;
      flex-direction: column;
      max-width: 480px;
    }
    .plot-card.small { max-width: 320px; }
    .plot-card img { width: 100%; height: auto; display: block; }
    .plot-card img:hover { opacity: 0.88; }
    .plot-card figcaption {
      padding: 0.4rem 0.75rem;
      font-size: 0.78rem;
      color: var(--muted);
      word-break: break-all;
    }
    details {
      background: var(--card-bg);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      margin-bottom: 0.75rem;
      overflow: hidden;
    }
    details > summary {
      cursor: pointer;
      padding: 0.75rem 1rem;
      font-weight: 600;
      color: var(--navy);
      background: #eaf6fb;
      list-style: none;
      user-select: none;
    }
    details > summary::before { content: "\\25B6  "; font-size: 0.75rem; }
    details[open] > summary::before { content: "\\25BC  "; }
    details > summary::-webkit-details-marker { display: none; }
    details > p, details > .plots-grid { padding: 0.75rem 1rem; }
    .opendap-link { color: var(--teal); text-decoration: none; font-size: 0.9rem; }
    .opendap-link:hover { text-decoration: underline; }
"""

    def _write_html(  # noqa: C901, PLR0912, PLR0913, PLR0915
        self,
        html_path: Path,
        title: str,
        png_paths: list[str],
        nc_files: list[str],
        stoqs_url: str | None = None,
    ) -> None:
        """Write a styled HTML page with deployment and per-log plot thumbnails."""
        # Deployment plot thumbnail cards
        depl_cards = ""
        for p in png_paths:
            if Path(p).exists():
                name = Path(p).name
                img_tag = f'        <img src="{name}" alt="{name}" loading="lazy">'
                depl_cards += (
                    f'      <figure class="plot-card">\n'
                    f'        <a href="{name}">{img_tag}</a>\n'
                    f"        <figcaption>{name}</figcaption>\n"
                    f"      </figure>\n"
                )
            else:
                self.logger.debug("Deployment PNG not found, skipping: %s", p)

        stoqs_card = ""
        if stoqs_url:
            _logo = (
                "https://github.com/stoqs/stoqs/raw/master"
                "/stoqs/static/images/STOQS_logo_gray1_689.png"
            )
            stoqs_card = (
                f'      <figure class="plot-card">\n'
                f'        <a href="{stoqs_url}">'
                f'<img src="{_logo}" alt="STOQS" loading="lazy" style="padding:1rem"></a>\n'
                f"        <figcaption>View in {stoqs_url.split('query')[0]}</figcaption>\n"
                f"      </figure>\n"
            )

        # Group nc_files by log directory (second-to-last URL component)
        grouped: dict[str, list[str]] = {}
        for url in nc_files:
            log_dir = url.rsplit("/", 2)[1]
            grouped.setdefault(log_dir, []).append(url)

        log_sections = ""
        for log_dir in sorted(grouped):
            inner = ""
            # Link to per-log HTML page if it exists
            per_log_html_url = ""
            for nc_url in grouped[log_dir]:
                candidate = nc_url.replace(
                    LRAUV_OPENDAP_BASE.rstrip("/"), BASE_LRAUV_WEB.rstrip("/")
                ).replace(f"_{FREQ}.nc", f"_{FREQ}.html")
                if self._url_exists(candidate):
                    per_log_html_url = candidate
                    break
            if per_log_html_url:
                inner += (
                    f'      <p><a class="opendap-link" href="{per_log_html_url}">'
                    f"&#128196;&nbsp;{log_dir} — per-log plots</a></p>\n"
                )
            for nc_url in grouped[log_dir]:
                nc_name = nc_url.rsplit("/", 1)[1]
                dap_form_url = nc_url + ".html"
                inner += (
                    f'      <p><a class="opendap-link" href="{dap_form_url}">'
                    f"&#128190;&nbsp;{nc_name} (OPeNDAP)</a></p>\n"
                )
                thumb_row = ""
                for png_url in self._png_urls_for_nc(nc_url):
                    if self._url_exists(png_url):
                        pname = png_url.rsplit("/", 1)[1]
                        thumb_row += (
                            f'        <figure class="plot-card small">\n'
                            f'          <a href="{png_url}">'
                            f'<img src="{png_url}" alt="{pname}" loading="lazy"></a>\n'
                            f"          <figcaption>{pname}</figcaption>\n"
                            f"        </figure>\n"
                        )
                    else:
                        self.logger.debug("Per-log PNG not found, skipping: %s", png_url)
                if thumb_row:
                    inner += f'      <div class="plots-grid">\n{thumb_row}      </div>\n'
            if inner:
                log_sections += (
                    f"    <details>\n      <summary>{log_dir}</summary>\n{inner}    </details>\n"
                )

        html_title_tag = title.replace("\n", " \u2014 ")
        html_h1 = title.replace("\n", "<br>")
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html_title_tag}</title>
  <style>{self._CSS}  </style>
</head>
<body>
  <header>
    <h1>{html_h1}</h1>
  </header>
  <main>
    <section>
      <h2>Deployment Plots</h2>
      <div class="plots-grid">
{depl_cards}{stoqs_card}      </div>
    </section>
    <section>
      <h2>Per-log Plots</h2>
{log_sections}    </section>
  </main>
</body>
</html>
"""
        html_path.write_text(html, encoding="utf-8")

    def _dlist_list(  # noqa: C901, PLR0912
        self,
        start_dt: datetime,
        end_dt: datetime,
        auv_name: str | None = None,
    ) -> list[str]:
        """Return dlist paths (relative to the scan base) whose date-range
        directory overlaps with *start_dt*–*end_dt*.

        Scans ``{base}/{auv}/missionlogs/{year}/`` for directories named
        ``YYYYMMDD_YYYYMMDD`` and checks whether they overlap the requested
        window.  The sibling ``.dlist`` file (``YYYYMMDD_YYYYMMDD.dlist``)
        is returned for each matching directory.
        """
        _vol = Path(LRAUV_VOL)
        base = _vol if _vol.is_dir() else Path(BASE_LRAUV_PATH)
        dlists: list[str] = []
        auv_dirs = (
            sorted(base.glob("*/missionlogs/"))
            if not auv_name
            else [base / auv_name / "missionlogs"]
        )
        for missionlogs_dir in auv_dirs:
            if not missionlogs_dir.is_dir():
                continue
            auv = missionlogs_dir.parent.name
            for year_dir in sorted(missionlogs_dir.glob("*/")):
                try:
                    year = int(year_dir.name)
                except ValueError:
                    continue
                if year < start_dt.year or year > end_dt.year:
                    continue
                for date_range_dir in sorted(year_dir.glob("*/")):
                    # Directory name is YYYYMMDD_YYYYMMDD
                    parts = date_range_dir.name.split("_")
                    if len(parts) != 2:  # noqa: PLR2004
                        continue
                    try:
                        dir_start = datetime.strptime(parts[0], "%Y%m%d").replace(tzinfo=UTC)
                        dir_end = datetime.strptime(parts[1], "%Y%m%d").replace(tzinfo=UTC)
                    except ValueError:
                        continue
                    # Overlap check: ranges overlap if dir_start <= end_dt and dir_end >= start_dt
                    if dir_start > end_dt or dir_end < start_dt:
                        continue
                    # The .dlist file is a sibling of the date_range directory, same name + .dlist
                    dlist_file = year_dir / f"{date_range_dir.name}.dlist"
                    if dlist_file.exists():
                        rel = f"{auv}/missionlogs/{year_dir.name}/{dlist_file.name}"
                        dlists.append(rel)
                        self.logger.info("Found dlist: %s", rel)
                    else:
                        self.logger.debug("No .dlist sibling found for %s", date_range_dir)
        return dlists

    def process_command_line(self) -> None:
        parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        mode = parser.add_mutually_exclusive_group(required=True)
        mode.add_argument(
            "--dlist",
            help=(
                "Path to the .dlist file, relative to BASE_LRAUV_PATH"
                " (e.g. tethys/missionlogs/2012/20120908_20120920.dlist)"
                " or an absolute path."
            ),
        )
        mode.add_argument(
            "--last_n_days",
            type=int,
            metavar="LAST_N_DAYS",
            help="Process deployments whose date range ends in the last N days.",
        )
        mode.add_argument(
            "--start",
            metavar="YYYYMMDD",
            help="Process deployments starting at or after this date.",
        )
        parser.add_argument(
            "--end",
            metavar="YYYYMMDD",
            default=None,
            help=(
                "End date for time-range mode (YYYYMMDD). Defaults to today when used with --start."
            ),
        )
        parser.add_argument(
            "--auv_name",
            default=None,
            help=(
                "Restrict dlist search to this AUV name (e.g. brizo, ahi)."
                " If not specified, all AUVs will be searched."
            ),
        )
        parser.add_argument(
            "-v",
            "--verbose",
            type=int,
            default=0,
            choices=[0, 1, 2],
            help="Verbosity level (0=warn, 1=info, 2=debug)",
        )
        parser.add_argument(
            "--update_ssds_provenance",
            action="store_true",
            help="Submit/update provenance records in the SSDS_Metadata database",
        )
        self.args = parser.parse_args()
        if self.args.start and not self.args.end:
            self.args.end = datetime.now(tz=UTC).strftime("%Y%m%d")


if __name__ == "__main__":
    dp = DeploymentPlotter()
    dp.process_command_line()
    args = dp.args
    dp.logger.setLevel(dp._log_levels[min(args.verbose, 2)])

    if args.dlist:
        dlists = [args.dlist]
    elif args.last_n_days:
        end_dt = datetime.now(tz=UTC)
        start_dt = end_dt - timedelta(days=args.last_n_days)
        dlists = dp._dlist_list(start_dt, end_dt, args.auv_name)
    else:  # --start [--end]
        start_dt = datetime.strptime(args.start, "%Y%m%d").replace(tzinfo=UTC)
        end_dt = datetime.strptime(args.end, "%Y%m%d").replace(tzinfo=UTC)
        dlists = dp._dlist_list(start_dt, end_dt, args.auv_name)

    if not dlists:
        dp.logger.warning("No dlist files found for the specified time range.")
        sys.exit(0)

    for dlist in dlists:
        dp.plot_deployment(
            dlist,
            verbose=args.verbose,
            update_ssds_provenance=args.update_ssds_provenance,
        )
