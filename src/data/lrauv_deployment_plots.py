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
import base64
import http
import logging
import os
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
from provenance import get_script_github_url, get_web_url, submit_process_run
from resample import FREQ, LRAUV_OPENDAP_BASE

ENV_LRAUV_NOTIFY = "LRAUV_NOTIFY"
ENV_SMTP_HOST = "SMTP_HOST"
ENV_SMTP_PORT = "SMTP_PORT"


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
        Uses join='outer' so every variable present in any log_file is retained
        (absent values filled with NaN).
        """
        if not nc_files:
            return None

        datasets = []
        for p in nc_files:
            try:
                datasets.append(xr.open_dataset(str(p)))
            except OSError as e:
                self.logger.warning("Skipping %s: %s", p, e)

        if not datasets:
            return None

        self.logger.info("Concatenating %d dataset(s) via xr.concat", len(datasets))
        return xr.concat(datasets, dim="time", join="outer")

    def _deployment_has_outputs(self, deployment_dir: Path, plot_name_stem: str) -> bool:
        """Return True if any per-deployment PNG already exists in *deployment_dir*."""
        return any(deployment_dir.glob(f"{plot_name_stem}_*.png"))

    def plot_deployment(  # noqa: C901, PLR0912, PLR0913, PLR0915
        self,
        dlist: str,
        verbose: int = 0,
        update_ssds_provenance: bool = False,  # noqa: FBT001, FBT002
        force: bool = False,  # noqa: FBT001, FBT002
        notify: str | None = None,
    ) -> None:
        """Main entry point: generate deployment-level plots from a .dlist path.

        Args:
            dlist: Path to .dlist file (relative to BASE_LRAUV_PATH or absolute).
            verbose: Verbosity level (0-2).
            update_ssds_provenance: Submit provenance records to SSDS_Metadata.
            force: Reprocess even when output PNGs already exist.
            notify: Email address or Slack webhook URL to notify after completion.
                    Falls back to the ``LRAUV_NOTIFY`` environment variable.
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
            plot_name_stem = raw_name.replace(" ", "_").replace("/", "_")
            self.logger.info("Deployment name: %s", raw_name)
        else:
            plot_name_stem = dlist_rel.stem
            self.logger.warning(
                "Could not parse deployment name from dlist; using %s",
                plot_name_stem,
            )

        if not force and self._deployment_has_outputs(deployment_dir, plot_name_stem):
            self.logger.info(
                "Outputs already exist for %s, skipping (use --force to reprocess)", dlist
            )
            return

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
                notify=notify,
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
        notify: str | None = None,
    ) -> None:
        """Fetch STOQS permalink and write per-PNG HTML pages."""
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
        _auv_name = dlist.split("/")[0]
        for png_path in png_paths:
            if Path(png_path).exists():
                per_png_html = Path(png_path).with_suffix(".html")
                self._write_per_png_html(
                    per_png_html,
                    html_title,
                    Path(png_path).name,
                    get_web_url(png_path),
                    stoqs_url,
                    nc_files,
                    auv_name=_auv_name,
                    png_file_path=Path(png_path),
                    other_png_paths=[p for p in png_paths if p != png_path],
                )
                self.logger.info("Per-PNG HTML written to %s", per_png_html)
        archiver = Archiver(add_handlers=True, clobber=True)
        archiver.logger.setLevel(self._log_levels[min(verbose, 2)])
        archiver.copy_lrauv_deployment(deployment_dir, plot_name_stem)
        html_paths = [
            Path(p).with_suffix(".html") for p in png_paths if Path(p).with_suffix(".html").exists()
        ]
        self._notify(notify or "", raw_name or plot_name_stem, html_paths, stoqs_url)
        if update_ssds_provenance:
            self._submit_provenance(
                deployment_dir=deployment_dir,
                dlist=dlist,
                plot_name_stem=plot_name_stem,
                raw_name=raw_name,
                png_paths=png_paths,
                nc_files=nc_files,
            )

    def _send_notify_email(
        self,
        recipient: str,
        deployment_name: str,
        html_paths: list[Path],
    ) -> None:
        """Send a plain-HTML email with the standard inline PNG and a single web link."""
        import smtplib  # noqa: PLC0415
        from email.mime.image import MIMEImage  # noqa: PLC0415
        from email.mime.multipart import MIMEMultipart  # noqa: PLC0415
        from email.mime.text import MIMEText  # noqa: PLC0415

        std_html = next(
            (p for p in html_paths if "2column_cmocean" in str(p)),
            html_paths[0] if html_paths else None,
        )
        std_png = std_html.with_suffix(".png") if std_html else None
        if std_png and not std_png.exists():
            std_png = None
        web_url = get_web_url(str(std_html)) if std_html else ""

        sent_on = datetime.now(tz=UTC).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        plain = (
            f"New LRAUV deployment plots: {deployment_name}\n\n"
            f"  View this and related information on the web:\n  {web_url}\n\n"
            f"Sent on: {sent_on}"
        )
        img_tag = (
            '<img src="cid:std_plot" alt="Standard plot" style="max-width:100%"><br>'
            if std_png
            else ""
        )
        html_body = (
            f"{img_tag}"
            f'<p><a href="{web_url}">View this and related information on the web</a></p>'
            f"<p><small>Sent on: {sent_on}</small></p>"
        )

        outer = MIMEMultipart("related")
        outer["Subject"] = f"New LRAUV deployment plots: {deployment_name}"
        outer["From"] = "auv-python@mbari.org"
        outer["To"] = recipient
        alt = MIMEMultipart("alternative")
        outer.attach(alt)
        alt.attach(MIMEText(plain, "plain"))
        alt.attach(MIMEText(html_body, "html"))
        if std_png:
            img = MIMEImage(std_png.read_bytes())
            img.add_header("Content-ID", "<std_plot>")
            img.add_header("Content-Disposition", "inline", filename=std_png.name)
            outer.attach(img)
        try:
            smtp_host = os.environ.get(ENV_SMTP_HOST, "localhost")
            smtp_port = int(os.environ.get(ENV_SMTP_PORT, "587"))
            with smtplib.SMTP(smtp_host, smtp_port) as s:
                s.starttls()
                s.send_message(outer)
            self.logger.info("Email notification sent to %s", recipient)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Email notification failed: %s", exc)

    def _notify(
        self,
        target: str,
        deployment_name: str,
        html_paths: list[Path],
        stoqs_url: str | None,
    ) -> None:
        """Send an email or Slack notification with links to the new deployment HTML pages.

        *target* is auto-detected:
        - starts with ``https://`` → treated as a Slack incoming-webhook URL
        - anything else → treated as an email address (sent via localhost SMTP)

        The ``LRAUV_NOTIFY`` environment variable can supply the target so it
        stays out of shell history and cron job command lines.
        """
        resolved = target or os.environ.get(ENV_LRAUV_NOTIFY, "")
        if not resolved:
            return

        if resolved.startswith("https://"):
            # Slack incoming webhook — include all plot links and STOQS URL
            import requests  # noqa: PLC0415

            plot_links = [(get_web_url(str(p)), self._plot_label(str(p))) for p in html_paths]
            lines = [f"New LRAUV deployment plots available: {deployment_name}", ""]
            for url, label in plot_links:
                lines.append(f"  {label}: {url}")
            if stoqs_url:
                after_scheme = stoqs_url.split("//", 1)[-1] if "//" in stoqs_url else stoqs_url
                stoqs_db_label = after_scheme.split("/")[1] if "/" in after_scheme else after_scheme
                lines.append("")
                lines.append(f"  STOQS ({stoqs_db_label}): {stoqs_url}")
            try:
                resp = requests.post(resolved, json={"text": "\n".join(lines)}, timeout=10)  # noqa: S113
                resp.raise_for_status()
                self.logger.info("Slack notification sent to webhook")
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("Slack notification failed: %s", exc)
        else:
            self._send_notify_email(resolved, deployment_name, html_paths)

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

        additional_resources: list[dict] = []

        for png_path in png_paths:
            if not Path(png_path).exists():
                self.logger.debug("PNG not found, skipping provenance: %s", png_path)
                continue
            png_resources = additional_resources + [
                {
                    "name": Path(png_path).name,
                    "uristring": get_web_url(png_path),
                    "description": f"Deployment quick look plot: {Path(png_path).name}",
                    "resourcetype_name": "Quick Look Plot",
                }
            ]
            per_png_html = Path(png_path).with_suffix(".html")
            if per_png_html.exists():
                png_resources.append(
                    {
                        "name": per_png_html.name,
                        "uristring": get_web_url(str(per_png_html)),
                        "description": f"Per-PNG HTML page for {per_png_html.name}",
                        "resourcetype_name": "html",
                    }
                )
            try:
                submit_process_run(
                    input_uris=input_uris,
                    producer_name=producer_name,
                    producer_description=producer_description,
                    pr_start=now,
                    pr_end=now,
                    script_name="src/data/lrauv_deployment_plots.py",
                    cmd_line_args=cmd_line,
                    additional_resources=png_resources,
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

    def _per_log_png_links(self, nc_urls: list[str]) -> list[tuple[str, str]]:
        """Return (url, label) pairs for each existing per-log PNG."""
        parts: list[tuple[str, str]] = []
        for nc_url in nc_urls:
            for png_url in self._png_urls_for_nc(nc_url):
                if self._url_exists(png_url):
                    pname = png_url.rsplit("/", 1)[1]
                    parts.append((png_url, pname))
        return parts

    def _write_per_png_html(  # noqa: C901, PLR0913
        self,
        html_path: Path,
        title: str,
        png_name: str,
        png_url: str,
        stoqs_url: str | None,
        nc_files: list[str],
        auv_name: str = "",
        png_file_path: Path | None = None,
        other_png_paths: list[str] | None = None,
    ) -> None:
        """Write a plain HTML page for one deployment PNG.

        Embeds the full-size PNG as a base64 data URI when *png_file_path* is
        supplied (so the image works in email clients that block external URLs).
        Falls back to the fully-qualified *png_url* otherwise.
        """
        # Group nc_files by log directory (second-to-last path component)
        grouped: dict[str, list[str]] = {}
        for url in nc_files:
            log_dir = url.rsplit("/", 2)[1]
            grouped.setdefault(log_dir, []).append(url)

        # Collect per-row data first so we can suppress empty columns
        nt = 'target="_blank" rel="noopener"'  # new-tab attributes for all links
        rows: list[dict] = []
        for log_dir in sorted(grouped):
            nc_urls = grouped[log_dir]
            log_stoqs_url = self._per_log_stoqs_url(nc_urls, auv_name, stoqs_url)
            png_links = self._per_log_png_links(nc_urls)
            rows.append(
                {
                    "dir": log_dir,
                    "plots": "<br>".join(f'<a href="{u}" {nt}>{lbl}</a>' for u, lbl in png_links),
                    "dap": "".join(
                        f'<a href="{nc_url}.html" {nt}>OPeNDAP</a>' for nc_url in nc_urls
                    ),
                    "stoqs": f'<a href="{log_stoqs_url}" {nt}>STOQS</a>' if log_stoqs_url else "",
                }
            )

        # Only show columns that have at least one non-empty cell
        show = {col: any(r[col] for r in rows) for col in ("plots", "dap", "stoqs")}
        col_labels = {"plots": "Quick Look Plots", "dap": "OPeNDAP", "stoqs": "STOQS"}

        header_cells = "<th>Log directory</th>" + "".join(
            f"<th>{col_labels[c]}</th>" for c in col_labels if show[c]
        )
        log_rows = ""
        for r in rows:
            data_cells = "".join(f"<td>{r[c]}</td>" for c in col_labels if show[c])
            log_rows += f"    <tr><td>{r['dir']}</td>{data_cells}</tr>\n"

        log_table = (
            "  <table>\n"
            f"    <thead><tr>{header_cells}</tr></thead>\n"
            "    <tbody>\n"
            f"{log_rows}"
            "    </tbody>\n"
            "  </table>\n"
        )

        other_plots_line = ""
        if other_png_paths:
            sibling_links = [
                f'<a href="{Path(p).with_suffix(".html").name}" {nt}>{Path(p).name}</a>'
                for p in other_png_paths
                if Path(p).exists()
            ]
            if sibling_links:
                other_plots_line = (
                    "<p>Other plots for this deployment: " + " | ".join(sibling_links) + "</p>\n"
                )

        stoqs_line = ""
        if stoqs_url:
            after_scheme = stoqs_url.split("//", 1)[-1] if "//" in stoqs_url else stoqs_url
            db_label = after_scheme.split("/")[1] if "/" in after_scheme else after_scheme
            stoqs_line = f'<p>View these data in <a href="{stoqs_url}" {nt}>{db_label}</a></p>\n'

        if png_file_path is not None and png_file_path.exists():
            b64 = base64.b64encode(png_file_path.read_bytes()).decode("ascii")
            img_src = f"data:image/png;base64,{b64}"
        else:
            img_src = png_url

        html_title_single = title.replace("\n", " \u2014 ")
        script_github_url = get_script_github_url("src/data/lrauv_deployment_plots.py")
        created_ts = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        footer = (
            "<hr>\n"
            "<p><small>Created by "
            f'<a href="{script_github_url}" {nt}>lrauv_deployment_plots.py</a>'
            f" on {created_ts}</small></p>\n"
        )
        html = (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n'
            "<head>\n"
            '  <meta charset="utf-8">\n'
            f"  <title>{html_title_single}</title>\n"
            "</head>\n"
            "<body>\n"
            f"  <h1>{html_title_single}</h1>\n"
            f'  <img src="{img_src}" alt="{png_name}">\n'
            f"  {other_plots_line}"
            f"  {stoqs_line}"
            "  <h2>Log files</h2>\n"
            f"{log_table}"
            f"{footer}"
            "</body>\n"
            "</html>\n"
        )
        html_path.write_text(html, encoding="utf-8")

    _PLOT_KINDS = ("2column_cmocean", "2column_biolume", "2column_planktivore")
    _PLOT_KIND_LABELS = {
        "2column_cmocean": "Standard",
        "2column_biolume": "Bioluminescence",
        "2column_planktivore": "Planktivore",
    }

    def _png_urls_for_nc(self, nc_url: str) -> list[str]:
        """Return web-accessible PNG URLs for all plot kinds for one OPeNDAP nc URL."""
        rel = nc_url.replace(LRAUV_OPENDAP_BASE.rstrip("/") + "/", "")
        base = BASE_LRAUV_WEB.rstrip("/") + "/" + rel[: -len(".nc")]
        return [f"{base}_{kind}.png" for kind in self._PLOT_KINDS]

    def _plot_label(self, path_str: str) -> str:
        """Return a human-readable label for a plot PNG or HTML path."""
        for kind, label in self._PLOT_KIND_LABELS.items():
            if kind in path_str:
                return label
        return Path(path_str).stem

    def _url_exists(self, url: str) -> bool:
        """Return True if the URL responds with HTTP 200 to a HEAD request."""
        try:
            req = urllib.request.Request(url, method="HEAD")  # noqa: S310
            with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310
                return resp.status == http.client.OK
        except (urllib.error.URLError, OSError):
            return False

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
        base = Path(LRAUV_VOL)
        if not base.is_dir():
            self.logger.error("%s does not exist.", LRAUV_VOL)
            self.logger.info("Is %s mounted?", LRAUV_VOL)
            sys.exit(1)
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
            nargs="?",
            const=1,
            default=0,
            choices=[0, 1, 2],
            help="Verbosity level (0=warn, 1=info, 2=debug)",
        )
        parser.add_argument(
            "--update_ssds_provenance",
            action="store_true",
            help="Submit/update provenance records in the SSDS_Metadata database",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help=(
                "Reprocess deployments even when output PNGs already exist."
                " By default, deployments with existing outputs are skipped."
            ),
        )
        parser.add_argument(
            "--notify",
            default="",
            metavar="EMAIL_OR_WEBHOOK",
            help=(
                "Send a notification when new plots are written. Provide an email"
                " address or a Slack incoming-webhook URL. Falls back to the"
                f" {ENV_LRAUV_NOTIFY} environment variable if not specified."
            ),
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
            force=args.force,
            notify=args.notify,
        )
