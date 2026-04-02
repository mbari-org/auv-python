"""Record data processing provenance in the SSDS_Metadata database.

Submits a single POST to ``/api/process-runs/``, which atomically resolves
or creates all related entities (Person, Software, DataContainers, Resources)
server-side.  Can be called from ``process.py`` after a successful processing
run, or standalone via the CLI at the bottom of this file.

See: https://github.com/mbari-org/auv-python/issues/144
"""

import argparse
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from socket import gethostname

import git
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ENV_SSDS_API_BASE = "SSDS_API_BASE"
SSDS_API_BASE = os.environ.get(ENV_SSDS_API_BASE, "https://mooring-ssds.shore.mbari.org/api")
GIT_WEB_BASE = "https://github.com/mbari-org/auv-python/blob"
REQUEST_TIMEOUT = 30

# Environment variable names used for authentication stubs.
ENV_SSDS_API_KEY = "SSDS_API_KEY"  # noqa: S105
ENV_SSDS_API_KEY_HEADER = "SSDS_API_KEY_HEADER"  # noqa: S105

# Maps local absolute path prefixes to OPeNDAP URL prefixes.
# Mirrors the Perl ``%web_lookup`` hash in ssds_util.pl.
# Computed from the project layout, matching BASE_LRAUV_PATH / BASE_PATH.
_PROJECT_DATA = Path(__file__).resolve().parent.parent.parent / "data"
PATH_TO_URL_MAP: dict[str, str] = {
    str(_PROJECT_DATA / "auv_data"): "http://dods.mbari.org/opendap/data/auvctd",
    str(_PROJECT_DATA / "lrauv_data"): "http://dods.mbari.org/opendap/data/lrauv",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_git_version() -> str:
    """Return the current git commit hash, or ``'unknown'``."""
    try:
        repo = git.Repo(Path(__file__).resolve().parent, search_parent_directories=True)
    except Exception:  # noqa: BLE001
        logger.debug("Could not determine git version", exc_info=True)
        return "unknown"
    else:
        return repo.head.commit.hexsha


def build_authenticated_session(
    api_key: str | None = None,
    api_key_header: str | None = None,
) -> requests.Session:
    """Return a requests.Session with auth headers from args or env.

    Resolution order:
    1) Explicit function args
    2) Environment variables

    Supported auth stubs:
    - API key via configurable header (default: X-API-Key)
    """
    session = requests.Session()

    resolved_api_key = api_key or os.environ.get(ENV_SSDS_API_KEY)
    resolved_api_key_header = (
        api_key_header or os.environ.get(ENV_SSDS_API_KEY_HEADER) or "X-API-Key"
    )

    if resolved_api_key:
        session.headers[resolved_api_key_header] = resolved_api_key

    return session


def get_dods_url(nc_file_path: str) -> str:
    """Translate a local NetCDF path to its OPeNDAP URL.

    Walks ``PATH_TO_URL_MAP`` looking for a matching prefix in the
    *resolved* path string.  Returns the original path unchanged if no
    match is found.
    """
    resolved = str(Path(nc_file_path).resolve())
    for local_prefix, url_prefix in PATH_TO_URL_MAP.items():
        if resolved.startswith(local_prefix):
            return resolved.replace(local_prefix, url_prefix, 1)
    return resolved


def get_git_url(script_name: str, version: str) -> str:
    """Return a GitHub web URL for *script_name* at *version*."""
    return f"{GIT_WEB_BASE}/{version}/{script_name}"


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------
def submit_process_run(  # noqa: PLR0913
    nc_file_path: str,
    input_uris: list[str],
    *,
    poc_email: str = "mccann@mbari.org",
    pr_start: str | None = None,
    pr_end: str | None = None,
    software_name: str = "auv-python",
    software_version: str | None = None,
    script_name: str = "src/data/process.py",
    cmd_line_args: str = "",
    log_file_url: str | None = None,
    additional_resources: list[dict] | None = None,
    api_key: str | None = None,
    api_key_header: str | None = None,
    api_base: str = SSDS_API_BASE,
    session: requests.Session | None = None,
    log: logging.Logger | None = None,
) -> dict | None:
    """Submit a ProcessRun to the SSDS_Metadata database via POST /api/process-runs/.

    The server resolves or creates all related entities (Person, Software,
    DataContainers, Resources) from the supplied natural keys.

    Returns the created ProcessRun dict on success, or raises on HTTP error.
    """
    log = log or logger
    session = session or build_authenticated_session(
        api_key=api_key,
        api_key_header=api_key_header,
    )
    resolved_api_key_header = (
        api_key_header or os.environ.get(ENV_SSDS_API_KEY_HEADER) or "X-API-Key"
    )
    if resolved_api_key_header not in session.headers:
        log.warning(
            "No SSDS API key header detected on session; set %s "
            "(or pass a pre-authenticated session). "
            "If the server uses cookie/SSO auth, this warning can be ignored.",
            ENV_SSDS_API_KEY,
        )
    software_version = software_version or _get_git_version()

    resources: list[dict] = [
        {
            "name": "processing_script",
            "uristring": get_git_url(script_name, software_version),
            "description": f"Processing script: {script_name}",
        },
    ]
    if cmd_line_args:
        resources.append(
            {
                "name": "command_line",
                "uristring": f"urn:cmdline:{gethostname()}",
                "description": cmd_line_args,
            }
        )
    if log_file_url:
        resources.append(
            {
                "name": "processing_log",
                "uristring": log_file_url,
                "description": "Processing log file",
            }
        )
    if additional_resources:
        resources.extend(additional_resources)

    payload = {
        "output_uri": get_dods_url(nc_file_path),
        "output_name": Path(nc_file_path).name,
        "input_uris": input_uris,
        "software_name": software_name,
        "software_version": software_version,
        "person_email": poc_email,
        "startdate": pr_start,
        "enddate": pr_end,
        "resources": resources,
    }

    url = f"{api_base}/process-runs/"
    resp = session.post(url, json=payload, timeout=REQUEST_TIMEOUT)
    if not resp.ok:
        err_txt = (
            f"{resp.status_code} {resp.reason} for url: {url}"
            f"\n  payload: {payload}\n  response: {resp.text}"
        )
        raise requests.HTTPError(err_txt, response=resp)
    process_run = resp.json()
    log.info("Provenance recorded: %s -> id=%s", url, process_run.get("id", "?"))
    return process_run


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def process_command_line() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("nc_file", help="Path to the output NetCDF file")
    parser.add_argument(
        "--input",
        dest="inputs",
        action="append",
        default=[],
        help="Input URI (may be repeated)",
    )
    parser.add_argument("--email", default="mccann@mbari.org", help="Point-of-contact email")
    parser.add_argument("--script", default="src/data/process.py", help="Processing script path")
    parser.add_argument("--cmd", default=" ".join(sys.argv), help="Command line string")
    parser.add_argument("--log-url", default=None, help="URL to processing log")
    parser.add_argument(
        "--api-base",
        default=SSDS_API_BASE,
        help="SSDS REST API base URL",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = process_command_line()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    now = datetime.now(tz=UTC).isoformat()
    result = submit_process_run(
        nc_file_path=args.nc_file,
        input_uris=args.inputs,
        poc_email=args.email,
        pr_start=now,
        pr_end=now,
        script_name=args.script,
        cmd_line_args=args.cmd,
        log_file_url=args.log_url,
        api_base=args.api_base,
    )
    sys.exit(0 if result else 1)
