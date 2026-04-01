"""Record data processing provenance in the SSDS_Metadata database.

Replaces the legacy Perl ``submitDStoNetCDFProcessRun`` with authenticated
REST API calls to the mooring-ssds service.  Every public helper is
designed to be called from ``process.py`` after a successful processing
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
GITHUB_REPO = "https://github.com/mbari-org/auv-python"
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


def _find_first(
    session: requests.Session,
    endpoint: str,
    params: dict,
    api_base: str = SSDS_API_BASE,
) -> dict | None:
    """GET /{endpoint}/ filtered by *params* — return first result or ``None``."""
    resp = session.get(f"{api_base}/{endpoint}/", params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    # DRF pagination wraps list results in {"count": N, "results": [...]}
    results = data.get("results", data) if isinstance(data, dict) else data
    return results[0] if isinstance(results, list) and results else None


def find_person_by_email(
    session: requests.Session,
    email: str,
    api_base: str = SSDS_API_BASE,
) -> dict | None:
    """``GET /persons?email=…`` — return the first match or ``None``."""
    return _find_first(session, "persons", {"email": email}, api_base)


def find_software(
    session: requests.Session,
    name: str,
    softwareversion: str,
    uristring: str,
    api_base: str = SSDS_API_BASE,
) -> dict | None:
    "``GET /software?name=…&softwareversion=…&uristring=…`` — return the first match or ``None``."
    return _find_first(
        session,
        "software",
        {"name": name, "softwareversion": softwareversion, "uristring": uristring},
        api_base,
    )


def find_datacontainer(
    session: requests.Session,
    uristring: str,
    api_base: str = SSDS_API_BASE,
) -> dict | None:
    """``GET /datacontainers?uristring=…`` — return the first match or ``None``."""
    return _find_first(session, "datacontainers", {"uristring": uristring}, api_base)


def find_resource(
    session: requests.Session,
    uristring: str,
    api_base: str = SSDS_API_BASE,
) -> dict | None:
    """``GET /resources?uristring=…`` — return the first match or ``None``."""
    return _find_first(session, "resources", {"uristring": uristring}, api_base)


def _create_entity(
    session: requests.Session,
    endpoint: str,
    payload: dict,
    api_base: str = SSDS_API_BASE,
) -> dict:
    """POST *payload* to *endpoint* and return the created entity."""
    url = f"{api_base}/{endpoint}/"
    resp = session.post(url, json=payload, timeout=REQUEST_TIMEOUT)
    if not resp.ok:
        err_txt = (
            f"{resp.status_code} {resp.reason} for url: {url}"
            f"\n  payload: {payload}\n  response: {resp.text}"
        )
        raise requests.HTTPError(err_txt, response=resp)
    return resp.json()


def _find_process_run_by_output(
    session: requests.Session,
    dods_url: str,
    api_base: str = SSDS_API_BASE,
) -> dict | None:
    """``GET /dataproducers?output_uri=…`` — find an existing ProcessRun."""
    return _find_first(session, "dataproducers", {"output_uri": dods_url}, api_base)


def _ensure_link(
    session: requests.Session,
    endpoint: str,
    params: dict,
    api_base: str = SSDS_API_BASE,
) -> None:
    """POST the association; silently ignore duplicate-key responses.

    Prefer POST-first over a GET-check-then-POST pattern: junction-table
    endpoints may not support filtering by FK query params, which would
    cause _find_first to return a false positive and silently skip the POST.
    """
    url = f"{api_base}/{endpoint}/"
    resp = session.post(url, json=params, timeout=REQUEST_TIMEOUT)
    if resp.ok:
        return
    # 409 Conflict, 400, or 500 whose body signals a unique/PK constraint violation.
    # Django can surface IntegrityError as a 500 when the view lacks explicit handling.
    _dup_markers = ("unique", "already exists", "duplicate key", "primary key constraint")
    body = resp.text.lower()
    if resp.status_code == requests.codes.conflict or any(m in body for m in _dup_markers):
        return
    err_txt = (
        f"{resp.status_code} {resp.reason} for url: {url}"
        f"\n  payload: {params}\n  response: {resp.text}"
    )
    raise requests.HTTPError(err_txt, response=resp)


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------
def submit_process_run(  # noqa: PLR0913 C901
    nc_file_path: str,
    input_uris: list[str],
    *,
    poc_email: str = "mccann@mbari.org",
    pr_start: str | None = None,
    pr_end: str | None = None,
    software_name: str = "auv-python",
    software_description: str = "AUV data processing pipeline",
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
    """Submit a ProcessRun provenance record to the SSDS_Metadata database.

    This is the direct Python equivalent of the legacy Perl function
    ``submitDStoNetCDFProcessRun``.

    Returns the ProcessRun entity dict on success, or ``None`` on failure.
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

    # a. Look up Person -------------------------------------------------------
    person = find_person_by_email(session, poc_email, api_base)
    if person is None:
        log.warning("Person with email %s not found – skipping provenance", poc_email)
        return None

    # b. Build Resource list --------------------------------------------------
    resources: list[dict] = []
    # Script URL
    script_url = get_git_url(script_name, software_version)
    resources.append(
        {
            "name": "processing_script",
            "uristring": script_url,
            "description": f"Processing script: {script_name}",
        }
    )
    # Command line
    if cmd_line_args:
        resources.append(
            {
                "name": "command_line",
                "uristring": f"urn:cmdline:{gethostname()}",
                "description": cmd_line_args,
            }
        )
    # Processing log
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

    # c. Create Software entity -----------------------------------------------
    software_payload = {
        "name": software_name,
        "version": "1",
        "description": software_description,
        "softwareversion": software_version,
        "uristring": f"{GITHUB_REPO}/tree/{software_version}",
    }
    software = find_software(
        session, software_name, software_version, software_payload["uristring"], api_base
    ) or _create_entity(session, "software", software_payload, api_base)

    # d. Get or create ProcessRun (DataProducer) ------------------------------
    output_url = get_dods_url(nc_file_path)
    existing = _find_process_run_by_output(session, output_url, api_base)
    if existing:
        pr_id = existing["id"]
        # Update with new timing / description
        update_payload = {
            "description": f"Processing run for {Path(nc_file_path).name}",
            "startdate": pr_start,
            "enddate": pr_end,
            "personid_fk": person["id"],
            "softwareid_fk": software["id"],
        }
        url = f"{api_base}/dataproducers/{pr_id}/"
        session.put(url, json=update_payload, timeout=REQUEST_TIMEOUT).raise_for_status()
        process_run = {**existing, **update_payload}
        log.info("Updated existing ProcessRun id=%s", pr_id)
    else:
        process_run = _create_entity(
            session,
            "dataproducers",
            {
                "name": f"Processing {Path(nc_file_path).name}",
                "description": f"Processing run for {Path(nc_file_path).name}",
                "version": "1",
                "startdate": pr_start,
                "enddate": pr_end,
                "dataproducertype": "ProcessRun",
                "personid_fk": person["id"],
                "softwareid_fk": software["id"],
            },
            api_base,
        )
        pr_id = process_run["id"]
        log.info("Created new ProcessRun id=%s", pr_id)

    # e. Get or create output DataContainer ------------------------------------
    output_dc = find_datacontainer(session, output_url, api_base)
    if output_dc is None:
        output_dc = _create_entity(
            session,
            "datacontainers",
            {
                "name": Path(nc_file_path).name,
                "version": "1",
                "uristring": output_url,
                "description": f"Processed NetCDF output: {Path(nc_file_path).name}",
                "dataproducerid_fk": pr_id,
            },
            api_base,
        )
    elif output_dc.get("dataproducerid_fk") != pr_id:
        dc_id = output_dc["id"]
        session.patch(
            f"{api_base}/datacontainers/{dc_id}/",
            json={"dataproducerid_fk": pr_id},
            timeout=REQUEST_TIMEOUT,
        ).raise_for_status()
        log.info("Linked existing DataContainer id=%s to ProcessRun id=%s", dc_id, pr_id)

    # f. Link relationships ---------------------------------------------------

    # Inputs — get or create DataContainer, then link via dataproducer-inputs
    for uri in input_uris:
        input_dc = find_datacontainer(session, uri, api_base) or _create_entity(
            session,
            "datacontainers",
            {
                "name": Path(uri).name,
                "version": "1",
                "uristring": uri,
                "description": f"Input: {Path(uri).name}",
            },
            api_base,
        )
        _ensure_link(
            session,
            "dataproducer-inputs",
            {"datacontainerid_fk": input_dc["id"], "dataproducerid_fk": pr_id},
            api_base,
        )

    # Resources — create each Resource then link via dataproducer-assoc-resource
    for res_payload in resources:
        res = find_resource(session, res_payload["uristring"], api_base) or _create_entity(
            session, "resources", {**res_payload, "version": "1"}, api_base
        )
        _ensure_link(
            session,
            "dataproducer-assoc-resource",
            {"dataproducerid_fk": pr_id, "resourceid_fk": res["id"]},
            api_base,
        )

    # g. Log result -----------------------------------------------------------
    log.info("Provenance recorded: %s", f"{api_base}/dataproducers/{pr_id}")
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
