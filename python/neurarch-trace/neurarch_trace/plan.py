"""`--plan` / `--share`: send the traced graph to `POST /api/v1/plan` and print the card.

Standard library only, and no torch import, so the wire contract can be tested
without a model. The server renders the card; this module sends the graph, prints
`text` verbatim, and turns every failure into one sentence the user can act on.

Nothing here runs unless the CLI was given `--plan` or `--share`.
"""
import json
import os
import socket
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Sequence
from urllib.parse import urlsplit

DEFAULT_API = "https://www.neurarch.com"
PLAN_PATH = "/api/v1/plan"
TIMEOUT_SECONDS = 30


class PlanError(Exception):
    """The plan could not be fetched. `graph_path` is where the graph still landed."""

    def __init__(self, message: str, graph_path: Optional[str] = None):
        super().__init__(message)
        self.graph_path = graph_path


def api_base(env: Optional[Dict[str, str]] = None) -> str:
    """`NEURARCH_API` if set, else the production host; never with a trailing slash."""
    env = os.environ if env is None else env
    return (env.get("NEURARCH_API") or DEFAULT_API).rstrip("/")


def api_host(base: str) -> str:
    """`www.neurarch.com` for the default base; whatever host the env pointed at otherwise."""
    return urlsplit(base).netloc or base


def build_source(target: str, inputs: Sequence[str], version: str) -> Dict[str, str]:
    return {
        "kind": "trace",
        "target": target,
        "input": " ".join(inputs),
        "tool": "neurarch-trace/%s" % version,
    }


def _server_message(body: bytes, fallback: str) -> str:
    try:
        data = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return fallback
    if isinstance(data, dict) and isinstance(data.get("error"), str) and data["error"]:
        return data["error"]
    return fallback


def request_plan(
    graph: Dict[str, Any],
    source: Dict[str, str],
    share: bool = False,
    base: Optional[Dict[str, Any]] = None,
    api: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: float = TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """POST the graph and return the decoded JSON response.

    Raises PlanError for a network failure, a non-2xx status (429 names
    NEURARCH_API_KEY), or a body that is not the documented shape.
    """
    api = (api or api_base()).rstrip("/")
    host = api_host(api)
    body: Dict[str, Any] = {"model": graph, "share": bool(share), "source": source}
    if base is not None:
        body["base"] = base
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": source.get("tool", "neurarch-trace"),
    }
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    req = urllib.request.Request(
        api + PLAN_PATH, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        raw = e.read()
        if e.code == 429:
            msg = _server_message(raw, "too many requests")
            raise PlanError(
                "%s rate-limited this request: %s. Set NEURARCH_API_KEY to raise the limit." % (host, msg))
        raise PlanError("%s returned HTTP %d: %s" % (host, e.code, _server_message(raw, e.reason or "error")))
    except (urllib.error.URLError, socket.timeout, ConnectionError, OSError) as e:
        reason = getattr(e, "reason", None) or e
        raise PlanError("could not reach %s (%s)" % (api, reason))

    try:
        data = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        raise PlanError("%s returned a response that is not JSON" % host)
    if not isinstance(data, dict) or not isinstance(data.get("text"), str):
        raise PlanError("%s returned a response without a plan text" % host)
    return data


def has_blocker(response: Dict[str, Any]) -> bool:
    """True when the server says the design will not run (`--fail-on-block` exits 2)."""
    plan = response.get("plan")
    run = plan.get("run") if isinstance(plan, dict) else None
    if not isinstance(run, dict):
        return False
    if run.get("legal") is False:
        return True
    blockers = run.get("blockers")
    return isinstance(blockers, list) and len(blockers) > 0


def print_plan(response: Dict[str, Any], share: bool, out=None, err=None) -> None:
    """Print `text` verbatim, then `Share: <url>` on its own last line when one came back.

    stdout carries only those two things so the output can be pasted anywhere;
    a `share_error` goes to stderr after the plan.
    """
    out = out or sys.stdout
    err = err or sys.stderr
    text = response["text"]
    out.write(text if text.endswith("\n") else text + "\n")
    if not share:
        return
    url = response.get("url")
    if isinstance(url, str) and url:
        out.write("Share: %s\n" % url)
        return
    reason = response.get("share_error")
    if not isinstance(reason, str) or not reason:
        reason = "the server returned no URL"
    err.write("neurarch-trace: the plan printed but no share link was made: %s\n" % reason)


__all__ = [
    "PlanError", "api_base", "api_host", "build_source", "request_plan", "has_blocker", "print_plan",
    "DEFAULT_API", "PLAN_PATH", "TIMEOUT_SECONDS",
]
