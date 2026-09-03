"""`--plan` / `--share` against a stub of `POST /api/v1/plan`.

The stub records every request and answers with whatever the test queued, so
these pin the wire contract (body shape, Authorization header, error paths) and
what the CLI puts on stdout (the plan text verbatim, then `Share: <url>` only
when asked). The server endpoint ships separately; nothing here reaches the
network.

The `neurarch_trace.plan` tests need no torch. The CLI tests trace a two-layer
`nn.Sequential` and are skipped when torch is not installed.
"""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neurarch_trace import __version__  # noqa: E402
from neurarch_trace.plan import PlanError, build_source, has_blocker, print_plan, request_plan  # noqa: E402

CANNED_TEXT = (
    "+-----------------------------------------+\n"
    "| ResNet18                    11.7M params |\n"
    "| Runs: yes            Fits: T4 (16 GB)   |\n"
    "+-----------------------------------------+\n"
)


def canned(share_url=None, share_error=None, blockers=(), legal=True):
    body = {
        "plan": {
            "version": 1,
            "summary": "ResNet18, 11.7M params",
            "model": {"name": "ResNet18"},
            "run": {"legal": legal, "blockers": list(blockers), "warnings": []},
            "cost": {"train_usd": 0.4},
            "diff": None,
        },
        "text": CANNED_TEXT,
        "markdown": "# ResNet18\n",
    }
    if share_url:
        body["url"] = share_url
    if share_error:
        body["share_error"] = share_error
    return body


class Stub:
    """One `http.server` in a thread; `requests` is what it saw, `responses` what it will say."""

    def __init__(self):
        self.requests = []
        self.responses = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length)
                stub.requests.append({
                    "path": self.path,
                    "headers": {k.lower(): v for k, v in self.headers.items()},
                    "body": json.loads(raw.decode("utf-8")),
                })
                status, payload = stub.responses.pop(0) if stub.responses else (200, canned())
                data = json.dumps(payload).encode("utf-8") if not isinstance(payload, bytes) else payload
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = "http://127.0.0.1:%d" % self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()

    @property
    def last(self):
        return self.requests[-1]


@pytest.fixture
def stub():
    s = Stub()
    yield s
    s.close()


GRAPH = {
    "id": "traced-m", "name": "m", "description": "",
    "components": [
        {"id": "input", "type": "input", "name": "input", "params": {}, "inputs": [], "outputs": ["l1"]},
        {"id": "l1", "type": "linear", "name": "fc", "params": {"inFeatures": 4, "outFeatures": 2},
         "inputs": ["input"], "outputs": ["output"]},
        {"id": "output", "type": "output", "name": "output", "params": {}, "inputs": ["l1"], "outputs": []},
    ],
    "connections": [
        {"id": "e0", "from": "input", "to": "l1", "fromPort": "bottom", "toPort": "top"},
        {"id": "e1", "from": "l1", "to": "output", "fromPort": "bottom", "toPort": "top"},
    ],
}
SOURCE = build_source("models/resnet.py:ResNet18", ["1,3,224,224"], __version__)


# ---- the wire contract, no torch ------------------------------------------------

def test_request_body_matches_the_contract(stub, monkeypatch):
    monkeypatch.delenv("NEURARCH_API_KEY", raising=False)
    resp = request_plan(GRAPH, SOURCE, share=False, api=stub.url)
    assert resp["text"] == CANNED_TEXT
    req = stub.last
    assert req["path"] == "/api/v1/plan"
    assert req["headers"]["content-type"] == "application/json"
    assert "authorization" not in req["headers"]
    body = req["body"]
    assert set(body) == {"model", "share", "source"}
    assert body["model"] == GRAPH
    assert body["share"] is False
    assert body["source"] == {
        "kind": "trace",
        "target": "models/resnet.py:ResNet18",
        "input": "1,3,224,224",
        "tool": "neurarch-trace/" + __version__,
    }


def test_base_and_share_and_api_key_are_sent_when_given(stub):
    base = dict(GRAPH, id="traced-base")
    request_plan(GRAPH, SOURCE, share=True, base=base, api=stub.url + "/", api_key="nk_test_123")
    req = stub.last
    assert req["path"] == "/api/v1/plan"
    assert req["headers"]["authorization"] == "Bearer nk_test_123"
    assert req["body"]["share"] is True
    assert req["body"]["base"] == base


def test_api_key_comes_from_the_env_only_when_the_cli_passes_it(stub, monkeypatch):
    # request_plan itself never reads the env; the CLI does, and passes it through.
    monkeypatch.setenv("NEURARCH_API_KEY", "nk_env")
    request_plan(GRAPH, SOURCE, api=stub.url)
    assert "authorization" not in stub.last["headers"]


def test_429_quotes_the_server_and_names_the_env_var(stub):
    stub.responses.append((429, {"error": "12 plans per hour per IP"}))
    with pytest.raises(PlanError) as e:
        request_plan(GRAPH, SOURCE, api=stub.url)
    msg = str(e.value)
    assert "12 plans per hour per IP" in msg
    assert "NEURARCH_API_KEY" in msg
    assert "127.0.0.1" in msg


def test_other_http_errors_carry_status_and_message(stub):
    stub.responses.append((400, {"error": "model.components must be a list"}))
    with pytest.raises(PlanError, match=r"HTTP 400: model\.components must be a list"):
        request_plan(GRAPH, SOURCE, api=stub.url)
    stub.responses.append((503, b"<html>bad gateway</html>"))
    with pytest.raises(PlanError, match=r"HTTP 503"):
        request_plan(GRAPH, SOURCE, api=stub.url)


def test_unreachable_host_is_one_sentence(stub):
    stub.close()  # the port is now closed; nothing listens there
    with pytest.raises(PlanError, match=r"^could not reach " + stub.url.replace(".", r"\.")):
        request_plan(GRAPH, SOURCE, api=stub.url)


def test_a_200_without_text_is_rejected(stub):
    stub.responses.append((200, {"plan": {}}))
    with pytest.raises(PlanError, match="without a plan text"):
        request_plan(GRAPH, SOURCE, api=stub.url)
    stub.responses.append((200, b"not json"))
    with pytest.raises(PlanError, match="not JSON"):
        request_plan(GRAPH, SOURCE, api=stub.url)


def test_print_plan_is_verbatim_and_share_line_is_last(capsys):
    print_plan(canned(), share=False)
    assert capsys.readouterr().out == CANNED_TEXT

    print_plan(canned(share_url="https://www.neurarch.com/p/abc123"), share=True)
    out, err = capsys.readouterr()
    assert out == CANNED_TEXT + "Share: https://www.neurarch.com/p/abc123\n"
    assert err == ""

    # A url in the response is not printed when the user did not ask to share.
    print_plan(canned(share_url="https://www.neurarch.com/p/abc123"), share=False)
    assert capsys.readouterr().out == CANNED_TEXT

    print_plan(canned(share_error="sharing is not enabled on this deployment"), share=True)
    out, err = capsys.readouterr()
    assert out == CANNED_TEXT and "Share:" not in out
    assert "sharing is not enabled on this deployment" in err

    # Text without a trailing newline still ends the line before Share:.
    print_plan(dict(canned(share_url="u"), text="one line"), share=True)
    assert capsys.readouterr().out == "one line\nShare: u\n"


def test_has_blocker_reads_run_legal_and_blockers():
    assert not has_blocker(canned())
    assert has_blocker(canned(blockers=["conv2d 'c1': in_channels 3 != upstream 16"]))
    assert has_blocker(canned(legal=False))
    assert not has_blocker({"text": "x"})


# ---- the CLI end to end (needs torch) -----------------------------------------

@pytest.fixture
def model_file(tmp_path):
    pytest.importorskip("torch")
    src = tmp_path / "m.py"
    src.write_text(
        "import torch.nn as nn\n"
        "def build():\n"
        "    return nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))\n"
    )
    return src


def cli_main():
    from neurarch_trace.cli import main
    return main


def test_cli_without_plan_sends_nothing_and_is_unchanged(stub, model_file, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("NEURARCH_API", stub.url)
    out = tmp_path / "m.neurarch.json"
    assert cli_main()([str(model_file) + ":build", "--input", "2,4", "-o", str(out)]) == 0
    assert stub.requests == []
    assert capsys.readouterr().out.startswith("wrote %s (3 layers" % out)
    assert json.loads(out.read_text())["components"]


def test_cli_plan_prints_only_the_plan_text(stub, model_file, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("NEURARCH_API", stub.url)
    monkeypatch.delenv("NEURARCH_API_KEY", raising=False)
    out = tmp_path / "m.neurarch.json"
    assert cli_main()([str(model_file) + ":build", "--input", "2,4", "-o", str(out), "--plan"]) == 0
    captured = capsys.readouterr()
    assert captured.out == CANNED_TEXT
    assert "Sending the graph (3 layers) to 127.0.0.1" in captured.err
    assert "add --share for a public link" in captured.err
    assert "wrote %s" % out in captured.err
    assert out.is_file()

    req = stub.last
    assert "authorization" not in req["headers"]
    body = req["body"]
    assert body["share"] is False and "base" not in body
    assert body["source"]["kind"] == "trace"
    assert body["source"]["target"] == str(model_file) + ":build"
    assert body["source"]["input"] == "2,4"
    assert body["source"]["tool"] == "neurarch-trace/" + __version__
    assert body["model"] == json.loads(out.read_text())


def test_cli_share_prints_the_url_last_and_sends_the_key(stub, model_file, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("NEURARCH_API", stub.url)
    monkeypatch.setenv("NEURARCH_API_KEY", "nk_env_key")
    stub.responses.append((200, canned(share_url="https://www.neurarch.com/p/q7x")))
    out = tmp_path / "m.neurarch.json"
    assert cli_main()([str(model_file) + ":build", "--input", "2,4", "-o", str(out), "--share"]) == 0
    captured = capsys.readouterr()
    assert captured.out == CANNED_TEXT + "Share: https://www.neurarch.com/p/q7x\n"
    assert "for the plan and a public link" in captured.err
    assert stub.last["body"]["share"] is True
    assert stub.last["headers"]["authorization"] == "Bearer nk_env_key"


def test_cli_share_error_prints_plan_then_message(stub, model_file, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("NEURARCH_API", stub.url)
    stub.responses.append((200, canned(share_error="sharing is not enabled")))
    out = tmp_path / "m.neurarch.json"
    assert cli_main()([str(model_file) + ":build", "--input", "2,4", "-o", str(out), "--share"]) == 0
    captured = capsys.readouterr()
    assert captured.out == CANNED_TEXT
    assert "sharing is not enabled" in captured.err


def test_cli_base_is_sent(stub, model_file, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("NEURARCH_API", stub.url)
    base_path = tmp_path / "base.neurarch.json"
    base_path.write_text(json.dumps(GRAPH))
    out = tmp_path / "m.neurarch.json"
    argv = [str(model_file) + ":build", "--input", "2,4", "-o", str(out), "--plan", "--base", str(base_path)]
    assert cli_main()(argv) == 0
    assert stub.last["body"]["base"] == GRAPH
    capsys.readouterr()

    assert cli_main()(argv[:-1] + [str(tmp_path / "missing.json")]) == 1
    assert "--base: no such file" in capsys.readouterr().err
    assert stub.requests and len(stub.requests) == 1  # nothing was sent for the bad --base


def test_cli_429_names_the_key_and_the_written_file(stub, model_file, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("NEURARCH_API", stub.url)
    stub.responses.append((429, {"error": "12 plans per hour per IP"}))
    out = tmp_path / "m.neurarch.json"
    assert cli_main()([str(model_file) + ":build", "--input", "2,4", "-o", str(out), "--plan"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "12 plans per hour per IP" in captured.err
    assert "NEURARCH_API_KEY" in captured.err
    assert "the graph was still written to %s" % out in captured.err
    assert out.is_file()


def test_cli_unreachable_host_still_writes_the_graph(stub, model_file, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("NEURARCH_API", stub.url)
    stub.close()
    out = tmp_path / "m.neurarch.json"
    assert cli_main()([str(model_file) + ":build", "--input", "2,4", "-o", str(out), "--plan"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "could not reach %s" % stub.url in captured.err
    assert "the graph was still written to %s" % out in captured.err
    assert out.is_file()


def test_cli_fail_on_block(stub, model_file, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("NEURARCH_API", stub.url)
    out = tmp_path / "m.neurarch.json"
    argv = [str(model_file) + ":build", "--input", "2,4", "-o", str(out), "--plan"]
    stub.responses.append((200, canned(blockers=["linear 'fc': inFeatures 4 != upstream 3"])))
    assert cli_main()(argv) == 0  # default: the plan printed, exit 0 even with a blocker
    assert capsys.readouterr().out == CANNED_TEXT

    stub.responses.append((200, canned(blockers=["linear 'fc': inFeatures 4 != upstream 3"])))
    assert cli_main()(argv + ["--fail-on-block"]) == 2
    assert capsys.readouterr().out == CANNED_TEXT  # the plan still prints before the exit code

    stub.responses.append((200, canned()))
    assert cli_main()(argv + ["--fail-on-block"]) == 0


def test_cli_plan_refuses_stdout_graph(stub, model_file, capsys, monkeypatch):
    monkeypatch.setenv("NEURARCH_API", stub.url)
    assert cli_main()([str(model_file) + ":build", "--input", "2,4", "-o", "-", "--plan"]) == 1
    assert "--plan cannot be combined with -o -" in capsys.readouterr().err
    assert stub.requests == []
