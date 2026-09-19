"""The autopatch: trace a forward pass the test is driving, not the tracer.

Two things are worth more than the rest of this file. `test_matches_the_cli`
says the graph you get for free is the graph you would have got by naming the
model and its input by hand, because a second-class graph would make the easy
path the wrong one. `test_leaves_the_run_alone` says what the autopatch is
allowed to do to a script it is a guest in: nothing. One forward pass, the mode
it found the model in, no flags set on anyone's tensors.
"""
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neurarch_trace import autopatch, trace_model  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PTH_FILE = "neurarch_autopatch.pth"


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flat = nn.Flatten()
        self.head = nn.Linear(8, 4)

    def forward(self, x):
        y = self.relu(self.bn(self.conv(x)))
        return self.head(self.flat(self.pool(y)))


@pytest.fixture(autouse=True)
def clean_state(monkeypatch, tmp_path):
    """Every test starts with the autopatch disarmed and writing into tmp_path."""
    for name in list(os.environ):
        if name.startswith("NEURARCH_"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NEURARCH_TRACE_OUT", str(tmp_path / "graph.neurarch.json"))
    autopatch._reset_for_tests()
    yield
    autopatch._reset_for_tests()


def arm():
    assert autopatch.install(force=True) is True


# ── arming ─────────────────────────────────────────────────────────────────


def test_enabled_reads_the_environment():
    assert autopatch.enabled({}) is False
    assert autopatch.enabled({"NEURARCH_TRACE": "0"}) is False
    assert autopatch.enabled({"NEURARCH_TRACE": ""}) is False
    for yes in ("1", "true", "TRUE", " yes ", "on"):
        assert autopatch.enabled({"NEURARCH_TRACE": yes}) is True


def test_install_is_a_no_op_without_the_variable():
    original = nn.Module.__call__
    assert autopatch.install() is False
    assert nn.Module.__call__ is original


def test_install_is_idempotent():
    arm()
    patched = nn.Module.__call__
    assert autopatch.install(force=True) is True
    assert nn.Module.__call__ is patched


def test_our_own_cli_is_never_wrapped(monkeypatch):
    monkeypatch.setenv("NEURARCH_TRACE", "1")
    monkeypatch.setattr(sys, "argv", ["/usr/local/bin/neurarch-trace", "m:M"])
    original = nn.Module.__call__
    assert autopatch.install() is False
    assert nn.Module.__call__ is original


# ── capturing ──────────────────────────────────────────────────────────────


def test_captures_the_first_forward_pass(tmp_path):
    arm()
    out = Tiny()(torch.randn(2, 3, 16, 16))
    assert out.shape == (2, 4)

    graph = json.loads(Path(os.environ["NEURARCH_TRACE_OUT"]).read_text())
    types = [c["type"] for c in graph["components"]]
    assert types[0] == "input" and types[-1] == "output"
    assert "conv2d" in types and "linear" in types
    assert graph["name"] == "Tiny"
    assert "autopatch" in graph["description"]


def test_restores_call_and_captures_only_once(tmp_path):
    original_before = autopatch._state["orig_call"]
    arm()
    assert nn.Module.__call__ is not autopatch._state["orig_call"]
    model = Tiny()
    model(torch.randn(1, 3, 16, 16))

    # The wrapper is gone from the second step on, so a training loop pays for
    # this exactly once.
    assert nn.Module.__call__ is autopatch._state["orig_call"] or not autopatch._state["patched"]
    assert autopatch._state["patched"] is False

    first = Path(os.environ["NEURARCH_TRACE_OUT"]).read_bytes()
    os.remove(os.environ["NEURARCH_TRACE_OUT"])
    for _ in range(3):
        model(torch.randn(1, 3, 16, 16))
    assert not Path(os.environ["NEURARCH_TRACE_OUT"]).exists(), "captured a second time"
    assert first  # and the first one was real
    assert original_before is None


def test_a_submodule_call_is_not_the_model(tmp_path):
    """The outermost call wins: calling the whole model records the whole model."""
    arm()
    model = Tiny()
    model(torch.randn(1, 3, 16, 16))
    graph = json.loads(Path(os.environ["NEURARCH_TRACE_OUT"]).read_text())
    types = [c["type"] for c in graph["components"]]
    assert types.count("conv2d") == 1 and types.count("linear") == 1
    assert len(types) > 4, "only the submodule was recorded"


def test_skip_ignores_warmup_forwards(monkeypatch, tmp_path):
    monkeypatch.setenv("NEURARCH_TRACE_SKIP", "2")
    arm()
    warm, real = Tiny(), Tiny()
    warm(torch.randn(1, 3, 16, 16))
    warm(torch.randn(1, 3, 16, 16))
    assert not Path(os.environ["NEURARCH_TRACE_OUT"]).exists()
    real(torch.randn(1, 3, 8, 8))
    graph = json.loads(Path(os.environ["NEURARCH_TRACE_OUT"]).read_text())
    assert graph["components"][0]["params"]["shape"] == [3, 8, 8]


def test_depth_and_name_are_honoured(monkeypatch, tmp_path):
    monkeypatch.setenv("NEURARCH_TRACE_NAME", "my-model")
    monkeypatch.setenv("NEURARCH_TRACE_DEPTH", "1")
    arm()
    Tiny()(torch.randn(1, 3, 16, 16))
    graph = json.loads(Path(os.environ["NEURARCH_TRACE_OUT"]).read_text())
    assert graph["name"] == "my-model"
    assert all("." not in c["name"] for c in graph["components"])


def test_a_module_called_without_tensors_does_not_burn_the_capture(tmp_path):
    class NoTensors(nn.Module):
        def forward(self, flag=False):
            return torch.zeros(1) if flag else None

    arm()
    NoTensors()()
    assert not Path(os.environ["NEURARCH_TRACE_OUT"]).exists()
    Tiny()(torch.randn(1, 3, 16, 16))
    assert Path(os.environ["NEURARCH_TRACE_OUT"]).exists(), "the real model was skipped"


def test_a_broken_model_does_not_become_our_exception(tmp_path):
    class Broken(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def forward(self, x):
            return self.fc(x) + torch.zeros(3, 3, 3)  # not broadcastable

    arm()
    with pytest.raises(RuntimeError) as exc:
        Broken()(torch.randn(1, 4))
    assert "neurarch" not in str(exc.value).lower()
    assert autopatch._state["patched"] is False, "left the wrapper installed after a failure"


# ── the guest contract ─────────────────────────────────────────────────────


def test_leaves_the_run_alone(tmp_path):
    """One forward pass, the mode it found, no flags set on the caller's batch."""
    calls = {"n": 0}

    class Counting(Tiny):
        def forward(self, x):
            calls["n"] += 1
            return super().forward(x)

    model = Counting()
    model.train()
    x = torch.randn(2, 3, 16, 16)

    arm()
    model(x)

    assert calls["n"] == 1, "ran the forward pass more than once"
    assert model.training is True, "changed the model's mode"
    assert x.requires_grad is False, "set requires_grad on the caller's tensor"
    assert x.grad is None
    assert Path(os.environ["NEURARCH_TRACE_OUT"]).exists()


def test_matches_the_cli(tmp_path):
    """The graph you get for free is the graph you get by asking for it."""
    torch.manual_seed(0)
    model = Tiny().eval()
    x = torch.randn(1, 3, 16, 16)
    by_hand = trace_model(model, [x.clone()], name="Tiny")

    autopatch._reset_for_tests()
    arm()
    model(x)
    observed = json.loads(Path(os.environ["NEURARCH_TRACE_OUT"]).read_text())

    assert [c["type"] for c in observed["components"]] == [c["type"] for c in by_hand["components"]]
    assert [c["name"] for c in observed["components"]] == [c["name"] for c in by_hand["components"]]
    assert [(e["from"], e["to"]) for e in observed["connections"]] == \
           [(e["from"], e["to"]) for e in by_hand["connections"]]
    assert [c.get("outputShape") for c in observed["components"]] == \
           [c.get("outputShape") for c in by_hand["components"]]


def test_no_grad_still_yields_a_graph(tmp_path):
    """Inference has no autograd chain; identity edges still carry a linear model."""
    arm()
    with torch.no_grad():
        Tiny()(torch.randn(1, 3, 16, 16))
    graph = json.loads(Path(os.environ["NEURARCH_TRACE_OUT"]).read_text())
    types = [c["type"] for c in graph["components"]]
    assert "conv2d" in types and "linear" in types
    assert graph["connections"], "no edges recovered at all"


# ── packaging ──────────────────────────────────────────────────────────────


def test_pth_is_a_single_executable_line():
    lines = (ROOT / PTH_FILE).read_text().splitlines()
    code = [ln for ln in lines if ln.strip() and not ln.startswith("#")]
    assert len(code) == 1, "a .pth line that is not a comment and not `import ...` becomes a sys.path entry"
    assert code[0].startswith("import "), "site.py only executes lines starting with `import`"
    assert "NEURARCH_TRACE" in code[0]
    compile(code[0], PTH_FILE, "exec")


def test_pth_does_nothing_without_the_variable(tmp_path):
    """The line runs in every interpreter in the environment; it must be free."""
    script = "import sys; print('torch' in sys.modules, 'neurarch_trace' in sys.modules)"
    env = {k: v for k, v in os.environ.items() if not k.startswith("NEURARCH_")}
    line = [ln for ln in (ROOT / PTH_FILE).read_text().splitlines()
            if ln.strip() and not ln.startswith("#")][0]
    out = subprocess.run([sys.executable, "-c", line + "\n" + script],
                         capture_output=True, text=True, env=env, cwd=str(ROOT))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False False", out.stdout


def test_pth_is_packaged(tmp_path):
    """build_py must leave the .pth at the root of the build tree.

    That is the whole mechanism: wheel content at the root of the build tree is
    installed at the root of site-packages, which is the only directory Python
    scans for .pth files. If this breaks, `pip install neurarch-trace` keeps
    working and the autopatch quietly stops existing.
    """
    pytest.importorskip("setuptools")
    build = tmp_path / "lib"
    out = subprocess.run(
        [sys.executable, "setup.py", "build_py", "--build-lib", str(build)],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    # Only the .pth is asserted: whether the package tree lands next to it
    # depends on the setuptools version reading pyproject.toml, which is
    # setuptools' business and varies by environment. That the *published*
    # wheel carries both is gated in .github/workflows/publish-pypi.yml, which
    # runs against a modern backend and refuses to publish without it.
    assert (build / PTH_FILE).is_file(), "build_py did not put %s at the root of the build tree" % PTH_FILE


def test_sdist_carries_the_pth():
    """`uv build` makes the wheel from the sdist; MANIFEST.in is what keeps the
    .pth in it. Without this the published wheel loses the autopatch and every
    test above still passes, because they all run from the source tree."""
    manifest = (ROOT / "MANIFEST.in").read_text()
    assert PTH_FILE in manifest


# ── the import order that actually happens ─────────────────────────────────


def test_torch_imported_after_arming(tmp_path):
    """The real sequence: the .pth arms in an interpreter with no torch in it.

    Every other test in this file arms an already-imported torch, so none of
    them touches `_TorchWatcher`, which is the only code path a real
    `NEURARCH_TRACE=1 python train.py` uses.
    """
    out_path = tmp_path / "later.neurarch.json"
    script = """
import sys
sys.path.insert(0, %r)
import neurarch_trace.autopatch as autopatch
assert autopatch.install(force=True)
assert "torch" not in sys.modules, "torch was already imported; this proves nothing"

import torch
import torch.nn as nn
assert autopatch._state["patched"], "the watcher did not fire on the torch import"

model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
model(torch.randn(3, 4))
print("ok")
""" % str(ROOT)
    env = {k: v for k, v in os.environ.items() if not k.startswith("NEURARCH_")}
    env["NEURARCH_TRACE_OUT"] = str(out_path)
    env["NEURARCH_TRACE_NAME"] = "later"
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                         env=env, cwd=str(tmp_path))
    assert run.returncode == 0, run.stdout + run.stderr
    graph = json.loads(out_path.read_text())
    assert [c["type"] for c in graph["components"]] == \
           ["input", "linear", "relu", "linear", "output"]


# ── the gate ───────────────────────────────────────────────────────────────


class _StubPlan(BaseHTTPRequestHandler):
    """Stands in for POST /api/v1/plan so the gate can be tested off the network."""

    blocked = True
    seen = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        _StubPlan.seen.append(json.loads(self.rfile.read(length).decode()))
        body = json.dumps({
            "text": "STUB PLAN CARD",
            "plan": {"run": {"legal": not _StubPlan.blocked,
                             "blockers": ["head: linear declares 43264, upstream emits 64"]
                                         if _StubPlan.blocked else []}},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def stub_plan(monkeypatch):
    _StubPlan.seen = []
    server = HTTPServer(("127.0.0.1", 0), _StubPlan)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("NEURARCH_API", "http://127.0.0.1:%d" % server.server_address[1])
    yield _StubPlan
    server.shutdown()


def test_nothing_is_sent_unless_asked(stub_plan):
    arm()
    Tiny()(torch.randn(1, 3, 16, 16))
    assert stub_plan.seen == [], "sent the graph without PLAN or SHARE"


def test_plan_sends_the_graph_and_prints_the_card(monkeypatch, stub_plan, capsys):
    monkeypatch.setenv("NEURARCH_TRACE_PLAN", "1")
    stub_plan.blocked = False
    arm()
    Tiny()(torch.randn(1, 3, 16, 16))
    assert len(stub_plan.seen) == 1
    sent = stub_plan.seen[0]
    assert sent["share"] is False
    assert "autopatch" in sent["source"]["tool"]
    assert [c["type"] for c in sent["model"]["components"]][0] == "input"
    # The card goes to stderr: stdout belongs to the script we are a guest in.
    captured = capsys.readouterr()
    assert "STUB PLAN CARD" in captured.err
    assert captured.out == ""


def test_fail_on_block_stops_the_run(monkeypatch, stub_plan):
    monkeypatch.setenv("NEURARCH_TRACE_PLAN", "1")
    monkeypatch.setenv("NEURARCH_TRACE_FAIL_ON_BLOCK", "1")
    stub_plan.blocked = True
    arm()
    with pytest.raises(SystemExit) as exit_info:
        Tiny()(torch.randn(1, 3, 16, 16))
    assert exit_info.value.code == 2


def test_a_blocker_without_the_flag_only_prints(monkeypatch, stub_plan, capsys):
    monkeypatch.setenv("NEURARCH_TRACE_PLAN", "1")
    stub_plan.blocked = True
    arm()
    Tiny()(torch.randn(1, 3, 16, 16))  # no SystemExit
    assert "STUB PLAN CARD" in capsys.readouterr().err


def test_an_unreachable_server_does_not_stop_the_run(monkeypatch, capsys):
    monkeypatch.setenv("NEURARCH_TRACE_PLAN", "1")
    monkeypatch.setenv("NEURARCH_TRACE_FAIL_ON_BLOCK", "1")
    monkeypatch.setenv("NEURARCH_API", "http://127.0.0.1:1")  # nothing listens
    arm()
    Tiny()(torch.randn(1, 3, 16, 16))
    assert "could not reach" in capsys.readouterr().err
    assert Path(os.environ["NEURARCH_TRACE_OUT"]).exists(), "lost the graph too"
