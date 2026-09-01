import json
import os
import subprocess
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neurarch_trace import trace_model  # noqa: E402
from neurarch_trace.cli import main  # noqa: E402


class BasicUnit(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.relu(y + x)


class ResCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU())
        self.layer1 = nn.Sequential(BasicUnit(16), BasicUnit(16))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = self.layer1(self.stem(x))
        return self.fc(torch.flatten(self.pool(x), 1))


class TransformerBlock(nn.Module):
    def __init__(self, d=32, heads=4):
        super().__init__()
        self.tok = nn.Embedding(100, d)
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.head = nn.Linear(d, 100)

    def forward(self, ids):
        x = self.tok(ids)
        h = self.norm1(x)
        x = x + self.attn(h, h, h)[0]
        x = x + self.mlp(self.norm2(x))
        return self.head(x)


class TwoTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(8, 4)
        self.b = nn.Linear(6, 4)
        self.out = nn.Linear(8, 2)

    def forward(self, x, y):
        return self.out(torch.cat([self.a(x), self.b(y)], dim=1))


def by_id(graph):
    return {c["id"]: c for c in graph["components"]}


def assert_loadable(graph):
    # The same checks `src/loader.ts` makes before any tool runs.
    assert isinstance(graph["components"], list) and isinstance(graph["connections"], list)
    ids = set()
    for c in graph["components"]:
        assert isinstance(c["id"], str) and c["id"] not in ids
        ids.add(c["id"])
        for k in ("type", "name", "position", "params", "inputs", "outputs"):
            assert k in c
    for e in graph["connections"]:
        assert e["from"] in ids and e["to"] in ids
        assert {"id", "from", "to", "fromPort", "toPort"} <= set(e)
    assert sum(c["type"] == "input" for c in graph["components"]) >= 1
    assert sum(c["type"] == "output" for c in graph["components"]) == 1
    json.loads(json.dumps(graph))


def test_residual_cnn():
    g = trace_model(ResCNN(), [torch.randn(2, 3, 32, 32)], name="rescnn")
    assert_loadable(g)
    c = by_id(g)
    types = [x["type"] for x in g["components"]]
    assert types.count("conv2d") == 5 and types.count("batchNorm") == 5
    # The shared ReLU is called twice per block; each call is its own node.
    assert types.count("relu") == 1 + 4
    assert types.count("add") == 2
    assert types.count("globalAvgPool2d") == 1 and types.count("linear") == 1
    assert len(g["components"]) == 21

    assert c["input"]["params"]["shape"] == [3, 32, 32]
    assert c["stem_0"]["inputShape"] == [3, 32, 32] and c["stem_0"]["outputShape"] == [16, 32, 32]
    assert c["stem_0"]["params"] == {"inChannels": 3, "outChannels": 16, "kernelSize": 3, "stride": 1, "padding": 1}
    assert c["fc"]["inputShape"] == [16] and c["fc"]["params"] == {"inFeatures": 16, "outFeatures": 10}
    assert c["output"]["inputShape"] == [10]

    add = c["add"]
    assert sorted(add["inputs"]) == ["layer1_0_bn2", "stem_2"]
    assert add["outputShape"] == [16, 32, 32] and add["scope"] == "layer1.0"
    assert c["layer1_0_relu_2"]["inputs"] == ["add"]
    assert c["layer1_0_conv1"]["scope"] == "layer1.0" and c["stem_1"]["scope"] == "stem"
    assert "scope" not in c["fc"]
    assert len(g["connections"]) == sum(len(x["inputs"]) for x in g["components"])


def test_transformer_block_with_mha_and_two_residuals():
    g = trace_model(TransformerBlock(), [torch.randint(0, 100, (1, 7))], name="tb")
    assert_loadable(g)
    c = by_id(g)
    assert c["tok"]["type"] == "embedding"
    assert c["tok"]["params"] == {"vocabSize": 100, "embeddingDim": 32}
    assert c["tok"]["inputShape"] == [7] and c["tok"]["outputShape"] == [7, 32]
    assert c["attn"]["type"] == "multiHeadAttention"
    assert c["attn"]["params"] == {"hiddenDim": 32, "numHeads": 4}
    assert c["norm1"]["params"] == {"normalizedShape": 32}
    assert c["mlp_1"]["type"] == "gelu" and c["mlp_1"]["scope"] == "mlp"
    adds = [x for x in g["components"] if x["type"] == "add"]
    assert len(adds) == 2
    assert sorted(adds[0]["inputs"]) == ["attn", "tok"]
    assert sorted(adds[1]["inputs"]) == ["add", "mlp_2"]
    assert c["head"]["inputs"] == ["add_2"] and c["output"]["inputShape"] == [7, 100]


def test_two_inputs_and_concat():
    g = trace_model(TwoTower(), [torch.randn(3, 8), torch.randn(3, 6)], name="tt")
    assert_loadable(g)
    c = by_id(g)
    assert c["input_0"]["params"]["shape"] == [8] and c["input_1"]["params"]["shape"] == [6]
    cat = [x for x in g["components"] if x["type"] == "concatenate"]
    assert len(cat) == 1 and sorted(cat[0]["inputs"]) == ["a", "b"]
    assert cat[0]["outputShape"] == [8] and c["out"]["inputs"] == [cat[0]["id"]]


def test_depth_cuts_blocks_into_custom_nodes():
    g = trace_model(ResCNN(), [torch.randn(1, 3, 32, 32)], name="d", depth=2)
    c = by_id(g)
    assert c["layer1_0"]["type"] == "customModule"
    assert c["layer1_0"]["params"]["className"] == "BasicUnit"
    assert c["layer1_0"]["params"]["paramCount"] == sum(p.numel() for p in BasicUnit(16).parameters())
    assert c["layer1_0"]["scope"] == "layer1" and c["layer1_1"]["inputs"] == ["layer1_0"]
    assert "add" not in c


def test_patch_embed_heuristic_and_custom_map():
    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_dim, self.num_heads = 8, 2
            self.qkv = nn.Linear(8, 24)

        def forward(self, x):
            return self.qkv(x)[..., :8]

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch = nn.Conv2d(3, 8, kernel_size=16, stride=16)
            self.attn = Attention()

        def forward(self, x):
            return self.attn(self.patch(x).flatten(2).transpose(1, 2))

    g = trace_model(M(), [torch.randn(1, 3, 32, 32)], name="pe", depth=1)
    c = by_id(g)
    assert c["patch"]["type"] == "patchEmbed"
    assert c["patch"]["params"] == {"patchSize": 16, "embedDim": 8, "inChans": 3}
    assert c["attn"]["type"] == "attention"
    assert c["attn"]["params"]["embedDim"] == 8 and c["attn"]["params"]["numHeads"] == 2
    assert c["attn"]["inputShape"] == [4, 8]


def test_cli_writes_file_and_reports_failures(tmp_path, capsys):
    src = tmp_path / "m.py"
    src.write_text(
        "import torch.nn as nn\n"
        "def build():\n"
        "    return nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))\n"
        "class Broken(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__(); self.fc = nn.Linear(4, 3)\n"
        "    def forward(self, x):\n"
        "        return self.fc(x[:, :2])\n"
    )
    out = tmp_path / "m.neurarch.json"
    assert main([str(src) + ":build", "--input", "2,4", "-o", str(out)]) == 0
    g = json.loads(out.read_text())
    assert_loadable(g)
    assert [x["type"] for x in g["components"]] == ["input", "linear", "relu", "linear", "output"]
    assert "scope" not in g["components"][1]

    assert main([str(src) + ":Broken", "--input", "1,4", "-o", str(tmp_path / "x.json")]) == 1
    err = capsys.readouterr().err
    assert err.startswith("neurarch-trace: forward pass failed") and "Traceback" not in err

    assert main([str(src) + ":nope", "--input", "1,4"]) == 1
    assert main(["not-a-target", "--input", "1,4"]) == 1
    assert main([str(src) + ":build"]) == 1


def test_module_entry_point_runs_as_subprocess(tmp_path):
    out = tmp_path / "s.neurarch.json"
    env = dict(os.environ, PYTHONPATH=os.path.join(os.path.dirname(__file__), ".."))
    r = subprocess.run(
        [sys.executable, "-m", "neurarch_trace", "torch.nn:Identity", "--input", "1,4", "-o", str(out)],
        capture_output=True, text=True, env=env, cwd=str(tmp_path),
    )
    # Identity is pass-through, so there is nothing to record: a clear one-line failure.
    assert r.returncode == 1 and r.stderr.strip().startswith("neurarch-trace:")
    r = subprocess.run(
        [sys.executable, "-m", "neurarch_trace", "torch.nn:Linear", "--input", "1,4", "-o", "-", "--verbose"],
        capture_output=True, text=True, env=env, cwd=str(tmp_path),
    )
    assert r.returncode == 1  # nn.Linear() needs args; --verbose keeps the traceback
    assert "Traceback" in r.stderr
