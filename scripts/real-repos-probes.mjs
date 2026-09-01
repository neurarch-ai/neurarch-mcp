#!/usr/bin/env node
// Minimal PyTorch snippets, one per construct that docs/REAL_REPOS_STUDY.md
// attributes a parse failure or a false finding to. Each is run through the
// vendored engine and the node list printed, so every roadmap item in the
// study is pinned by an input small enough to fix against.
//
//   node scripts/real-repos-probes.mjs
//
// NUL means graphFromPyTorchSource returned null. The expected column is what a
// correct parse would produce; the gap between the two columns is the roadmap.

import { graphFromPyTorchSource } from '../src/vendor/engine.bundle.mjs';

const H = 'import math\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n';
const probes = [
  { name: "baseline nn.Linear + nn.Conv2d ints", expected: "conv, fc", code: `
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)
        self.fc = nn.Linear(16, 10)
    def forward(self, x):
        return self.fc(self.conv(x))` },
  { name: "torch.nn.Linear fully-qualified prefix", expected: "fc", code: `
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(16, 10)
    def forward(self, x):
        return self.fc(x)` },
  { name: "ModuleList generator expression", expected: "layers x4", code: `
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(16, 16) for _ in range(4))
    def forward(self, x):
        for l in self.layers: x = l(x)
        return x` },
  { name: "ModuleList list comprehension, literal range", expected: "layers x4", code: `
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(16, 16) for _ in range(4)])
    def forward(self, x):
        for l in self.layers: x = l(x)
        return x` },
  { name: "ModuleList list comprehension over same-file block, literal args", expected: "fc, act x4", code: `
class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, d)
        self.act = nn.GELU()
    def forward(self, x):
        return self.act(self.fc(x))
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Block(16) for _ in range(4)])
    def forward(self, x):
        for l in self.layers: x = l(x)
        return x` },
  { name: "ModuleList over same-file block, config-object args (HF style)", expected: "fc, act xN, head", code: `
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc = nn.Linear(config.hidden_size, config.hidden_size)
        self.act = nn.GELU()
    def forward(self, x):
        return self.act(self.fc(x))
class Net(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        self.head = nn.Linear(config.hidden_size, config.vocab_size)
    def forward(self, x):
        for l in self.layers: x = l(x)
        return self.head(x)` },
  { name: "ModuleList() then append in for loop", expected: "layers x4", code: `
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(4):
            self.layers.append(nn.Linear(16, 16))
    def forward(self, x):
        for l in self.layers: x = l(x)
        return x` },
  { name: "nn.Sequential(*list) built in a loop", expected: "conv, relu x3", code: `
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for i in range(3):
            layers += [nn.Conv2d(16, 16, 3), nn.ReLU()]
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)` },
  { name: "helper class imported from another file", expected: "inc (opaque), out", code: `
from .parts import DoubleConv
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.inc = DoubleConv(3, 64)
        self.out = nn.Conv2d(64, 2, 1)
    def forward(self, x):
        return self.out(self.inc(x))` },
  { name: "same-file helper class with plain int args", expected: "c1, c2, out", code: `
class DoubleConv(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.c1 = nn.Conv2d(i, o, 3)
        self.c2 = nn.Conv2d(o, o, 3)
    def forward(self, x):
        return self.c2(self.c1(x))
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.inc = DoubleConv(3, 64)
        self.out = nn.Conv2d(64, 2, 1)
    def forward(self, x):
        return self.out(self.inc(x))` },
  { name: "same-file helper wrapped in parentheses (milesial style)", expected: "c1, out", code: `
class DoubleConv(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.c1 = nn.Conv2d(i, o, 3)
    def forward(self, x):
        return self.c1(x)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.inc = (DoubleConv(3, 64))
        self.out = nn.Conv2d(64, 2, 1)
    def forward(self, x):
        return self.out(self.inc(x))` },
  { name: "x.view(...) flatten in forward before Linear", expected: "conv, flatten, fc", code: `
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 8, 3)
        self.fc = nn.Linear(8 * 26 * 26, 10)
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)` },
  { name: "torch.flatten(x, 1) in forward before Linear", expected: "conv, flatten, fc", code: `
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 8, 3)
        self.fc = nn.Linear(8 * 26 * 26, 10)
    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        return self.fc(x)` },
  { name: "F.relu between two linears (nested call)", expected: "w1, relu, w2 (in that order)", code: `
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(16, 32)
        self.w2 = nn.Linear(32, 16)
    def forward(self, x):
        return self.w2(F.relu(self.w1(x)))` },
  { name: "nn.Parameter positional embedding added in forward", expected: "pos (PE), attn", code: `
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, 64, 16))
        self.attn = nn.MultiheadAttention(16, 4)
    def forward(self, x):
        x = x + self.pos
        return self.attn(x, x, x)[0]` },
  { name: "helper class defined AFTER the model (last-class heuristic)", expected: "conv, fc", code: `
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)
        self.fc = nn.Linear(16, 10)
    def forward(self, x):
        return self.fc(self.conv(x))
class LayerNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
    def forward(self, x):
        return x * self.w` },
  { name: "multi-line class header", expected: "fc", code: `
class Net(
    nn.Module,
):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 10)
    def forward(self, x):
        return self.fc(x)` },
  { name: "factory function returning nn.Sequential (PaLM style)", expected: "embedding, linear, linear", code: `
def PaLM(dim, depth):
    return nn.Sequential(nn.Embedding(1000, dim), nn.Linear(dim, dim), nn.Linear(dim, 1000))` },
  { name: "subclass of nn.Transformer with super().__init__(kwargs)", expected: "emb, encoder, dec", code: `
class Net(nn.Transformer):
    def __init__(self, ntoken, ninp):
        super().__init__(d_model=ninp, nhead=2, num_encoder_layers=2)
        self.emb = nn.Embedding(ntoken, ninp)
        self.dec = nn.Linear(ninp, ntoken)
    def forward(self, x):
        return self.dec(self.encoder(self.emb(x)))` },
  { name: "yaml-driven parse_model (ultralytics style)", expected: "model (opaque)", code: `
class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model, self.save = parse_model(cfg, ch=[3])
    def forward(self, x):
        return self.model(x)` },
  { name: "Conv1d with unresolved width", expected: "conv1, gelu, fc", code: `
class Net(nn.Module):
    def __init__(self, n_state):
        super().__init__()
        self.conv1 = nn.Conv1d(80, n_state, kernel_size=3, padding=1)
        self.fc = nn.Linear(n_state, 10)
    def forward(self, x):
        return self.fc(F.gelu(self.conv1(x)))` },
  { name: "Sequential containing helper with unresolved width", expected: "classifier_0 dropout, classifier_1 linear", code: `
class Net(nn.Module):
    def __init__(self, dropout=0.2, num_classes=1000):
        super().__init__()
        self.last_channel = 1280
        self.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(self.last_channel, num_classes))
    def forward(self, x):
        return self.classifier(x)` },
  { name: "sub-module created by a factory call (get_down_block / create_conv2d)", expected: "stem (opaque), fc", code: `
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = create_conv2d(3, 32, 3, stride=2)
        self.fc = nn.Linear(32, 10)
    def forward(self, x):
        return self.fc(self.stem(x))` },
  { name: "norm_layer partial / callable alias", expected: "fc, norm", code: `
from functools import partial
class Net(nn.Module):
    def __init__(self, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.fc = nn.Linear(16, 16)
        self.norm = norm_layer(16)
    def forward(self, x):
        return self.norm(self.fc(x))` },
  { name: "setattr loop (stylegan2)", expected: "b0..b2, out", code: `
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        for i in range(3):
            setattr(self, f"b{i}", nn.Linear(16, 16))
        self.out = nn.Linear(16, 1)
    def forward(self, x):
        return self.out(x)` },
  { name: "multi-line constructor call", expected: "proj, fc", code: `
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(
            3,
            16,
            kernel_size=3,
        )
        self.fc = nn.Linear(16, 10)
    def forward(self, x):
        return self.fc(self.proj(x))` },
  { name: "ModuleList generator over zip", expected: "layers xN", code: `
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
    def forward(self, x):
        for l in self.layers: x = l(x)
        return x` },
  { name: "conditional expression constructor", expected: "classifier", code: `
class Net(nn.Module):
    def __init__(self, num_labels=10):
        super().__init__()
        self.classifier = (
            nn.Linear(768, num_labels) if num_labels > 0 else nn.Identity()
        )
    def forward(self, x):
        return self.classifier(x)` },
  { name: "wrapper call around constructor zero_module(nn.Conv2d(...))", expected: "proj_in, proj_out", code: `
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_in = nn.Conv2d(16, 16, 1)
        self.proj_out = zero_module(nn.Conv2d(16, 16, 1))
    def forward(self, x):
        return self.proj_out(self.proj_in(x))` },
  { name: "torch.nn.ModuleList of same-file layer, torch.nn prefix everywhere", expected: "fc x4", code: `
class Layer(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = torch.nn.Linear(d, d)
    def forward(self, x):
        return self.fc(x)
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([Layer(16) for _ in range(4)])
    def forward(self, x):
        for l in self.layers: x = l(x)
        return x` },
  { name: "nn.Sequential(*builder(...))", expected: "conv_stem, blocks (opaque), classifier", code: `
class Net(nn.Module):
    def __init__(self, block_args):
        super().__init__()
        self.conv_stem = nn.Conv2d(3, 32, 3, stride=2)
        self.blocks = nn.Sequential(*builder(32, block_args))
        self.classifier = nn.Linear(1280, 1000)
    def forward(self, x):
        return self.classifier(self.blocks(self.conv_stem(x)))` },
  { name: "forward with control flow (if has_mask) and nested calls", expected: "input_emb, decoder", code: `
class Net(nn.Module):
    def __init__(self, ntoken, ninp):
        super().__init__()
        self.input_emb = nn.Embedding(ntoken, ninp)
        self.decoder = nn.Linear(ninp, ntoken)
        self.init_weights()
    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.input_emb.weight, -initrange, initrange)
    def forward(self, src, has_mask=True):
        if has_mask:
            src = src * 2
        src = self.input_emb(src) * math.sqrt(self.ninp)
        output = self.decoder(src)
        return F.log_softmax(output, dim=-1)` },
  { name: "same-file helper class, model class FIRST in file", expected: "c1, c2, out", code: `
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.inc = DoubleConv(3, 64)
        self.out = nn.Conv2d(64, 2, 1)
    def forward(self, x):
        return self.out(self.inc(x))
class DoubleConv(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.c1 = nn.Conv2d(i, o, 3)
        self.c2 = nn.Conv2d(o, o, 3)
    def forward(self, x):
        return self.c2(self.c1(x))` },
  { name: "raise ValueError with a multi-line triple-quoted message in __init__", expected: "encoder, decoder", code: `
class Net(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.encoder = nn.Embedding(10, 16)
        if k:
            raise ValueError("""line one
                 line two""")
        self.decoder = nn.Linear(16, 10)
    def forward(self, x):
        return self.decoder(self.encoder(x))` },
  { name: "try/except with a single-line raise in __init__ (control)", expected: "encoder, decoder", code: `
class Net(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.encoder = nn.Embedding(10, 16)
        try:
            n = {'a': 1}[k]
        except KeyError:
            raise ValueError('bad')
        self.decoder = nn.Linear(16, 10)
    def forward(self, x):
        return self.decoder(self.encoder(x))` },
];

const rows = [];
for (const { name, expected, code } of probes) {
  const m = graphFromPyTorchSource(H + code.trimStart(), 'probe');
  const layers = m ? m.components.filter((c) => c.type !== 'input' && c.type !== 'output') : null;
  rows.push({ name, expected, got: layers ? layers.map((c) => `${c.name}:${c.type}`).join(', ') : 'NUL', count: layers ? layers.length : 'NUL' });
}
const w = Math.max(...rows.map((r) => r.name.length));
console.log(`${'construct'.padEnd(w)} | got | expected`);
for (const r of rows) console.log(`${r.name.padEnd(w)} | ${String(r.count).padStart(3)} ${r.got} | ${r.expected}`);
