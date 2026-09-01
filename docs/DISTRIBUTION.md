# Where this server is listed, and how each listing is kept current

Every registry wants the same metadata; it is prepared once here. Order
matters: the official registry first, because the directories crawl it.

| Where | Mechanism | Status | On each release |
|---|---|---|---|
| npm | `npm publish` | live | version bump, `npm publish` |
| Official MCP Registry | `server.json` + `mcp-publisher publish` (GitHub auth for the `io.github.neurarch-ai/*` namespace) | listed since 0.10 | bump `version` in both places in `server.json`, republish |
| Smithery | `smithery.yaml` in repo root; claim the server at smithery.ai/new with the GitHub repo | file added 0.13, claim pending | nothing; Smithery re-reads the repo |
| Glama | `glama.json` in repo root; claim at glama.ai/mcp/servers | file added 0.13, claim pending | nothing |
| PulseMCP, mcp.so | submission form (pulsemcp.com/submit, mcp.so/submit) with npm + GitHub URLs | pending | nothing |
| Docker MCP Catalog | PR to github.com/docker/mcp-registry adding `servers/neurarch/server.yaml` (below) | pending | bump the image tag in the PR |
| Claude Desktop | `.mcpb` from `npm run build:mcpb`, attached to the GitHub release; submit to the Desktop Extension directory from the Anthropic form | 0.13 bundle built | rebuild, attach to release |
| Cursor directory | cursor.directory/mcp submit form; the README's Add-to-Cursor deeplink | pending | nothing |
| awesome-mcp-servers | PR to github.com/punkpeye/awesome-mcp-servers (entry below) | pending | nothing |
| VS Code | the `vscode.dev/redirect/mcp/install` badge in the README | live | nothing |

## awesome-mcp-servers entry

Under **Data Science Tools** (alphabetical):

```
- [neurarch-ai/neurarch-mcp](https://github.com/neurarch-ai/neurarch-mcp) 📇 🏠 🍎 🪟 🐧 - Structural awareness of a PyTorch model for the agent: layers, params, FLOPs, blast radius, the design linter, the full readiness/cost/deployment verdict, and which of k candidate designs to train. Offline, no key.
```

## Docker MCP Catalog `server.yaml`

```yaml
name: neurarch
image: mcp/neurarch
type: server
meta:
  category: data-science
  tags:
    - pytorch
    - machine-learning
    - lint
    - neural-network
about:
  title: Neurarch
  description: Structural awareness of a PyTorch model for the agent: layers, params, FLOPs, blast radius, the design linter, the full readiness/cost/deployment verdict, and which of k candidate designs to train. Reads a .py, a .neurarch.json, a bundled reference architecture (zoo:<id>) or a Hugging Face repo (hf:<org/name>, opt-in network).
  icon: https://neurarch.com/favicon.png
source:
  project: https://github.com/neurarch-ai/neurarch-mcp
  branch: main
  dockerfile: Dockerfile
config:
  description: The server runs hosted-style inside the container; pass model_path (zoo:/hf:) or model_source (inline text) on each call, or mount a directory and pass a path.
  secrets:
    - name: neurarch.hf_token
      env: HF_TOKEN
      example: hf_xxx
```

The catalog builds from `Dockerfile` in this repo, which starts the server in
hosted mode with `--hf`; a mounted model directory works through `model_path`.

## Launch checklist (once per major release)

1. `npm publish`, then `mcp-publisher publish`.
2. GitHub release with the `.mcpb` attached and the CHANGELOG section as the body.
3. Claim / resubmit the directories above.
4. Post: the real-repos study (`docs/REAL_REPOS_STUDY.md`) is the hook, the GIF is the visual, the tiny-vit.py planted bug is the reproducible demo.
5. Every outbound link carries `?utm_source=<channel>` to `https://www.neurarch.com/mcp`, so the channel that actually brings people is measurable.
