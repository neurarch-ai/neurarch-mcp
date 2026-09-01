# Hosting neurarch-mcp

The stdio server needs the agent and the model file on one machine. The hosted
form is the same binary with no model on disk, serving Streamable HTTP, so a
client with no filesystem to share (claude.ai connectors, a phone, a CI job in
someone else's repo) can still ask. Every call names its model:

- `model_path: "zoo:<id>"` for a bundled reference architecture,
- `model_path: "hf:<org/name>"` for a Hugging Face repo (the server runs with `--hf`),
- `model_source: "<text>"` for a `.neurarch.json` document or PyTorch source pasted inline.

## Run it

```bash
npm run build
NEURARCH_MCP_TOKEN=$(openssl rand -hex 16) \
  node dist/index.js --http --host=0.0.0.0 --hf --tools=core
```

`GET /health` returns `{ ok: true, hosted: true }`. Point an MCP client at
`https://<host>/mcp` with `Authorization: Bearer <token>`.

## Deploy

`Dockerfile` and `fly.toml` are in the repo root:

```bash
fly launch --copy-config --no-deploy
fly secrets set NEURARCH_MCP_TOKEN=$(openssl rand -hex 16)
fly deploy
```

Any container host works the same way; the image listens on 8787.

## What the hosted server sees

Exactly what a client sends it: the model text or the reference it names. A
pasted `model_source` is parsed in memory and not written anywhere; `hf:`
configs are cached on the container's disk for a day. There is no account, no
per-user state and no log of graphs. Corpus reporting (`NEURARCH_REPORT=1`)
is off unless the deploy sets it, and it never carries the graph.

This is a different privacy contract from the local server, which opens no
socket at all. Say so wherever a hosted URL is published.
