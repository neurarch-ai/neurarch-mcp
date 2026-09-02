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

`Dockerfile` and `fly.toml` are in the repo root. The image's default command
is hosted stdio (what Docker MCP Toolkit runs); `fly.toml`'s `[processes]`
overrides it to HTTP. First deploy:

```bash
fly apps create neurarch-mcp
fly secrets set NEURARCH_MCP_TOKEN=$(openssl rand -hex 16) --stage
fly deploy --remote-only
fly ips allocate-v4 --shared && fly ips allocate-v6   # if the deploy could not provision IPs
```

Any container host works the same way; the image listens on 8787.

The production instance lives at `https://neurarch-mcp.fly.dev/mcp` and runs
in PUBLIC read-only mode (`NEURARCH_MCP_PUBLIC_HOST=neurarch-mcp.fly.dev`, no
token): anyone may call the read tools, write tools are not exposed, and the
DNS-rebinding check still pins the hostname. Connect a client with:

```json
{ "mcpServers": { "neurarch": { "type": "http", "url": "https://neurarch-mcp.fly.dev/mcp" } } }
```

claude.ai (web): Settings, Connectors, Add custom connector, paste the URL, no
authentication. To run a private instance instead, set NEURARCH_MCP_TOKEN and
leave NEURARCH_MCP_PUBLIC_HOST unset; clients then need the bearer header.

## What the hosted server sees

Exactly what a client sends it: the model text or the reference it names. A
pasted `model_source` is parsed in memory and not written anywhere; `hf:`
configs are cached on the container's disk for a day. There is no account, no
per-user state and no log of graphs. Corpus reporting (`NEURARCH_REPORT=1`)
is off unless the deploy sets it, and it never carries the graph.

This is a different privacy contract from the local server, which opens no
socket at all. Say so wherever a hosted URL is published.
