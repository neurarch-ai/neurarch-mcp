# neurarch-mcp

Model Context Protocol server that exposes a [Neurarch](https://neurarch.com) model graph to Claude Code, Cursor, Windsurf, Codex, and any other MCP-aware AI agent.

The agent gets **structural awareness** of your neural network: layer list, parameter counts, FLOPs, blast-radius impact analysis, and Mermaid diagrams, without you pasting 200 lines of `nn.Module` into chat.

## Why

When you ask Claude Code "rewrite the training loop for my new encoder block", it sees your `train.py` but not your model. It guesses at shapes and parameter counts. `neurarch-mcp` plugs the structured model graph into the same conversation so the agent can answer "what depends on `attn_8`?" or "where does the parameter budget actually live?" with one call instead of ten.

## Install

No install. Just point your MCP client at it via `npx`:

```jsonc
// ~/.claude/mcp_servers.json (Claude Code)
{
  "mcpServers": {
    "neurarch": {
      "command": "npx",
      "args": ["-y", "neurarch-mcp", "/abs/path/to/your-model.neurarch.json"]
    }
  }
}
```

Cursor (`.cursor/mcp.json`), Windsurf, and Codex use the same `command` + `args` shape under their respective config keys.

To produce the model file: open your model in the [Neurarch](https://neurarch.com) app, then **File → Save (.json)**. The MCP server reads that file directly.

## Tools

| Tool | What it does |
|---|---|
| `get_model_summary` | One-shot overview: layer count, total params, dominant types, input/output shape. |
| `get_layer` | Full definition of one layer by name: params, shapes, notes, upstream/downstream ids. |
| `find_layers` | Search layers by type and/or name regex. |
| `layer_impact` | Blast radius of changing a layer or matched set. Flags shape-sensitive and weight-carrying downstream layers. |
| `param_count_by_block` | Parameter counts grouped by block / scope / type. |
| `flops_by_block` | MAC counts (FLOPs ÷ 2) grouped by block / scope / type. |
| `mermaid_diagram` | Render the model as Mermaid `flowchart TD` syntax. Truncates past 60 layers. |
| `list_blocks` | List collapsed groups (or scope-derived blocks if none): members, params, FLOPs. |

`layer_impact` is the headline tool. Before the agent recommends `delete every conv_X`, it can call `layer_impact` and tell the user "this rewires 8 downstream layers, 3 of which carry weights and will need rebuild."

## Example prompt (Claude Code)

After wiring the server, in Claude Code:

> Look at the Neurarch model. Where do the parameters actually live, and which block would shrink the model fastest if I cut it in half?

The agent calls `get_model_summary`, then `param_count_by_block`, then `layer_impact` on the heaviest block, and writes a recommendation grounded in the actual numbers from your model.

## What this is not

- **Not a generic codebase indexer.** This serves one `.neurarch.json` file. For codebase structure, use [GitNexus](https://github.com/abhigyanpatwari/GitNexus) or similar.
- **Not connected to your Neurarch workspace.** v1 reads a saved JSON file only. Live editing happens in the Neurarch web app.

## Development

```bash
git clone https://github.com/neurarch-ai/neurarch-mcp
cd neurarch-mcp
npm install
npm run build                 # tsup → dist/index.js
node dist/index.js --help     # confirm bin works
```

The package vendors a small set of pure-TypeScript utilities (model types, parameter and FLOP estimators, impact analyzer) from the main Neurarch repo. They live under `src/lib/` and have no runtime dependencies beyond `@modelcontextprotocol/sdk`.

## License

MIT. See [LICENSE](./LICENSE).

## Links

- [Neurarch](https://neurarch.com) — the visual neural-network editor that produces the model files this server reads.
- [Model Context Protocol](https://modelcontextprotocol.io) — the spec this server implements.
- [npm](https://www.npmjs.com/package/neurarch-mcp) — package page.
