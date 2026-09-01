# Hosted neurarch-mcp: the same server over Streamable HTTP with no model on
# disk. Callers pass model_path (zoo:/hf:) or model_source (inline text) on
# every call. See docs/HOSTED.md.
FROM node:22-alpine
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --omit=dev --ignore-scripts
COPY dist ./dist
COPY zoo ./zoo
COPY README.md LICENSE ./
ENV NODE_ENV=production
EXPOSE 8787
# stdio by default, which is how Docker MCP Toolkit drives a container: no
# model is mounted, so every call names one (zoo:, hf:, or model_source).
# --hf is on because a hosted server exists to answer about models it does
# not have. For an HTTP deployment override the command:
#   node dist/index.js --http --host=0.0.0.0 --hf --tools=core
# and set NEURARCH_MCP_TOKEN (required before write tools may leave loopback).
ENTRYPOINT ["node", "dist/index.js"]
CMD ["--hosted", "--hf", "--tools=core"]
