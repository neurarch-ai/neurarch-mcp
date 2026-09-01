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
HEALTHCHECK --interval=30s --timeout=3s CMD wget -qO- http://127.0.0.1:8787/health || exit 1
# --hf is on: a hosted server exists to answer about models it does not have,
# and hf:<org/name> is the one way in that needs no upload. The token is
# required by the server itself before it will bind to 0.0.0.0 with anything
# but read tools; set NEURARCH_MCP_TOKEN in the deploy.
CMD ["node", "dist/index.js", "--http", "--host=0.0.0.0", "--hf", "--tools=core"]
