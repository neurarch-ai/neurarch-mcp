# Hosted neurarch-mcp: the same server over Streamable HTTP with no model on
# disk. Callers pass model_path (zoo:/hf:) or model_source (inline text) on
# every call. See docs/HOSTED.md.
#
# Two stages, because the image has to build from a bare git checkout. dist/ is
# gitignored, so a single-stage `COPY dist` only worked on a machine that had
# already run `npm run build`; a registry building from the repository (Glama,
# Docker MCP Toolkit) failed at that line, the server never started, and the
# listing could not be scored.
FROM node:22-alpine AS build
WORKDIR /app
COPY package.json package-lock.json tsconfig.json tsup.config.ts ./
RUN npm ci --ignore-scripts
COPY src ./src
RUN npm run build

FROM node:22-alpine
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --omit=dev --ignore-scripts
COPY --from=build /app/dist ./dist
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
