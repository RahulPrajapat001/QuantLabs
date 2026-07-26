# QuantLab product UI (Next.js 14, standalone output).
#
# Three stages: dependencies, build, runtime. The runtime carries Next's
# `standalone` bundle — the server plus only the node_modules it actually
# imports — so the shipped image has no build toolchain and no dev dependencies.
#
# The design tokens and Plotly templates live in ../config and are imported by
# the app, so the build context is the repo root, not frontend/. See
# docker-compose.yml.

# ------------------------------------------------------------------- deps ----
FROM node:20-alpine AS deps
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json ./
# `npm ci` for a lockfile-exact install; a floating dependency in a production
# image is an unreproducible image.
RUN npm ci

# ------------------------------------------------------------------ build ----
FROM node:20-alpine AS build
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY frontend/ ./
# Generated design tokens, imported by tailwind.config.ts and the chart layer.
COPY config/ ../config/

# Next inlines NEXT_PUBLIC_* into the client bundle at build time, so this must
# be a build arg — setting it only at runtime would leave the browser calling
# whatever was baked in. It is the URL the *browser* resolves, not the compose
# service name.
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL     NEXT_TELEMETRY_DISABLED=1
RUN npm run build

# ---------------------------------------------------------------- runtime ----
FROM node:20-alpine AS runtime
WORKDIR /app

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000

# node:alpine ships a `node` user (uid 1000); use it rather than root.
COPY --from=build --chown=node:node /app/.next/standalone ./
COPY --from=build --chown=node:node /app/.next/static ./.next/static
COPY --from=build --chown=node:node /app/public ./public

USER node
EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD wget -q --spider http://127.0.0.1:3000/ || exit 1

CMD ["node", "server.js"]
