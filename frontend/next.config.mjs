/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Emits .next/standalone: the server plus only the node_modules actually
  // imported, so the production image needs no build toolchain. Required by
  // frontend/Dockerfile.
  output: "standalone",
  // The design tokens and Plotly templates live outside frontend/ so the
  // Streamlit dashboard consumes the same files. Next needs to be told it may
  // trace above its own root. (Top-level in 15; still under `experimental` in 14.)
  experimental: {
    outputFileTracingRoot: new URL("..", import.meta.url).pathname,
  },
  env: {
    // Server-rendered and client code both read this. Set NEXT_PUBLIC_API_URL
    // to point at a non-local API; there is no hardcoded production host.
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000",
  },
};

export default nextConfig;
