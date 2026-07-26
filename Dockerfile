# QuantLab API / dashboard image.
#
# Multi-stage: dependencies are resolved and installed into a virtualenv in the
# builder, then only that virtualenv and the source are copied into a slim
# runtime. Build tools (gcc, headers for scientific wheels) never reach the
# shipped image.
#
# WeasyPrint is the reason this image is not `python:slim` with pip and nothing
# else. It renders the PDF tearsheet through Pango and Cairo, which are *system*
# libraries, not Python packages — `pip install weasyprint` succeeds and then
# fails at first render with "cannot load library 'gobject-2.0'". The runtime
# stage installs them explicitly, and `docker-compose` runs a smoke render on
# startup so a missing library fails the container rather than the first user's
# report.

# ---------------------------------------------------------------- builder ----
FROM python:3.11-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# Build-only: compilers and headers for any wheel without a manylinux build.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy only what pip needs to resolve, so a source-only change does not
# invalidate the dependency layer.
COPY pyproject.toml README.md ./
COPY src/quantlab/__init__.py src/quantlab/__init__.py

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# The extras the production image actually serves: data platform, research
# engine, API, PDF reports, PostgreSQL driver. `dev` is deliberately absent.
RUN pip install --upgrade pip setuptools wheel \
    && pip install ".[data,research,api,reports,postgres,dashboard]"

# ---------------------------------------------------------------- runtime ----
FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    # JSON logs: the container is not a TTY, but be explicit rather than relying
    # on isatty() detection inside an orchestrator.
    QUANTLAB_LOG_LEVEL=INFO

# Runtime system libraries.
#
# The Pango/Cairo/GDK-Pixbuf set is what WeasyPrint dlopen()s at render time.
# libpq5 is the PostgreSQL client library. curl is here for the compose
# healthcheck and nothing else.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libcairo2 \
        libpango-1.0-0 \
        libpangocairo-1.0-0 \
        libgdk-pixbuf-2.0-0 \
        libffi8 \
        shared-mime-info \
        fonts-dejavu-core \
        libpq5 \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

# Non-root. A research API has no business running as uid 0, and the container
# writes only to /app/data, which is owned by this user.
RUN groupadd --system --gid 1001 quantlab \
    && useradd --system --uid 1001 --gid quantlab --create-home quantlab

WORKDIR /app

COPY --chown=quantlab:quantlab pyproject.toml alembic.ini README.md ./
COPY --chown=quantlab:quantlab src/ ./src/
COPY --chown=quantlab:quantlab scripts/ ./scripts/
COPY --chown=quantlab:quantlab dashboard/ ./dashboard/
COPY --chown=quantlab:quantlab config/ ./config/
COPY --chown=quantlab:quantlab docs/ ./docs/

# Install the project itself (not its dependencies — those came from the venv) so
# the `quantlab` console script and `src` layout resolve.
RUN pip install --no-deps --no-cache-dir -e . \
    && mkdir -p /app/data/raw /app/data/features \
    && chown -R quantlab:quantlab /app/data /app/src

USER quantlab

# Fail fast at build time if the PDF stack is not actually loadable. This is the
# check that would otherwise be deferred to the first tearsheet request in
# production, which is exactly the wrong moment to discover a missing .so.
RUN python -c "import weasyprint; weasyprint.HTML(string='<p>ok</p>').write_pdf()" \
    && echo "WeasyPrint OK: Pango/Cairo present and a PDF rendered"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "quantlab.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
