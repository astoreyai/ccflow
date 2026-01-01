# ccflow - Claude Code CLI Middleware
# Subscription-based usage (no API keys required)
#
# Build:
#   docker build -t ccflow .
#
# Run (mount Claude credentials):
#   docker run -v ~/.claude:/root/.claude ccflow
#
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Build dependencies
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir build hatchling \
    && pip wheel --no-cache-dir --wheel-dir /wheels \
    pydantic pydantic-settings structlog aiosqlite \
    prometheus-client \
    fastapi uvicorn websockets

# -----------------------------------------------------------------------------
# Stage 2: Runtime
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

LABEL maintainer="Aaron Storey <aaron@kymera.systems>"
LABEL description="ccflow - Claude Code CLI Middleware"
LABEL version="0.1.0"

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # ccflow config
    CCFLOW_LOG_LEVEL=INFO \
    CCFLOW_METRICS_PORT=9090 \
    CCFLOW_API_PORT=8080

WORKDIR /app

# Install runtime dependencies + Node.js for Claude CLI
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    gnupg \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" | tee /etc/apt/sources.list.d/nodesource.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g @anthropic-ai/claude-code \
    && npm cache clean --force

# Install Python wheels from builder
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels

# Copy application
COPY src/ /app/src/
COPY pyproject.toml README.md py.typed /app/

# Install ccflow
RUN pip install --no-cache-dir -e .

# Create non-root user for production
RUN useradd --create-home --shell /bin/bash ccflow \
    && mkdir -p /home/ccflow/.claude \
    && chown -R ccflow:ccflow /app /home/ccflow

# Copy entrypoint
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose ports
# 8080 - API server
# 9090 - Prometheus metrics
EXPOSE 8080 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default to non-root user
USER ccflow
WORKDIR /home/ccflow

ENTRYPOINT ["/entrypoint.sh"]
CMD ["serve"]
