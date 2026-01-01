#!/bin/bash
set -e

# =============================================================================
# ccflow Docker Entrypoint
# =============================================================================
# Handles Claude CLI authentication and service startup
#
# Usage:
#   entrypoint.sh serve     - Start API server (default)
#   entrypoint.sh worker    - Start pool worker
#   entrypoint.sh shell     - Interactive shell
#   entrypoint.sh <command> - Run arbitrary command
# =============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[ccflow]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[ccflow]${NC} $1"
}

log_error() {
    echo -e "${RED}[ccflow]${NC} $1"
}

# -----------------------------------------------------------------------------
# Check Claude CLI authentication
# -----------------------------------------------------------------------------
check_claude_auth() {
    log_info "Checking Claude CLI authentication..."

    # Check if credentials directory is mounted
    if [ ! -d "$HOME/.claude" ]; then
        log_warn "~/.claude directory not found"
        log_warn "Mount your Claude credentials: -v ~/.claude:/home/ccflow/.claude"
        return 1
    fi

    # Check if we can run claude
    if ! command -v claude &> /dev/null; then
        log_error "Claude CLI not found in PATH"
        return 1
    fi

    # Try a quick auth check (claude doctor or similar)
    # The CLI will indicate if not authenticated
    if claude --version &> /dev/null; then
        log_info "Claude CLI available: $(claude --version 2>/dev/null || echo 'unknown version')"
        return 0
    else
        log_error "Claude CLI check failed"
        return 1
    fi
}

# -----------------------------------------------------------------------------
# Wait for dependencies
# -----------------------------------------------------------------------------
wait_for_deps() {
    # If using external services (Redis, etc.), wait for them here
    # Currently ccflow is self-contained
    :
}

# -----------------------------------------------------------------------------
# Start API server
# -----------------------------------------------------------------------------
start_server() {
    log_info "Starting ccflow API server..."
    log_info "  API Port: ${CCFLOW_API_PORT:-8080}"
    log_info "  Metrics Port: ${CCFLOW_METRICS_PORT:-9090}"
    log_info "  Log Level: ${CCFLOW_LOG_LEVEL:-INFO}"

    exec python -m ccflow.server \
        --host 0.0.0.0 \
        --port "${CCFLOW_API_PORT:-8080}" \
        --metrics-port "${CCFLOW_METRICS_PORT:-9090}" \
        --log-level "${CCFLOW_LOG_LEVEL:-INFO}"
}

# -----------------------------------------------------------------------------
# Start pool worker
# -----------------------------------------------------------------------------
start_worker() {
    log_info "Starting ccflow pool worker..."
    log_info "  Workers: ${CCFLOW_POOL_WORKERS:-4}"

    exec python -m ccflow.worker \
        --workers "${CCFLOW_POOL_WORKERS:-4}"
}

# -----------------------------------------------------------------------------
# Interactive shell
# -----------------------------------------------------------------------------
start_shell() {
    log_info "Starting interactive shell..."
    exec /bin/bash
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
main() {
    log_info "ccflow container starting..."

    # Check Claude CLI (warn but don't fail - might be API-only mode)
    if ! check_claude_auth; then
        log_warn "Claude CLI not authenticated - CLI features will not work"
        log_warn "To authenticate, run: docker exec -it <container> claude login"
    fi

    # Wait for any dependencies
    wait_for_deps

    # Handle commands
    case "${1:-serve}" in
        serve)
            start_server
            ;;
        worker)
            start_worker
            ;;
        shell)
            start_shell
            ;;
        claude)
            # Pass through to claude CLI
            shift
            exec claude "$@"
            ;;
        python)
            # Pass through to python
            shift
            exec python "$@"
            ;;
        *)
            # Run arbitrary command
            exec "$@"
            ;;
    esac
}

main "$@"
