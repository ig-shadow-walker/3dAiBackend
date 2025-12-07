#!/bin/bash

# Run script for 3D Generative Models Backend (Single Worker Mode)
# This script starts the FastAPI server with embedded scheduler
#
# For multi-worker deployments with separate scheduler service, use:
#   ./scripts/run_multiworker.sh
#
# Usage:
#   ./scripts/run_server.sh [OPTIONS]
#
# Options:
#   --user-auth-enabled     Enable user authentication (default: false)
#   --debug                 Enable debug mode (default: false)
#   --help                  Show this help message

set -e

# Parse command line arguments
USER_AUTH_ENABLED="false"
DEBUG_MODE="false"

while [[ $# -gt 0 ]]; do
    case $1 in
        --user-auth-enabled)
            USER_AUTH_ENABLED="true"
            shift
            ;;
        --debug)
            DEBUG_MODE="true"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --user-auth-enabled     Enable user authentication (default: false)"
            echo "  --debug                 Enable debug mode (default: false)"
            echo "  --help                  Show this help message"
            echo ""
            echo "Environment Variables:"
            echo "  P3D_HOST               Server host address (default: 0.0.0.0)"
            echo "  P3D_PORT               Server port (default: 7842)"
            echo "  P3D_RELOAD             Enable auto-reload for development (default: false)"
            echo "  P3D_LOG_LEVEL          Logging level (default: info)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "🚀 Starting 3D Generative Models Backend (Single Worker Mode)..."

# Check if configuration files exist
if [ ! -f "config/system.yaml" ]; then
    echo "❌ Configuration file config/system.yaml not found"
    echo "Please run ./scripts/setup.sh to create configuration files"
    exit 1
fi

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Essential configuration parameters
export P3D_USER_AUTH_ENABLED="$USER_AUTH_ENABLED"
export P3D_DEBUG="$DEBUG_MODE"

# optionally enable this if thumbnail rendering is INCORRECT
# Xvfb :2 -screen 0 1024x768x16 & 
# export DISPLAY=:2

# Default values
HOST=${P3D_HOST:-"0.0.0.0"}
PORT=${P3D_PORT:-7842}
RELOAD=${P3D_RELOAD:-"false"}
LOG_LEVEL=${P3D_LOG_LEVEL:-"info"}

echo "📋 Configuration:"
echo "   Host: $HOST"
echo "   Port: $PORT"
echo "   Reload: $RELOAD"
echo "   Log Level: $LOG_LEVEL"
echo "   User Auth: $USER_AUTH_ENABLED"
echo "   Debug Mode: $DEBUG_MODE"
echo ""

# Start server based on environment
if [ "$RELOAD" = "true" ]; then
    echo "🔄 Starting development server with auto-reload..."
    uvicorn api.main:app \
        --host "$HOST" \
        --port "$PORT" \
        --reload \
        --log-level "$LOG_LEVEL"
else
    echo "🚀 Starting production server (current ONLY a single worker supported)..."
    # Single worker
    uvicorn api.main:app \
        --host "$HOST" \
        --port "$PORT" \
        --log-level "$LOG_LEVEL"

fi
