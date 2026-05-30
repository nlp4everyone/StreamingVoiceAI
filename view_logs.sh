#!/bin/bash

COMPOSE_FILE="docker/docker-compose.yml"
PROJECT_NAME="streaming_stt"

SERVICE="${1:-}"
LINES="${2:-100}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()   { echo -e "${GREEN}[INFO]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

command -v docker >/dev/null 2>&1 || error "Docker is not installed"

usage() {
    echo "Usage: $0 [service] [lines]"
    echo ""
    echo "  service   Service name to tail (default: all services)"
    echo "            Available: web"
    echo "  lines     Number of lines to show initially (default: 100)"
    echo ""
    echo "Examples:"
    echo "  $0               # tail all services"
    echo "  $0 web           # tail web service only"
    echo "  $0 web 200       # tail web service, last 200 lines"
}

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
    exit 0
fi

if [[ -n "$SERVICE" ]]; then
    log "Tailing logs for service: $SERVICE (last $LINES lines)"
    sudo docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" logs -f --tail="$LINES" "$SERVICE"
else
    log "Tailing logs for all services (last $LINES lines)"
    sudo docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" logs -f --tail="$LINES"
fi
