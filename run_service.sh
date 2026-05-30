#!/bin/bash
set -e

COMPOSE_FILE="docker/docker-compose.yml"
PROJECT_NAME="streaming_stt"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()   { echo -e "${GREEN}[INFO]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

command -v docker >/dev/null 2>&1       || error "Docker is not installed"
sudo docker compose version >/dev/null 2>&1  || error "Docker Compose plugin is not installed"
sudo docker info >/dev/null 2>&1             || error "Docker daemon is not running"
nvidia-smi >/dev/null 2>&1              || warn "nvidia-smi not found — GPU may not be available for the ray service"

log "Building images..."
sudo docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" build

log "Starting services..."
sudo docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" up -d

