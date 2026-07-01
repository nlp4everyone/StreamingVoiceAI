COMPOSE_FILE := docker/docker-compose.yml
PROJECT_NAME := streaming_stt
SERVICE ?=
LINES ?= 100

.PHONY: up build down restart logs ps

build:
	@command -v docker >/dev/null 2>&1 || { echo "Docker is not installed"; exit 1; }
	sudo docker compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) build

up: build
	@sudo docker compose version >/dev/null 2>&1 || { echo "Docker Compose plugin is not installed"; exit 1; }
	@sudo docker info >/dev/null 2>&1 || { echo "Docker daemon is not running"; exit 1; }
	@nvidia-smi >/dev/null 2>&1 || echo "[WARN] nvidia-smi not found — GPU may not be available for the ray service"
	sudo docker compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) up -d

down:
	sudo docker compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) down

restart: down up

logs:
	sudo docker compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) logs -f --tail=$(LINES) $(SERVICE)

ps:
	sudo docker compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) ps
