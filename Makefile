# Motion-Amplified Camouflage Breaking
# Unified Makefile for Backend & Frontend Development

.PHONY: all dev run install install-backend install-frontend samples build test clean help

# Environment detection
VENV ?= $(CURDIR)/.venv
PYTHON := $(shell if [ -f "$(VENV)/bin/python" ]; then echo "$(VENV)/bin/python"; else which python3; fi)
UVICORN := $(shell if [ -f "$(VENV)/bin/uvicorn" ]; then echo "$(VENV)/bin/uvicorn"; else which uvicorn; fi)
PIP := $(shell if [ -f "$(VENV)/bin/pip" ]; then echo "$(VENV)/bin/pip"; else which pip; fi)
NPM ?= npm

all: dev

help:
	@echo "Available commands:"
	@echo "  make dev         - Start both backend and frontend servers concurrently"
	@echo "  make install     - Install dependencies for both backend (.venv) and frontend"
	@echo "  make samples     - Run scripts/make_samples.py to precompute demo video samples"
	@echo "  make build       - Build the production frontend distribution"
	@echo "  make test        - Run synthetic pipeline smoke test"
	@echo "  make clean       - Remove cached files and temporary artifacts"

install: install-backend install-frontend

install-backend:
	@echo "[*] Setting up Python virtual environment and dependencies..."
	@if [ ! -d "$(VENV)" ]; then python3 -m venv $(VENV); fi
	@$(VENV)/bin/pip install --upgrade pip
	@$(VENV)/bin/pip install -r backend/requirements.txt

install-frontend:
	@echo "[*] Installing frontend npm packages..."
	@cd frontend && $(NPM) install

# Starts both backend (FastAPI on port 8000) and frontend (Vite on port 5173)
dev run:
	@echo "================================================================="
	@echo "  Starting Motion-Amplified Camouflage Breaking dev servers"
	@echo "  - Backend:  http://localhost:8000 (API & Docs: /docs)"
	@echo "  - Frontend: http://localhost:5173"
	@echo "  Press Ctrl+C to stop both processes."
	@echo "================================================================="
	@trap 'kill 0' EXIT INT TERM; \
	(cd backend && $(UVICORN) app.main:app --reload --port 8000) & \
	(cd frontend && $(NPM) run dev) & \
	wait

samples:
	@echo "[*] Generating precomputed video samples..."
	@$(PYTHON) scripts/make_samples.py

build:
	@echo "[*] Building frontend..."
	@cd frontend && $(NPM) run build

test:
	@echo "[*] Running synthetic pipeline test..."
	@$(PYTHON) test_pipeline.py

clean:
	@echo "[*] Cleaning bytecode and build artifacts..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@rm -rf frontend/dist
