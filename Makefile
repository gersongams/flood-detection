.PHONY: install redis worker run stop clean help

# Default target
help:
	@echo "Flood Risk Detector - Available commands:"
	@echo ""
	@echo "  make worker     - Start Celery worker (terminal 1)"
	@echo "  make run        - Start Streamlit app (terminal 2)"
	@echo ""
	@echo "  make install    - Install dependencies"
	@echo "  make stop       - Stop all processes"
	@echo "  make clean      - Clean up data and cache files"
	@echo ""

# Install dependencies
install:
	uv sync

# Start Redis using Docker
redis:
	@docker compose up -d

# Run Celery worker (foreground with logs) - run in terminal 1
# Uses --pool=solo to avoid fork() which breaks MPS (Apple GPU)
worker: redis
	uv run celery -A app.celery_app worker -l info --pool=solo

# Run Streamlit app (foreground with logs) - run in terminal 2
run: redis
	uv run streamlit run app/main.py

# Stop all processes
stop:
	@pkill -f "streamlit run" || true
	@pkill -f "celery -A app.celery_app" || true
	@docker compose down || true
	@echo "Stopped"

# Clean up generated files
clean:
	@rm -rf data/images/* data/models/*
	@rm -rf __pycache__ app/__pycache__ app/**/__pycache__
	@echo "Cleaned"
