.PHONY: install lint test lock

install:
	@echo "Installing dependencies from lockfile..."
	conda-lock install -n chimera --dev

lint:
	@echo "Running linters..."
	ruff check .
	black .
	isort .

test:
	@echo "Running tests..."
	pytest

lock:
	@echo "Generating conda-lock.yml..."
	conda-lock -f environment.yml -p linux-64 --lockfile conda-lock.yml