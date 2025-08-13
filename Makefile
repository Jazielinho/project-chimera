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

# Day 2: GO/NO-GO #1 (Prueba de Vida) targets
day2-setup:
	@echo "Setting up directories for Day 2..."
	@mkdir -p data/flickr8k/processed
	@mkdir -p reports
	@mkdir -p logs

day2-data:
	@echo "Downloading and preparing Flickr8k data..."
	python scripts/download_prepare_data.py --force

day2-bench:
	@echo "Benchmarking dataloader..."
	python scripts/bench_dataloader.py

day2-sanity:
	@echo "Running sanity check training..."
	python -m chimera.train.train_sanity --monitor_gpu \
		--batch_size 16 \
		--learning_rate 5e-4 \
		--num_workers 8 \
		--grad_accum_steps 8 \
		--embed_dim 256

day2-full: day2-setup day2-data day2-bench day2-sanity
	@echo "Day 2 complete! Check reports/ for results."