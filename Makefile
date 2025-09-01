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
.PHONY: day2-setup day2-data day2-bench day2-sanity day2-full

day2-setup:
	@echo "Setting up directories for Day 2..."
	@mkdir -p data/flickr8k/processed
	@mkdir -p reports
	@mkdir -p logs

day2-data: day2-setup
	@echo "Downloading and preparing Flickr8k data..."
	python scripts/download_prepare_data.py

day2-sanity: day2-data
	@echo "Running sanity check training..."
	python -m chimera.train.train_sanity

day2-bench: day2-data
	@echo "Benchmarking dataloader..."
	python scripts/bench_dataloader.py

# Target principal para ejecutar todo el Día 2
day2-full: day2-data day2-sanity
	@echo "Day 2 complete! Check MLflow UI and reports/ for results."


# Day 3: Image Encoder targets
.PHONY: day3-bench-image-encoder day3-full

day3-bench-image-encoder: day2-data
	@echo "Benchmarking image encoder..."
	python scripts/bench_image_encoder.py --batch-size 64 --visualize-arch --log-mlflow
	@echo "Benchmark report generated at reports/bench_image_encoder.md"
	@echo "Architecture summary generated at reports/architecture_image_encoder.md"
	@echo "Results logged to MLflow for reproducibility"

day3-full: day3-bench-image-encoder
	@echo "Day 3 complete! Image encoder benchmarked and tested."

bench-image-encoder: day3-bench-image-encoder

# Day 4: Text Encoder targets
.PHONY: day4-bench-text-encoder day4-full

day4-bench-text-encoder: day2-data
	@echo "Benchmarking text encoder..."
	python scripts/bench_text_encoder.py --batch-size 64 --visualize-arch --log-mlflow ; \
	if [ -f reports/architecture_text_encoder.md ]; then \
		echo "Architecture summary generated at reports/architecture_text_encoder.md"; \
	else \
		echo "Architecture summary FAILED (file not found)"; \
	fi
	@echo "Benchmark report generated at reports/bench_text_encoder.md"
	@echo "Results logged to MLflow for reproducibility"

day4-full: day4-bench-text-encoder
	@echo "Day 4 complete! Text encoder benchmarked and tested."

bench-text-encoder: day4-bench-text-encoder

# Day 5: Contrastive loss
.PHONY: day4-bench-text-encoder day4-full

day5-bench-contrastive-loss:
	@echo "Benchmarking contrastive loss..."
	python scripts/bench_contrastive_loss.py --batch-size 64 --log-mlflow
	@echo "Benchmark report generated at reports/bench_contrastive_loss.md"
	@echo "Results logged to MLflow for reproducibility"

day5-full: day5-bench-contrastive-loss
	@echo "Day 5 complete! Contrastive loss benchmarked and tested."


# Day 6: Training targets
.PHONY: day6-train day6-monitor day6-full

day6-train:
	@echo "Starting training for one epoch..."
	python scripts/run_training.py

day6-monitor:
	@echo "Running GPU monitoring..."
	nvidia-smi dmon -s pucm -d 2 -o TD -f logs/dmon_train.log

day6-analyze-metrics:
	@echo "Analyzing training metrics..."
	@echo "GPU Utilization from dmon_train.log:" \
	&& grep -v "#" logs/dmon_train.log | awk '{ sum_sm += $$6; count++ } END { print "Average SM%: " sum_sm/count "%" }'
	@echo "Check MLflow UI for complete metrics visualization"

day6-full: day6-train
	@echo "Day 6 complete! Check MLflow UI and logs for results"
	@echo "Run 'make day6-analyze-metrics' to analyze GPU utilization"