# Convenience targets. Run `make help` to list them.
.DEFAULT_GOAL := help
PYTHON ?= python

.PHONY: help setup data train evaluate predict clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## Install Python dependencies
	$(PYTHON) -m pip install -r requirements.txt

data:  ## Generate the synthetic sample dataset
	$(PYTHON) tools/generate_sample_data.py --out data/sample --per-class 150

train:  ## Train using configs/default.yaml
	$(PYTHON) train.py --config configs/default.yaml

evaluate:  ## Evaluate the most recent run (override with RUN=outputs/...)
	$(PYTHON) evaluate.py --run-dir $(or $(RUN),$(shell ls -dt outputs/*/ | head -1))

clean:  ## Remove generated data and training outputs
	rm -rf data/sample outputs
