.PHONY: setup check test lint verify-data
setup:        ## editable install of the package
	pip install -e .
check:        ## verify the environment (GPU, libs, paths)
	python scripts/check_env.py
test:
	pytest -q
lint:
	ruff check src tests scripts
verify-data:  ## verify MRNet folder layout and labels
	python scripts/verify_data.py
