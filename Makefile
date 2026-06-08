.PHONY: install install-dev train lint format test clean
install:
	pip install -r requirements.txt
install-dev: install
	pip install pytest pytest-cov black isort flake8
train:
	python scripts/feature_engineering.py && python scripts/train_model.py
lint:
	flake8 scripts/ tests/ --max-line-length=100 --ignore=E501,W503
format:
	black --line-length 100 scripts/ tests/
test:
	pytest tests/ -v --cov=scripts --cov-report=term-missing
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage htmlcov/ .pytest_cache/ models/