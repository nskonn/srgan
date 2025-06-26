.PHONY: lint format check-types security-check

lint:
	@echo "Running flake8..."
	flake8 src/
	@echo "Running pylint..."
	pylint src/

format:
	@echo "Running isort..."
	isort src/
	@echo "Running black..."
	black src/

check-types:
	@echo "Running mypy..."
	mypy src/

security-check:
	@echo "Running bandit..."
	bandit -r src/

ci-check: lint check-types security-check