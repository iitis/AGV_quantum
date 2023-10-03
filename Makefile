init:
	@pip3 install -r requirements.txt
test:
	@rm -rf tests/__pycache__/
	@PYTHONPATH=. pytest -q --durations=10 -n auto --cov=. --cov-report term --cov-fail-under 85 tests/
lint:
	@pylint --fail-under=9 *.py

clean:
	@rm -rf *.jpg

.PHONY: init test clean
