init:
	@pip3 install -r requirements.txt
test:
	@rm -rf tests/__pycache__/
	@PYTHONPATH=. pytest -q --durations=10 --cov=. --cov-report term --cov-fail-under 75 tests/
lint:
	@pylint --fail-under=4.5 *.py

clean:
	@rm -rf *.jpg

.PHONY: init test clean
