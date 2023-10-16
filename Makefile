format:
	ruff .
	black .
	mypy .

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

lm:
	python3 lfd_assignment3_lm.py -s -t data/test.txt

lstm:
	python3 lfd_assignment3_lstm.py -s -t data/test.txt