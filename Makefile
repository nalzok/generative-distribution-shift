.PHONY: spiral mnist

spiral:
	pipenv run python3 -m spiral.train

mnist:
	pipenv run python3 -m mnist.train
