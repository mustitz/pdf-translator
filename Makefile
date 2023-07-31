# This Makefile is used to run pylint for all python source

all: Makefile horse.py
	pylint --rcfile=.pylintrc *.py
