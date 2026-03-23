
SRC=src
BUILD=build

.PHONY: build run

build:
	gcc -g -fopenmp $(SRC)/main.c -o $(BUILD)/main

run:
	build/main
