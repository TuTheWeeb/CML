
SRC=src
BUILD=build

.PHONY: build run

build:
	gcc -fopenmp $(SRC)/main.c -o $(BUILD)/main

run:
	build/main
