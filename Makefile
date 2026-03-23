
CC=gcc

SRC=src
BUILD=build

FLAGS=-fopenmp -O3 -march=native

.PHONY: build run

build:
	$(CC) $(FLAGS) $(SRC)/main.c -o $(BUILD)/main


run:
	build/main
