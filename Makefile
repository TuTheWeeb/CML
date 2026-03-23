
CC=gcc

SRC=src
BUILD=build

.PHONY: build run

build:
	$(CC) -g -fopenmp $(SRC)/main.c -o $(BUILD)/main

run:
	build/main
