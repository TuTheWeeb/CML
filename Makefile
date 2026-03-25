
CC=gcc

SRC=src
BUILD=build

FLAGS=-fopenmp -O3 -Wall -Werror -march=native

.PHONY: build run

build: src/*
	$(CC) $(FLAGS) $(SRC)/main.c -o $(BUILD)/main


run: build/main
	build/main
