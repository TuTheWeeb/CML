
CC=gcc

SRC=src
BUILD=build

FLAGS=-g -fopenmp -O3 -Wall -Werror 

.PHONY: build run

build: src/*
	$(CC) $(FLAGS) $(SRC)/main.c -o $(BUILD)/main


run: build/main
	build/main
