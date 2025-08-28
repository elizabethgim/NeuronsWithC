#include <stdio.h>
#include "neuron.h"

const int PATTERN_COUNT = 10;
const int HIDDEN_NODES = 8;

const int OUTPUT_NODES=4;
const int INPUT_NODES=9;

double input[PATTERN_COUNT][INPUT_NODES] = {
	{1, 1, 1, 1, 1, 1, 0}, // 0
	{0, 1, 1, 0, 0, 0, 0}, // 1
	{1, 1, 0, 1, 1, 0, 1}, // 2
	{1, 1, 1, 1, 0, 0, 1}, // 3
	{0, 1, 1, 0, 0, 1, 1}, // 4
	{1, 0, 1, 1, 0, 1, 1}, // 5
	{0, 0, 1, 1, 1, 1, 1}, // 6
	{1, 1, 1, 0, 0, 0, 0}, // 7
	{1, 1, 1, 1, 1, 1, 1}, // 8
	{1, 1, 1, 1, 0, 0, 1} // 9
};

double target[PATTERN_COUNT][INPUT_NODES] = {
	{0, 0, 0, 0},
	{0, 0, 0, 1},
	{0, 0, 1, 0},
	{0, 0, 1, 1},
	{0, 1, 0, 0},
	{0, 1, 0, 1},
	{0, 1, 1, 0},
	{0, 1, 1, 1},
	{1, 0, 0, 0},
	{1, 0, 0, 1},
};   

double hidden[HIDDEN_NODES];
double output[OUTPUT_NODES];
double weight[INPUT_NODES][HIDDEN_NODES];
double bias[HIDDEN_NODES];

double output_backpropagation[OUTPUT_NODES];
double input_backpropagation[INPUT_NODES];

int main(){
	try {
		forward(*input, *weight, bias, output, INPUT_NODES, OUTPUT_NODES, LINEAR);
		return 0;
	} catch(const char* error){
		printf(error);
	}
}