#include <stdio.h>
#include "neuron.h"

const int PATTERN_COUNT = 10;
const int HIDDEN_NODES = 8;

const int INPUT_NODES=7;
const int OUTPUT_NODES=4;

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

double output_b[OUTPUT_NODES];
double input_b[INPUT_NODES];

double DweightE[INPUT_NODES][HIDDEN_NODES];
double DbiasE[INPUT_NODES][HIDDEN_NODES];

double learning_rate = 0.5;
double EPOCH_COUNT = 10000;

int main(){

	for(int h=0;h<HIDDEN_NODES;h++){
		feed_forward(*input, *weight, bias, output, INPUT_NODES, OUTPUT_NODES, LINEAR);
		back_propagation(output_b, (const double *)weight, *input, input_b, OUTPUT_NODES, INPUT_NODES, SIGMOID);
	}

}