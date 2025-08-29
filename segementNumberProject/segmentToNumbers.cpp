#include <stdio.h>
#include "../source/myann.h"

const int PATTERN_COUNT = 10;
const int INPUT_NODES=7;
const int HIDDEN_NODES = 8;
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
double weightH[INPUT_NODES][HIDDEN_NODES];
double biasH[HIDDEN_NODES];
double weightO[HIDDEN_NODES][HIDDEN_NODES];
double biasO[OUTPUT_NODES];

double output_b[OUTPUT_NODES];
double hidden_b[HIDDEN_NODES];

double DoutputE[OUTPUT_NODES];
double DweightOE[INPUT_NODES][HIDDEN_NODES];
double DbiasOE[INPUT_NODES][HIDDEN_NODES];

double DweightHE[INPUT_NODES][HIDDEN_NODES];
double DbiasHE[INPUT_NODES][HIDDEN_NODES];



int main(){
	for(int epoch=1;epoch<=1000000;epoch++){
		feed_forward(input[2], (const double*)weightH, biasH, hidden, INPUT_NODES, HIDDEN_NODES, SIGMOID);
		feed_forward(hidden, (const double*)weightO, biasO, output, HIDDEN_NODES, OUTPUT_NODES, SIGMOID);
		
		double Error = get_error(target[2], output, OUTPUT_NODES, MSE);

		if(Error < 0.0001){
			printf("epoch = %d\n", epoch);
			printf("Error = %f\n", Error);
			
			for(int i=0;i<OUTPUT_NODES;i++){
				printf("output[%d] = %f\n", i, output[i]);
			}
			break;
		}
		

		get_DoutputE(target[2], output, DoutputE, OUTPUT_NODES);

		prepare_back_propagation(DoutputE, output, output_b, OUTPUT_NODES, SIGMOID);
		back_propagation(output_b, (const double*)weightO, hidden, hidden_b, OUTPUT_NODES, INPUT_NODES, SIGMOID);
		
		get_gradients((double *)DweightOE, (double *)DbiasOE, output_b, hidden, HIDDEN_NODES, OUTPUT_NODES);
		get_gradients((double *)DweightHE, (double *)DbiasHE, hidden_b, input[2], INPUT_NODES, HIDDEN_NODES);

		double learning_rate = 0.5;

		apply_gradient((double *)DweightOE, (double *)DbiasOE, learning_rate, (double *)weightO, biasO, HIDDEN_NODES, OUTPUT_NODES);
		apply_gradient((double *)DweightHE, (double *)DbiasHE, learning_rate, (double *)weightH, biasH, INPUT_NODES, HIDDEN_NODES);

		
	}
}