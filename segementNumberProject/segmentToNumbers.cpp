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
	for(int epoch=1;epoch<=1000;epoch++){
		for(int p = 0;p<PATTERN_COUNT;p++){	
			feed_forward(input[p], (const double*)weightH, biasH, hidden, INPUT_NODES, HIDDEN_NODES, SIGMOID);
			feed_forward(hidden, (const double*)weightO, biasO, output[p], HIDDEN_NODES, OUTPUT_NODES, SIGMOID);
			
			double Error = get_error(target[p], output[p], OUTPUT_NODES, MSE);

			// if(Error < 0.0001){
			// 	printf("epoch = %d\n", epoch);
			// 	printf("Error = %f\n", Error);
				
			// 	for(int i=0;i<OUTPUT_NODES;i++){
			// 		printf("output[%d] = %f\n", i, output[i]);
			// 	}
			// 	break;
			// }
			

			get_DoutputE(target[p], output[p], DoutputE, OUTPUT_NODES);

			prepare_back_propagation(DoutputE, output[p], output_b, OUTPUT_NODES, SIGMOID);
			back_propagation(output_b, (const double*)weightO, hidden, hidden_b, OUTPUT_NODES, INPUT_NODES, SIGMOID);
			
			get_gradients((double *)DweightOE, (double *)DbiasOE, output_b, hidden, HIDDEN_NODES, OUTPUT_NODES);
			get_gradients((double *)DweightHE, (double *)DbiasHE, hidden_b, input[p], INPUT_NODES, HIDDEN_NODES);

			double learning_rate = 0.5;

			apply_gradient((double *)DweightOE, (double *)DbiasOE, learning_rate, (double *)weightO, biasO, HIDDEN_NODES, OUTPUT_NODES);
			apply_gradient((double *)DweightHE, (double *)DbiasHE, learning_rate, (double *)weightH, biasH, INPUT_NODES, HIDDEN_NODES);

		}
		if(epoch%100==0) printf(".");
	}
	printf("\n");

	for(int pc=0;pc<PATTERN_COUNT;pc++){
		printf("target %d : ", pc);
		for(int on=0;on<OUTPUT_NODES;on){
			printf("%.0f ", target[pc][on]);
		}
		printf("pattern %d : ", pc);
		for(int on=0;on<OUTPUT_NODES;on++){
			printf("%.2f ",output[pc][on]);
		}
	}
}