#include <stdio.h>
#include "../source/myann.h"

const int PATTERN_COUNT = 10;
const int INPUT_NODES= 7;
const int HIDDEN_NODES = 8;
const int OUTPUT_NODES= 4;

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


double target[PATTERN_COUNT][OUTPUT_NODES] = {
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
double output[PATTERN_COUNT][OUTPUT_NODES];
double weightH[INPUT_NODES][HIDDEN_NODES];
double biasH[HIDDEN_NODES];
double weightO[HIDDEN_NODES][OUTPUT_NODES];
double biasO[OUTPUT_NODES];

double output_b[OUTPUT_NODES];
double hidden_b[HIDDEN_NODES];

double DoutputE[OUTPUT_NODES];
double DweightOE[HIDDEN_NODES][OUTPUT_NODES];
double DbiasOE[OUTPUT_NODES];

double DweightHE[INPUT_NODES][HIDDEN_NODES];
double DbiasHE[HIDDEN_NODES];

int shuffled_pattern[PATTERN_COUNT];

void print_weight(
	const double* weightH,
	const int INPUT_NODES,
	const int HIDDEN_NODES
);

void print_result(
	const double *target,
	const double *output
);

int main(){
	initialize_weight((double *)weightH, biasH, INPUT_NODES, OUTPUT_NODES);
	initialize_weight((double *)weightO, biasO, INPUT_NODES, OUTPUT_NODES);

	for(int pc=0;pc<PATTERN_COUNT;pc++){
		shuffled_pattern[pc] = pc;
	}

	for(int epoch=1;epoch<=1000;epoch++){

		int tmp_a = 0;
		int tmp_b = 0;

		for(int pc=0;pc<PATTERN_COUNT;pc++){
			tmp_a=rand()%PATTERN_COUNT;
			tmp_b=shuffled_pattern[pc];
			shuffled_pattern[pc] = shuffled_pattern[tmp_a];
			shuffled_pattern[tmp_a] = tmp_b;
		}

		double sum_error = 0.;
		for(int p = 0;p<PATTERN_COUNT;p++){	
			feed_forward(input[p], (const double*)weightH, biasH, hidden, INPUT_NODES, HIDDEN_NODES, SIGMOID);
			feed_forward(hidden, (const double*)weightO, biasO, output[p], HIDDEN_NODES, OUTPUT_NODES, SIGMOID);
			
			double Error = get_error(target[p], output[p], OUTPUT_NODES, MSE);
			sum_error += Error;
			
			get_DoutputE(target[p], output[p], DoutputE, OUTPUT_NODES);

			prepare_back_propagation(DoutputE, output[p], output_b, OUTPUT_NODES, SIGMOID);
			back_propagation(output_b, (const double*)weightO, hidden, hidden_b, OUTPUT_NODES, INPUT_NODES, SIGMOID);
			
			get_gradients((double *)DweightOE, (double *)DbiasOE, output_b, hidden, HIDDEN_NODES, OUTPUT_NODES);
			get_gradients((double *)DweightHE, (double *)DbiasHE, hidden_b, input[p], INPUT_NODES, HIDDEN_NODES);

			double learning_rate = 0.5;

			apply_gradient((double *)DweightOE, (double *)DbiasOE, learning_rate, (double *)weightO, biasO, HIDDEN_NODES, OUTPUT_NODES);
			apply_gradient((double *)DweightHE, (double *)DbiasHE, learning_rate, (double *)weightH, biasH, INPUT_NODES, HIDDEN_NODES);
			
			
		}
		
		# define CNT_LOOP 100
		static int cnt_loop = CNT_LOOP;
		cnt_loop --;
		if(cnt_loop==0)
			cnt_loop = CNT_LOOP;
		else
			continue;
		
		printf("sum error: %f\n", sum_error);

		print_weight((double *)weightH, INPUT_NODES, HIDDEN_NODES);

		if(sum_error < 0.0004) break;
	}

	print_result((double *)target, (double *)output);
}

void print_weight(
	const double* weightH,
	const int INPUT_NODES,
	const int HIDDEN_NODES
){
	for(int i=0;i<INPUT_NODES;i++){
		for(int j=0;j<HIDDEN_NODES;j++){
			printf("%7.3f ", weightH[i*HIDDEN_NODES+j]);
		}
		printf("\n");
	}
}

void print_result(
	const double *target,
	const double *output
){
	printf("\n");

	for(int pc=0;pc<PATTERN_COUNT;pc++){
		printf("target %d : ", pc);
		for(int on=0;on<OUTPUT_NODES;on++){
			printf("%.0f ", target[pc*OUTPUT_NODES+on]);
		}

		printf("pattern %d : ", pc);
		for(int on=0;on<OUTPUT_NODES;on++){
			printf("%.2f ",output[pc*OUTPUT_NODES+on]);
		}
		printf("\n");
	}
}