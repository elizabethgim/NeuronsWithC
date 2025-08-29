#include <stdio.h>
#include <math.h>
#include "myann.h"

const int INPUT_NODES = 2;
const int HIDDEN_NODES = 2;
const int OUTPUT_NODES = 2;

double input[INPUT_NODES] = {0.05, 0.10};
double target[OUTPUT_NODES] = {0.01, 0.99};
double hidden[HIDDEN_NODES];
double output[OUTPUT_NODES];

double weightH[INPUT_NODES][HIDDEN_NODES] = {{0.15, 0.25}, {0.20, 0.30}};
double biasH[HIDDEN_NODES] = {0.35, 0.35};

double weightO[INPUT_NODES][OUTPUT_NODES] = {{0.40, 0.50}, {0.45, 0.55}};
double biasO[OUTPUT_NODES] = {0.60, 0.60};

double output_b[OUTPUT_NODES];
double input_b[INPUT_NODES];
double DoutputE[OUTPUT_NODES];

double hidden_b[HIDDEN_NODES];

double DweightE[INPUT_NODES][HIDDEN_NODES];
double DbiasE[HIDDEN_NODES];

double learning_rate = 0.05;

int main(){
    for(long epoch=1; epoch<=1000000;epoch++){
        feed_forward(input, (const double*)weightH, biasH, hidden, INPUT_NODES, HIDDEN_NODES, SIGMOID);
        feed_forward(hidden, (const double*)weightO, biasO, output, HIDDEN_NODES, OUTPUT_NODES, SIGMOID);
        
        double Error = get_error(target, output, OUTPUT_NODES);

        if(Error < 0.0001){
            printf("epoch = %d\n", epoch);
            printf("Error = %f\n", Error);
            printf("output[0] = %f\n", output[0]);
            printf("output[1] = %f\n", output[1]);
        }

        get_DoutputE(target, output, DoutputE, OUTPUT_NODES);

        prepare_back_propagation(DoutputE, output, output_b, OUTPUT_NODES, SIGMOID);
        back_propagation(output_b, (const double*)weightO, hidden, hidden_b, OUTPUT_NODES, INPUT_NODES, SIGMOID);
        
        get_gradient((double *)DweightE, DbiasE, hidden_b, input, INPUT_NODES, HIDDEN_NODES);
        apply_gradient((double *)DweightE, DbiasE, learning_rate, (double *)weightH, biasH, INPUT_NODES, OUTPUT_NODES);
    }
    
    return 0;
}