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

double weightH[INPUT_NODES][HIDDEN_NODES] = {{0.15, 0.25}, { 0.20, 0.30}};
double biasH[HIDDEN_NODES] = {0.35, 0.35};

double weightO[INPUT_NODES][OUTPUT_NODES] = {{0.40, 0.50}, { 0.45, 0.55}};
double biasO[OUTPUT_NODES] = {0.60, 0.60};

double output_b[OUTPUT_NODES];
double input_b[INPUT_NODES];

int main(){
    feed_forward(input, (const double*)weightH, biasH, hidden, INPUT_NODES, HIDDEN_NODES, SIGMOID);
    feed_forward(hidden, (const double*)weightO, biasO, output, HIDDEN_NODES, OUTPUT_NODES, SIGMOID);
    
    get_error(target, output, OUTPUT_NODES);
    return 0;
}