#include <stdio.h>

const int INPUT_NODE = 2;
const int OUTPUT_NODE = 3;

double input[INPUT_NODE] = {1, 2};
double output[OUTPUT_NODE];
double weight[INPUT_NODE][OUTPUT_NODE] = {{1, 2, 3}, { 4, 5, 6}};
double bias[OUTPUT_NODE] = {7, 8, 9};

void feed_forward(
    const double *input,
    const double (*weight)[OUTPUT_NODE],
    const double *bias,
    double *output,
    const int OUTPUT_NODES,
    const int INPUT_NODES
) {

    for(int j=0;j<OUTPUT_NODES;j++){
        double prediction = 0;
        
        for(int i = 0; i<INPUT_NODES;i++){
            prediction += input[i] * weight[i][j];
        }
        
        prediction += bias[j];
        output[j] = prediction;
        printf("output[%d] = %f\n", j, output[j]);
    }
}

int main(){
    feed_forward(input, weight, bias, output, OUTPUT_NODE, INPUT_NODE);

    return 0;
}