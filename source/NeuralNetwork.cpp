#include <stdio.h>

const int INPUT_NODE = 2;
const int OUTPUT_NODE = 3;

double input[INPUT_NODE] = {1, 2};
double output[OUTPUT_NODE];
double weight[INPUT_NODE][OUTPUT_NODE] = {{1, 2, 3}, { 4, 5, 6}};
double bias[OUTPUT_NODE] = {7, 8, 9};

double output_b[OUTPUT_NODE];

void feed_forward(
    const double *input,
    const double *weight,
    const double *bias,
    double *output,
    const int OUTPUT_NODES,
    const int INPUT_NODES
) {

    for(int j=0;j<OUTPUT_NODES;j++){
        double prediction = 0;

        for(int i = 0; i<INPUT_NODES;i++){
            prediction += input[i] * weight[i*OUTPUT_NODES+j];
        }
        
        prediction += bias[j];
        output[j] = prediction;
        printf("output[%d] = %f\n", j, output[j]);
    }
}

void back_propagation(
    const double *output_b, // feed_forward output
    const double * weight,
    double *input_b, // backpropagation output
    const int OUTPUT_NODES,
    const int INPUT_NODES
){
    for(int i=0; i<INPUT_NODES;i++){
        double sum =0.0;
        for(int j=0;j<OUTPUT_NODES;j++){
            sum += output_b[j] * weight[i*OUTPUT_NODES+j];
        }
        input_b[i] = sum;
        printf("output_b[%d] = %f\n", i, input_b[i]);
    }
}

int main(){
    feed_forward(input, (const double*)weight, bias, output, OUTPUT_NODE, INPUT_NODE);
    back_propagation(output, (const double*)weight, input, OUTPUT_NODE, INPUT_NODE);
    return 0;
}