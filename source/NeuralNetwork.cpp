#include <stdio.h>
#include <math.h>

const int INPUT_NODE = 2;
const int OUTPUT_NODE = 3;

double input[INPUT_NODE] = {1, 2};
double output[OUTPUT_NODE];
double weight[INPUT_NODE][OUTPUT_NODE] = {{1, 2, 3}, { 4, 5, 6}};
double bias[OUTPUT_NODE] = {7, 8, 9};

double output_b[OUTPUT_NODE];

typedef enum _activation{
    LINEAR = 0,
    SIGMOID = 1,
    RELU = 2
} activation_t;

void feed_forward(
    const double *input,
    const double *weight,
    const double *bias,
    double *output,
    const int OUTPUT_NODES,
    const int INPUT_NODES,
    activation_t act
) {

    for(int j=0;j<OUTPUT_NODES;j++){
        double prediction = 0;

        for(int i = 0; i<INPUT_NODES;i++){
            prediction += input[i] * weight[i*OUTPUT_NODES+j];
        }
        
        prediction += bias[j];
        output[j] = prediction;

         if(act==LINEAR){
            output[j] = prediction;
        }else if(act == SIGMOID){
            output[j] = 1.0 /(1.0+exp(-prediction));
        }else if(act==RELU){
            output[j] = prediction > 0? prediction : 0;
        }

        printf("output[%d] = %f\n", j, output[j]);
    }
}

void back_propagation(
    const double *output_b, // feed_forward output
    const double * weight,
    double *input_f, // feed_forward input
    double *input_b, // backpropagation output
    const int OUTPUT_NODES,
    const int INPUT_NODES,
    activation_t act
){
    for(int j=0; j<INPUT_NODES;j++){
        double sum =0.0;
        for(int i=0;i<OUTPUT_NODES;i++){
            sum += output_b[j] * weight[j*OUTPUT_NODES+i];
        }

        if(act==LINEAR){
            input_b[j] = sum;
        }else if(act == SIGMOID){
            input_b[j] = input_f[j] * (1 - input_f[j]) *sum ;
        }else if(act==RELU){
            input_b[j] = (input_f[j] > 0? 1 : 0) * sum;
        }

        printf("input_b[%d] = %f\n", j, input_b[j]);
    }
}

int main(){
    feed_forward(input, (const double*)weight, bias, output, OUTPUT_NODE, INPUT_NODE, SIGMOID);
    back_propagation(output, (const double*)weight, input, input, OUTPUT_NODE, INPUT_NODE, SIGMOID);
    return 0;
}