#include <math.h>
#include "activation_type.h"


void feed_forward(
    const double *input,
    const double *weight,
    const double *bias,
    double *output,
    const int INPUT_NODES,
    const int OUTPUT_NODES,
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
    const double *output_b, // backpropagation input
    const double * weight,
    double *input_f, // feed_forward input
    double *input_b, // backpropagation output
    const int OUTPUT_NODES,
    const int INPUT_NODES,
    activation_t act
){
    for(int j=0; j<OUTPUT_NODES;j++){
        double sum =0.0;
        for(int i=0;i<INPUT_NODES;i++){
            sum += output_b[i] * weight[i*OUTPUT_NODES+j];
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
