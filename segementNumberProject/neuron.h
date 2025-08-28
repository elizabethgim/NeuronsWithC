#include <math.h>
#include "activation_type.h"


double setting_output(activation_t act, double prediction){
    if(act==LINEAR){
        return prediction;
    }else if(act == SIGMOID){
        return 1.0 /(1.0+exp(-prediction));
    }else if(act==RELU){
        return prediction > 0? prediction : 0;
    } else{
        throw "Error: Non-valid activation type. Only 'LINEAR', 'SIGMOID', 'RELU' is available.";
    }
}

void forward(
    const double *input,
    const double *weight,
    const double *bias,
    double *output,
    const int OUTPUT_NODES,
    const int INPUT_NODES,
    activation_t act
) {
    double prediction = 0;

    for(int j=0;j<OUTPUT_NODES;j++){
        for(int i = 0; i<INPUT_NODES;i++){
            prediction += input[i] * weight[i*OUTPUT_NODES+j];
        }
        
        prediction += bias[j];

        // output[j] = setting_output(act, prediction);

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