#include <stdio.h>
#include <math.h>

typedef enum _activation{
    LINEAR = 0,
    SIGMOID = 1,
    RELU = 2
} activation_t;

typedef enum _loss{
    MSE = 0,
    CEE = 1
} loss_t;

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

        // printf("output[%d] = %f\n", j, output[j]);
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
            sum += output_b[i] * weight[j*OUTPUT_NODES+i];
        }

        if(act==LINEAR){
            input_b[j] = sum;
        }else if(act == SIGMOID){
            input_b[j] = input_f[j] * (1 - input_f[j]) *sum ;
        }else if(act==RELU){
            input_b[j] = (input_f[j] > 0? 1 : 0) * sum;
        }

        // printf("input_b[%d] = %f\n", j, input_b[j]);
    }
}

double get_error(
    const double *target,
    const double *output,
    const int OUTPUT_NODES,
    loss_t loss
){
    double Error=0.0;
    double error[OUTPUT_NODES];
    double sum_error = 0.0;

    for(int i=0;i<OUTPUT_NODES;i++){
        if(loss == MSE){
            error[i] = 0.5 *(output[i]-target[i])*(output[i]-target[i]);
        } else if(loss == CEE){
            error[i] = - target[i]*log(output[i]);
        }
        // printf("error[%d] = %f\n", i, error[i]);
        sum_error += error[i];
    }

    Error = sum_error;
    // printf("Error = %f\n", Error);
    return Error;

}

void get_DoutputE(
    const double *target, 
    const double *output, 
    double *DoutputE, 
    const int OUTPUT_NODES)
{
    for(int i=0;i<OUTPUT_NODES;i++){
        DoutputE[i] = output[i]-target[i];
        // printf("Doutput[%i]=%f\n",i, DoutputE[i]);
    }
}

void prepare_back_propagation(
    const double *DoutputE, 
    const double *output,
    double *output_b,
    const int OUTPUT_NODES,
    activation_t act
){
    for(int i=0;i<OUTPUT_NODES;i++){
        if(act==LINEAR){
            output_b[i] = DoutputE[i];
        }else if(act == SIGMOID){
            output_b[i] = output[i] * (1 - output[i]) * DoutputE[i] ;
        }else if(act==RELU){
            output_b[i] = (output[i] > 0? 1 : 0) * DoutputE[i];
        }
        
        // printf("output_b[%d] = %f\n", i, output_b[i]);

    }
    
}

void get_gradients(
    double *DweightE,
    double *DbiasE,
    const double *output_b,
    const double *input_f,
    const int F_INPUT_NODES,
    const int OUTPUT_NODES
){
    for(int j=0;j<OUTPUT_NODES;j++){
        for(int i=0;i<F_INPUT_NODES;i++){
            DweightE[i*OUTPUT_NODES+j] = input_f[i] * output_b[j];
            // printf("DweightE[%d][%d] = %f\n", i, j, DweightE[i*OUTPUT_NODES+j]);
        }
        DbiasE[j] = 1 * output_b[j];
        // printf("DbiasE[%d]=%f\n", j, DbiasE[j]);
    }
}

void apply_gradient(
    double *DweightE,
    double *DbiasE,
    double learning_rate,
    double *weight,
    double *bias,
    const int INPUT_NODES,
    const int OUTPUT_NODES
){
    for(int j=0; j<OUTPUT_NODES;j++){
        for(int i=0; i<INPUT_NODES;i++){
            weight[i*OUTPUT_NODES+j] -= learning_rate*DweightE[i*OUTPUT_NODES+j];
            // printf("weight[%d][%d]=%f\n", i,j,weight[i*OUTPUT_NODES+j]);
        }
        bias[j] -= learning_rate*DbiasE[j];
        // printf("bias[%d]=%f\n",j, bias[j]);
    }


}

const double INITIAL_WEIGHT_MAX = 0.5;

void initialize_weight(
    double *weight,
    double *bias,
    int INPUT_NODES,
    int OUTPUT_NODES
){
    double rand_num;
    for(int j=0;j<OUTPUT_NODES;j++){
        for(int i=0;i<INPUT_NODES;i++){
            rand_num = double(rand()%1000)/1000;
            weight[i*OUTPUT_NODES+j]=2.0*(rand_num-0.5)*INITIAL_WEIGHT_MAX;
            printf("%6.3f ", weight[i*OUTPUT_NODES+j]);
        }

        rand_num = double(rand()%1000)/1000;
        bias[j] = 2.0*(rand_num-0.5)*INITIAL_WEIGHT_MAX;
        printf("%6.3f\n ",bias[j]);
    }
    printf("\n");
}