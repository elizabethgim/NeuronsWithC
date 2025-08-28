#include <stdio.h>
#include <math.h>



double compute(double x1, double w1, double x2, double w2, double b) {
	return w1 * x1 + w2 * x2 + b;
}

int main() {
	double i1 = 0.05, i2 = 0.10;
	double t1 = 0.01, t2 = 0.99;
    

	double w1 = 0.15, w3 = 0.25;
	double w2 = 0.20, w4 = 0.30;

	double w5 = 0.40, w7 = 0.50;
    double w6 = 0.45, w8 = 0.55;

    double b1 = 0.35, b2 = 0.35;
    double b3 = 0.60, b4 = 0.60;

	for (int i = 0; i < 20; i++) {

		double h1 = compute(i1, w1, i2, w2, b1);
		double h2 = compute(i1, w3, i2, w4, b2);

		double o1 = compute(h1, w5, h2, w6, b3);
        double o2 = compute(h1, w7, h2, w8, b4);

		double error1 = 0.5 * pow(o1 - t1, 2);
        double error2 = 0.5 * pow(o2 - t2, 2);

        double error = error1 + error2;

        printf("h1: %.4f\t h2: %.4f\no1: %.4f\t o2: %.4f\ne1: %.4f\t e2: %.4f\nE: %.4f", h1, h2, o1, o2, error1, error2, error);
		double learning_rate = 0.05;

		double ob1 = o1 - t1;

		double dw5e = h1 * ob1;
		double dw6e = h2 * ob1;

		double dw1e = i1 * w5 * ob1;
		double dw2e = i2 * w5 * ob1;
		double dw3e = i1 * w6 * ob1;
		double dw4e = i2 * w6 * ob1;

		w6 = w6 - learning_rate * dw6e;
		w5 = w5 - learning_rate * dw5e;
		w4 = w4 - learning_rate * dw4e;
		w3 = w3 - learning_rate * dw3e;
		w2 = w2 - learning_rate * dw2e;
		w1 = w1 - learning_rate * dw1e;

		printf("o1: %f\t o2: %f\n", o1, o2);
		printf("w1: %f, w2: %f, w3: %f, w4: %f, w5: %f, w6: %f\n", w1, w2, w3, w4, w5, w6);
	}
	
	

	return 0;
}