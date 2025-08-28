#include <stdio.h>
#include <math.h>



double compute(double x1, double w1, double x2, double w2) {
	return w1 * x1 + w2 * x2;
}

int main() {
	double i1 = 2, i2 = 3;
	double t = 1;

	double w1 = 0.11, w3 = 0.12;
	double w2 = 0.21, w4 = 0.08;

	double w5 = 0.14, w6 = 0.15;


	for (int i = 0; i < 20; i++) {
		//double h1 = i1 * w1 + i2 * w2;
	//double h2 = i2 * w3 + i2 * w4;
		double h1 = compute(i1, w1, i2, w2);
		double h2 = compute(i1, w3, i2, w4);

		double o = compute(h1, w5, h2, w6);
		double error = 0.5 * pow(o - t, 2);

		double learning_rate = 0.05;

		double ob = o - t;

		double dw5e = h1 * ob;
		double dw6e = h2 * ob;

		double dw1e = i1 * w5 * ob;
		double dw2e = i2 * w5 * ob;
		double dw3e = i1 * w6 * ob;
		double dw4e = i2 * w6 * ob;

		w6 = w6 - learning_rate * dw6e;
		w5 = w5 - learning_rate * dw5e;
		w4 = w4 - learning_rate * dw4e;
		w3 = w3 - learning_rate * dw3e;
		w2 = w2 - learning_rate * dw2e;
		w1 = w1 - learning_rate * dw1e;

		printf("o: %f\n", o);
		// printf("w1: %f, w2: %f, w3: %f, w4: %f, w5: %f, w6: %f\n", w1, w2, w3, w4, w5, w6);
	}
	
	

	return 0;
}