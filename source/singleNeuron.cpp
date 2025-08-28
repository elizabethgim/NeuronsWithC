#include <stdio.h>
#include <math.h>

double prediction(double x, double w, double b) {
	return w * x + b;
}

int main() {
	double xs[] = { -1.0, 0.0, 0.1, 2.0, 3.0, 4.0 };
	double ys[] = { -2.0, 1.0, 4.0, 7.0, 10.0, 13.0 };

	double w = 10;
	double b = 10;

	double p = 0;
	double error = 0;

	double learning_rate = 1;

	for (int i = 0; i < 6; i++) {
		p = prediction(xs[i], w, b);
		error = 0.5 * pow((p - ys[i]), 2);

		double dpe = p - ys[i];
		double dpw = xs[i];
		double dbp = 1;

		double dwe = dpw * dpe;
		double dbe = dbp * dpe;

		printf("w: %.2f, b: %.2f, dwe:%.2f, dbe: %.2f\n", w, b, dwe, dbe);

		w = w - learning_rate * dwe;
		b = b - learning_rate * dbe;
	}

	printf("final: \t w: %.2f, b: %.2f\n", w, b);

	return 0;
}