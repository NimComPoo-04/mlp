#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#define WEIGHT_COUNT 2

float weights[WEIGHT_COUNT] = {0};
float bias = 0;

struct test_input
{
	float x[WEIGHT_COUNT];
	float e;
} tests[4] = {

#define NOT

#if defined(AND)
	{0, 0, 0},
	{0, 1, 0},
	{1, 0, 0},
	{1, 1, 1}
#elif defined(OR)
	{0, 0, 0},
	{0, 1, 1},
	{1, 0, 1},
	{1, 1, 1}
#elif defined(NAND)
	{0, 0, 1},
	{0, 1, 1},
	{1, 0, 1},
	{1, 1, 0}
#elif defined(NOR)
	{0, 0, 1},
	{0, 1, 0},
	{1, 0, 0},
	{1, 1, 0}
#elif defined(NOT)
	{0, 0, 1},
	{0, 1, 1},
	{1, 0, 0},
	{1, 1, 0}
#endif
};

void randomize()
{
	srand(time(0));

	bias = 1.0f * rand() / RAND_MAX;

	for(int i = 0; i < WEIGHT_COUNT; i++)
		weights[i] = 1.0f * rand() / RAND_MAX;
}

float sigmoid(float x)
{
	return 1 / (1 + expf(-x));
}

float forward(float *x)
{
	float out = bias;
	for(int i = 0; i < WEIGHT_COUNT; i++)
		out += weights[i] * x[i];
	return sigmoid(out);
}

float cost()
{
	int count = sizeof tests / sizeof tests[0];
	float out = 0;

	for(int i = 0; i < count; i++)
	{
		float e = forward(tests[i].x) - tests[i].e;
		out += e * e;
	}

	return out / count;
}

void dcost(float dw[WEIGHT_COUNT], float *db)
{
	int count = sizeof tests / sizeof tests[0];
	for(int i = 0; i < count; i++)
	{
		float y = forward(tests[i].x);
		float t = 2 * (y - tests[i].e) * y * (1 - y);

		*db += t / count;
		for(int j = 0; j < WEIGHT_COUNT; j++)
			dw[j] += t * tests[i].x[j] / count;
	}
}

void train()
{
	float rate = 1;

	float dw[WEIGHT_COUNT];
	float db;

	for(int i = 0; i < 1000 * 10; i++)
	{
		float c = cost();

		memset(dw, 0, sizeof dw);
		db = 0;

		dcost(dw, &db);

		bias -= db * rate;

		for(int j = 0; j < WEIGHT_COUNT; j++)
			weights[j] -= dw[j] * rate;

		printf("cost = %f  bias = %f  weight = ", c, bias);
		for(int j = 0; j < WEIGHT_COUNT; j++)
			printf("%f  ", weights[j]);
		puts("");
	}
}

int main(void)
{
	randomize();

	train();

	for(int i = 0; i < 4; i++)
	{
		float f = forward(tests[i].x);

		printf("%d | %f  .  %f  =  %f\n", i + 1, tests[i].x[0], tests[i].x[1], f);
	}

	return 0;
}
