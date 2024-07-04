#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "mlp.h"

//TANH

#define PI 3.1415
#define count 4000

float tb[count * 2] = {0};
traning_data_t td[count] = {0};
mlp_t model;

void setup_training()
{
	for(int i = 0; i < count; i++)
	{
		tb[i * 2] = 	 1.0 * i / count;
		tb[i * 2 + 1] =  sinf(2 * PI * tb[i * 2]);

		td[i].input = tb + i * 2;
		td[i].expected = tb + i * 2 + 1;
	}
}

void draw_model(void);

int main(void)
{
	printf("..sine using neural network..\n\n");

	setup_training();

	int layers[] = {1, 5, 7, 1};

	model = mlp_create(layers, sizeof layers / sizeof layers[0], layers[0]);
	mlp_t dmod  = mlp_copy_arch(&model);

	printf("..initial state before training..\n");
	mlp_print(&model);

	float dc_da[1] = {0};
	float output[1] = {0};
	float cost = 0;

	int epochs = 2000 * 100;

	int tk = 0;
	int chunk_size = 500;

	for(int k = 0; k < epochs; k++)
	{
		cost = mlp_train(&model, &dmod, td + tk * chunk_size, chunk_size, output, dc_da, 0.03);

		if(k % (epochs / 20) == 0)
			printf("cost = %f\n", cost);

		tk = (tk + 1) % (count / chunk_size);
	}

	/*
	puts("");
	printf("..final state after training..\n");
	mlp_print(&model);

	for(int i = 0; i < count; i++)
	{
		mlp_forward(&model, td[i].input, output);
		printf("(%f, %f)\n", td[i].input[0] * 2 * PI, output[0]);
	}
	*/

	draw_model();
}

#define width 800
#define height 800

char buffer[height][width][3];

void draw_model(void)
{
	float out;

	for(float i = 0; i <= 2 * PI; i += 2 * PI / count)
	{
		float actual = sinf(i);

		int w = i / (2 * PI) * width;
		int h = ((1.0f - actual) / 2.0 * height);

		buffer[h][w][0] = 0xff;
		buffer[h][w][1] = 0;
		buffer[h][w][2] = 0xff;

		float in = i / (2 * PI);
		mlp_forward(&model, &in, &out);

		h = ((1.0f - out) / 2.0 * height);

		buffer[h][w][0] = 0;
		buffer[h][w][1] = 0xff;
		buffer[h][w][2] = 0xff;
	}

	fprintf(stderr, "P6 %d %d 255\n", width, height);
	fwrite(buffer, sizeof buffer, 1, stderr);
}
