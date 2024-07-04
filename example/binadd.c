#include <stdio.h>
#include <stdlib.h>

#include "mlp.h"

int main(void)
{
	printf("..binary Adder using neural network..\n\n");

	int layers[] = {3, 2};

	mlp_t model = mlp_create(layers, sizeof layers / sizeof layers[0], layers[0]);
	mlp_t dmod  = mlp_copy_arch(&model);

	printf("..fnitial State before training..\n");
	mlp_print(&model);

	traning_data_t td[] = {
		{(float []){0, 0, 0}, (float []){0, 0}},
		{(float []){0, 0, 1}, (float []){0, 1}},
		{(float []){0, 1, 0}, (float []){0, 1}},
		{(float []){0, 1, 1}, (float []){1, 0}},
		{(float []){1, 0, 0}, (float []){0, 1}},
		{(float []){1, 0, 1}, (float []){1, 0}},
		{(float []){1, 1, 0}, (float []){1, 0}},
		{(float []){1, 1, 1}, (float []){1, 1}},
	};

	float dc_da[2] = {0};
	float output[2] = {0};
	float cost = 0;

	int epochs = 1000 * 500;
	for(int k = 0; k < epochs; k++)
	{
		cost = mlp_train(&model, &dmod, td, sizeof td / sizeof td[0], output, dc_da, 1E-1);

		if(k % (epochs / 10) == 0)
			printf("cost = %f\n", cost);
	}

	puts("");
	printf("..final state after training..\n");
	mlp_print(&model);


	for(int i = 0; i < sizeof td / sizeof td[0]; i++)
	{
		mlp_forward(&model, td[i].input, output);

		printf("%d | %f %f %f = %f %f\n",
				i + 1,
				td[i].input[0],
				td[i].input[1],
				td[i].input[2],
				output[0], output[1]);
	}

	float input[] = {1, 0, 1};
	mlp_forward(&model, input, output);

	puts("");

	return 0;
}
