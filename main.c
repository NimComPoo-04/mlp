#include <stdio.h>
#include <stdlib.h>

#include "mlp.h"

// A ^ B = A NAND B) (A NDNA)

int main(void)
{
	int layers[] = {2, 2, 1};

	mlp_t model = mlp_create(layers, sizeof layers / sizeof layers[0], layers[0]);
	mlp_t dmod  = mlp_copy_arch(&model);

	traning_data_t td[] = {
		{(float []){0, 0}, (float []){1}},
		{(float []){0, 1}, (float []){0.7}},
		{(float []){1, 0}, (float []){0}},
		{(float []){1, 1}, (float []){1}},
	};

	float dc_da = 0;
	float output = 0;
	float cost = 0;

	for(int k = 0; k < 1000 * 100; k++)
	{
		cost = mlp_train(&model, &dmod, td, sizeof td / sizeof td[0], &output, &dc_da, 0.15);
	}

	mlp_print(&model);

	printf("cost = %f\n", cost);

	for(int i = 0; i < sizeof td / sizeof td[0]; i++)
	{
		mlp_forward(&model, td[i].input, &output);
		printf("%d | %f * %f = %f \n", i + 1, td[i].input[0], td[i].input[1], output);
	}

	return 0;
}
