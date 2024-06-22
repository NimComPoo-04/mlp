#include <stdio.h>
#include <stdlib.h>

#include "mlp.h"

// A ^ B = A NAND B) (A NDNA)

int main(void)
{
	int layers[] = {2, 2, 1};

	mlp_t model = mlp_create(layers, sizeof layers / sizeof layers[0], layers[0]);
	mlp_t dmod  = mlp_copy_arch(&model);

	float input[][2] = {
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	};

	float expected[] = {
		0.7, 0.3, 0.1, 0.56
	};

#define count (sizeof expected / sizeof expected[0])
	float output[count] = {0};
	float tmp[count] = {0};

	float cost = 0;

	//while(getc(stdin) == '\n')
	{

		for(int k = 0; k < 1000 * 100; k++)
		{
			for(int i = 0; i < count; i++)
			{
				mlp_forward(&model, input[i], output + i);
				mlp_backprop(&model, &dmod, output + i, expected + i, tmp + i);
			}

			mlp_apply_grad(&model, &dmod, 2 * count, 1E-2);
		}

		mlp_print(&model);


		printf("output = ");

		cost = 0;
		for(int i = 0; i < count; i++)
		{
			printf("%f  ", output[i]);
			cost += (output[i] - expected[i]) * (output[i] - expected[i]);
		}
		cost /= count;

		puts("");
		printf("cost = %f\n\n", cost);
	}

	return 0;
}
