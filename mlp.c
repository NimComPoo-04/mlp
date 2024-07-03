#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "mlp.h"

void mlp_layer_create(mlp_layer_t *m, int rows, int cols, int randomize)
{
	srand(time(0));

	if(randomize)
		srand(time(0));

	m->rows = rows;
	m->cols = cols;

	m->weights = calloc(sizeof(float), rows * cols);
	m->bias = calloc(sizeof(float), rows);
	m->in = calloc(sizeof(float), cols);

	for(int i = 0; i < rows * cols; i++)
		m->weights[i] = (randomize ? RAND_VAL * rand() / RAND_MAX : 0);

	for(int i = 0; i < rows; i++)
		m->bias[i] = (randomize ? RAND_VAL * rand() / RAND_MAX : 0);
}

// if you want to define your own activation for some odd reason
#ifndef MLP_ACTIVATION_DEFINED

float mlp_activation(float x)
{
#if defined(SIGMOID)
	 return 1.0 / (1.0 + expf(-x));
#elif defined(TANH)
	 return tanh(x);
#elif defined(RELU)
	 return (x > 0 ? x : 0);
#else
#error "Activation Function Not Defined"
#endif
}

float mlp_d_activation(float a)
{
#if defined(SIGMOID)
	return a * (1 - a);
#elif defined(TANH)
	return 1 - a * a;
#elif defined(RELU)
	 return (a > 0 ? 1 : 0);
#else
#error "Activation Function Not Defined"
#endif
}
#endif

void mlp_layer_forward(mlp_layer_t *layers, int count, float *exp)
{
	for(int i = 0; i < count; i++)
	{
		float *out = (i + 1 < count ? layers[i + 1].in : exp);

		for(int j = 0; j < layers[i].rows; j++)
		{
			out[j] = layers[i].bias[j];
			for(int k = 0; k < layers[i].cols; k++)
			{
				out[j] += layers[i].in[k] * layers[i].weights[j * layers[i].cols + k];
			}
			out[j] = mlp_activation(out[j]);
		}
	}
}

mlp_t mlp_create(int *layers, int count, int inputs)
{
	mlp_t m = {0};

	m.count = count;
	m.layers = calloc(count, sizeof(mlp_layer_t));

	mlp_layer_create(m.layers, layers[0], inputs, 1);
	for(int i = 1; i < count; i++)
		mlp_layer_create(m.layers + i, layers[i], layers[i - 1], 1);

	return m;
}

mlp_t mlp_copy_arch(mlp_t *t)
{
	mlp_t m = {0};

	m.count = t->count;
	m.layers = calloc(t->count, sizeof(mlp_layer_t));

	for(int i = 0; i < t->count; i++)
		mlp_layer_create(m.layers + i, t->layers[i].rows, t->layers[i].cols, 0);

	return m;
}

void mlp_forward(mlp_t *m, float *f, float *out)
{
	memcpy(m->layers[0].in, f, sizeof(float) * m->layers[0].cols);
	mlp_layer_forward(m->layers, m->count, out);
}

void mlp_print(mlp_t *m)
{
	for(int i = 0; i < m->count; i++)
	{
		printf("layer  = %d\n", i + 1);

#ifdef DISPLAY_IN
		printf("in     = ");
		for(int j = 0; j < m->layers[i].cols; j++)
			printf("%f  ", m->layers[i].in[j]);

		puts("");
#endif /* DISPLAY_IN */

		printf("bias   = ");
		for(int j = 0; j < m->layers[i].rows; j++)
			printf("%f  ", m->layers[i].bias[j]);

		puts("");

		printf("weight = ");
		for(int k = 0; k < m->layers[i].cols; k++)
			printf("%f  ", m->layers[i].weights[k]);
		puts("");
		for(int j = 1; j < m->layers[i].rows; j++)
		{
			printf("         ");
			for(int k = 0; k < m->layers[i].cols; k++)
				printf("%f  ", m->layers[i].weights[j * m->layers[i].cols + k]);
			puts("");
		}
		puts("");
	}
}

void mlp_free(mlp_t *m)
{
	for(int i = 0; i < m->count; i++)
	{
		free(m->layers[i].weights);
		free(m->layers[i].bias);
		free(m->layers[i].in);
	}

	free(m->layers);
}

// dw -> weights
// db -> bias
// da -> in
void mlp_layer_backprop(mlp_layer_t *m, mlp_layer_t *d_m, float *a, float *dc_da)
{
	memset(d_m->in, 0, d_m->cols * sizeof(float));

	for(int i = 0; i < d_m->rows; i++)
	{
		float t = dc_da[i] * mlp_d_activation(a[i]);

		d_m->bias[i] += t;

		for(int j = 0; j < d_m->cols; j++)
		{
			d_m->weights[i * d_m->cols + j] += t * m->in[j];
			d_m->in[j] += t * m->weights[i * d_m->cols + j];
		}
	}
}

void mlp_backprop(mlp_t *m, mlp_t *d_m, float *out, float *exp, float *dc_da)
{
	for(int i = 0; i < m->layers[m->count-1].rows ; i++)
	{
		dc_da[i] = 2 * (out[i] - exp[i]);
	}

	for(int i = m->count - 1; i >= 0; i--)
	{
		mlp_layer_backprop(m->layers + i, d_m->layers + i, out, dc_da);

		out = m->layers[i].in;
		dc_da = d_m->layers[i].in;
	}
}

void mlp_apply_grad(mlp_t *m, mlp_t *d_m, float count, float rate)
{
	for(int k = 0; k < m->count; k++)
	{
		for(int i = 0; i < m->layers[k].rows * m->layers[k].cols; i++)
		{
			m->layers[k].weights[i] -= rate * d_m->layers[k].weights[i] / 1.0f * count;
			d_m->layers[k].weights[i] = 0;
		}

		for(int i = 0; i < m->layers[k].rows; i++)
		{
			m->layers[k].bias[i] -= rate * d_m->layers[k].bias[i] / 1.0f * count;
			d_m->layers[k].bias[i]  = 0;
		}
	}
}

float mlp_train(mlp_t *m, mlp_t *dm, traning_data_t *td, int count, float *out, float *dc_da, float rate)
{
	float cost = 0;

	for(int i = 0; i < count; i++)
	{
		mlp_forward(m, td[i].input, out);

		for(int j = 0; j < m->layers[m->count - 1].rows; j++)
		{
			float f = (out[j] - td[i].expected[j]); 
			cost += f * f;
		}

		mlp_backprop(m, dm, out, td[i].expected, dc_da);
	}

	mlp_apply_grad(m, dm, 2 * count, rate);

	return cost / count;
}

