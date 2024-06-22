#ifndef _MLP_H_
#define _MLP_H_

typedef struct
{
	int rows;		// number of perceptron in the current layer
	int cols;		// number of perceptron in the previous layer

	float *weights;		// weight matrix  (rows x cols)
	float *bias;		// bias vector (rows x 1)
	float *in;		// input vector saved for backprop (cols x 1)
} mlp_layer_t;

typedef struct
{
	int count;
	mlp_layer_t *layers;
} mlp_t;

void mlp_layer_create(mlp_layer_t *m, int rows, int cols, int randomize);
void mlp_layer_forward(mlp_layer_t *layers, int count, float *exp);

// count is the number of layers,
// layers array contain the number of neurons in each layer
// must contain a input layer
mlp_t mlp_create(int *layers, int count, int inputs);
mlp_t mlp_copy_arch(mlp_t *m);

// returns the last (in) vector
void mlp_forward(mlp_t *m, float *f, float *out);

// params: model, storage for derivatives, output of neuralnet, expected output of neuralnet, a storage space to store dc_da
// expecting d_m to be zero so that backprop can be called for each training data
void mlp_backprop(mlp_t *model, mlp_t *d_m, float *out, float *exp, float *dc_da);
void mlp_layer_backprop(mlp_layer_t *m, mlp_layer_t *d_m, float *a, float *dc_da);
void mlp_apply_grad(mlp_t *m, mlp_t *d_m, float count, float rate);

typedef struct
{
	float *input;
	float *expected;
} traning_data_t;

float mlp_train(mlp_t *m, mlp_t *dm, traning_data_t *td, int count, float *out, float *dc_da, float rate);

void mlp_print(mlp_t *m);
void mlp_free(mlp_t *m);

#endif
