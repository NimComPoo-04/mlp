#ifndef _NNVIS_H_
#define _NNVIS_H_

#include <raylib.h>
#include <mlp.h>

typedef struct
{
	float *points;
	int size;

	int start;	// current size

	float max;
} cost_graph_t;

void nnvis_draw_nnet(int posx, int posy, int width, int height, mlp_t *mlp);

cost_graph_t nnvis_create_costgraph(int size);
void nnvis_draw_cost(int posx, int posy, int width, int height, cost_graph_t *cg, float cost);

#endif
