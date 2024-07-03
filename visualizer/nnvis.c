#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "nnvis.h"

static inline float sigmoidf(float x)
{
	return 1 / (1 + expf(-x));
}

void nnvis_draw_nnet(int posx, int posy, int width, int height, mlp_t *mlp)
{
	// 1 for input and 2 for padding
	int count = mlp->count + 2;

	float wid = 1.0 * width / count;	// wid is the width of each sector
	float dw  = wid * 0.24;			// dw is 1/10 of the width of each thing

	// getting the size of the circles and the positions of the circle

	int row_max = 0;
	for(int i = 0; i < mlp->count; i++)
		row_max = (row_max > mlp->layers[i].rows ? row_max : mlp->layers[i].rows);

	float dh =  1.0 * height / row_max;

	float radii = (2 * dw > dh ? dh : 2 * dw) * 0.5 * 0.9;

	// Drawing the cirlces
	
	Color col;
	col.a = 0xff;

	float x, y, nx, ny, t;

	for(int i = 2; i < count; i++)
	{
		x = i * wid + posx;
		y = (height - mlp->layers[i - 2].rows * dh + dh) / 2.0 + posy;

		for(int j = 0; j < mlp->layers[i - 2].rows; j++)
		{
			nx = x - wid;
			ny = (height - mlp->layers[i - 2].cols * dh + dh) / 2.0 + posy;

			for(int k = 0; k < mlp->layers[i - 2].cols; k++)
			{
				t = sigmoidf(mlp->layers[i - 2].weights[j * mlp->layers[i - 2].cols + k]);
				col = ColorFromHSV(t * 360, 1, 1);

				DrawLineEx((Vector2){x, y + j * dh}, (Vector2){nx, ny + k * dh}, t * 3 + 1, col);
			}
		}
	}

	for(int i = 2; i < count; i++)
	{
		x = i * wid + posx;
		y = (height - mlp->layers[i - 2].rows * dh + dh) / 2.0 + posy;

		for(int j = 0; j < mlp->layers[i - 2].rows; j++)
		{
			t = sigmoidf(mlp->layers[i - 2].bias[j]);
			col = ColorFromHSV(t * 360, 1, 1);

			if(i == count - 1)
				DrawCircle(x, y + j * dh, radii + 3, DARKBLUE);
			DrawCircle(x, y + j * dh, radii, col);
		}
	}

	// First ball lol
	x = wid + posx;
	y = (height - mlp->layers[0].cols * dh + dh) / 2.0 + posy;

	for(int i = 0; i < mlp->layers[0].cols; i++)
	{
		t = sigmoidf(mlp->layers[0].in[i]);
		col = ColorFromHSV(t * 360, 1, 1);

		DrawCircle(x, y + i * dh, radii + 3, DARKBLUE);
		DrawCircle(x, y + i * dh, radii, col);
	}
}

cost_graph_t nnvis_create_costgraph(int size)
{
	cost_graph_t cg;
	cg.points = calloc(sizeof(float), size);
	cg.size = size;
	cg.start = 0;
	cg.max = -1;

	return cg;
}

void nnvis_draw_cost(int posx, int posy, int width, int height, cost_graph_t *cg, float cost)
{
	cg->start = (cg->start + 1) % cg->size;
	cg->points[cg->start] = cost;

	cg->max = -1;
	for(int i = 0; i < cg->size; i++)
		cg->max = cg->max < cg->points[i] ? cg->points[i] : cg->max;

	float dx = 1.0 * width / cg->size;
	float x = 0;

	for(int i = 0; i < cg->size - 1; i++)
	{
		float y = height - cg->points[(i + cg->start + 1) % cg->size] / cg->max * (height - 1);
		float y2 = height - cg->points[(i + cg->start + 2) % cg->size] / cg->max * (height - 1);

		x += dx;

	//	DrawPixel(x + posx, y + posy, RED);
		DrawLineEx((Vector2){x + posx, y + posy}, (Vector2){x + posx + dx, y2 + posy}, 2, RED);
	}

	DrawLine(posx, height + posy, width + posx, height + posy, WHITE);
	DrawText("0", posx + 5, height + posy - 30, 24, WHITE);

	static char buffer[256];
	sprintf(buffer, "%.4f", cg->max);

	DrawLine(posx, posy, width + posx, posy, GREEN);
	DrawText(buffer, posx + 5, posy + 5, 24, GREEN);
}
