#include "nnvis.h"

#include <math.h>

#include <raylib.h>
#include <mlp.h>

#define PI 3.1415
#define count 2000

float tb[count * 2] = {0};
traning_data_t td[count] = {0};
mlp_t mlp;

void setup_training()
{
	for(int i = 0; i < count; i++)
	{
		tb[i * 2] = 	 1.0 * i / count;
		tb[i * 2 + 1] = sinf(2 * PI * tb[i * 2]) * cosf(3 * 2 * PI * tb[i * 2]) * tb[i * 2];

		td[i].input = tb + i * 2;
		td[i].expected = tb + i * 2 + 1;
	}
}

void draw_image(int x, int y, int , int);

int main(void)
{
	SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_VSYNC_HINT | FLAG_WINDOW_UNDECORATED);
	InitWindow(1000, 800, "...");

	setup_training();

	float dc_da[1] = {0};
	float output[1] = {0};
	float cost = 0;

	int tk = 0;
	int chunk_size = 500;
	float rate = 1E-6;

	int layers[] = {5, 7, 1};

	mlp = mlp_create(layers, sizeof layers / sizeof(int), 1);
	mlp_t dmod  = mlp_copy_arch(&mlp);

	cost_graph_t cg = nnvis_create_costgraph(100);
	
	int iteration = 0;

	int should_play = 0;

	while(!WindowShouldClose())
	{
		PollInputEvents();
		// cost = mlp_train(&mlp, &dmod, td, count, output, dc_da, 3E-8);

		if(should_play)
		{
			for(int i = 0; i < 10; i++)
			{
				cost = mlp_train(&mlp, &dmod, td, chunk_size, output, dc_da, rate);
				iteration++;
			}

			static int t = 4;
			if(cost < 0.003 && chunk_size < count)
			{
				rate = rate / 4;
				chunk_size += 500;
			}
		}

		if(IsKeyDown(KEY_SPACE))
			should_play = 1;

		BeginDrawing();

		ClearBackground(BLACK);
		nnvis_draw_nnet(0, 0, GetScreenWidth() / 2, GetScreenHeight() / 2, &mlp);
		nnvis_draw_cost(0, GetScreenHeight() / 2, GetScreenWidth() / 2, GetScreenHeight() / 2, &cg, cost);

		draw_image(GetScreenWidth() / 2, 0, GetScreenWidth() / 2, GetScreenHeight());

		DrawFPS(0, 0);
		static char buffer[100] = {0};
		sprintf(buffer, "%d Iter", iteration);
		DrawText(buffer, 0, 30, 20, WHITE);

		EndDrawing();
	}

	CloseWindow();
	return 0;
}

void draw_image(int x, int y, int width, int height)
{
	float out;

	DrawLine(x, y, x, y + height, BROWN);

	DrawLine(x + width / 2, y, x + width / 2, y + height, WHITE);
	DrawLine(x, y + height / 2, x + width, y + height / 2, WHITE);

	float prevout = 0;
	float prevoutexp = 0;
	float prevw = 0;

	float actual = 0;

	float w = 0;
	float h = ((1.0f - actual) / 2.0 * height);

	prevoutexp = h;
	prevw = w;

	float in = 0;
	mlp_forward(&mlp, &in, &out);

	h = ((1.0f - out) / 2.0 * height);
	prevout = h;

	for(float i = 2 * PI / count; i <= 2 * PI; i += 2 * PI / count)
	{
		actual = sinf(i) * cosf(3 * i) * i / (2 * PI);

		w = i / (2 * PI) * width;
		h = ((1.0f - actual) / 2.0 * height);

		DrawLineEx((Vector2){w + x, h + y}, (Vector2){prevw + x, prevoutexp + y}, 2, PURPLE);
		prevoutexp = h;

		in = i / (2 * PI);
		mlp_forward(&mlp, &in, &out);

		h = ((1.0f - out) / 2.0 * height);

		DrawLineEx((Vector2){w + x, h + y}, (Vector2){prevw + x, prevout + y}, 2, ORANGE);

		prevout = h;

		prevw = w;
	}
}
