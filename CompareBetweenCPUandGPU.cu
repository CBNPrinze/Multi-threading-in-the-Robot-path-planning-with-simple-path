#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>
#include <iostream>  
using namespace std;

#define PI 3.14159265

struct lis {

	double basex = 100;
	double basey = 0; // vi tri cua baseS

	double delta = 0.1; // khoang thoi gian rat nho

	double maxGoc = 50; // chenh lech goc quay cua duong di xa nhat voi duong di goc
	double goc = 1; // chenh lech goc quay giua 2 duong ke tiep
	int n = 2 * maxGoc / goc + 1; // so luong se chay
	int maxC = 1; // tan cua goc nhin lon nhat co the nhin thay duoc

	double R = 100; // khoang cach toi da toi base
	double v = 2; // van toc robot
	double w = maxGoc * (v / R) / (n / 2) * PI / 180; // toc do goc cua robot
	int maxT = 500; //thoi gian toi da chay trong 1 lan set

	int numcoll = 2;
	double coll[4] = { 90,70,20,20 }; // so luong va vi tri chuong ngai vat
};


struct location {
	double x[100000], y[100000]; //vi tri hien tai
	double alpha[100000]; // goc van toc
};

void RoRunH(location* loc, int tId) {
	lis d;
	int dtime = d.maxT;
	int t = 1;
	while (t < dtime) {
		loc[tId].alpha[t] = d.w * (tId - d.n / 2) * d.delta + loc[tId].alpha[t - 1];
		loc[tId].x[t] = d.v * cos(loc[tId].alpha[t]) * d.delta + loc[tId].x[t - 1];
		loc[tId].y[t] = d.v * sin(loc[tId].alpha[t]) * d.delta + loc[tId].y[t - 1];
		t++;
	}
}

void RoCheckH(location* loc, int* timeSt, int tId) {
	lis d;
	int t = 0;
	while (t < d.maxT) {
		if ((loc[tId].x[t] >= d.coll[0]) &&
			(loc[tId].x[t] <= d.coll[0] + d.coll[2]) &&
			(loc[tId].y[t] >= d.coll[1]) &&
			(loc[tId].y[t] <= d.coll[1] + d.coll[3])) {
			timeSt[tId] = t;
			break;
		}
		t++;
	}
}

__global__ void RoRun(location* loc) {
	lis d;
	int tId = threadIdx.x;
	int dtime = d.maxT;
	int t = 1;
	while (t < dtime) {
		loc[tId].alpha[t] = d.w * (tId - d.n / 2) * d.delta + loc[tId].alpha[t - 1];
		loc[tId].x[t] = d.v * cos(loc[tId].alpha[t]) * d.delta + loc[tId].x[t - 1];
		loc[tId].y[t] = d.v * sin(loc[tId].alpha[t]) * d.delta + loc[tId].y[t - 1];
		t++;
	}
}

__global__ void RoCheck(location* loc, int* timeSt) {
	lis d;
	int tId = threadIdx.x;
	int t = 0;
	while (t < d.maxT) {
		if ((loc[tId].x[t] >= d.coll[0]) &&
			(loc[tId].x[t] <= d.coll[0] + d.coll[2]) &&
			(loc[tId].y[t] >= d.coll[1]) &&
			(loc[tId].y[t] <= d.coll[1] + d.coll[3])) {
			timeSt[tId] = t;
			break;
		}
		t++;
	}
}

int main() {
	lis h;


	location* loc;
	int* timeSt;

	loc = (location*)malloc(sizeof(location) * h.n);
	timeSt = (int*)malloc(sizeof(int) * h.n);

	for (int i = 0;i < h.n;i++) { // khoi tao cho tung luong
		loc[i].x[0] = h.basex;
		loc[i].y[0] = h.basey;
		loc[i].alpha[0] = PI / 2;
		timeSt[i] = h.maxT;
	}

	location* d_loc = nullptr;
	int* d_timeSt = nullptr;

	cudaMalloc((void**)&d_loc, sizeof(location) * h.n);
	cudaMalloc((void**)&d_timeSt, sizeof(int) * h.n);

	cudaMemcpy(d_loc, loc, sizeof(location) * h.n, cudaMemcpyHostToDevice);
	cudaMemcpy(d_timeSt, timeSt, sizeof(int) * h.n, cudaMemcpyHostToDevice);

	clock_t begin, end;
	begin = clock();
	for (int i = 1;i < h.n;i++) {
		RoRunH(loc, i);
	}
	for (int i = 1;i < h.n;i++) {
		RoCheckH(loc, timeSt, i);
	}
	end = clock();
	double duration = double(end - begin) / CLOCKS_PER_SEC * 1000;


	cout << "CPU Time: " << duration << " ms \n";

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
    
	cudaEventRecord(start);

	RoRun << <1, h.n >> > (d_loc);

	RoCheck << <1, h.n >> > (d_loc, d_timeSt);

	cudaEventRecord(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cout << "GPU Time: "<< milliseconds << " ms \n";

	cudaFree(d_loc);
	cudaFree(d_timeSt);
	free(loc);
	free(timeSt);
}