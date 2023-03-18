#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/imgproc.hpp>
using namespace cv;
using namespace std;

#define PI 3.14159265

struct lis {

	double basex = 100;
	double basey = 0; // vi tri cua s

	double delta = 0.1; // khoang thoi gian rat nho

	double maxAl = 50; // chenh lech goc quay cua duong di xa nhat voi duong di goc
	double goc = 2; // chenh lech goc quay giua 2 duong ke tiep
	int n = 2 * maxAl / goc + 1; // so luong se chay
	int maxC = 1; // tan cua goc nhin lon nhat co the nhin thay duoc

	double v = 2; // van toc robot
	int maxT = 500; //thoi gian toi da chay trong 1 lan set
	double w = maxAl / (n / 2) * PI / 180 / maxT / delta; // toc do goc cua robot

	int numcoll = 2;
	double coll[4] = { 70,80,35,20 }; // so luong va vi tri chuong ngai vat
};


struct location {
	double x[100000], y[100000]; //vi tri hien tai
	double alpha[100000]; // goc van toc
};

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

	Mat img = cv::Mat(700, 1000, CV_8UC3);
	img = cv::Scalar(0, 0, 0);

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

	RoRun << <1, h.n >> > (d_loc);

	cudaMemcpy(loc, d_loc, sizeof(location) * h.n, cudaMemcpyDeviceToHost);

	RoCheck << <1, h.n >> > (d_loc, d_timeSt);

	cudaMemcpy(timeSt, d_timeSt, sizeof(int) * h.n, cudaMemcpyDeviceToHost);

	cudaFree(d_loc);
	cudaFree(d_timeSt);

	Rect r = Rect(h.coll[0] * 5, h.coll[1] * 5, h.coll[2] * 5, h.coll[3] * 5);

	for (int i = 0;i < h.n;i++) {
		for (int t = 0;t < timeSt[i];t++) {
			img.at<cv::Vec3b>(loc[i].y[t] * 5, loc[i].x[t] * 5) = cv::Vec3b(0, 0, 255);
		}
	}

	cv::rectangle(img, r, cv::Scalar(0, 128, 255), 1);

	//hiển thị ảnh
	imshow("Image", img);

	waitKey(0); // Wait for a keystroke in the window
	getchar();

	free(loc);
	free(timeSt);

}
