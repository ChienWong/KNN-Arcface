#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <device_functions.h>

#include<device_launch_parameters.h>
#include<device_functions.h>
#include<math_functions.h>
#include <cuda_runtime.h>
#include <vector>

#define  PI 3.1415926
struct Matrix
{
	int width;
	int height;
	float *elements;
};
__device__ float getElement(Matrix *A, int row, int col)
{
	return A->elements[row * A->width + col];
}
__device__ void setElement(Matrix *A, int row, int col, float value)
{
	A->elements[row * A->width + col] = value;
}
__global__ void matMulKernel(Matrix *A, Matrix *B, Matrix *C)
{

	//该方法仅适用于矩阵乘向量，block数量等于rows,threadpreblock数量等于cols
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;
	__shared__ float shared[512];
	shared[tid] = getElement(A, bid, tid) * getElement(B, tid, 0);
	for (int stride = 2; stride <= 512; stride *= 2) {
		if (tid%stride == 0) {
			shared[tid] += shared[tid + stride / 2];
		}
		__syncthreads();
	}
	if (tid == 0) {
		shared[0] = acos(shared[0]) / PI * 180;
		setElement(C, bid, 0, shared[0]);
	}
}
std::vector<float> getAllangle(std::vector<const float*> set,const float* _f) {
	Matrix *A, *B, *C;
	// 申请托管内存
	cudaMallocManaged((void**)&A, sizeof(Matrix));
	cudaMallocManaged((void**)&B, sizeof(Matrix));
	cudaMallocManaged((void**)&C, sizeof(Matrix));
	A->height = set.size();
	A->width = 512;
	B->height = 512;
	B->width = 1;
	C->width = 1;
	C->height = set.size();
	cudaMallocManaged((void**)&A->elements, A->width * A->height * sizeof(float));
	cudaMallocManaged((void**)&B->elements, B->width * B->height * sizeof(float));
	cudaMallocManaged((void**)&C->elements, C->width * C->height * sizeof(float));
	for (int i = 0; i < set.size(); i++) {
		for (int j = 0; j < 512; j++) {
			A->elements[i * 512 + j] = set[i][j];
		}
	}
	for (int i = 0; i < 512; i++) {
		B->elements[i] = _f[i];
	}
	dim3 blockSize(512);
	dim3 gridSize(A->height);
	matMulKernel <<< gridSize, blockSize >>>(A, B, C);
	const cudaError_t error = cudaDeviceSynchronize();
	if (error != cudaSuccess)\
	{
		printf("ERROR: %s:%d,", __FILE__, __LINE__);
		printf("code:%d,reason:%s\n", error, cudaGetErrorString(error));
	}
	std::vector<float> temp;
	for (int i = 0; i < C->height; i++) {
		printf("the element %d : %f\n", i, C->elements[i]);
		temp.push_back(C->elements[i]);
	}
	cudaFree(A->elements);
	cudaFree(B->elements);
	cudaFree(C->elements);
	cudaFree(A);
	cudaFree(B);
	cudaFree(C);
	return temp;
}