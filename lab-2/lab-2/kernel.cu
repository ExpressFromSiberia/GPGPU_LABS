#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>
#include <windows.h>
#include <math.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <random>

using namespace std;

const unsigned BLOCK_SIZE = 32;
const unsigned int DIM = 2000;

enum MemType
{
    COMMON,
    SHARED,
};

void Matrix_Creator(double* matrix_A, double* matrix_B) {

    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<double> distribution(-30, 30);

    for (int i = 0; i < DIM; i++) {
        for (int j = 0; j < DIM; j++) {
            int	k = DIM * i + j;
            matrix_A[k] = distribution(generator);
            matrix_B[k] = distribution(generator);
        }
    }
}

double Matrix_Subtraction(double* matrix_CP, double* matrix_GP) {

    double max_deviation = 0;

    for (int i = 0; i < DIM * DIM; i++) {
        double deviation = abs(matrix_CP[i] - matrix_GP[i]);
        if (deviation > max_deviation)
            max_deviation = deviation;
    }

    return max_deviation;
}

void Matrix_Multiply_CP(double* Matrix_A, double* Matrix_B, double* Matrix_C) {

    LARGE_INTEGER start, stop, freq;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&start);

    for (size_t i = 0; i < DIM; ++i) {
        for (size_t j = 0; j < DIM; ++j) {
            Matrix_C[i * DIM + j] = 0;
            for (size_t k = 0; k < DIM; ++k) {
                Matrix_C[i * DIM + j] += Matrix_A[i * DIM + k] * Matrix_B[k * DIM + j];
            }
        }
    }

    QueryPerformanceCounter(&stop);

    size_t time_delta = stop.QuadPart - start.QuadPart;
    cout << "CPU time: " << static_cast<float>(time_delta) / freq.QuadPart << " seconds" << endl;
}

__global__ void Matrix_Multiply_GP(double* Matrix_A, double* Matrix_B, unsigned int n, double* Matrix_C) {

    double sum = 0.0;
    int   ia = n * blockDim.y * blockIdx.y + n * threadIdx.y;
    int   jb = blockDim.x * blockIdx.x + threadIdx.x;

    if (ia >= n * n || jb >= n)
        return;

    for (int k = 0; k < n; k++) {
        sum += Matrix_A[ia + k] * Matrix_B[jb + k * n];
    }

    Matrix_C[ia + jb] = sum;
}

__global__ void Matrix_Multiply_GP_Shared(double* Matrix_A, double* Matrix_B, unsigned int n, double* Matrix_C)
{
    int subMatrixA_ind = n * BLOCK_SIZE * blockIdx.y;
    int aEnd = subMatrixA_ind + n - 1;

    int aStep = BLOCK_SIZE;
    int subMatrixB_ind = BLOCK_SIZE * blockIdx.x;
    int bStep = BLOCK_SIZE * n;
    double sum = 0.0;

    for (int ia = subMatrixA_ind, ib = subMatrixB_ind; ia <= aEnd; ia += aStep, ib += bStep)
    {
        __shared__ double as[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ double bs[BLOCK_SIZE][BLOCK_SIZE];

        if (subMatrixA_ind + n * threadIdx.y < n * n && ia - subMatrixA_ind + threadIdx.x < n)
            as[threadIdx.y][threadIdx.x] = Matrix_A[ia + n * threadIdx.y + threadIdx.x];
        else
            as[threadIdx.y][threadIdx.x] = 0;
        if (ib - subMatrixB_ind + n * threadIdx.y < n * n && subMatrixB_ind + threadIdx.x < n)
            bs[threadIdx.y][threadIdx.x] = Matrix_B[ib + n * threadIdx.y + threadIdx.x];
        else
            bs[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; k++)
            sum += as[threadIdx.y][k] * bs[k][threadIdx.x];

        __syncthreads();
    }

    int ic = n * BLOCK_SIZE * blockIdx.y + BLOCK_SIZE * blockIdx.x;
    if (BLOCK_SIZE * blockIdx.y + threadIdx.y < n && BLOCK_SIZE * blockIdx.x + threadIdx.x < n)
        Matrix_C[ic + n * threadIdx.y + threadIdx.x] = sum;
}

void Matrix_GP_Ligature(double* Matrix_A, double* Matrix_B, double* Matrix_C, MemType type) {

    int numBytes = DIM * DIM * sizeof(double);
    double* adev = NULL;
    double* bdev = NULL;
    double* cdev = NULL;

    cudaMalloc((void**)&adev, numBytes);
    cudaMalloc((void**)&bdev, numBytes);
    cudaMalloc((void**)&cdev, numBytes);

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((unsigned int)ceil((double)DIM / threads.x), (unsigned int)ceil((double)DIM / threads.y));
    cudaEvent_t start, stop;
    float gpuTime = 0.0f;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cudaMemcpy(adev, Matrix_A, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(bdev, Matrix_B, numBytes, cudaMemcpyHostToDevice);

    if (type == COMMON)
        Matrix_Multiply_GP << <blocks, threads >> > (adev, bdev, DIM, cdev);
    if (type == SHARED)
        Matrix_Multiply_GP_Shared << <blocks, threads >> > (adev, bdev, DIM, cdev);

    cudaMemcpy(Matrix_C, cdev, numBytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);

    if (type == COMMON)
        cout << "GPU time: " << gpuTime / 1000.0f << " seconds" << endl;
    if (type == SHARED)
        cout << "GPU shared memory time: " << gpuTime / 1000.0f << " seconds" << endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(adev);
    cudaFree(bdev);
    cudaFree(cdev);
}

int main(int argc, char* argv[]) {

    double* Matrix_A = new double[DIM * DIM];
    double* Matrix_B = new double[DIM * DIM];
    Matrix_Creator(Matrix_A, Matrix_B);
    double* Matrix_C_CP = new double[DIM * DIM];
    double* Matrix_C_GP = new double[DIM * DIM];
    double* Matrix_C_GP_Shared = new double[DIM * DIM];

    Matrix_Multiply_CP(Matrix_A, Matrix_B, Matrix_C_CP);

    Matrix_GP_Ligature(Matrix_A, Matrix_B, Matrix_C_GP, COMMON);
    double deviation = Matrix_Subtraction(Matrix_C_CP, Matrix_C_GP);
    cout << "Max deviation in matrices values = " << deviation << endl;

    Matrix_GP_Ligature(Matrix_A, Matrix_B, Matrix_C_GP_Shared, SHARED);
    double deviation_shared = Matrix_Subtraction(Matrix_C_CP, Matrix_C_GP_Shared);
    cout << "Max deviation in matrices values = " << deviation_shared << endl;

    delete[] Matrix_A;
    delete[] Matrix_B;
    delete[] Matrix_C_CP;
    delete[] Matrix_C_GP;
    delete[] Matrix_C_GP_Shared;

    return 0;
}