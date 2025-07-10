#include <cstdlib>
#include <random>
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define N 512

std::random_device rd; // hardware based seed
std::mt19937 gen(rd()); // mersenne twister engine
std::uniform_int_distribution<> dis(0, 99); // 0~99 distribution

void random_ints(int* des, int size){
    for(int i = 0; i < size; i++){
        des[i] = dis(gen);
    }
}

__global__ void add(int *a, int *b, int *c){
    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

int main(void){
    int *a, *b, *c; // host
    int *d_a, *d_b, *d_c; // device
    int size = N * sizeof(int);

    // allocate device(gpu) memory
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_b, size);

    // allocate host memory
    a = (int *)malloc(size);  random_ints(a, N);
    b = (int *)malloc(size);  random_ints(b, N);
    c = (int *)malloc(size);

    // host values
    std::cout << "a:\n";
    for(int i = 0; i < size; i++){
        std::cout << a[i] << " ";
    }
    std::cout << "b:\n";
    for(int i = 0; i < size; i++){
        std::cout << b[i] << " ";
    }

    // copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // launch add() kernel on gpu with n blocks
    add<<<N,1>>>(d_a, d_b, d_c);

    // copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // cleanup
    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); 
    return 0;
}