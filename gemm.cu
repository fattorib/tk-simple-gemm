/*
    bf16 tensorcore GEMM
*/

#include <cuda.h>

#include <iostream>

#include "common.hpp"
#include "kernels.hpp"
#include "kittens.cuh"
using namespace kittens;

#define CudaCheckError() __cudaCheckError(__FILE__, __LINE__)
inline void __cudaCheckError(const char *file, const int line) {
	cudaError err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n", file, line,
		        cudaGetErrorString(err));
		exit(-1);
	}
	// More careful checking. However, this will affect performance.
	// Comment away if needed.
	err = cudaDeviceSynchronize();
	if (cudaSuccess != err) {
		fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
		        file, line, cudaGetErrorString(err));
		exit(-1);
	}
}

int main() {
	const size_t M = 4096;
	const size_t K = 4096;
	const size_t N = 4096;

	constexpr size_t numelA = M * K;
	constexpr size_t numelB = K * N;
	constexpr size_t numelC = M * N;

	bf16 *A, *B, *C, *C_ref, *dA, *dB, *dC;
	A = new bf16[numelA];
	B = new bf16[numelB];
	C = new bf16[numelC];
	C_ref = new bf16[numelC];

	int *cache_l2;
	size_t numel_l2 = 256e6;

	common::fill_random(A, numelA);
	common::fill_random(B, numelB);

	cudaMalloc(&dA, numelA * sizeof(bf16));
	cudaMalloc(&dB, numelB * sizeof(bf16));
	cudaMalloc(&dC, numelC * sizeof(bf16));

	// following Triton benchmarks -> before each bench iter write 256MB to
	// clear L2
	cudaMalloc(&cache_l2, numel_l2 * sizeof(int));
	cudaMalloc(&dC, numelC * sizeof(bf16));

	cudaMemcpy(dA, A, numelA * sizeof(bf16), cudaMemcpyHostToDevice);
	cudaMemcpy(dB, B, numelB * sizeof(bf16), cudaMemcpyHostToDevice);

	float *milliseconds;
	cudaEvent_t *start, *stop;

	int iters = 125;
	int warmup = 25;

	start = new cudaEvent_t[iters];
	stop = new cudaEvent_t[iters];
	milliseconds = new float[iters];

	for (int i = 0; i < iters; i++) {
		common::zero_<int>(cache_l2, numel_l2);

		cudaEventCreate(&start[i]);
		cudaEventCreate(&stop[i]);

		cudaEventRecord(start[i]);
		gemm_128x128::launch_gemm<M, N, K>(dA, dB, dC);
		cudaEventRecord(stop[i]);
		cudaEventSynchronize(stop[i]);

		cudaEventElapsedTime(&milliseconds[i], start[i], stop[i]);
	}

	CudaCheckError();

	double total = 0.0;

	for (auto s = warmup; s < iters; s++) {
		total += milliseconds[s];
	}
	double elapsed_time = total * 1e-3;
	double flops = (iters - warmup) * (2 * M * N * K);
	std::cout << "Problem Size: " << M << " x " << N << " x " << K << std::endl;
	std::cout << "Total Elapsed Time: " << elapsed_time << "s" << std::endl;
	std::cout << "TFLOP/s " << (flops * 1e-12) / elapsed_time << std::endl;

	cudaMemcpy(C, dC, numelC * sizeof(bf16), cudaMemcpyDeviceToHost);

	if (M <= 1024) {
		// perform CPU GEMM and check output
		common::cpu_gemm_tt(A, B, C_ref, M, N, K);
		common::check_error(C, C_ref, numelC);
	}

	delete[] A;
	delete[] B;
	delete[] C;
	delete[] start;
	delete[] stop;
	delete[] milliseconds;

	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);
}
