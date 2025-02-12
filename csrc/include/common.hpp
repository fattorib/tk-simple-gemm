#include <cuda_bf16.h>

#include <cassert>
#include <cmath>
#include <iostream>
#include <random>

#include "kittens.cuh"

#ifndef COMMON_H
#define COMMON_H

namespace common {

void fill_random(__nv_bfloat16 *A, size_t numel, float scale = 1.0) {
	std::random_device rd{};
	std::mt19937 gen{rd()};
	std::normal_distribution<float> d{0.0, scale};
	for (int i = 0; i < numel; i++) {
		A[i] = __float2bfloat16(d(gen));
	}
}

void fill_zeros(__nv_bfloat16 *A, size_t numel) {
	for (int i = 0; i < numel; i++) {
		A[i] = __float2bfloat16(0.0f);
	}
}

//  following BLAS naming conventions, this is performs a row major GEMM [m,k] @
//  [k,n] -> [m,n] between bf16 inputs and an fp32 accumulator
void cpu_gemm_tt(const __nv_bfloat16 *A, const __nv_bfloat16 *B,
                 __nv_bfloat16 *C, int m, int n, int k) {
	int lda, ldb, ldc;

	// rowstrides
	// A is (row) so stride is (k,1)
	// B is (row) so stride is (n,1)
	// C is (row) so stride is (n,1)
	lda = k;
	ldb = n;
	ldc = n;

	for (int r = 0; r < m; r++) {
		for (int c = 0; c < n; c++) {
			float tmp = 0.0f;
			for (int inner = 0; inner < k; inner++) {
				tmp += __bfloat162float(A[r * lda + inner]) *
				       __bfloat162float(B[inner * ldb + c]);
			}
			C[r * ldc + c] = __float2bfloat16(tmp);
		}
	}
}

void check_error(const __nv_bfloat16 *arr, const __nv_bfloat16 *ref,
                 int numel) {
	float rel_err;
	float abs_err;

	float max_rel_error = -INFINITY;
	float max_abs_error = -INFINITY;

	float total_diff = 0.0f;
	float diff_norm = 0.0f;
	float norm = 0.0f;

	float a_elem, r_elem, max_a_elem, max_r_elem, max_abs_a_elem,
	    max_abs_r_elem;

	for (int i = 0; i < numel; i++) {
		a_elem = __bfloat162float(arr[i]);
		r_elem = __bfloat162float(ref[i]);

		if (arr[i] != arr[i]) {
			printf("arr: (%7.6f) ref: (%7.6f) \n", a_elem, r_elem);
			throw std::runtime_error(
			    "ERROR: NaN value encountered in output array");
			break;
		}

		if (ref[i] != ref[i]) {
			printf("arr: (%7.6f) ref: (%7.6f) \n", a_elem, r_elem);
			throw std::runtime_error(
			    "ERROR: NaN value encountered in reference array");
		}

		if (r_elem != 0) {
			rel_err = std::abs(a_elem - r_elem) / std::abs(r_elem);
		}

		abs_err = std::abs(a_elem - r_elem);

		total_diff += abs_err;

		max_rel_error = std::fmaxf(max_rel_error, rel_err);
		max_abs_error = std::fmaxf(max_abs_error, abs_err);

		if (max_rel_error == rel_err) {
			max_a_elem = a_elem;
			max_r_elem = r_elem;
		}

		if (max_abs_error == abs_err) {
			max_abs_a_elem = a_elem;
			max_abs_r_elem = r_elem;
		}

		diff_norm += std::pow((a_elem - r_elem), 2.0);
		norm += std::pow((r_elem), 2.0);
	}

	float linalg_rel_error = std::pow(diff_norm, 0.5) / std::pow(norm, 0.5);
	printf("Maximum relative error: (%1.8f)\n", max_rel_error);
	printf("Maximum absolute error: (%1.8f)\n", max_abs_error);
	printf("Linalg relative error: (%1.8f)\n", linalg_rel_error);
	printf("Average abs error: (%1.8f)\n", total_diff / float(numel));
	printf("Offending max relative error (Actual, Expected): (%1.8f, %1.8f)\n",
	       max_a_elem, max_r_elem);
	printf("Offending max absolute error (Actual, Expected): (%1.4f, %1.4f)\n",
	       max_abs_a_elem, max_abs_r_elem);
}

u_int ceil_div(u_int a, u_int b) { return (a + b - 1) / b; }

template <typename T = __nv_bfloat16>
__global__ void zero_kernel(T *ptr) {
	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int nt = blockDim.x;
	int block_ptr_start = nt * bx;
	int idx = block_ptr_start + tx;

	ptr[idx] = T(0.0);
}

template <typename T = __nv_bfloat16>
void zero_(T *ptr, const int numel) {
	int BLOCKSIZE = 128;
	const dim3 grid(ceil_div(numel, BLOCKSIZE));
	zero_kernel<<<grid, BLOCKSIZE>>>(ptr);
}

}  // namespace common

#endif
