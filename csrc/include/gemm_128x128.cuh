/*
    4 warp GEMM computing a 128x128 output tile
*/

#include "common.hpp"
#include "kittens.cuh"

#ifndef GEMM_128x128_H
#define GEMM_128x128_H

namespace gemm_128x128 {

constexpr int MMA_M = 64;
constexpr int MMA_N = 64;
constexpr int MMA_K = 32;

constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;
constexpr int BLOCK_K = 32;

constexpr int NUM_WORKERS = 4;

/*
    - Each warp computes TC GEMMs of size [64, 32] @ [32, 64] = [64, 64]
    - Each TB writes a 128 x 128 tile
    - The warps in the output TB tile are laid out as follows:
        +-------+-------+
        | warp0 | warp1 |
        +-------+-------+
        | warp2 | warp3 |
        +-------+-------+
*/

constexpr int PIPE_STAGES = 2;

constexpr int LOAD_GROUPS = 2;  // 2 groups of 2 workers each

constexpr int BLOCK_SIZE = NUM_WORKERS * kittens::WARP_THREADS;

using namespace kittens;

using shared_tileA = st_bf<MMA_M, MMA_K>;
using shared_tileB = st_bf<MMA_K, MMA_N>;
using shared_tileC = st_bf<MMA_M, MMA_N>;

using reg_tileA = rt_bf<MMA_M, MMA_K>;
using reg_tileB =
    rt_bf<MMA_K, MMA_N,
          kittens::ducks::rt_layout::col>;  // this has to be col major
using reg_tileC = rt_fl<MMA_M, MMA_N>;

template <int M, int K>
using a_gl = gl<bf16, 1, 1, M, K, shared_tileA>;
template <int K, int N>
using b_gl = gl<bf16, 1, 1, K, N, shared_tileB>;
template <int M, int N>
using c_gl = gl<bf16, 1, 1, M, N, shared_tileC>;

template <int M, int N, int K>
struct gemm_globals {
	a_gl<M, K> a;
	b_gl<K, N> b;
	c_gl<M, N> c;
};

template <int M, int N, int K>
gemm_globals<M, N, K> gemm_init(bf16 *d_A, bf16 *d_B, bf16 *d_C) {
	using globals = gemm_globals<M, N, K>;

	a_gl<M, K> a_arg{d_A, nullptr, nullptr, nullptr, nullptr};
	b_gl<K, N> b_arg{d_B, nullptr, nullptr, nullptr, nullptr};
	c_gl<M, N> c_arg{d_C, nullptr, nullptr, nullptr, nullptr};

	globals g(a_arg, b_arg, c_arg);

	return g;
}

template <int M, int N, int K>
__global__ __launch_bounds__(BLOCK_SIZE, 1) void gemm(
    const __grid_constant__ gemm_globals<M, N, K> g) {
	using load_group = kittens::group<(NUM_WORKERS / LOAD_GROUPS)>;

	auto workerid = kittens::warpid();

	auto row_worker = workerid / 2;
	auto col_worker = workerid % 2;

	auto load_id = load_group::groupid();

	constexpr int LOAD_BLOCKS = NUM_WORKERS / load_group::GROUP_WARPS;

	// TB swizzling for L2 -> not a large speedup
	// const int GROUP_SIZE = 8;
	// u_int linear_idx = blockIdx.x + blockIdx.y * gridDim.x;
	// u_int block_per_grp = GROUP_SIZE * gridDim.y;
	// u_int grp_id =  linear_idx / block_per_grp;
	// u_int firstBlock = grp_id * GROUP_SIZE;

	// u_int groupSizeM = min((gridDim.y - firstBlock), GROUP_SIZE);

	// int warp_col = (firstBlock + (linear_idx % groupSizeM)) * LOAD_GROUPS;
	// int warp_row = ((linear_idx % block_per_grp) / groupSizeM) * LOAD_GROUPS;

	int warp_row = LOAD_GROUPS * blockIdx.y;
	int warp_col = LOAD_GROUPS * blockIdx.x;

	extern __shared__ alignment_dummy __shm[];
	shared_allocator al((int *)&__shm[0]);

	shared_tileA(&a_s)[LOAD_BLOCKS][PIPE_STAGES] =
	    al.allocate<shared_tileA, LOAD_BLOCKS, PIPE_STAGES>();
	shared_tileB(&b_s)[LOAD_BLOCKS][PIPE_STAGES] =
	    al.allocate<shared_tileB, LOAD_BLOCKS, PIPE_STAGES>();

	// this is required for vectorized store, just re-use a_s memory
	shared_tileC(&c_s)[LOAD_BLOCKS][LOAD_BLOCKS] =
	    reinterpret_cast<shared_tileC(&)[LOAD_BLOCKS][LOAD_BLOCKS]>(a_s);

	// warp level tiles
	reg_tileA ar_bf;
	reg_tileB br_bf;

	reg_tileC cr_fl;

	zero(cr_fl);

	int numKtile = K / MMA_K;

	int tic = 0;

	load_group::load_async<2, true>(a_s[load_id][tic], g.a,
	                                {warp_row + load_id, 0});
	load_group::load_async<2, true>(b_s[load_id][tic], g.b,
	                                {0, warp_col + load_id});

	for (int inner = 0; inner < numKtile;
	     inner++, tic = (tic + 1) % PIPE_STAGES) {
		int next_load_idx = inner + 1;
		if (next_load_idx < numKtile) {
			int next_tic = (tic + 1) % PIPE_STAGES;

			load_group::load_async<2, true>(
			    a_s[load_id][next_tic], g.a,
			    {warp_row + load_id, next_load_idx});
			load_group::load_async<2, true>(
			    b_s[load_id][next_tic], g.b,
			    {next_load_idx, warp_col + load_id});
			load_async_wait<2>();  // NOTE: N > 2 doesn't work, in total this
			                       // kernel makes 4 async load groups so this
			                       // creates a sliding window of 2
		}

		else
			load_async_wait();

		__syncthreads();

		load(ar_bf, a_s[row_worker][tic]);
		load(br_bf, b_s[col_worker][tic]);
		mma_AB(cr_fl, ar_bf, br_bf, cr_fl);
	}

	__syncthreads();

	store(c_s[row_worker][col_worker], cr_fl);
	store<2, false>(g.c, c_s[row_worker][col_worker],
	                {warp_row + row_worker, warp_col + col_worker});
}

template <int M, int N, int K>
void launch_gemm(bf16 *A, bf16 *B, bf16 *C) {
	const dim3 grid(common::ceil_div(N, BLOCK_N * LOAD_GROUPS),
	                common::ceil_div(M, BLOCK_M * LOAD_GROUPS));

	gemm_globals<M, N, K> g = gemm_init<M, N, K>(A, B, C);

	unsigned long mem_size = 50000;
	cudaDeviceSynchronize();
	cudaFuncSetAttribute(gemm<M, N, K>,
	                     cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);

	gemm<M, N, K><<<grid, BLOCK_SIZE, mem_size>>>(g);
}
}  // namespace gemm_128x128
#endif