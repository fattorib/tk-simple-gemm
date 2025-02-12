# ThunderKittens - Simple GEMM

This repo contains a performant tensorcore GEMM kernel written in ThunderKittens (*and another slower kernel lol*). For square matrices, the 4-warp, 128x128x32 kernel is within ~98% of cuBLAS and Triton. Thunderkittens is quite nice to use, and while it includes a few example GEMM kernels, these 1) use H100 specific features (WGMMA) and 2) use the author's load-compute-store-finish (LCSF) programming model. This repo intends to provide an example of a simple GEMM kernel that is still fast.  

## Benchmarks

Benchmarks performed on an 4096x4096x4096 problem with bfloat16 inputs and float accumulation on an RTX 4070. Triton kernel is taken from [here](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#sphx-glr-getting-started-tutorials-03-matrix-multiplication-py):

| Kernel                     | TFLOPs |
|----------------------------|--------|
| ThunderKittens (this repo) |   61.1 |
| cuBLAS                     |   61.4 |
| Triton                     |   62.2 |

# Compile

## Setup 
Clone repo with:
```bash
git clone --recurse-submodules https://github.com/fattorib/tk-simple-gemm.git
```

This code has been tested in the following environment:
- gcc 11.4.0
- nvcc 12.6
- RTX 4070 
- ubuntu22.04

All development work was performed in the `nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04` docker image. 

## Build
To build the GEMM kernel (defaults to 128x128x32 kernel), run:

```bash 
make gemm
```

to run the kernel and benchmark it, run:
```
./gemm.bin
```

your output should be something like:

```bash
Problem Size: 4096 x 4096 x 4096
Total Elapsed Time: 0.225039s
TFLOP/s 61.0734
```

# Citations

```bibtex
@misc{spector2024thunderkittenssimplefastadorable,
      title={ThunderKittens: Simple, Fast, and Adorable AI Kernels}, 
      author={Benjamin F. Spector and Simran Arora and Aaryan Singhal and Daniel Y. Fu and Christopher Ré},
      year={2024},
      eprint={2410.20399},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.20399}, 
}
```