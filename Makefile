STD		 =-std=c++20
GPU_FLAGS =-DKITTENS_4090 -arch=sm_89
NVCCFLAGS=-DNDEBUG -Xcompiler=-fPIE --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing --use_fast_math -forward-unknown-to-host-compiler -O3 -Xnvlink=--verbose -Xptxas=--verbose -Xptxas=--warn-on-spills -MD -MT -MF -x cu -lrt -lpthread -ldl -lcuda

gemm: 
	nvcc $(STD) -I ThunderKittens/include -I csrc/include $(GPU_FLAGS) $(NVCCFLAGS) gemm.cu -o gemm.bin