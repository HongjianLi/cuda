CC=clang -O2
NVCC=nvcc -use_fast_math

concurrentKernels: concurrentKernels.o concurrentKernels.cubin
	$(CC) -o $@ $< -L${CUDA_ROOT}/lib64 -lcuda

concurrentKernels.o: concurrentKernels.cpp
	$(CC) -o $@ $< -c -I${CUDA_ROOT}/include

concurrentKernels.cubin: concurrentKernels.cu
	$(NVCC) -o $@ $< -arch=sm_11 -cubin

clean:
	rm -f concurrentKernels concurrentKernels.o concurrentKernels.cubin
