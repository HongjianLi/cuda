CC=clang -O2
NVCC=nvcc -ftz=true -prec-div=false -prec-sqrt=false -use_fast_math

zeroCopy: zeroCopy.o zeroCopy.cubin
	$(CC) -o $@ $< -L${CUDA_ROOT}/lib64 -lcuda

zeroCopy.o: zeroCopy.cpp
	$(CC) -o $@ $< -c -I${CUDA_ROOT}/include

zeroCopy.cubin: zeroCopy.cu
	$(NVCC) -o $@ $< -arch=sm_30 -cubin

clean:
	rm -f zeroCopy zeroCopy.o zeroCopy.cubin
