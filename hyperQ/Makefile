CC=clang -O2
NVCC=nvcc -use_fast_math

hyperQ: hyperQ.o hyperQ.cubin
	$(CC) -o $@ $< -L${CUDA_ROOT}/lib64 -lcuda

hyperQ.o: hyperQ.cpp
	$(CC) -o $@ $< -c -I${CUDA_ROOT}/include

hyperQ.cubin: hyperQ.cu
	$(NVCC) -o $@ $< -arch=sm_11 -cubin

clean:
	rm -f hyperQ hyperQ.o hyperQ.cubin
