CC=clang -O2
NVCC=nvcc -ftz=true -prec-div=false -prec-sqrt=false -use_fast_math

vectorAdd: vectorAdd.o vectorAdd.cubin
	$(CC) -o $@ $< -L${CUDA_ROOT}/lib64 -lcuda

vectorAdd.o: vectorAdd.cpp
	$(CC) -o $@ $< -c -I${CUDA_ROOT}/include

vectorAdd.cubin: vectorAdd.cu
	$(NVCC) -o $@ $< -arch=sm_11 -cubin

clean:
	rm -f vectorAdd vectorAdd.o vectorAdd.cubin
