CC=clang++ -std=c++11 -O2
NVCC=nvcc

multiDevice: multiDevice.o multiDevice.cubin
	$(CC) -o $@ $< -L${CUDA_ROOT}/lib64 -lcuda

multiDevice.o: multiDevice.cpp
	$(CC) -o $@ $< -c -I${CUDA_ROOT}/include

multiDevice.cubin: multiDevice.cu
	$(NVCC) -o $@ $< -arch=sm_11 -cubin

clean:
	rm -f multiDevice multiDevice.o multiDevice.cubin
