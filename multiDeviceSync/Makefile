CC=clang++ -std=c++11 -O2 -fsanitize=thread
NVCC=nvcc

all: multiDevice monte_carlo.cubin

multiDevice: multiDevice.o io_service_pooled.o
	$(CC) -o $@ $^ -lboost_system -lboost_filesystem -L${CUDA_ROOT}/lib64 -lcuda

multiDevice.o: multiDevice.cpp
	$(CC) -o $@ $< -c -I${CUDA_ROOT}/include

io_service_pooled.o: io_service_pooled.cpp
	$(CC) -o $@ $< -c

monte_carlo.cubin: monte_carlo.cu
	$(NVCC) -o $@ $< -arch=sm_11 -cubin

clean:
	rm -f multiDevice multiDevice.o io_service_pooled.o monte_carlo.cubin
