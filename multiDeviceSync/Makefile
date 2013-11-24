CC=clang++ -std=c++11 -O2 -fsanitize=thread
NVCC=nvcc

all: multiDevice monte_carlo.cubin

multiDevice: multiDevice.o io_service_pool.o
	$(CC) -o $@ $^ -L${BOOST_ROOT}/lib -lboost_system -lboost_filesystem -L${CUDA_ROOT}/lib64 -lcuda

multiDevice.o: multiDevice.cpp
	$(CC) -o $@ $< -c -I${BOOST_ROOT} -I${CUDA_ROOT}/include

io_service_pool.o: io_service_pool.cpp
	$(CC) -o $@ $< -c -I${BOOST_ROOT}

monte_carlo.cubin: monte_carlo.cu
	$(NVCC) -o $@ $< -arch=sm_11 -cubin

clean:
	rm -f multiDevice multiDevice.o io_service_pool.o monte_carlo.cubin
