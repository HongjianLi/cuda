CC=clang -O2

deviceQuery: deviceQuery.o
	$(CC) -o $@ $^ -L${CUDA_ROOT}/lib64 -lcuda

deviceQuery.o: deviceQuery.cpp
	$(CC) -o $@ $< -c -I${CUDA_ROOT}/include

clean:
	rm -f deviceQuery deviceQuery.o
