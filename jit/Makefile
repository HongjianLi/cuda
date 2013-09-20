CC=clang -O2

jit: jit.o
	$(CC) -o $@ $^ -L${CUDA_ROOT}/lib64 -lcuda

%.o: %.cpp 
	$(CC) -o $@ $< -c -I${CUDA_ROOT}/include

clean:
	rm -f jit jit.o
