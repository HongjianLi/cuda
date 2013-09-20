CC=clang -O2

bandwidthTest: bandwidthTest.o
	$(CC) -o $@ $^ -lcuda

%.o: %.cpp
	$(CC) -o $@ $< -c -I${CUDA_ROOT}/include

clean:
	rm -f bandwidthTest *.o
