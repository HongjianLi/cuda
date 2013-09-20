CC=clang++ -std=c++11 -O2
NVCC=nvcc -ptx -ftz=true -prec-div=false -prec-sqrt=false -use_fast_math# -src-in-ptx -g -G -Xcompiler -rdynamic -lineinfo -maxrregcount=N
ES=~/idock/embedSources.sh

vectorAdd: vectorAdd.o
	$(CC) -o $@ $^ -lcuda

vectorAdd.o: vectorAdd.cpp vectorAdd.h
	$(CC) -o $@ $< -c -I${CUDA_ROOT}/include

vectorAdd.h: vectorAdd_35.ptx vectorAdd_30.ptx vectorAdd_20.ptx vectorAdd_13.ptx vectorAdd_12.ptx vectorAdd_11.ptx vectorAdd_10.ptx
	$(ES) $^ > $@

%_10.ptx: %.cu
	$(NVCC) -o $@ $< -arch=compute_10

%_11.ptx: %.cu
	$(NVCC) -o $@ $< -arch=compute_11

%_12.ptx: %.cu
	$(NVCC) -o $@ $< -arch=compute_12

%_13.ptx: %.cu
	$(NVCC) -o $@ $< -arch=compute_13

%_20.ptx: %.cu
	$(NVCC) -o $@ $< -arch=compute_20

%_30.ptx: %.cu
	$(NVCC) -o $@ $< -arch=compute_30

%_35.ptx: %.cu
	$(NVCC) -o $@ $< -arch=compute_35

clean:
	rm -f vectorAdd vectorAdd.o vectorAdd_??.ptx
