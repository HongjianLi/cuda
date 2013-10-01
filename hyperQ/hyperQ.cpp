#include <time.h>
#include "../cu_helper.h"

int main(int argc, char *argv[])
{
	const int num_milliseconds = 10;
	const int num_streams = 32;
	CUdevice device;
	CUcontext context;
	CUmodule module;
	CUfunction kernel0, kernel1;
	CUstream *streams;
	CUevent beg, end;
	int clock_rate;
	clock_t num_clocks;
	float elapsed;
	int i;
	checkCudaErrors(cuInit(0));
	checkCudaErrors(cuDeviceGet(&device, 0));
	checkCudaErrors(cuDeviceGetAttribute(&clock_rate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device));
	checkCudaErrors(cuCtxCreate(&context, CU_CTX_SCHED_AUTO, device));
	checkCudaErrors(cuModuleLoad(&module, "hyperQ.cubin"));
	checkCudaErrors(cuModuleGetFunction(&kernel0, module, "kernel0"));
	checkCudaErrors(cuModuleGetFunction(&kernel1, module, "kernel1"));
	num_clocks = clock_rate * num_milliseconds;
	streams = (CUstream*)malloc(sizeof(CUstream) * num_streams);
	for (i = 0; i < num_streams; ++i)
	{
		checkCudaErrors(cuStreamCreate(&streams[i], CU_STREAM_DEFAULT));
	}
	checkCudaErrors(cuEventCreate(&beg, CU_EVENT_DEFAULT));
	checkCudaErrors(cuEventCreate(&end, CU_EVENT_DEFAULT));
	checkCudaErrors(cuEventRecord(beg, 0));
	for (int i = 0; i < num_streams; ++i)
	{
		void* args[] = { &num_clocks };
		checkCudaErrors(cuLaunchKernel(kernel0, 1, 1, 1, 1, 1, 1, 0, streams[i], args, NULL));
		checkCudaErrors(cuLaunchKernel(kernel1, 1, 1, 1, 1, 1, 1, 0, streams[i], args, NULL));
	}
	checkCudaErrors(cuEventRecord(end, 0));
	checkCudaErrors(cuEventSynchronize(end));
	checkCudaErrors(cuEventElapsedTime(&elapsed, beg, end));
	printf("%d streams, each %d kernels, each %d ms\n", num_streams, 2, num_milliseconds);
	printf("       SM <= 1.3: %d ms\n", num_milliseconds * 2 * num_streams);
	printf("2.0 <= SM <= 3.0: %d ms\n", num_milliseconds * (num_streams + 1));
	printf("3.5 <= SM       : %d ms\n", num_milliseconds * 2);
	printf("sample execution: %.3f ms\n", elapsed);
	cuEventDestroy(end);
	cuEventDestroy(beg);
	for (int i = 0; i < num_streams; ++i)
	{
		cuStreamDestroy(streams[i]);
	}
	free(streams);
	checkCudaErrors(cuModuleUnload(module));
	checkCudaErrors(cuCtxDestroy(context));
}
