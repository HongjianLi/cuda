#include <time.h>
#include "../cu_helper.h"

int main(int argc, char *argv[])
{
	const int num_milliseconds = 10;
	const int num_kernels = 2;
	const int num_streams = 32;
	CUdevice device;
	CUcontext context;
	CUmodule module;
	CUfunction function;
	CUstream *streams;
	CUevent beg, end;
	int clock_rate, cc_major, cc_minor;
	clock_t num_clocks;
	float elapsed;
	int s, k;
	checkCudaErrors(cuInit(0));
	checkCudaErrors(cuDeviceGet(&device, 0));
	checkCudaErrors(cuDeviceGetAttribute(&cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
	checkCudaErrors(cuDeviceGetAttribute(&cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
	checkCudaErrors(cuDeviceGetAttribute(&clock_rate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device));
	checkCudaErrors(cuCtxCreate(&context, CU_CTX_SCHED_AUTO, device));
	checkCudaErrors(cuModuleLoad(&module, "hyperQ.cubin"));
	checkCudaErrors(cuModuleGetFunction(&function, module, "hyperQ"));
	num_clocks = clock_rate * num_milliseconds;
	streams = (CUstream*)malloc(sizeof(CUstream) * num_streams);
	for (s = 0; s < num_streams; ++s)
	{
		checkCudaErrors(cuStreamCreate(&streams[s], CU_STREAM_DEFAULT));
	}
	checkCudaErrors(cuEventCreate(&beg, CU_EVENT_DEFAULT));
	checkCudaErrors(cuEventCreate(&end, CU_EVENT_DEFAULT));
	checkCudaErrors(cuEventRecord(beg, 0));
	void* args[] = { &num_clocks };
	for (s = 0; s < num_streams; ++s)
	{
		for (k = 0; k < num_kernels; ++k)
		{
			checkCudaErrors(cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, streams[s], args, NULL));
		}
	}
	checkCudaErrors(cuEventRecord(end, 0));
	checkCudaErrors(cuEventSynchronize(end));
	checkCudaErrors(cuEventElapsedTime(&elapsed, beg, end));
	cuEventDestroy(end);
	cuEventDestroy(beg);
	for (s = 0; s < num_streams; ++s)
	{
		cuStreamDestroy(streams[s]);
	}
	free(streams);
	checkCudaErrors(cuModuleUnload(module));
	checkCudaErrors(cuCtxDestroy(context));
	printf("%d streams, each %d kernels, each %d ms\n", num_streams, num_kernels, num_milliseconds);
	printf("       SM <= 1.3:%4d ms\n", num_milliseconds * num_kernels * num_streams);
	printf("2.0 <= SM <= 3.0:%4d ms\n", num_milliseconds * (1 + (num_kernels - 1) * num_streams));
	printf("3.5 <= SM       :%4d ms\n", num_milliseconds * num_kernels);
	printf("       SM == %d.%d:%4d ms\n", cc_major, cc_minor, (int)elapsed);
}
