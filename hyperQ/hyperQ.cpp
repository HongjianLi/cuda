#include <time.h>
#include "../cu_helper.h"

int main(int argc, char *argv[])
{
	int num_streams = 32;
	float kernel_time = 10;
	int clock_rate;
	clock_t time_clocks;
	CUdevice device;
	CUcontext context;
	CUmodule module;
	CUfunction kernel_A, kernel_B, sum;
	clock_t *h_a;
	CUdeviceptr d_a;
	CUstream *streams;
	float elapsed_time;
	CUevent beg, end, *events;
	int i;
	checkCudaErrors(cuInit(0));
	checkCudaErrors(cuDeviceGetAttribute(&clock_rate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, 0));
	checkCudaErrors(cuDeviceGet(&device, 0));
	checkCudaErrors(cuCtxCreate(&context, CU_CTX_SCHED_AUTO, device));
	checkCudaErrors(cuModuleLoad(&module, "hyperQ.cubin"));
	checkCudaErrors(cuModuleGetFunction(&kernel_A, module, "kernel_A"));
	checkCudaErrors(cuModuleGetFunction(&kernel_B, module, "kernel_B"));
	checkCudaErrors(cuModuleGetFunction(&sum, module, "sum"));
	checkCudaErrors(cuMemAllocHost((void **)&h_a, sizeof(clock_t)));
	checkCudaErrors(cuMemAlloc(&d_a, 2 * num_streams * sizeof(clock_t)));
	streams = (CUstream*)malloc(sizeof(CUstream) * num_streams);
	for (i = 0; i < num_streams; ++i)
	{
		checkCudaErrors(cuStreamCreate(&streams[i], CU_STREAM_DEFAULT));
	}
	checkCudaErrors(cuEventCreate(&beg, CU_EVENT_DEFAULT));
	checkCudaErrors(cuEventCreate(&end, CU_EVENT_DEFAULT));
	time_clocks = (clock_t)(clock_rate * kernel_time);
	clock_t total_clocks = 0;
	checkCudaErrors(cuEventRecord(beg, 0));
	for (int i = 0; i < num_streams; ++i)
	{
		CUdeviceptr d_i0 = d_a + sizeof(clock_t) * i * 2;
		CUdeviceptr d_i1 = d_a + sizeof(clock_t) * i * 2 + 1;
		void* args[] = { &d_i0, &time_clocks };
		checkCudaErrors(cuLaunchKernel(kernel_A, 1, 1, 1, 1, 1, 1, 0, streams[i], args, NULL));
		args[0] = &d_i1;
		checkCudaErrors(cuLaunchKernel(kernel_B, 1, 1, 1, 1, 1, 1, 0, streams[i], args, NULL));
	}
	checkCudaErrors(cuEventRecord(end, 0));
	int num_streams_2 = num_streams * 2;
	void* args[] = { &d_a, &num_streams_2 };
	checkCudaErrors(cuLaunchKernel(sum, 1, 1, 1, 32, 1, 1, 0, 0, args, NULL));
//	checkCudaErrors(cuMemcpyDtoHAsync(h_a, d_a, sizeof(clock_t), 0));
	checkCudaErrors(cuMemcpyDtoH(h_a, d_a, sizeof(clock_t)));
	checkCudaErrors(cuEventSynchronize(end));
	checkCudaErrors(cuEventElapsedTime(&elapsed_time, beg, end));
	printf("Expected time for serial execution of %d sets of kernels is between approx. %.3fms and %.3fms\n", num_streams, (num_streams + 1) * kernel_time, 2 * num_streams *kernel_time);
	printf("Expected time for fully concurrent execution of %d sets of kernels is approx. %.3fms\n", num_streams, 2 * kernel_time);
	printf("Measured time for sample = %.3fms\n", elapsed_time);
	if (h_a[0] <= time_clocks * num_streams * 2)
	{
		printf("Test failed!\n");
	}
	for (int i = 0; i < num_streams; ++i)
	{
		cuStreamDestroy(streams[i]);
	}
	free(streams);
	cuEventDestroy(end);
	cuEventDestroy(beg);
	cuMemFree(d_a);
	cuMemFreeHost(h_a);
	checkCudaErrors(cuModuleUnload(module));
	checkCudaErrors(cuCtxDestroy(context));
}
