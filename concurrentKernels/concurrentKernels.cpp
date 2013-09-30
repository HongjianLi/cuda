#include <time.h>
#include "../cu_helper.h"

int main(int argc, char *argv[])
{
	int num_kernels = 8;
	int num_streams = num_kernels + 1;
	int num_bytes = sizeof(clock_t) * num_kernels;
	float kernel_time = 10;
	int clock_rate;
	clock_t time_clocks;
	CUdevice device;
	CUcontext context;
	CUmodule module;
	CUfunction clock_block, sum;
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
	checkCudaErrors(cuModuleLoad(&module, "concurrentKernels.cubin"));
	checkCudaErrors(cuModuleGetFunction(&clock_block, module, "clock_block"));
	checkCudaErrors(cuModuleGetFunction(&sum, module, "sum"));
	checkCudaErrors(cuMemAllocHost((void **)&h_a, num_bytes));
	checkCudaErrors(cuMemAlloc(&d_a, num_bytes));
	streams = (CUstream*)malloc(sizeof(CUstream) * num_streams);
	for (i = 0; i < num_streams; ++i)
	{
		checkCudaErrors(cuStreamCreate(&streams[i], CU_STREAM_DEFAULT));
	}
	checkCudaErrors(cuEventCreate(&beg, CU_EVENT_DEFAULT));
	checkCudaErrors(cuEventCreate(&end, CU_EVENT_DEFAULT));
	events = (CUevent*)malloc(sizeof(CUevent) * num_kernels);
	for (i = 0; i < num_kernels; ++i)
	{
		checkCudaErrors(cuEventCreate(&events[i], CU_EVENT_DISABLE_TIMING));
	}
	time_clocks = (clock_t)(clock_rate * kernel_time);
	checkCudaErrors(cuEventRecord(beg, 0));
	for (i = 0; i < num_kernels; ++i)
	{
		CUdeviceptr addr = d_a + i;
		void* args[] = { &addr, &time_clocks };
		checkCudaErrors(cuLaunchKernel(clock_block, 1, 1, 1, 1, 1, 1, 0, streams[i], args, NULL));
		checkCudaErrors(cuEventRecord(events[i], streams[i]));
		checkCudaErrors(cuStreamWaitEvent(streams[num_streams-1], events[i], 0));
	}
	void* args[] = { &d_a, &num_kernels };
	checkCudaErrors(cuLaunchKernel(sum, 1, 1, 1, 32, 1, 1, 0, streams[num_streams - 1], args, NULL));
	checkCudaErrors(cuMemcpyDtoHAsync(h_a, d_a, sizeof(clock_t), streams[num_streams - 1]));
	checkCudaErrors(cuEventRecord(end, 0));
	checkCudaErrors(cuEventSynchronize(end));
	checkCudaErrors(cuEventElapsedTime(&elapsed_time, beg, end));
	printf("Expected time for serial execution of %d kernels = %.3fms\n", num_kernels, kernel_time * num_kernels);
	printf("Expected time for concurrent execution of %d kernels = %.3fms\n", num_kernels, kernel_time);
	printf("Measured time for sample = %.3fms\n", elapsed_time);
	if (h_a[0] <= time_clocks * num_kernels)
	{
		printf("Test failed!\n");
	}
	for (int i = 0; i < num_kernels; ++i)
	{
		cuEventDestroy(events[i]);
	}
	free(events);
	cuEventDestroy(end);
	cuEventDestroy(beg);
	for (int i = 0; i < num_streams; ++i)
	{
		cuStreamDestroy(streams[i]);
	}
	free(streams);
	cuMemFree(d_a);
	cuMemFreeHost(h_a);
	checkCudaErrors(cuModuleUnload(module));
	checkCudaErrors(cuCtxDestroy(context));
}
