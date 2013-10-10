#include <math.h>
#include "../cu_helper.h"

int main(int argc, char *argv[])
{
	const unsigned int lws = 256;
	const unsigned int gws = 1024 * lws;

	checkCudaErrors(cuInit(0));
	int num_devices;
	checkCudaErrors(cuDeviceGetCount(&num_devices));
	for (int d = 0; d < num_devices; ++d)
	{
		printf("DEVICE %d\n", d);
		char name[256];
		checkCudaErrors(cuDeviceGetName(name, sizeof(name), d));
		printf("DEVICE NAME: %s\n", name);
		int can_map_host_memory;
		checkCudaErrors(cuDeviceGetAttribute(&can_map_host_memory, CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, 0));
		printf("CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY: %d\n", can_map_host_memory);
		CUdevice device;
		checkCudaErrors(cuDeviceGet(&device, d));
		CUcontext context;
		checkCudaErrors(cuCtxCreate(&context, CU_CTX_MAP_HOST, device));

		CUmodule module;
		checkCudaErrors(cuModuleLoad(&module, "zeroCopy.cubin")); // nvcc -cubin -arch=sm_11 vectorAdd.cu
		CUfunction function;
		checkCudaErrors(cuModuleGetFunction(&function, module, "zeroCopy"));

		float* h_a;
		float* h_b;
		float* h_c;
		checkCudaErrors(cuMemHostAlloc((void**)&h_a, sizeof(float) * gws, CU_MEMHOSTALLOC_DEVICEMAP));
		checkCudaErrors(cuMemHostAlloc((void**)&h_b, sizeof(float) * gws, CU_MEMHOSTALLOC_DEVICEMAP));
		checkCudaErrors(cuMemHostAlloc((void**)&h_c, sizeof(float) * gws, CU_MEMHOSTALLOC_DEVICEMAP));
		for (int i = 0; i < gws; ++i)
		{
			h_a[i] = rand() / (float)RAND_MAX;
			h_b[i] = rand() / (float)RAND_MAX;
		}
		CUdeviceptr d_a;
		CUdeviceptr d_b;
		CUdeviceptr d_c;
		checkCudaErrors(cuMemHostGetDevicePointer(&d_a, h_a, 0));
		checkCudaErrors(cuMemHostGetDevicePointer(&d_b, h_b, 0));
		checkCudaErrors(cuMemHostGetDevicePointer(&d_c, h_c, 0));
		void* args[] = { &d_a, &d_b, &d_c };
		checkCudaErrors(cuLaunchKernel(function, gws / lws, 1, 1, lws, 1, 1, 0, NULL, args, NULL));
		checkCudaErrors(cuCtxSynchronize());
		bool passed = true;
		for (int i = 0; i < gws; ++i)
		{
			const float ref = h_a[i] + h_b[i];
			if (fabs(h_c[i] - ref) > 1e-7)
			{
				printf("i = %d, ref = %f, h_c[i] = %f\n", i, ref, h_c[i]);
				passed = false;
				break;
			}
		}
		printf("zeroCopy %s\n\n", passed ? "passed" : "failed");
		checkCudaErrors(cuMemFreeHost(h_a));
		checkCudaErrors(cuMemFreeHost(h_b));
		checkCudaErrors(cuMemFreeHost(h_c));
		checkCudaErrors(cuModuleUnload(module));
		checkCudaErrors(cuCtxDestroy(context));
	}
}
