#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "../cu_helper.h"
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

double shrDeltaT()
{
#if defined(_WIN32)
	static LARGE_INTEGER old_time;
	LARGE_INTEGER new_time, freq;
	QueryPerformanceFrequency(&freq);
	QueryPerformanceCounter(&new_time);
	const double DeltaT = ((double)new_time.QuadPart - (double)old_time.QuadPart) / (double)freq.QuadPart;
	old_time = new_time;
	return DeltaT;
#elif defined(__unix__)
	static struct timeval old_time;
	struct timeval new_time;
	gettimeofday(&new_time, NULL);
	const double DeltaT = ((double)new_time.tv_sec + 1.0e-6 * (double)new_time.tv_usec) - ((double)old_time.tv_sec + 1.0e-6 * (double)old_time.tv_usec);
	old_time.tv_sec  = new_time.tv_sec;
	old_time.tv_usec = new_time.tv_usec;
	return DeltaT;
#elif defined (__APPLE__) || defined (MACOSX)
	static time_t old_time;
	time_t new_time;
	new_time  = clock();
	const double DeltaT = double(new_time - old_time) / CLOCKS_PER_SEC;
	old_time.tv_sec  = new_time.tv_sec;
	old_time.tv_usec = new_time.tv_usec;
	return DeltaT;
#else
	return 0;
#endif
}

int main(int argc, char* argv[])
{
	const int n = 4;
	const size_t sizes[n] = { 3 << 10, 15 << 10, 15 << 20, 100 << 20 };
	const int iterations[n] = { 60000, 60000, 300, 30 };
	printf("device,device name,size (B),memory,access,direction,time (s),bandwidth (MB/s)\n");

	checkCudaErrors(cuInit(0));
	int num_devices = 0;
	checkCudaErrors(cuDeviceGetCount(&num_devices));
	for (int d = 0; d < num_devices; ++d)
	{
		char device_name[256];
		checkCudaErrors(cuDeviceGetName(device_name, sizeof(device_name), d));

		CUdevice device;
		checkCudaErrors(cuDeviceGet(&device, d));
		CUcontext context;
		checkCudaErrors(cuCtxCreate(&context, 0, device));

		for (int s = 0; s < n; ++s)
		{
			const size_t size = sizes[s];
			const int iteration = iterations[s];
			const double bandwidth_unit = (double)size * iteration / (1 << 20);
			void* h_p;
			CUdeviceptr d_p;
			double time;
			double bandwidth;
			checkCudaErrors(cuMemAlloc(&d_p, size));

			// allocate pageable h_p
			h_p = malloc(size);
			// --memory=pageable --access=direct --htod
			shrDeltaT();
			for (int i = 0; i < iteration; ++i)
			{
				checkCudaErrors(cuMemcpyHtoD(d_p, h_p, size));
			}
			checkCudaErrors(cuCtxSynchronize());
			time = shrDeltaT();
			bandwidth = bandwidth_unit / time;
			printf("%d,%s,%lu,%s,%s,%s,%.3f,%.0f\n", d, device_name, size, "pageable", "direct", "HtoD", time, bandwidth);
			// --memory=pageable --access=direct --dtoh
			shrDeltaT();
			for (int i = 0; i < iteration; ++i)
			{
				checkCudaErrors(cuMemcpyDtoH(h_p, d_p, size));
			}
			checkCudaErrors(cuCtxSynchronize());
			time = shrDeltaT();
			bandwidth = bandwidth_unit / time;
			printf("%d,%s,%lu,%s,%s,%s,%.3f,%.0f\n", d, device_name, size, "pageable", "direct", "DtoH", time, bandwidth);
			// deallocate pageable h_p
			free(h_p);

			// allocate pinned h_p
	        checkCudaErrors(cuMemHostAlloc(&h_p, size, 0));
			// --memory=pinned --access=direct --htod
			shrDeltaT();
			for (int i = 0; i < iteration; ++i)
			{
				checkCudaErrors(cuMemcpyHtoDAsync(d_p, h_p, size, 0));
			}
			checkCudaErrors(cuCtxSynchronize());
			time = shrDeltaT();
			bandwidth = bandwidth_unit / time;
			printf("%d,%s,%lu,%s,%s,%s,%.3f,%.0f\n", d, device_name, size, "pinned", "direct", "HtoD", time, bandwidth);
			// --memory=pinned --access=direct --dtoh
			shrDeltaT();
			for (int i = 0; i < iteration; ++i)
			{
				checkCudaErrors(cuMemcpyDtoHAsync(h_p, d_p, size, 0));
			}
			checkCudaErrors(cuCtxSynchronize());
			time = shrDeltaT();
			bandwidth = bandwidth_unit / time;
			printf("%d,%s,%lu,%s,%s,%s,%.3f,%.0f\n", d, device_name, size, "pinned", "direct", "DtoH", time, bandwidth);
			// deallocate pinned h_p
			checkCudaErrors(cuMemFreeHost(h_p));

			// allocate writecombined h_p
	        checkCudaErrors(cuMemHostAlloc(&h_p, size, CU_MEMHOSTALLOC_WRITECOMBINED));
			// --memory=writecombined --access=direct --htod
			shrDeltaT();
			for (int i = 0; i < iteration; ++i)
			{
				checkCudaErrors(cuMemcpyHtoDAsync(d_p, h_p, size, 0));
			}
			checkCudaErrors(cuCtxSynchronize());
			time = shrDeltaT();
			bandwidth = bandwidth_unit / time;
			printf("%d,%s,%lu,%s,%s,%s,%.3f,%.0f\n", d, device_name, size, "writecombined", "direct", "HtoD", time, bandwidth);
			// --memory=writecombined --access=direct --dtoh
			shrDeltaT();
			for (int i = 0; i < iteration; ++i)
			{
				checkCudaErrors(cuMemcpyDtoHAsync(h_p, d_p, size, 0));
			}
			checkCudaErrors(cuCtxSynchronize());
			time = shrDeltaT();
			bandwidth = bandwidth_unit / time;
			printf("%d,%s,%lu,%s,%s,%s,%.3f,%.0f\n", d, device_name, size, "writecombined", "direct", "DtoH", time, bandwidth);
			// deallocate writecombined h_p
			checkCudaErrors(cuMemFreeHost(h_p));

			checkCudaErrors(cuMemFree(d_p));
		}
		checkCudaErrors(cuCtxDestroy(context));
	}
}
