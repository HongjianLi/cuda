#include <vector>
#include <mutex>
#include <condition_variable>
#include <ctime>
#include "../cu_helper.h"
using namespace std;

static mutex m;
static condition_variable completion;
vector<int> idleDevices;

void spin(const clock_t num_clocks)
{
	for (const clock_t threshold = clock() + num_clocks; clock() < threshold;);
}

// The callback will block later work in the stream until it is finished. Callbacks must return promptly. Callbacks must not make any CUDA API calls.
void CUDA_CB callback(CUstream stream, CUresult error, void* data)
{
	checkCudaErrors(error);
	lock_guard<mutex> guard(m);
//	idleDevices.push_back(*(int*)data);
	completion.notify_one();
}

int main(int argc, char* argv[])
{
	// Initialize constants.
	const size_t num_ligands = 20;
	clock_t num_clocks = 1e+9;

	// Initialize the CUDA driver API.
	checkCudaErrors(cuInit(0));

	// Get the number of devices with compute capability 1.0 or greater that are available for execution.
	int num_devices;
	checkCudaErrors(cuDeviceGetCount(&num_devices));

	// Initialize containers of contexts and functions.
	vector<CUcontext> contexts;
	vector<CUfunction> functions;
	contexts.reserve(num_devices);
	functions.reserve(num_devices);

	// Populate containers of contexts and functions.
	for (int i = 0; i < num_devices; ++i)
	{
		// Get a device handle.
		CUdevice device;
		checkCudaErrors(cuDeviceGet(&device, i));

		// Filter devices with compute capability 1.1 or greater, which is required by cuStreamAddCallback and cuMemHostGetDevicePointer.
		int compute_capability_major;
		checkCudaErrors(cuDeviceGetAttribute(&compute_capability_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
		if (compute_capability_major == 1)
		{
			int compute_capability_minor;
			checkCudaErrors(cuDeviceGetAttribute(&compute_capability_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
			if (compute_capability_minor < 1) continue;
		}

		// Create context, module and function.
		CUcontext context;
		checkCudaErrors(cuCtxCreate(&context, CU_CTX_SCHED_AUTO, device));
		CUmodule module;
		checkCudaErrors(cuModuleLoad(&module, "multiDevice.cubin")); // nvcc -cubin -arch=sm_11 -G, and use cuda-gdb
		CUfunction function;
		checkCudaErrors(cuModuleGetFunction(&function, module, "spin"));

		// Enqueue a synchronization event to make sure the device is ready for execution.
		size_t idx = contexts.size();
		checkCudaErrors(cuStreamAddCallback(0, []CUDA_CB (CUstream stream, CUresult error, void* data)
		{
			checkCudaErrors(error);
			lock_guard<mutex> guard(m);
			printf("%d\n", *(int*)data);
			idleDevices.push_back(*(int*)data);
			completion.notify_one();
		}, &idx, 0));

		// Pop the current context.
		checkCudaErrors(cuCtxPopCurrent(NULL));

		// Save context and function.
		contexts.push_back(context);
		functions.push_back(function);
	}

	// Update the number of devices with compute capability 1.1 or greater.
	num_devices = contexts.size();

	size_t num_completed_tasks = 0;
//	while (num_completed_tasks < num_ligands)
	{
		unique_lock<mutex> lock(m);
		completion.wait(lock);

		// Make a context current.
//		checkCudaErrors(cuCtxPushCurrent(contexts[i]));

		// Pop the current context.
//		checkCudaErrors(cuCtxPopCurrent(NULL));
	}


	// Cleanup.
	for (int i = 0; i < num_devices; ++i)
	{
		checkCudaErrors(cuCtxDestroy(contexts[i]));
	}
}
