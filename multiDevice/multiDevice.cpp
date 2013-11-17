#include <ctime>
#include <boost/filesystem/operations.hpp>
#include "../cu_helper.h"
#include "io_service_pooled.hpp"
using namespace std;
using namespace boost::filesystem;

void spin(const clock_t num_clocks)
{
	for (const clock_t threshold = clock() + num_clocks; clock() < threshold;);
}

class ligand
{
public:
	ligand(const path p)
	{
		spin(1e+4); // Parse file.
	}

	void populate(int* h_l)
	{
		spin(1e+3); // Write data.
	}

	void write() const
	{
		spin(1e+5);
	}
//private:
	int dev;
	int* h_l;
	int* h_e;
	CUdeviceptr d_s;
	CUdeviceptr d_l;
};

int main(int argc, char* argv[])
{
	// Initialize constants.
	const size_t num_ligands = 20;
	const unsigned int lws = 256;
	const unsigned int gws = 32 * lws;

	srand(time(0));
	io_service_pooled io(2);
	io.init(0);
	mutex m;
	condition_variable completion;
	vector<int> idle;

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
		checkCudaErrors(cuCtxCreate(&context, CU_CTX_SCHED_AUTO | CU_CTX_MAP_HOST, device));
		CUmodule module;
		checkCudaErrors(cuModuleLoad(&module, "monte_carlo.cubin")); // nvcc -cubin -arch=sm_11 -G, and use cuda-gdb
		CUfunction function;
		checkCudaErrors(cuModuleGetFunction(&function, module, "monte_carlo"));
		CUdeviceptr d_p;
		checkCudaErrors(cuModuleGetGlobal(&d_p, NULL, module, "p"));
		float* h_p;
		checkCudaErrors(cuMemAllocHost((void**)&h_p, sizeof(float) * 16));
		for (int i = 0; i < 16; ++i)
		{
			h_p[i] = rand() / (float)RAND_MAX;
		}
		checkCudaErrors(cuMemcpyHtoDAsync(d_p, h_p, sizeof(float) * 16, 0));

		// Enqueue a synchronization event to make sure the device is ready for execution.
		size_t idx = contexts.size();
		idle.push_back(0);
		checkCudaErrors(cuStreamAddCallback(0, []CUDA_CB (CUstream stream, CUresult error, void* data)
		{
			checkCudaErrors(error);
			printf("%d\n", *reinterpret_cast<int*>(data));
//			lock_guard<mutex> guard(m);
//			idle.push_back(*(int*)data);
//			completion.notify_one();
		}, &idx, 0));

		// Pop the current context.
		checkCudaErrors(cuCtxPopCurrent(NULL));

		// Save context and function.
		contexts.push_back(context);
		functions.push_back(function);
	}

	const directory_iterator const_dir_iter; // A default constructed directory_iterator acts as the end iterator.
	for (directory_iterator dir_iter("."); dir_iter != const_dir_iter; ++dir_iter)
	{
		printf("%s\n", dir_iter->path().c_str());
//		io.post([&]()
//		{
//		});

		// Parse the ligand.
		ligand lig(dir_iter->path());

		// Wait until a device is ready for execution.
		int dev;
		{
			unique_lock<mutex> lock(m);
			if (idle.empty()) completion.wait(lock);
			dev = idle.back();
			idle.pop_back();
		}

		// Check for new atom types
		{
			lock_guard<mutex> guard(m);
			if (false)
			{
				io.init(6);
				io.post([&]()
				{
					io.done();
				});
				io.sync();
				for (int i = 0; i < num_devices; ++i)
				{
					checkCudaErrors(cuCtxPushCurrent(contexts[i]));
//					checkCudaErrors(cuMemAlloc(&d_e, sizeof(float) * 1e+7));
//					checkCudaErrors(cuMemcpyHtoD(d_e, h_e, sizeof(float) * 1e+7));
					checkCudaErrors(cuCtxPopCurrent(NULL));
				}
			}
		}

		checkCudaErrors(cuCtxPushCurrent(contexts[dev]));
		checkCudaErrors(cuMemHostAlloc((void**)&lig.h_l, sizeof(float) * lws, CU_MEMHOSTALLOC_DEVICEMAP));
		for (int i = 0; i < lws; ++i)
		{
			lig.h_l[i] = rand() / (float)RAND_MAX;
		}
		checkCudaErrors(cuMemHostGetDevicePointer(&lig.d_l, lig.h_l, 0));
		checkCudaErrors(cuMemAlloc(&lig.d_s, sizeof(float) * gws));
		checkCudaErrors(cuMemsetD32Async(lig.d_s, 0, gws, 0));
		checkCudaErrors(cuLaunchKernel(functions[dev], 1, 1, 1, 1, 1, 1, 0, NULL, (void*[]){ &lig.d_s, &lig.d_l }, NULL));
		checkCudaErrors(cuMemHostAlloc((void**)&lig.h_e, sizeof(float) * lws, 0));
		checkCudaErrors(cuMemcpyDtoHAsync(lig.h_e, lig.d_s, sizeof(float) * lws, 0));
/*		checkCudaErrors(cuStreamAddCallback(0, []CUDA_CB (CUstream stream, CUresult error, void* data)
		{
			printf("callback\n");
			checkCudaErrors(error);
			auto io = (io_service_pooled*)data;
			io->done();
			ligand* lig = (ligand*)data;
			checkCudaErrors(error);
			lock_guard<mutex> guard(m);
			idle.push_back(lig->dev);
			completion.notify_one();
			io.post([&]()
			{
				lig->write();
				safe_printf();
				checkCudaErrors(cuCtxPushCurrent(contexts[lig->dev]));
				checkCudaErrors(cuMemFree(lig->d_s));
				checkCudaErrors(cuMemFreeHost(lig->h_e));
				checkCudaErrors(cuMemFreeHost(lig->h_l));
				checkCudaErrors(cuCtxPopCurrent(NULL));
			});
		}, &lig, 0));*/
		checkCudaErrors(cuCtxPopCurrent(NULL));
	}

	// Wait until all results are written.
	io.sync();

	// Cleanup.
	for (auto& c : contexts)
	{
		checkCudaErrors(cuCtxDestroy(c));
	}
}
