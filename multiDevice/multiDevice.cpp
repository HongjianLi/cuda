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

template <typename T>
class safe_vector : public vector<T>
{
public:
	void safe_push_back(const T x)
	{
		lock_guard<mutex> guard(m);
		this->push_back(x);
		cv.notify_one();
	}
	T safe_pop_back()
	{
		unique_lock<mutex> lock(m);
		if (this->empty()) cv.wait(lock);
		const T x = this->back();
		this->pop_back();
		return x;
	}
private:
	mutex m;
	condition_variable cv;
};

class ligand
{
public:
	ligand(const path p) : p(p)
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
	path p;
	int dev;
	int* h_l;
	int* h_e;
	CUdeviceptr d_s;
	CUdeviceptr d_l;
};

template <typename T>
class callback_data_wrapper
{
public:
	callback_data_wrapper(const T e, safe_vector<T>& v, ligand&& lig_) : e(e), v(v), lig(move(lig_)) {}
	const T e;
	safe_vector<T>& v;
	ligand lig;
};

int main(int argc, char* argv[])
{
	// Initialize constants.
	const unsigned int lws = 256;
	const unsigned int gws = 32 * lws;

	// Initialize variables.
	srand(time(0));
	io_service_pooled io(2);

	// Initialize the CUDA driver API.
	checkCudaErrors(cuInit(0));

	// Get the number of devices with compute capability 1.0 or greater that are available for execution.
	int num_devices;
	checkCudaErrors(cuDeviceGetCount(&num_devices));

	// Initialize containers of contexts and functions.
	vector<CUcontext> contexts;
	vector<CUfunction> functions;
	safe_vector<int> idle;
	contexts.reserve(num_devices);
	functions.reserve(num_devices);
	idle.reserve(num_devices);

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

		// Initialize symbols.
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
/*		callback_data_wrapper<int> w(contexts.size(), idle);
		checkCudaErrors(cuStreamAddCallback(0, []CUDA_CB (CUstream stream, CUresult error, void* data)
		{
			checkCudaErrors(error);
			auto w = reinterpret_cast<callback_data_wrapper<int>*>(data);
			w->v.safe_push_back(w->e);
		}, &w, 0));*/

		// Pop the current context.
		checkCudaErrors(cuCtxPopCurrent(NULL));

		// Save context and function.
		contexts.push_back(context);
		functions.push_back(function);
	}
	idle.resize(contexts.size());
	iota(idle.begin(), idle.end(), 0);

	// Initialize a default constructed directory_iterator acts as the end iterator.
	const directory_iterator const_dir_iter;
	for (directory_iterator dir_iter("."); dir_iter != const_dir_iter; ++dir_iter)
	{
		printf("%s\n", dir_iter->path().c_str());
//		io.post([&]()
//		{
//		});

		// Parse the ligand.
		ligand lig(dir_iter->path());

		// Wait until a device is ready for execution.
		int dev = idle.safe_pop_back();

		// Check for new atom types
		{
//			lock_guard<mutex> guard(m);
			if (false)
			{
				io.init(1);
				io.post([&]()
				{
					io.done();
				});
				io.sync();
				for (auto& context : contexts)
				{
					checkCudaErrors(cuCtxPushCurrent(context));
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
		callback_data_wrapper<int> w(dev, idle, move(lig));
		checkCudaErrors(cuStreamAddCallback(0, []CUDA_CB (CUstream stream, CUresult error, void* data)
		{
			checkCudaErrors(error);
			auto w = reinterpret_cast<callback_data_wrapper<int>*>(data);
			printf("loop callback, dev = %d, lig = %s\n", w->e, w->lig.p.c_str());
			w->v.safe_push_back(w->e);
/*			auto io = (io_service_pooled*)data;
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
			});*/
		}, &w, 0));
		checkCudaErrors(cuCtxPopCurrent(NULL));
	}

	// Wait until all results are written.
//	io.sync();

	// Cleanup.
	for (auto& context : contexts)
	{
		checkCudaErrors(cuCtxDestroy(context));
	}
	printf("exiting\n");
}
