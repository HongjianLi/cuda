#include <iostream>
#include <iomanip>
#include <ctime>
#include <boost/filesystem/operations.hpp>
#include "../cu_helper.h"
#include "io_service_pool.hpp"
using namespace boost::filesystem;

inline void spin(const clock_t num_clocks)
{
	for (const clock_t threshold = clock() + num_clocks; clock() < threshold;);
}

class ligand
{
public:
	explicit ligand(const path p) : p(p)
	{
		spin(1e+4); // Parse file.
	}

	void populate(int* h_l)
	{
		spin(1e+3); // Write data.
	}

	void write(const float* ex) const
	{
		spin(1e+5);
	}

	path p;
	vector<int> atoms;
};

class safe_function
{
public:
	void operator()(function<void(void)>&& f)
	{
		lock_guard<mutex> guard(m);
		f();
	}
private:
	mutex m;
};

template <typename T>
class safe_counter
{
public:
	void init(const T z)
	{
		n = z;
		i = 0;
	}
	void increment()
	{
		lock_guard<mutex> guard(m);
		if (++i == n) cv.notify_one();
	}
	void wait()
	{
		unique_lock<mutex> lock(m);
		cv.wait(lock);
	}
private:
	mutex m;
	condition_variable cv;
	T n;
	T i;
};

template <typename T>
class safe_vector : public vector<T>
{
public:
	using vector<T>::vector;
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

int main(int argc, char* argv[])
{
	// Initialize constants.
	const unsigned int lws = 256;
	const unsigned int gws = 32 * lws;
	const unsigned int num_threads = 2;

	// Get the number of devices with compute capability 1.0 or greater that are available for execution.
	cout << "Detecting CUDA devices" << endl;
	checkCudaErrors(cuInit(0));
	int num_devices;
	checkCudaErrors(cuDeviceGetCount(&num_devices));
	if (!num_devices)
	{
		cerr << "No CUDA devices detected" << endl;
		return 2;
	}

	// Initialize variables.
	cout.setf(ios::fixed, ios::floatfield);
	cout << setprecision(2);
	srand(time(0));
	auto h_p = static_cast<float*>(malloc(sizeof(float) * 16));
	for (int i = 0; i < 16; ++i)
	{
		h_p[i] = rand() / static_cast<float>(RAND_MAX);
	}

	// Create an io service for host.
	io_service_pool ioh(num_threads);
	safe_counter<size_t> cnt;

	// Initialize containers of contexts and functions.
	vector<CUcontext> contexts(num_devices);
	vector<CUfunction> functions(num_devices);
	vector<int> can_map_host_memory(num_devices);

	// Populate containers of contexts and functions.
	for (int dev = 0; dev < num_devices; ++dev)
	{
		// Get a device handle from an ordinal.
		CUdevice device;
		checkCudaErrors(cuDeviceGet(&device, dev));

		// Check if the device can map host memory into CUDA address space.
		checkCudaErrors(cuDeviceGetAttribute(&can_map_host_memory[dev], CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, device));

		// Create context, module and function.
		checkCudaErrors(cuCtxCreate(&contexts[dev], CU_CTX_SCHED_AUTO | CU_CTX_MAP_HOST, device));
		CUmodule module;
		checkCudaErrors(cuModuleLoad(&module, "monte_carlo.fatbin")); // nvcc -cubin -arch=sm_11 -G, and use cuda-gdb
		checkCudaErrors(cuModuleGetFunction(&functions[dev], module, "monte_carlo"));

		// Initialize symbols.
		CUdeviceptr d_p;
		checkCudaErrors(cuModuleGetGlobal(&d_p, NULL, module, "p"));
		checkCudaErrors(cuMemcpyHtoD(d_p, h_p, sizeof(float) * 16));

		// Pop the current context.
		checkCudaErrors(cuCtxPopCurrent(NULL));
	}

	// Initialize a vector of idle devices.
	safe_vector<int> idle(num_devices);
	iota(idle.begin(), idle.end(), 0);

	// Create an io service for device.
	io_service_pool iod(num_devices);
	safe_function safe_print;

	// Loop over the ligands in the specified folder.
	const directory_iterator const_dir_iter;
	for (directory_iterator dir_iter("."); dir_iter != const_dir_iter; ++dir_iter)
	{
		// Parse the ligand.
		ligand lig(dir_iter->path());

		// Check for new atom types
		const size_t num_types = 4;
		if (num_types)
		{
			cnt.init(num_types);
			for (size_t i = 0; i < num_types; ++i)
			{
				ioh.post([&]()
				{
					cnt.increment();
				});
			}
			cnt.wait();
			const size_t numBytes = sizeof(float) * 1e+7;
			for (auto& context : contexts)
			{
				checkCudaErrors(cuCtxPushCurrent(context));
//				checkCudaErrors(cuMemAlloc(&d_e, numBytes));
//				checkCudaErrors(cuMemcpyHtoD(d_e, h_e, numBytes));
				checkCudaErrors(cuCtxPopCurrent(NULL));
			}
		}

		// Wait until a device is ready for execution.
		const int dev = idle.safe_pop_back();

		iod.post(bind<void>([&,dev](const ligand& lig)
		{
			checkCudaErrors(cuCtxPushCurrent(contexts[dev]));
			float* h_l;
			checkCudaErrors(cuMemHostAlloc((void**)&h_l, sizeof(float) * lws, CU_MEMHOSTALLOC_DEVICEMAP));
			for (int i = 0; i < lws; ++i)
			{
				h_l[i] = rand() / static_cast<float>(RAND_MAX);
			}
			CUdeviceptr d_l;
			if (can_map_host_memory[dev])
			{
				checkCudaErrors(cuMemHostGetDevicePointer(&d_l, h_l, 0));
			}
			else
			{
				checkCudaErrors(cuMemAlloc(&d_l, sizeof(float) * lws));
				checkCudaErrors(cuMemcpyHtoDAsync(d_l, h_l, sizeof(float) * lws, NULL));
			}
			CUdeviceptr d_s;
			checkCudaErrors(cuMemAlloc(&d_s, sizeof(float) * gws));
			checkCudaErrors(cuMemsetD32Async(d_s, 0, gws, NULL));
			void* params[] = { &d_s, &d_l };
			checkCudaErrors(cuLaunchKernel(functions[dev], gws / lws, 1, 1, lws, 1, 1, 0, NULL, params, NULL));
			float* h_e;
			checkCudaErrors(cuMemHostAlloc((void**)&h_e, sizeof(float) * lws, 0));
			checkCudaErrors(cuMemcpyDtoHAsync(h_e, d_s, sizeof(float) * lws, NULL));
			checkCudaErrors(cuCtxSynchronize());
			checkCudaErrors(cuMemFree(d_s));
			if (!can_map_host_memory[dev])
			{
				checkCudaErrors(cuMemFree(d_l));
			}
			for (int i = 0; i < lws; ++i)
			{
				const float actual = h_e[i];
				const float expected = h_l[i] * 2.0f + 1.0f + h_p[i % 16];
				if (fabs(actual - expected) > 1e-7)
				{
					printf("h_e[%d] = %f, expected = %f\n", i, actual, expected);
					break;
				}
			}
			checkCudaErrors(cuMemFreeHost(h_l));
			lig.write(h_e);
			safe_print([&]()
			{
				cout << setw(1) << dev << setw(3) << 0 << setw(20) << lig.p.filename().string();
				for (int i = 0; i < 9; ++i)
				{
					cout << setw(6) << h_e[i];
				}
				cout << endl;
			});
			checkCudaErrors(cuMemFreeHost(h_e));
			checkCudaErrors(cuCtxPopCurrent(NULL));
			idle.safe_push_back(dev);
		}, move(lig)));
	}

	// Wait until the io service for host has finished all its tasks.
	ioh.wait();

	// Wait until the io service for device has finished all its tasks.
	iod.wait();

	// Destroy contexts.
	for (auto& context : contexts)
	{
		checkCudaErrors(cuCtxDestroy(context));
	}
}
