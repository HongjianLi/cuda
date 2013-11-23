#include <vector>
#include <future>
#include <ctime>
#include <boost/asio/io_service.hpp>
#include <boost/filesystem/operations.hpp>
#include "../cu_helper.h"
using namespace std;
using namespace boost::asio;
using namespace boost::filesystem;

inline void spin(const clock_t num_clocks)
{
	for (const clock_t threshold = clock() + num_clocks; clock() < threshold;);
}

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

	path p;
};

template <typename T>
class safe_counter
{
public:
	safe_counter(const T n) : n(n), i(0) {}
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
	const unsigned int concurrency = 2;

	// Initialize variables.
	srand(time(0));
	auto h_p = static_cast<float*>(malloc(sizeof(float) * 16));
	for (int i = 0; i < 16; ++i)
	{
		h_p[i] = rand() / static_cast<float>(RAND_MAX);
	}

	// Create a thread pool.
	vector<future<void>> futures;
	io_service io;
	unique_ptr<io_service::work> w(new io_service::work(io));
	for (int i = 0; i < concurrency; ++i)
	{
		futures.emplace_back(async(launch::async, [&]()
		{
			io.run();
		}));
	}

	// Initialize the CUDA driver API.
	checkCudaErrors(cuInit(0));

	// Get the number of devices with compute capability 1.0 or greater that are available for execution.
	int num_devices;
	checkCudaErrors(cuDeviceGetCount(&num_devices));

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
		checkCudaErrors(cuModuleLoad(&module, "monte_carlo.cubin")); // nvcc -cubin -arch=sm_11 -G, and use cuda-gdb
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

	// Loop over the ligands in the specified folder.
	const directory_iterator const_dir_iter;
	for (directory_iterator dir_iter("."); dir_iter != const_dir_iter; ++dir_iter)
	{
		// Parse the ligand.
		const ligand lig(dir_iter->path());

		// Check for new atom types
		const size_t num_types = 4;
		if (num_types)
		{
			safe_counter<size_t> c(num_types);
			for (size_t i = 0; i < num_types; ++i)
			{
				io.post([&]()
				{
					c.increment();
				});
			}
			c.wait();
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
		printf("main, lig = %s, dev = %d\n", lig.p.c_str(), dev);

		io.post(bind<void>([&,dev](const ligand lig)
		{
			printf("work, lig = %s, dev = %d\n", lig.p.c_str(), dev);

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
			checkCudaErrors(cuLaunchKernel(functions[dev], gws / lws, 1, 1, lws, 1, 1, 0, NULL, (void*[]){ &d_s, &d_l }, NULL));
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
			//		lig->write();
			//		safe_printf();
			checkCudaErrors(cuMemFreeHost(h_e));
			checkCudaErrors(cuCtxPopCurrent(NULL));
			idle.safe_push_back(dev);
		}, move(lig)));
	}

	// Wait until the io service has finished all its tasks.
	printf("io.destroy();\n");
	w.reset();
	for (auto& f : futures)
	{
		f.get();
	}

	// Destroy contexts.
	printf("cuCtxDestroy\n");
	for (auto& context : contexts)
	{
		checkCudaErrors(cuCtxDestroy(context));
	}
	printf("exiting\n");
}
