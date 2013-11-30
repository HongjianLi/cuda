#include <iostream>
#include <iomanip>
#include <ctime>
#include <boost/filesystem/operations.hpp>
#include "../cu_helper.h"
#include "io_service_pool.hpp"
using namespace boost::filesystem;

void spin(const clock_t num_clocks)
{
	for (const clock_t threshold = clock() + num_clocks; clock() < threshold;);
}

class ligand
{
public:
	explicit ligand(const path& p) : filename(p.filename())
	{
		spin(1e+4);
	}

	void encode(float* const ligh, const unsigned int lws) const
	{
		for (int i = 0; i < lws; ++i)
		{
			ligh[i] = rand() / static_cast<float>(RAND_MAX);
		}
		spin(1e+3);
	}

	void write(const float* const cnfh) const
	{
		spin(1e+5);
	}

	path filename;
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
		if (i < n) cv.wait(lock);
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


template <typename T>
class callback_data_base
{
public:
	callback_data_base(const T dev, safe_vector<T>& idle) : dev(dev), idle(idle) {}
	const T dev;
	safe_vector<T>& idle;
};

template <typename T>
class callback_data : public callback_data_base<T>
{
public:
	callback_data(const T dev, safe_vector<T>& idle, io_service_pool& io, vector<float*>& cnfh, vector<float*>& ligh, vector<float>& prmh, safe_function& safe_print, size_t& num_ligands, ligand&& lig_) : callback_data_base<T>(dev, idle), io(io), cnfh(cnfh), ligh(ligh), prmh(prmh), safe_print(safe_print), num_ligands(num_ligands), lig(move(lig_)) {}
	io_service_pool& io;
	vector<float*>& cnfh;
	vector<float*>& ligh;
	vector<float> & prmh;
	safe_function& safe_print;
	size_t& num_ligands;
	ligand lig;
};

int main(int argc, char* argv[])
{
	// Initialize constants.
	const unsigned int lws = 256;
	const unsigned int gws = 32 * lws;
	const unsigned int num_threads = thread::hardware_concurrency();

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
	srand(time(0));
	vector<float> prmh(16);
	for (auto& prm : prmh)
	{
		prm = rand() / static_cast<float>(RAND_MAX);
	}

	// Create an io service pool for host.
	io_service_pool ioh(num_threads);
	safe_counter<size_t> cnt;
	safe_function safe_print;

	// Initialize containers of contexts, streams and functions.
	vector<CUcontext> contexts;
	vector<CUstream> streams;
	vector<CUfunction> functions;
	vector<float*> ligh;
	vector<CUdeviceptr> ligd;
	vector<CUdeviceptr> slnd;
	vector<float*> cnfh;
	safe_vector<int> idle;
	contexts.reserve(num_devices);
	streams.reserve(num_devices);
	functions.reserve(num_devices);
	ligh.reserve(num_devices);
	ligd.reserve(num_devices);
	slnd.reserve(num_devices);
	cnfh.reserve(num_devices);
	idle.reserve(num_devices);
	for (int dev = 0; dev < num_devices; ++dev)
	{
		// Get a device handle.
		CUdevice device;
		checkCudaErrors(cuDeviceGet(&device, dev));

		// Filter devices with compute capability 1.1 or greater, which is required by cuStreamAddCallback and cuMemHostGetDevicePointer.
		int compute_capability_major;
		checkCudaErrors(cuDeviceGetAttribute(&compute_capability_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
		if (compute_capability_major == 1)
		{
			int compute_capability_minor;
			checkCudaErrors(cuDeviceGetAttribute(&compute_capability_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
			if (compute_capability_minor < 1) continue;
		}

		// Create context.
		CUcontext context;
		checkCudaErrors(cuCtxCreate(&context, CU_CTX_SCHED_AUTO | CU_CTX_MAP_HOST, device));

		// Create stream.
		CUstream stream;
		checkCudaErrors(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

		// Load module.
		CUmodule module;
		checkCudaErrors(cuModuleLoad(&module, "multiDevice.fatbin")); // nvcc -fatbin -arch=sm_11 -G, and use cuda-gdb

		// Get function from module.
		CUfunction function;
		checkCudaErrors(cuModuleGetFunction(&function, module, "monte_carlo"));

		// Initialize symbols.
		CUdeviceptr prmd;
		size_t prms;
		checkCudaErrors(cuModuleGetGlobal(&prmd, &prms, module, "p"));
		assert(prms == sizeof(float) * 16);
		checkCudaErrors(cuMemcpyHtoD(prmd, prmh.data(), prms));

		// Allocate ligh, ligd, slnd and cnfh.
		checkCudaErrors(cuMemHostAlloc((void**)&ligh[dev], sizeof(float) * lws, CU_MEMHOSTALLOC_DEVICEMAP));
		checkCudaErrors(cuMemHostGetDevicePointer(&ligd[dev], ligh[dev], 0));
		checkCudaErrors(cuMemAlloc(&slnd[dev], sizeof(float) * gws));
		checkCudaErrors(cuMemHostAlloc((void**)&cnfh[dev], sizeof(float) * lws, 0));

		// Enqueue a synchronization event to make sure the device is ready for execution.
		checkCudaErrors(cuStreamAddCallback(NULL, []CUDA_CB (CUstream stream, CUresult error, void* data)
		{
			checkCudaErrors(error);
			const auto cbd = unique_ptr<callback_data_base<int>>(reinterpret_cast<callback_data_base<int>*>(data));
			cbd->idle.safe_push_back(cbd->dev);
		}, new callback_data_base<int>(contexts.size(), idle), 0));

		// Pop the current context.
		checkCudaErrors(cuCtxPopCurrent(NULL));

		// Save context, stream and function.
		contexts.push_back(context);
		streams.push_back(stream);
		functions.push_back(function);
	}

	// Loop over the ligands in the specified folder.
	size_t num_ligands = 0;
	cout.setf(ios::fixed, ios::floatfield);
	cout << "ID              Ligand D  pKd 1     2     3     4     5     6     7     8     9" << endl << setprecision(2);
	for (directory_iterator dir_iter("."), const_dir_iter; dir_iter != const_dir_iter; ++dir_iter)
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

			const size_t map_bytes = sizeof(float) * 1e+5;
			for (auto& context : contexts)
			{
				checkCudaErrors(cuCtxPushCurrent(context));
				float* maph;
				checkCudaErrors(cuMemHostAlloc((void**)&maph, map_bytes, 0));
				CUdeviceptr mapd;
				checkCudaErrors(cuMemAlloc(&mapd, map_bytes));
				checkCudaErrors(cuMemcpyHtoD(mapd, maph, map_bytes));
				checkCudaErrors(cuMemFreeHost(maph));
				checkCudaErrors(cuCtxPopCurrent(NULL));
			}
		}

		// Wait until a device is ready for execution.
		const int dev = idle.safe_pop_back();

		// Push the context of the chosen device.
		checkCudaErrors(cuCtxPushCurrent(contexts[dev]));

		// Encode the current ligand.
		lig.encode(ligh[dev], lws);

		const size_t lig_bytes = sizeof(float) * lws;

		// Clear the solution buffer.
		checkCudaErrors(cuMemsetD32Async(slnd[dev], 0, gws, streams[dev]));

		// Launch kernel.
		void* params[] = { &slnd[dev], &ligd[dev] };
		checkCudaErrors(cuLaunchKernel(functions[dev], gws / lws, 1, 1, lws, 1, 1, lig_bytes, streams[dev], params, NULL));

		// Copy conformations from device memory to host memory.
		checkCudaErrors(cuMemcpyDtoHAsync(cnfh[dev], slnd[dev], sizeof(float) * lws, streams[dev]));

		// Add callback.
		checkCudaErrors(cuStreamAddCallback(streams[dev], []CUDA_CB (CUstream stream, CUresult error, void* data)
		{
			checkCudaErrors(error);
			const auto cbdp = reinterpret_cast<callback_data<int>*>(data);

			cbdp->io.post([=]()
			{
				const auto cbd = unique_ptr<callback_data<int>>(cbdp);
				const auto dev = cbd->dev;
				const auto& cnfh = cbd->cnfh;
				const auto& ligh = cbd->ligh;
				const auto& prmh = cbd->prmh;
				auto& safe_print = cbd->safe_print;
				auto& num_ligands = cbd->num_ligands;
				auto& lig = cbd->lig;

				// Validate results.
				for (int i = 0; i < lws; ++i)
				{
					const float actual = cnfh[dev][i];
					const float expected = ligh[dev][i] * 2.0f + 1.0f + prmh[i % 16];
					if (fabs(actual - expected) > 1e-7)
					{
						printf("cnfh[%d] = %f, expected = %f\n", i, actual, expected);
						break;
					}
				}

				// Write conformations.
				lig.write(cnfh[dev]);

				// Output and save ligand stem and predicted affinities.
				safe_print([&]()
				{
					cout << setw(2) << ++num_ligands << setw(20) << lig.filename.string() << setw(2) << dev << ' ';
					for (int i = 0; i < 9; ++i)
					{
						cout << setw(6) << cnfh[dev][i];
					}
					cout << endl;
				});

				// Signal the main thread to post another task.
				cbd->idle.safe_push_back(cbd->dev);
			});
		}, new callback_data<int>(dev, idle, ioh, cnfh, ligh, prmh, safe_print, num_ligands, move(lig)), 0));

		// Pop the context after use.
		checkCudaErrors(cuCtxPopCurrent(NULL));
	}

	// Wait until the io service for host has finished all its tasks.
	ioh.wait();

	// Destroy contexts.
	for (auto& context : contexts)
	{
		checkCudaErrors(cuCtxDestroy(context));
	}

	cout << "Writing log records of " << num_ligands << " ligands to the log file" << endl;
}
