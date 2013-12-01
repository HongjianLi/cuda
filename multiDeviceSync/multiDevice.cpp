#include <iostream>
#include <iomanip>
#include <ctime>
#include <fstream>
#include <boost/filesystem/operations.hpp>
#include "../cu_helper.h"
#include "io_service_pool.hpp"
using namespace boost::filesystem;

void spin(const clock_t num_clocks)
{
	for (const clock_t threshold = clock() + num_clocks; clock() < threshold;);
}

class scoring_function
{
public:
	static const size_t n = 15;
};

class atom
{
public:
	atom() : xs(rand() % scoring_function::n) {}
	size_t xs;
};

class receptor
{
public:
	array<int, 3> num_probes;
	size_t num_probes_product;
	vector<vector<float>> maps;
	explicit receptor(const path& p) : num_probes({100, 80, 70}), num_probes_product(1), maps(scoring_function::n)
	{
		for (size_t i = 0; i < 3; ++i)
		{
			num_probes_product *= num_probes[i];
		}
	}
	void populate(const scoring_function& sf, const vector<size_t>& xs, const size_t z)
	{
		spin(1e+5);
	}
};

class ligand
{
public:
	explicit ligand(const path& p) : filename(p.filename()), atoms(rand() % 10)
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
	vector<atom> atoms;
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

int main(int argc, char* argv[])
{
	// Initialize constants.
	const unsigned int lws = 256;
	const unsigned int gws = 32 * lws;
	const unsigned int num_threads = thread::hardware_concurrency();

	// Initialize variables.
	srand(time(0));
	vector<float> prmh(16);
	for (auto& prm : prmh)
	{
		prm = rand() / static_cast<float>(RAND_MAX);
	}

	cout << "Creating an io service pool of " << num_threads << " worker threads for host" << endl;
	io_service_pool ioh(num_threads);
	safe_counter<size_t> cnt;

	cout << "Precalculating a scoring function of " << scoring_function::n << " atom types in parallel" << endl;
	scoring_function sf;

	path receptor_path;
	cout << "Parsing receptor " << receptor_path << endl;
	receptor rec(receptor_path);

	cout << "Detecting CUDA devices" << endl;
	checkCudaErrors(cuInit(0));
	int num_devices;
	checkCudaErrors(cuDeviceGetCount(&num_devices));
	if (!num_devices)
	{
		cerr << "No CUDA devices detected" << endl;
		return 2;
	}
	vector<CUdevice> devices(num_devices);
	vector<int> can_map_host_memory(num_devices);
	for (int dev = 0; dev < num_devices; ++dev)
	{
		// Get a device handle from an ordinal.
		checkCudaErrors(cuDeviceGet(&devices[dev], dev));

		// Check if the device can map host memory into CUDA address space.
		checkCudaErrors(cuDeviceGetAttribute(&can_map_host_memory[dev], CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, devices[dev]));
	}

	cout << "Compiling modules for " << num_devices << " devices" << endl;
	std::ifstream ifs("multiDevice.fatbin", ios::binary);
	auto image = vector<char>((istreambuf_iterator<char>(ifs)), istreambuf_iterator<char>());
	vector<CUcontext> contexts(num_devices);
	vector<CUstream> streams(num_devices);
	vector<CUfunction> functions(num_devices);
	vector<vector<size_t>> xst(num_devices);
	vector<float*> ligh(num_devices);
	vector<CUdeviceptr> ligd(num_devices);
	vector<CUdeviceptr> slnd(num_devices);
	vector<float*> cnfh(num_devices);
	for (int dev = 0; dev < num_devices; ++dev)
	{
		// Create context.
		checkCudaErrors(cuCtxCreate(&contexts[dev], CU_CTX_SCHED_AUTO | (can_map_host_memory[dev] ? CU_CTX_MAP_HOST : 0), devices[dev]));

		// Create stream.
		checkCudaErrors(cuStreamCreate(&streams[dev], CU_STREAM_NON_BLOCKING));

		// Load module.
		CUmodule module;
		checkCudaErrors(cuModuleLoadData(&module, image.data()));

		// Get function from module.
		checkCudaErrors(cuModuleGetFunction(&functions[dev], module, "monte_carlo"));

		// Initialize symbols.
		CUdeviceptr prmd;
		size_t prms;
		checkCudaErrors(cuModuleGetGlobal(&prmd, &prms, module, "p"));
		assert(prms == sizeof(float) * 16);
		checkCudaErrors(cuMemcpyHtoD(prmd, prmh.data(), prms));

		// Reserve space for xst.
		xst[dev].reserve(sf.n);

		// Allocate ligh, ligd, slnd and cnfh.
		checkCudaErrors(cuMemHostAlloc((void**)&ligh[dev], sizeof(float) * lws, can_map_host_memory[dev] ? CU_MEMHOSTALLOC_DEVICEMAP : 0));
		if (can_map_host_memory[dev])
		{
			checkCudaErrors(cuMemHostGetDevicePointer(&ligd[dev], ligh[dev], 0));
		}
		else
		{
			checkCudaErrors(cuMemAlloc(&ligd[dev], sizeof(float) * lws));
		}
		checkCudaErrors(cuMemAlloc(&slnd[dev], sizeof(float) * gws));
		checkCudaErrors(cuMemHostAlloc((void**)&cnfh[dev], sizeof(float) * lws, 0));

		// Pop the current context.
		checkCudaErrors(cuCtxPopCurrent(NULL));
	}
	image.clear();

	// Initialize a vector of idle devices.
	safe_vector<int> idle(num_devices);
	iota(idle.begin(), idle.end(), 0);

	cout << "Creating an io service pool of " << num_devices << " worker threads for device" << endl;
	io_service_pool iod(num_devices);
	safe_function safe_print;

	// Perform docking for each ligand in the input folder.
	size_t num_ligands = 0;
	cout.setf(ios::fixed, ios::floatfield);
	cout << "ID              Ligand D  pKd 1     2     3     4     5     6     7     8     9" << endl << setprecision(2);
	for (directory_iterator dir_iter("."), const_dir_iter; dir_iter != const_dir_iter; ++dir_iter)
	{
		// Parse the ligand.
		ligand lig(dir_iter->path());

		// Find atom types that are presented in the current ligand but not presented in the grid maps.
		vector<size_t> xs;
		for (const atom& a : lig.atoms)
		{
			const size_t t = a.xs;
			if (rec.maps[t].empty())
			{
				rec.maps[t].resize(rec.num_probes_product);
				xs.push_back(t);
			}
		}

		// Create grid maps on the fly if necessary.
		if (xs.size())
		{
			// Create grid maps in parallel.
			cnt.init(rec.num_probes[2]);
			for (size_t z = 0; z < rec.num_probes[2]; ++z)
			{
				ioh.post([&,z]()
				{
					rec.populate(sf, xs, z);
					cnt.increment();
				});
			}
			cnt.wait();
		}

		// Wait until a device is ready for execution.
		const int dev = idle.safe_pop_back();

		iod.post(bind<void>([&,dev](ligand& lig)
		{
			// Push the context of the chosen device.
			checkCudaErrors(cuCtxPushCurrent(contexts[dev]));

			// Find atom types that are presented in the current ligand but are not yet copied to device memory.
			vector<size_t> xs;
			for (const atom& a : lig.atoms)
			{
				const size_t t = a.xs;
				if (find(xst[dev].cbegin(), xst[dev].cend(), t) == xst[dev].cend())
				{
					xst[dev].push_back(t);
					xs.push_back(t);
				}
			}

			// Copy grid maps from host memory to device memory if necessary.
			if (xs.size())
			{
				const size_t map_bytes = sizeof(float) * rec.num_probes_product;
				for (const auto t : xs)
				{
					CUdeviceptr mapd;
					checkCudaErrors(cuMemAlloc(&mapd, map_bytes));
					checkCudaErrors(cuMemcpyHtoD(mapd, rec.maps[t].data(), map_bytes));
				}
			}

			// Encode the current ligand.
			lig.encode(ligh[dev], lws);

			// Copy ligand from host memory to device memory if necessary.
			const size_t lig_bytes = sizeof(float) * lws;
			if (!can_map_host_memory[dev])
			{
				checkCudaErrors(cuMemcpyHtoDAsync(ligd[dev], ligh[dev], lig_bytes, streams[dev]));
			}

			// Clear the solution buffer.
			checkCudaErrors(cuMemsetD32Async(slnd[dev], 0, gws, streams[dev]));

			// Launch kernel.
			void* params[] = { &slnd[dev], &ligd[dev] };
			checkCudaErrors(cuLaunchKernel(functions[dev], gws / lws, 1, 1, lws, 1, 1, lig_bytes, streams[dev], params, NULL));

			// Copy conformations from device memory to host memory.
			checkCudaErrors(cuMemcpyDtoHAsync(cnfh[dev], slnd[dev], sizeof(float) * lws, streams[dev]));

			// Synchronize.
			checkCudaErrors(cuStreamSynchronize(streams[dev]));

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

			// Pop the context after use.
			checkCudaErrors(cuCtxPopCurrent(NULL));

			// Signal the main thread to post another task.
			idle.safe_push_back(dev);
		}, move(lig)));
	}

	// Wait until the io service for host has finished all its tasks.
	ioh.wait();

	// Wait until the io service for device has finished all its tasks.
	iod.wait();
	assert(idle.size() == num_devices);

	// Destroy contexts.
	for (auto& context : contexts)
	{
		checkCudaErrors(cuCtxDestroy(context));
	}

	cout << "Writing log records of " << num_ligands << " ligands to the log file" << endl;
}
