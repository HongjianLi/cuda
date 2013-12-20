#include <iostream>
#include <iomanip>
#include <ctime>
#include <array>
#include <numeric>
#include <fstream>
#include <boost/filesystem/operations.hpp>
#include "../cu_helper.h"
#include "io_service_pool.hpp"
using namespace boost::filesystem;

void spin(const size_t n)
{
	for (size_t i = 0; i < n; ++i)
	{
		rand();
	}
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
	size_t map_bytes;
	vector<vector<float>> maps;
	explicit receptor(const path& p) : num_probes({ 100, 80, 70 }), num_probes_product(num_probes[0] * num_probes[1] * num_probes[2]), map_bytes(sizeof(float) * num_probes_product), maps(scoring_function::n)
	{
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
		for (const auto& a : atoms) xs[a.xs] = true;
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
	array<bool, scoring_function::n> xs;
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
class callback_data
{
public:
	callback_data(io_service_pool& io, const unsigned int lws, const T dev, vector<float*>& cnfh, vector<float*>& ligh, vector<float>& prmh, ligand&& lig_, safe_function& safe_print, size_t& num_ligands, safe_vector<T>& idle) : io(io), lws(lws), dev(dev), cnfh(cnfh), ligh(ligh), prmh(prmh), lig(move(lig_)), safe_print(safe_print), num_ligands(num_ligands), idle(idle) {}
	io_service_pool& io;
	const unsigned int lws;
	const T dev;
	const vector<float*>& cnfh;
	const vector<float*>& ligh;
	const vector<float> & prmh;
	ligand lig;
	safe_function& safe_print;
	size_t& num_ligands;
	safe_vector<T>& idle;
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

	cout << "Creating an io service pool of " << num_threads << " worker threads" << endl;
	io_service_pool io(num_threads);
	safe_counter<size_t> cnt;
	safe_function safe_print;

	cout << "Precalculating a scoring function of " << scoring_function::n << " atom types in parallel" << endl;
	scoring_function sf;

	path receptor_path;
	cout << "Parsing receptor " << receptor_path << endl;
	receptor rec(receptor_path);

	cout << "Detecting CUDA devices with compute capability 1.1 or greater" << endl;
	checkCudaErrors(cuInit(0));
	int num_devices;
	checkCudaErrors(cuDeviceGetCount(&num_devices));
	if (!num_devices)
	{
		cerr << "No CUDA devices detected" << endl;
		return 2;
	}
	vector<CUdevice> devices;
	devices.reserve(num_devices);
	for (int dev = 0; dev < num_devices; ++dev)
	{
		// Get a device handle.
		CUdevice device;
		checkCudaErrors(cuDeviceGet(&device, dev));

		// Filter devices with compute capability 1.1 or greater, which is required by cuMemHostGetDevicePointer and cuStreamAddCallback.
		int compute_capability_major;
		checkCudaErrors(cuDeviceGetAttribute(&compute_capability_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
		if (compute_capability_major == 1)
		{
			int compute_capability_minor;
			checkCudaErrors(cuDeviceGetAttribute(&compute_capability_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
			if (compute_capability_minor < 1) continue;
		}

		// Save the device handle.
		devices.push_back(device);
	}
	num_devices = devices.size();

	cout << "Compiling modules for " << num_devices << " devices" << endl;
	std::ifstream ifs("multiDevice.fatbin", ios::binary);
	auto image = vector<char>((istreambuf_iterator<char>(ifs)), istreambuf_iterator<char>());
	vector<CUcontext> contexts(num_devices);
	vector<CUstream> streams(num_devices);
	vector<CUfunction> functions(num_devices);
	vector<array<CUdeviceptr, sf.n>> mpsd(num_devices);
	vector<float*> ligh(num_devices);
	vector<CUdeviceptr> ligd(num_devices);
	vector<CUdeviceptr> slnd(num_devices);
	vector<float*> cnfh(num_devices);
	for (int dev = 0; dev < num_devices; ++dev)
	{
		// Create context.
		checkCudaErrors(cuCtxCreate(&contexts[dev], CU_CTX_SCHED_AUTO | CU_CTX_MAP_HOST, devices[dev]));

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

		// Allocate ligh, ligd, slnd and cnfh.
		checkCudaErrors(cuMemHostAlloc((void**)&ligh[dev], sizeof(float) * lws, CU_MEMHOSTALLOC_DEVICEMAP));
		checkCudaErrors(cuMemHostGetDevicePointer(&ligd[dev], ligh[dev], 0));
		checkCudaErrors(cuMemAlloc(&slnd[dev], sizeof(float) * gws));
		checkCudaErrors(cuMemHostAlloc((void**)&cnfh[dev], sizeof(float) * lws, 0));

		// Pop the current context.
		checkCudaErrors(cuCtxPopCurrent(NULL));
	}
	image.clear();

	// Initialize a vector of idle devices.
	safe_vector<int> idle(num_devices);
	iota(idle.begin(), idle.end(), 0);

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
		for (size_t t = 0; t < sf.n; ++t)
		{
			if (lig.xs[t] && rec.maps[t].empty())
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
				io.post([&,z]()
				{
					rec.populate(sf, xs, z);
					cnt.increment();
				});
			}
			cnt.wait();
		}

		// Wait until a device is ready for execution.
		const int dev = idle.safe_pop_back();

		// Push the context of the chosen device.
		checkCudaErrors(cuCtxPushCurrent(contexts[dev]));

		// Copy grid maps from host memory to device memory if necessary.
		for (size_t t = 0; t < sf.n; ++t)
		{
			if (lig.xs[t] && !mpsd[dev][t])
			{
				checkCudaErrors(cuMemAlloc(&mpsd[dev][t], rec.map_bytes));
				checkCudaErrors(cuMemcpyHtoD(mpsd[dev][t], rec.maps[t].data(), rec.map_bytes));
			}
		}

		// Encode the current ligand.
		lig.encode(ligh[dev], lws);

		// Compute the number of shared memory bytes.
		const size_t lig_bytes = sizeof(float) * lws;

		// Clear the solution buffer.
		checkCudaErrors(cuMemsetD32Async(slnd[dev], 0, gws, streams[dev]));

		// Launch kernel.
		void* params[] = { &slnd[dev], &ligd[dev] };
		checkCudaErrors(cuLaunchKernel(functions[dev], gws / lws, 1, 1, lws, 1, 1, lig_bytes, streams[dev], params, NULL));

		// Copy conformations from device memory to host memory.
		checkCudaErrors(cuMemcpyDtoHAsync(cnfh[dev], slnd[dev], sizeof(float) * lws, streams[dev]));

		// Add a callback to the compute stream.
		checkCudaErrors(cuStreamAddCallback(streams[dev], [](CUstream stream, CUresult error, void* data)
		{
			checkCudaErrors(error);
			const shared_ptr<callback_data<int>> cbd(reinterpret_cast<callback_data<int>*>(data));
			cbd->io.post([=]()
			{
				const auto   lws = cbd->lws;
				const auto   dev = cbd->dev;
				const auto& cnfh = cbd->cnfh;
				const auto& ligh = cbd->ligh;
				const auto& prmh = cbd->prmh;
				auto& lig = cbd->lig;
				auto& safe_print = cbd->safe_print;
				auto& num_ligands = cbd->num_ligands;
				auto& idle = cbd->idle;

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
				idle.safe_push_back(dev);
			});
		}, new callback_data<int>(io, lws, dev, cnfh, ligh, prmh, move(lig), safe_print, num_ligands, idle), 0));

		// Pop the context after use.
		checkCudaErrors(cuCtxPopCurrent(NULL));
	}

	// Synchronize contexts.
	for (auto& context : contexts)
	{
		checkCudaErrors(cuCtxPushCurrent(context));
		checkCudaErrors(cuCtxSynchronize());
		checkCudaErrors(cuCtxPopCurrent(NULL));
	}

	// Wait until the io service pool has finished all its tasks.
	io.wait();
	assert(idle.size() == num_devices);

	// Destroy contexts.
	for (auto& context : contexts)
	{
		checkCudaErrors(cuCtxDestroy(context));
	}

	cout << "Writing log records of " << num_ligands << " ligands to the log file" << endl;
}
