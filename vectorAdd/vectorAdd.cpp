#include <string.h>
#include <math.h>
#include "../cu_helper.h"
#include "vectorAdd.h"

inline static int cci(const int compute_capability_major, const int compute_capability_minor)
{
	static const int ccs[] = { 35, 30, 20, 13, 12, 11, 10 };
	const int cc = 10 * compute_capability_major + compute_capability_minor;
	int i;
	for (i = 0; cc < ccs[i]; ++i);
	return i;
}

CUDA_CB void callback(CUstream stream, CUresult error, void* data)
{
	checkCudaErrors(error);
	printf("callback, h_e[0] = %f\n", *(float*)data);
}

int main(int argc, char* argv[])
{
	const unsigned int lws = 256;
	const unsigned int gws = 32 * lws;
	float* p_l = (float*)malloc(sizeof(float) * lws);
	for (int i = 0; i < lws; ++i)
	{
		p_l[i] = rand() / (float)RAND_MAX;
	}

	checkCudaErrors(cuInit(0));
	int num_devices = 0;
	checkCudaErrors(cuDeviceGetCount(&num_devices));
	for (int d = 0; d < num_devices; ++d)
	{
		int compute_capability_major;
		checkCudaErrors(cuDeviceGetAttribute(&compute_capability_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, d));
		int compute_capability_minor;
		checkCudaErrors(cuDeviceGetAttribute(&compute_capability_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, d));
		char name[256];
		checkCudaErrors(cuDeviceGetName(name, sizeof(name), d));
		printf("%d: %s, SM %d.%d\n", d, name, compute_capability_major, compute_capability_minor);
		const char* const source = sources[cci(compute_capability_major, compute_capability_minor)];

		CUdevice device;
		checkCudaErrors(cuDeviceGet(&device, d));
		CUcontext context;
		checkCudaErrors(cuCtxCreate(&context, CU_CTX_SCHED_AUTO/*CU_CTX_SCHED_YIELD*/, device));
//		checkCudaErrors(cuCtxSetCacheConfig(CU_FUNC_CACHE_PREFER_L1));
//		checkCudaErrors(cuCtxSetSharedMemConfig(CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE));

		const unsigned int numOptions = 6;
		CUjit_option* options = (CUjit_option*)malloc(sizeof(CUjit_option) * numOptions);
		void** optionValues = (void**)malloc(sizeof(void*) * numOptions);
		options[0] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
		size_t/*unsigned int*/ info_log_buffer_size_bytes = 1024;
		optionValues[0] = (void*)info_log_buffer_size_bytes;
		options[1] = CU_JIT_INFO_LOG_BUFFER;
		char* info_log_buffer = (char*)malloc(sizeof(char) * info_log_buffer_size_bytes);
		optionValues[1] = info_log_buffer;
		options[2] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
		size_t/*unsigned int*/ error_log_buffer_size_bytes = 1024;
		optionValues[2] = (void*)error_log_buffer_size_bytes;
		options[3] = CU_JIT_ERROR_LOG_BUFFER;
		char* error_log_buffer = (char*)malloc(sizeof(char) * error_log_buffer_size_bytes);
		optionValues[3] = error_log_buffer;
		options[4] = CU_JIT_LOG_VERBOSE;
		size_t/*int*/ log_verbose = 1;
		optionValues[4] = (void*)log_verbose;
		options[5] = CU_JIT_MAX_REGISTERS;
		size_t/*unsigned int*/ max_registers = 32;
		optionValues[5] = (void*)max_registers;
//		options[6] = CU_JIT_CACHE_MODE;
//		CUjit_cacheMode cache_mode = CU_JIT_CACHE_OPTION_NONE; // CU_JIT_CACHE_OPTION_CG, CU_JIT_CACHE_OPTION_CA
//		optionValues[6] = (void*)cache_mode;
//		options[7] = CU_JIT_GENERATE_DEBUG_INFO;
//		int generated_debug_info = 1;
//		optionValues[7] = (void*)generated_debug_info;
//		options[8] = CU_JIT_GENERATE_LINE_INFO;
//		int generated_line_info = 1;
//		optionValues[8] = (void*)generated_line_info;

		CUmodule module;
//		checkCudaErrors(cuModuleLoadData(&module, source)); // cuModuleLoadData passes valgrind --leak-check
//		checkCudaErrors(cuModuleLoadDataEx(&module, source, numOptions, options, optionValues)); // cuModuleLoadDataEx sometimes fails valgrind --leak-check
		checkCudaErrors(cuModuleLoad(&module, "vectorAdd.cubin")); // nvcc -cubin -arch=sm_11 -G, and use cuda-gdb instead of gdb
		printf("%s\n", info_log_buffer);
		printf("%s\n", error_log_buffer);
		free(error_log_buffer);
		free(info_log_buffer);
		free(optionValues);
		free(options);

		CUfunction function;
		checkCudaErrors(cuModuleGetFunction(&function, module, "vectorAdd"));
		int max_threads_per_block;
		checkCudaErrors(cuFuncGetAttribute(&max_threads_per_block, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, function));
		printf("CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK: %d\n", max_threads_per_block);
		int shared_size_bytes;
		checkCudaErrors(cuFuncGetAttribute(&shared_size_bytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, function));
		printf("CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES: %d\n", shared_size_bytes);
		int const_size_bytes;
		checkCudaErrors(cuFuncGetAttribute(&const_size_bytes, CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, function));
		printf("CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES: %d\n", const_size_bytes);
		int local_size_bytes;
		checkCudaErrors(cuFuncGetAttribute(&local_size_bytes, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, function));
		printf("CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES: %d\n", local_size_bytes);
		int num_regs;
		checkCudaErrors(cuFuncGetAttribute(&num_regs, CU_FUNC_ATTRIBUTE_NUM_REGS, function));
		printf("CU_FUNC_ATTRIBUTE_NUM_REGS: %d\n", num_regs);
		int ptx_version;
		checkCudaErrors(cuFuncGetAttribute(&ptx_version, CU_FUNC_ATTRIBUTE_PTX_VERSION, function));
		printf("CU_FUNC_ATTRIBUTE_PTX_VERSION: %d\n", ptx_version);
		int binary_version;
		checkCudaErrors(cuFuncGetAttribute(&binary_version, CU_FUNC_ATTRIBUTE_BINARY_VERSION, function));
		printf("CU_FUNC_ATTRIBUTE_BINARY_VERSION: %d\n", binary_version);
//		checkCudaErrors(cuFuncSetCacheConfig(function, CU_FUNC_CACHE_PREFER_L1));
//		checkCudaErrors(cuFuncSetSharedMemConfig(function, CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE/*CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE, CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE*/));

		float* h_p;
		checkCudaErrors(cuMemAllocHost((void**)&h_p, sizeof(float) * 16));
		for (int i = 0; i < 16; ++i)
		{
			h_p[i] = rand() / (float)RAND_MAX;
		}
		CUdeviceptr d_p;
		checkCudaErrors(cuModuleGetGlobal(&d_p, NULL, module, "p"));
		checkCudaErrors(cuMemcpyHtoD(d_p, h_p, sizeof(float) * 16));

		CUstream stream;
		checkCudaErrors(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING/*CU_STREAM_DEFAULT*/));
// TODO: simpleZeroCopy, HtoD
		float* h_l;
		float* h_e;
		checkCudaErrors(cuMemAllocHost((void**)&h_l, sizeof(float) * lws));
		checkCudaErrors(cuMemAllocHost((void**)&h_e, sizeof(float) * lws));
		memcpy(h_l, p_l, sizeof(float) * lws);
		CUdeviceptr d_s;
		CUdeviceptr d_l;
		checkCudaErrors(cuMemAlloc(&d_s, sizeof(float) * gws));
		checkCudaErrors(cuMemAlloc(&d_l, sizeof(float) * lws));
//		checkCudaErrors(cuMemsetD32(d_s, 0, gws));
		checkCudaErrors(cuMemsetD32Async(d_s, 0, gws, stream));
		checkCudaErrors(cuMemcpyHtoDAsync(d_l, h_l, sizeof(float) * lws, stream));
		void* args[] = { &d_s, &d_l };
//		checkCudaErrors(cuLaunchKernel(function, gws / lws, 1, 1, lws, 1, 1, 0/*dynamic smem*/, NULL/*stream*/, args, NULL));
//		checkCudaErrors(cuCtxSynchronize());
		checkCudaErrors(cuLaunchKernel(function, gws / lws, 1, 1, lws, 1, 1, sizeof(float) * lws, stream, args, NULL));
		checkCudaErrors(cuMemcpyDtoHAsync(h_e, d_s, sizeof(float) * lws, stream));
		checkCudaErrors(cuStreamAddCallback(stream, callback, h_e, 0)); // The callback will block later work in the stream until it is finished. Callbacks must return promptly.
		CUevent complete;
		checkCudaErrors(cuEventCreate(&complete, CU_EVENT_DISABLE_TIMING));
		checkCudaErrors(cuEventRecord(complete, stream));
		checkCudaErrors(cuEventSynchronize(complete));
//		if (cuEventQuery(complete) == CUDA_SUCCESS);
//		checkCudaErrors(cuStreamWaitEvent(stream, complete, 0)); // analogue to user event in OpenCL
//		checkCudaErrors(cuStreamSynchronize(stream));
		bool passed = true;
		for (int i = 0; i < lws; i++)
		{
			const float ref = p_l[i] * 2.0f + 1.0f + h_p[i % 16];
			if (fabs(h_e[i] - ref) > 1e-7)
			{
				printf("i = %d, ref = %f, h_e[i] = %f\n", i, ref, h_e[i]);
				passed = false;
				break;
			}
		}
		printf("vectorAdd %s\n\n", passed ? "passed" : "failed");
		checkCudaErrors(cuMemFree(d_l));
		checkCudaErrors(cuMemFree(d_s));
		checkCudaErrors(cuMemFreeHost(h_e));
		checkCudaErrors(cuMemFreeHost(h_l));
		checkCudaErrors(cuMemFreeHost(h_p));

		checkCudaErrors(cuModuleUnload(module));
		checkCudaErrors(cuCtxDestroy(context));
	}
	free(p_l);
}
