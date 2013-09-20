#include "../cu_helper.h"

int main(int argc, char* argv[])
{
	if (argc != 2)
	{
		printf("jit vectorAdd.ptx\n");
		return 0;
	}

	FILE* source_file = fopen(argv[1], "rb");
	fseek(source_file, 0, SEEK_END);
	const size_t source_length = ftell(source_file);
	fseek(source_file, 0, SEEK_SET);
	char* const source = (char*)malloc(sizeof(char) * (source_length + 1));
	fread(source, sizeof(char), source_length, source_file);
	fclose(source_file);
	source[source_length] = '\0';

	checkCudaErrors(cuInit(0));
	int num_devices = 0;
	checkCudaErrors(cuDeviceGetCount(&num_devices));
	for (int i = 0; i < num_devices; ++i)
	{
		int compute_capability_major;
		checkCudaErrors(cuDeviceGetAttribute(&compute_capability_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, i));
		int compute_capability_minor;
		checkCudaErrors(cuDeviceGetAttribute(&compute_capability_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, i));
		char name[256];
		checkCudaErrors(cuDeviceGetName(name, sizeof(name), i));
		printf("%d: %s, SM %d.%d\n", i, name, compute_capability_major, compute_capability_minor);

		CUdevice device;
		checkCudaErrors(cuDeviceGet(&device, i));
		CUcontext context;
		checkCudaErrors(cuCtxCreate(&context, 0, device));
//		checkCudaErrors(cuCtxSetCacheConfig(CU_FUNC_CACHE_PREFER_L1));
//		checkCudaErrors(cuCtxSetSharedMemConfig(CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE));

		const unsigned int numOptions = 6;
		CUjit_option* options = (CUjit_option*)malloc(sizeof(CUjit_option) * numOptions);
		void** optionValues = (void**)malloc(sizeof(void*) * numOptions);
		options[0] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
		size_t info_log_buffer_size_bytes = 1024;
		optionValues[0] = (void*)info_log_buffer_size_bytes;
		options[1] = CU_JIT_INFO_LOG_BUFFER;
		char* info_log_buffer = (char*)malloc(sizeof(char) * info_log_buffer_size_bytes);
		optionValues[1] = info_log_buffer;
		options[2] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
		size_t error_log_buffer_size_bytes = 1024;
		optionValues[2] = (void*)error_log_buffer_size_bytes;
		options[3] = CU_JIT_ERROR_LOG_BUFFER;
		char* error_log_buffer = (char*)malloc(sizeof(char) * error_log_buffer_size_bytes);
		optionValues[3] = error_log_buffer;
		options[4] = CU_JIT_MAX_REGISTERS;
		size_t max_registers = 32;
		optionValues[4] = (void*)max_registers;
		options[5] = CU_JIT_LOG_VERBOSE;
		size_t log_verbose = 1;
		optionValues[5] = (void*)log_verbose;
		CUmodule module;
//		checkCudaErrors(cuModuleLoadData(&module, source)); // cuModuleLoadData passes valgrind --leak-check=full
		checkCudaErrors(cuModuleLoadDataEx(&module, source, numOptions, options, optionValues));
		printf("%s\n", info_log_buffer);
		printf("%s\n", error_log_buffer);
		free(error_log_buffer);
		free(info_log_buffer);
		free(optionValues);
		free(options);

//		CUfunction function;
//		checkCudaErrors(cuModuleGetFunction(&function, module, "vectorAdd"));
//		int max_threads_per_block;
//		checkCudaErrors(cuFuncGetAttribute(&max_threads_per_block, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, function));
//		printf("CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK: %d\n", max_threads_per_block);
//		int shared_size_bytes;
//		checkCudaErrors(cuFuncGetAttribute(&shared_size_bytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, function));
//		printf("CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES: %d\n", shared_size_bytes);
//		int const_size_bytes;
//		checkCudaErrors(cuFuncGetAttribute(&const_size_bytes, CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, function));
//		printf("CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES: %d\n", const_size_bytes);
//		int local_size_bytes;
//		checkCudaErrors(cuFuncGetAttribute(&local_size_bytes, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, function));
//		printf("CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES: %d\n", local_size_bytes);
//		int num_regs;
//		checkCudaErrors(cuFuncGetAttribute(&num_regs, CU_FUNC_ATTRIBUTE_NUM_REGS, function));
//		printf("CU_FUNC_ATTRIBUTE_NUM_REGS: %d\n", num_regs);
//		int ptx_version;
//		checkCudaErrors(cuFuncGetAttribute(&ptx_version, CU_FUNC_ATTRIBUTE_PTX_VERSION, function));
//		printf("CU_FUNC_ATTRIBUTE_PTX_VERSION: %d\n", ptx_version);
//		int binary_version;
//		checkCudaErrors(cuFuncGetAttribute(&binary_version, CU_FUNC_ATTRIBUTE_BINARY_VERSION, function));
//		printf("CU_FUNC_ATTRIBUTE_BINARY_VERSION: %d\n", binary_version);

		checkCudaErrors(cuModuleUnload(module));
		checkCudaErrors(cuCtxDestroy(context));
	}
	free(source);
}
