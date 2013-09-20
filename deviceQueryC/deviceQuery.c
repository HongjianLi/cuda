#include "../cu_helper.h"

int main(int argc, char* argv[])
{
	int num_devices;
	int i;
	char name[256];
	size_t total_mem;
	int compute_capability_major;
	int compute_capability_minor;
	int multiprocessor_count;
	int max_shared_memory_per_block;
	int total_constant_memory;
	int max_registers_per_block;
	int ecc_enabled;
	int kernel_exec_timeout;
	int clock_rate;
	int memory_clock_rate;
	int global_memory_bus_width;
	int l2_cache_size;
	int warp_size;
	int max_threads_per_multiprocessor;
	int max_threads_per_block;
	int max_block_dim_x;
	int max_block_dim_y;
	int max_block_dim_z;
	int max_grid_dim_x;
	int max_grid_dim_y;
	int max_grid_dim_z;
	int gpu_overlap;
	int async_engine_count;
	int integrated;
	int can_map_host_memory;
	int concurrent_kernels;
	int tcc_driver;
	int unified_addressing;
	int pci_bus_id;
	int pci_device_id;
	int compute_mode;
	printf("DEVICE,DEVICE NAME,TOTAL MEM (MB),COMPUTE CAPABILITY MAJOR,COMPUTE CAPABILITY MINOR,MULTIPROCESSOR COUNT,MAX SHARED MEMORY PER BLOCK (KB),TOTAL CONSTANT MEMORY (KB),MAX REGISTERS PER BLOCK,ECC ENABLED,KERNEL EXEC TIMEOUT,CLOCK RATE (MHz),MEMORY CLOCK RATE (MHz),GLOBAL MEMORY BUS WIDTH (b),L2 CACHE SIZE (KB),WARP SIZE,MAX THREADS PER MULTIPROCESSOR,MAX THREADS PER BLOCK,MAX BLOCK DIM X,MAX BLOCK DIM Y,MAX BLOCK DIM Z,MAX GRID DIM X,MAX GRID DIM Y,MAX GRID DIM Z,GPU OVERLAP,ASYNC ENGINE COUNT,INTEGRATED,CAN MAP HOST MEMORY,CONCURRENT KERNELS,TCC DRIVER,UNIFIED ADDRESSING,PCI BUS ID,PCI DEVICE ID,COMPUTE MODE\n");
	checkCudaErrors(cuInit(0));
	checkCudaErrors(cuDeviceGetCount(&num_devices));
	for (i = 0; i < num_devices; ++i)
	{
		checkCudaErrors(cuDeviceGetName(name, sizeof(name), i));
		checkCudaErrors(cuDeviceTotalMem(&total_mem, i));
		checkCudaErrors(cuDeviceGetAttribute(&compute_capability_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, i));
		checkCudaErrors(cuDeviceGetAttribute(&compute_capability_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, i));
		checkCudaErrors(cuDeviceGetAttribute(&multiprocessor_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, i));
		checkCudaErrors(cuDeviceGetAttribute(&max_shared_memory_per_block, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, i));
		checkCudaErrors(cuDeviceGetAttribute(&total_constant_memory, CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY, i));
		checkCudaErrors(cuDeviceGetAttribute(&max_registers_per_block, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, i));
		checkCudaErrors(cuDeviceGetAttribute(&ecc_enabled, CU_DEVICE_ATTRIBUTE_ECC_ENABLED, i));
		checkCudaErrors(cuDeviceGetAttribute(&kernel_exec_timeout, CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT, i));
		checkCudaErrors(cuDeviceGetAttribute(&clock_rate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, i));
		checkCudaErrors(cuDeviceGetAttribute(&memory_clock_rate, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, i));
		checkCudaErrors(cuDeviceGetAttribute(&global_memory_bus_width, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, i));
		checkCudaErrors(cuDeviceGetAttribute(&l2_cache_size, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, i));
		checkCudaErrors(cuDeviceGetAttribute(&warp_size, CU_DEVICE_ATTRIBUTE_WARP_SIZE, i));
		checkCudaErrors(cuDeviceGetAttribute(&max_threads_per_multiprocessor, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, i));
		checkCudaErrors(cuDeviceGetAttribute(&max_threads_per_block, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, i));
		checkCudaErrors(cuDeviceGetAttribute(&max_block_dim_x, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, i));
		checkCudaErrors(cuDeviceGetAttribute(&max_block_dim_y, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, i));
		checkCudaErrors(cuDeviceGetAttribute(&max_block_dim_z, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, i));
		checkCudaErrors(cuDeviceGetAttribute(&max_grid_dim_x, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, i));
		checkCudaErrors(cuDeviceGetAttribute(&max_grid_dim_y, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, i));
		checkCudaErrors(cuDeviceGetAttribute(&max_grid_dim_z, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, i));
		checkCudaErrors(cuDeviceGetAttribute(&gpu_overlap, CU_DEVICE_ATTRIBUTE_GPU_OVERLAP, i));
		checkCudaErrors(cuDeviceGetAttribute(&async_engine_count, CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT, i));
		checkCudaErrors(cuDeviceGetAttribute(&integrated, CU_DEVICE_ATTRIBUTE_INTEGRATED, i));
		checkCudaErrors(cuDeviceGetAttribute(&can_map_host_memory, CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, i));
		checkCudaErrors(cuDeviceGetAttribute(&concurrent_kernels, CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS, i));
		checkCudaErrors(cuDeviceGetAttribute(&tcc_driver, CU_DEVICE_ATTRIBUTE_TCC_DRIVER, i));
		checkCudaErrors(cuDeviceGetAttribute(&unified_addressing, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, i));
		checkCudaErrors(cuDeviceGetAttribute(&pci_bus_id, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, i));
		checkCudaErrors(cuDeviceGetAttribute(&pci_device_id, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, i));
		checkCudaErrors(cuDeviceGetAttribute(&compute_mode, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, i));
		printf("%d,%s,%lu,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n", i, name, total_mem / 1048576, compute_capability_major, compute_capability_minor, multiprocessor_count, max_shared_memory_per_block / 1024, total_constant_memory / 1024, max_registers_per_block, ecc_enabled, kernel_exec_timeout, clock_rate / 1000, memory_clock_rate / 1000, global_memory_bus_width, l2_cache_size / 1024, warp_size, max_threads_per_multiprocessor, max_threads_per_block, max_block_dim_x, max_block_dim_y, max_block_dim_z, max_grid_dim_x, max_grid_dim_y,max_grid_dim_z, gpu_overlap, async_engine_count, integrated, can_map_host_memory, concurrent_kernels, tcc_driver, unified_addressing, pci_bus_id, pci_device_id, compute_mode);
	}
}
