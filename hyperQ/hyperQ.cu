__device__ void spin(const clock_t num_clocks)
{
	for (const clock_t threshold = clock() + num_clocks; clock() < threshold;);
}

extern "C" __global__ void kernel0(clock_t num_clocks)
{
	spin(num_clocks);
}

extern "C" __global__ void kernel1(clock_t num_clocks)
{
	spin(num_clocks);
}
