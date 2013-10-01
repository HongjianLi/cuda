__device__ void spin(const clock_t num_clocks)
{
    const clock_t threshold = clock() + num_clocks;
    while (clock() < threshold);
}

extern "C" __global__ void kernelA(clock_t num_clocks)
{
    spin(num_clocks);
}

extern "C" __global__ void kernelB(clock_t num_clocks)
{
    spin(num_clocks);
}
