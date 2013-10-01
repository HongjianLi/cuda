__device__ void clock_block(clock_t *d_o, clock_t clock_count)
{
    clock_t start_clock = clock();
    clock_t clock_offset = 0;
    while (clock_offset < clock_count)
    {
        clock_offset = clock() - start_clock;
    }
    d_o[0] = clock_offset;
}

extern "C" __global__ void kernel_A(clock_t *d_o, clock_t clock_count)
{
    clock_block(d_o, clock_count);
}

extern "C" __global__ void kernel_B(clock_t *d_o, clock_t clock_count)
{
    clock_block(d_o, clock_count);
}

extern "C" __global__ void sum(clock_t *d_clocks, int N)
{
    __shared__ clock_t s_clocks[32];
    clock_t my_sum = 0;
    for (int i = threadIdx.x; i < N; i += blockDim.x)
    {
        my_sum += d_clocks[i];
    }
    s_clocks[threadIdx.x] = my_sum;
    __syncthreads();
    for (int i = warpSize / 2 ; i > 0 ; i /= 2)
    {
        if (threadIdx.x < i)
        {
            s_clocks[threadIdx.x] += s_clocks[threadIdx.x + i];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        d_clocks[0] = s_clocks[0];
    }
}
