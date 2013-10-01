extern "C" __global__ void spin(const clock_t num_clocks)
{
    const clock_t threshold = clock() + num_clocks;
    while (clock() < threshold);
}
