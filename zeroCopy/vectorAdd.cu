extern "C" __global__ void vectorAdd(const float* const a, const float* const b, float* const c)
{
	const int gid = blockDim.x * blockIdx.x + threadIdx.x;
	c[gid] = a[gid] + b[gid];
}
