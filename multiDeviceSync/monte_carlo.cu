__constant__ float p[16];
extern __shared__ float q[];

extern "C" __global__ void monte_carlo(float* const s, const float* const l)
{
	const int gid = blockDim.x * blockIdx.x + threadIdx.x;
	const int lid = threadIdx.x;
	q[lid] = l[lid];
	for (int j = 0; j < 4e+2; ++j)
	for (int i = 0; i < 5e+2; ++i)
	s[gid] = s[gid] + q[lid] * 2.0f + 1.0f + p[lid % 16];
	s[gid] = 0;
	s[gid] = s[gid] + q[lid] * 2.0f + 1.0f + p[lid % 16];
}
