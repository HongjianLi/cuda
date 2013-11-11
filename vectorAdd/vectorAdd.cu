__constant__ float p[16];
extern __shared__ float q[];

extern "C" __global__ void vectorAdd(float* const s, const float* const l)
{
	const int gid = blockDim.x * blockIdx.x + threadIdx.x;
	const int lid = threadIdx.x;
	q[lid] = l[lid];
	s[gid] = s[gid] + q[lid] * 2.0f + 1.0f + p[lid % 16];
//#pragma unroll
}
