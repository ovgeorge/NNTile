/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/conv2d/cuda.cu
 * 2D-Convolution between 2 matrices
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-09-14
 * */

#include <algorithm>

#include "nntile/kernel/conv2d/cuda.hh"

namespace nntile {
namespace kernel {
namespace conv2d {

template <typename T>
static __global__ void cuda_kernel(Index nx, Index ny, const T* src, Index mx,
								   Index my, const T* kernel, T* dst)
//! Compute full discrete linear convolution of two 2-dimensional arrays on
//! CUDA.
/*!@param[in] nx: Size of the first axis of src array
 * @param[in] ny: Size of the second axis of src array
 * @param[in] src: Input contiguous nx-by-ny array
 * @param[in] mx: Size of the first axis of kernel array
 * @param[in] my: Size of the second axis of kernel array
 * @param[in] kernel: Input contiguous mx-by-my array
 * @param[out] dst: Output contiguous (nx+my)-by-(ny+my) array
 * */
{
  Index i0 = threadIdx.x + blockIdx.x * blockDim.x,
        i1 = threadIdx.y + blockIdx.y * blockDim.y;
  if (i1 >= nx+mx-1 || i0 >= ny+my-1)
    return;
  T res = 0;
  for(Index j0=i0-my+1; j0 <= i0; j0++) {
  	for(Index j1=i1-mx+1; j1 <= i1; j1++) {
  		res += src[j1 * ny + j0] * kernel[(i1-j1) * my + (i0-j0)];
  	}
  }
  dst[i1 * (my+ny-1) + i0] = res;

}

template <typename T>
void cuda(cudaStream_t stream, Index nx, Index ny, const T* src, Index mx,
          Index my, const T* kernel, T* dst) noexcept
//! Per-element addition of a tensor and a broadcasted slice on CUDA
/*! This is a host function that does the following operations:
 *      dst[i,l,j] = beta*dst[i,l,j] + alpha*src[i,j]
 *
 * @param[in] m: Size of the first mode of src and dst tensors
 * @param[in] n: Size of the last mode of src and dst tensors
 * @param[in] k: Size of the middle mode of dst tensor
 * @param[in] alpha: Scalar factor for src
 * @param[in] src: Input contiguous m-by-n array
 * @param[in] beta: Scaling factor for dst
 * @param[inout] dst: Input and output contiguous m-by-k-by-n array
 * */
{
  // Both source and destination are Fortran-contiguous
  dim3 threads(std::min(int(ny), 8), std::min(int(nx), 8));
  dim3 blocks((ny + threads.x - 1) / threads.x, (nx + threads.y - 1) / threads.y);
  (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(nx, ny, src, mx, my, kernel, dst);
}

// Explicit instantiation
template void cuda<fp32_t>(cudaStream_t stream, Index nx, Index ny,
                           const fp32_t* src, Index mx, Index my,
                           const fp32_t* kernel, fp32_t* dst) noexcept;

template void cuda<fp64_t>(cudaStream_t stream, Index nx, Index ny,
                           const fp64_t* src, Index mx, Index my,
                           const fp64_t* kernel, fp64_t* dst) noexcept;

}  // namespace conv2d
}  // namespace kernel
}  // namespace nntile
