/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/transpose/cuda.cu
 * Transpose operation on buffers on CUDA
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-09-14
 * */

#include "nntile/kernel/transpose/cuda.hh"
#include <algorithm>

namespace nntile
{
namespace kernel
{
namespace transpose
{

template<typename T>
static __global__
void cuda_kernel(Index m, Index n, T alpha, const T* src, T* dst)
//! Transpose buffers on CPU
/*! dst[i,j] = alpha * src[j,i]
 *
 * @param[in] m: Number of rows of src and columns of dst
 * @param[in] n: Number of columns of src and rows of dst
 * @param[in] alpha: Scalar multiplier
 * @param[in] src: Source tensor
 * @param[out] dst: Destination of the add operation
 * */
{
    Index i = threadIdx.x + blockIdx.x*blockDim.x;
    Index j = i / m;
    i = i - j*m;
    if(i < m and j < n)
    {
        dst[i*n+j] = alpha * src[i+j*m];
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, T alpha, const T* src, T* dst)
    noexcept
//! Transpose buffers on CPU
/*! dst[i,j] = alpha * src[j,i]
 *
 * @param[in] m: Number of rows of src and columns of dst
 * @param[in] n: Number of columns of src and rows of dst
 * @param[in] alpha: Scalar multiplier
 * @param[in] src: Source tensor
 * @param[out] dst: Destination of the add operation
 * */
{
    // Both source and destination are Fortran-contiguous
    dim3 threads(32);
    dim3 blocks((m*n+threads.x-1)/threads.x);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(m, n, alpha, src, dst);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index m, Index n, fp32_t alpha,
        const fp32_t* src, fp32_t* dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index m, Index n, fp64_t alpha,
        const fp64_t* src, fp64_t* dst)
    noexcept;

} // namespace tranpose
} // namespace kernel
} // namespace nntile

