/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/add_slice/cuda.cu
 * Per-element addition of a tensor and a broadcasted slice on CUDA
 *
 * @version 1.0.0
 * */

#include "nntile/kernel/add_slice/cuda.hh"
#include <algorithm>
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::add_slice
{

template<typename T>
static __global__
void cuda_kernel(Index m, Index n, Index k, Index mk, T alpha, const T *src,
        T beta, T *dst)
//! Per-element addition of a tensor and a broadcasted slice on CUDA
/*! This is a global function that does the following operations:
 *      dst[i,l,j] = beta*dst[i,l,j] + alpha*src[i,j]
 *
 * @param[in] m: Size of the first mode of src and dst tensors
 * @param[in] n: Size of the last mode of src and dst tensors
 * @param[in] k: Size of the middle mode of dst tensor
 * @param[in] mk: Product of m and k
 * @param[in] alpha: Scalar factor for src
 * @param[in] src: Input contiguous m-by-n array
 * @param[in] beta: Scaling factor for dst
 * @param[inout] dst: Input and output contiguous m-by-k-by-n array
 * */
{
    Index i0 = threadIdx.x + blockIdx.x*blockDim.x,
          i1 = threadIdx.y + blockIdx.y*blockDim.y,
          i2 = threadIdx.z + blockIdx.z*blockDim.z;
    constexpr T zero = 0;
    if(i2 < k and i1 < n and i0 < m)
    {
        // Pointer to a corresponding fiber of the output array dst
        T *dst_fiber = dst + i1*mk + i0;
        // Value to add to the output fiber
        const T src_val = alpha * src[i1*m+i0];
        // Overwrite or update output depending on beta
        if(beta == zero)
        {
            // Set output value
            dst_fiber[i2*m] = src_val;
        }
        else
        {
            // Read value from the output
            T &dst_val = dst_fiber[i2*m];
            // And update it
            dst_val = beta*dst_val + src_val;
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, scal_t alpha,
        const T *src_, scal_t beta, T *dst_)
    noexcept
//! Per-element addition of a tensor and a broadcasted slice on CUDA
/*! This is a host function that does the following operations:
 *      dst[i,l,j] = beta*dst[i,l,j] + alpha*src[i,j]
 *
 * @param[in] m: Size of the first mode of src and dst tensors
 * @param[in] n: Size of the last mode of src and dst tensors
 * @param[in] k: Size of the middle mode of dst tensor
 * @param[in] alpha: Scalar factor for src
 * @param[in] src_: Input contiguous m-by-n array
 * @param[in] beta: Scaling factor for dst
 * @param[inout] dst_: Input and output contiguous m-by-k-by-n array
 * */
{
    // Both source and destination are Fortran-contiguous
    dim3 threads(std::min(int(m), 8), std::min(int(n), 8),
            std::min(int(k), 16));
    dim3 blocks((m+threads.x-1)/threads.x, (n+threads.y-1)/threads.y,
            (k+threads.z-1)/threads.z);
    using Y = typename CUDAComputeType<T>::value;
    auto src = reinterpret_cast<const Y *>(src_);
    auto dst = reinterpret_cast<Y *>(dst_);
    (cuda_kernel<Y>)<<<blocks, threads, 0, stream>>>(m, n, k, m*k, Y{alpha},
            src, Y{beta}, dst);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index m, Index n, Index k, scal_t alpha,
        const fp32_t *src, scal_t beta, fp32_t *dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index m, Index n, Index k, scal_t alpha,
        const fp64_t *src, scal_t beta, fp64_t *dst)
    noexcept;

} // namespace nntile::kernel::add_slice
