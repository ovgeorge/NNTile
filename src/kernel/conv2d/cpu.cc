/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/conv2d/cpu.cc
 * 2D-Convolution between 2 matrices
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-28
 * */

#include "nntile/kernel/conv2d/cpu.hh"

namespace nntile {
namespace kernel {
namespace conv2d {

template <typename T>
void cpu(Index src_n, Index src_m, Index src_offset_n, Index src_offset_m,
         const T *src, Index kernel_n, Index kernel_m, Index kernel_offset_n,
         Index kernel_offset_m, const T *kernel, Index dst_n, Index dst_m,
         Index dst_offset_n, Index dst_offset_m, T *dst) noexcept
//! Compute full discrete linear convolution of two 2-dimensional arrays on
//! CPU.
/*!@param[in] nx: Size of the first axis of src array
 * @param[in] ny: Size of the second axis of src array
 * @param[in] src: Input contiguous nx-by-ny array
 * @param[in] mx: Size of the first axis of kernel array
 * @param[in] my: Size of the second axis of kernel array
 * @param[in] kernel: Input contiguous mx-by-my array
 * @param[out] dst: Output contiguous (nx+my)-by-(ny+my) array
 * */
{
    for(Index i1 = src_offset_n; i1 < src_offset_n + src_n; ++i1)
    {
        for(Index j1 = kernel_offset_n; j1 < kernel_offset_n + kernel_n; ++j1)
        {
            Index dst_1 = i1 + j1;
            if(dst_1 < dst_offset_n || dst_offset_n + dst_n <= dst_1)
                continue;
            for(Index i2 = src_offset_m; i2 < src_offset_m + src_m; ++i2)
            {
                for(Index j2 = kernel_offset_m; j2 < kernel_offset_m + kernel_m;
                    ++j2)
                {
                    Index dst_2 = i2 + j2;
                    if(dst_2 < dst_offset_m || dst_offset_m + dst_m <= dst_2)
                        continue;
                    dst[(dst_1 - dst_offset_n) * dst_n + (dst_2 - dst_offset_m)] +=
                        src[(i1 - src_offset_n) * src_n + (i2 - src_offset_m)] *
                        kernel[(j1 - kernel_offset_n) * kernel_n + (j2 - kernel_offset_m)];
				}
            }
        }
    }
}

// Explicit instantiation
template void cpu<fp32_t>(Index src_n, Index src_m, Index src_offset_n,
                          Index src_offset_m, const fp32_t *src, Index kernel_n,
                          Index kernel_m, Index kernel_offset_n,
                          Index kernel_offset_m, const fp32_t *kernel,
                          Index dst_n, Index dst_m, Index dst_offset_n,
                          Index dst_offset_m, fp32_t *dst) noexcept;

template void cpu<fp64_t>(Index src_n, Index src_m, Index src_offset_n,
                          Index src_offset_m, const fp64_t *src, Index kernel_n,
                          Index kernel_m, Index kernel_offset_n,
                          Index kernel_offset_m, const fp64_t *kernel,
                          Index dst_n, Index dst_m, Index dst_offset_n,
                          Index dst_offset_m, fp64_t *dst) noexcept;

}  // namespace conv2d
}  // namespace kernel
}  // namespace nntile
