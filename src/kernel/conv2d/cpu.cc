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
void cpu(Index nx, Index ny, const T* src, Index mx, Index my, const T* kernel,
         T* dst) noexcept
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
  // Assume size of dst == size of src + size of kernel
  // Cycle over column of the output buffer dst
  for (Index i2 = 0; i2 < nx + mx - 1; ++i2) {
    // Cycle over row of the output buffer dst
    for (Index i1 = 0; i1 < ny + my - 1; ++i1) {
      dst[i1 + (ny+my-1) * i2] = 0;
    }
  }

  // Cycle over column of the output buffer dst
  for (Index i2 = 0; i2 < nx; ++i2) {
    // Cycle over row of the output buffer dst
    for (Index i1 = 0; i1 < ny; ++i1) {
      for (Index j2 = 0; j2 < mx; ++j2) {
        // Cycle over row of the output buffer dst
        for (Index j1 = 0; j1 < my; ++j1) {
          dst[i1 + j1 + (ny + my - 1) * (i2 + j2)] +=
              src[i1 + ny * i2] * kernel[j1 + my * j2];
        }
      }
    }
  }
}

// Explicit instantiation
template void cpu<fp32_t>(Index nx, Index ny, const fp32_t* src, Index mx,
                          Index my, const fp32_t* kernel, fp32_t* dst) noexcept;

template void cpu<fp64_t>(Index nx, Index ny, const fp64_t* src, Index mx,
                          Index my, const fp64_t* kernel, fp64_t* dst) noexcept;

}  // namespace conv2d
}  // namespace kernel
}  // namespace nntile
