/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/conv2d/cuda.hh
 * 2D-Convolution between 2 matrices
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-09-14
 * */

#include <nntile/base_types.hh>
#include <cuda_runtime.h>

namespace nntile {
namespace kernel {
namespace conv2d {

template <typename T>
void cuda(cudaStream_t stream, Index nx, Index ny, const T* src, Index mx,
          Index my, const T* kernel, T* dst) noexcept;

}  // namespace conv2d
}  // namespace kernel
}  // namespace nntile
