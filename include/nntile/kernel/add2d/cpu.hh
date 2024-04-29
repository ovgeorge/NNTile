/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/add2d/cpu.hh
 * Add operation on buffers on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @author Aleksandr Katrutsa
 * @date 2023-07-01
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile
{
namespace kernel
{
namespace add2d
{

// Apply add2d for buffers on CPU
template <typename T>
void cpu(Index nx, Index ny, T alpha, const T *src, Index ld_src, T beta,
         T *dst, Index ld_dst) noexcept;

} // namespace add2d
} // namespace kernel
} // namespace nntile
