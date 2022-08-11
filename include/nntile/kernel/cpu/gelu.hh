/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/cpu/gelu.hh
 * GeLU operation on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-04
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile
{
namespace kernel
{
namespace cpu
{

// GeLU operation on a buffer
template<typename T>
void gelu(Index nelems, T *data)
    noexcept;

} // namespace cpu
} // namespace kernel
} // namespace nntile
