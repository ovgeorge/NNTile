/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/strassen/cpu.hh
 * Strassen multiplication operation on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-10-26
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/constants.hh>

namespace nntile
{
namespace kernel
{
namespace strassen
{

// Per-element product of two buffers
template <typename T>
void cpu(TransOp transA, TransOp transB,
                          Index M, Index N, Index K, T alpha,
                          const T *A, const T *B, T beta, T *C) noexcept;

} // namespace prod
} // namespace kernel
} // namespace nntile

