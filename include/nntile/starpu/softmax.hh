/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu/softmax.hh
 * Softmax operation for StarPU buffer
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-12-09
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/starpu/config.hh>
#include <nntile/defs.h>

namespace nntile
{
namespace starpu
{
namespace softmax
{

//! Structure for arguments
struct args_t
{
    Index m;
    Index n;
    Index k;
};

// Apply softmax along middle axis of StarPU buffer on CPU
template<typename T>
void cpu(void *buffers[], void *cl_args)
    noexcept;

#ifdef NNTILE_USE_CUDA
// Apply softmax along middle axis of StarPU buffer on CUDA
template<typename T>
void cuda(void *buffers[], void *cl_args)
    noexcept;
#endif // NNTILE_USE_CUDA

extern Codelet codelet_fp32, codelet_fp64;

template<typename T>
constexpr Codelet *codelet()
{
    throw std::runtime_error("Non-supported type");
    return nullptr;
}

template<>
constexpr Codelet *codelet<fp32_t>()
{
    return &codelet_fp32;
}

template<>
constexpr Codelet *codelet<fp64_t>()
{
    return &codelet_fp64;
}

void init();

void restrict_where(uint32_t where);

void restore_where();

template<typename T>
void submit(Index m, Index n, Index k, Handle maxsumnorm, Handle dst);

} // namespace softmax
} // namespace starpu
} // namespace nntile
