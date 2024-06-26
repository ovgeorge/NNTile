/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu/add.hh
 * Add operation on StarPU buffers
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @author Aleksandr Katrutsa
 * @date 2023-07-02
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/defs.h>
#include <nntile/starpu/config.hh>

namespace nntile
{
namespace starpu
{
namespace add2d
{

//! Structure for arguments
template <typename T> struct args_t
{
    Index nx;
    Index ny;
    T alpha;
    Index offset_src;
    Index ld_src;
    T beta;
    Index offset_dst;
    Index ld_dst;
};

// Apply add for StarPU buffers on CPU
template <typename T> void cpu(void *buffers[], void *cl_args) noexcept;

#ifdef NNTILE_USE_CUDA
// Apply add for StarPU buffers on CUDA
template <typename T> void cuda(void *buffers[], void *cl_args) noexcept;
#endif // NNTILE_USE_CUDA

extern Codelet codelet_fp32, codelet_fp64;

template <typename T> constexpr Codelet *codelet()
{
    throw std::runtime_error("Non-supported type");
    return nullptr;
}

template <> constexpr Codelet *codelet<fp32_t>()
{
    return &codelet_fp32;
}

template <> constexpr Codelet *codelet<fp64_t>()
{
    return &codelet_fp64;
}

void init();

void restrict_where(uint32_t where);

void restore_where();

template <typename T>
void submit(Index nx, Index ny, T alpha, Handle src, Index offset_src,
            Index ld_src, T beta, Handle dst, Index offset_dst, Index ld_dst);

} // namespace add2d
} // namespace starpu
} // namespace nntile
