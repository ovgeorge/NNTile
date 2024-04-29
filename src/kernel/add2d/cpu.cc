/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/add2d/cpu.cc
 * Add operation on buffers on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @author Aleksandr Katrutsa
 * @date 2023-07-02
 * */

#include "nntile/kernel/add2d/cpu.hh"

namespace nntile
{
namespace kernel
{
namespace add2d
{

template <typename T>
void cpu(Index nx, Index ny, T alpha, const T *src, Index ld_src, T beta,
         T *dst, Index ld_dst) noexcept
//! Add of two buffers on CPU
/*! Performs the following operation:
 *      dst[i, j] = alpha*src[i, j] + beta*dst[i, j],
 * where alpha and beta are non-zero scalars.
 *
 * @param[in] nx: Size of the src and dst tensors along first axis
 * @param[in] ny: Size of the src and dst tensors along second axis
 * @param[in] alpha: Scalar multiplier for the src tensor
 * @param[in] src: Source tensor
 * @param[in] ld_src: number of elements between each elements along second axis
 * for the src tensor
 * @param[in] beta: Scalar multiplier for the dst tensor
 * @param[inout] dst: Destination of the add2d operation
 * @param[in] ld_src: number of elements between each elements along second axis
 * for the dst tensor
 * */
{
    for(Index j = 0; j < ny; ++j)
        for(Index i = 0; i < nx; ++i)
            dst[i + j * ld_dst] =
                alpha * src[i + j * ld_src] + beta * dst[i + j * ld_dst];
}

// Explicit instantiation
template void cpu<fp32_t>(Index nx, Index ny, fp32_t alpha, const fp32_t *src,
                          Index ld_src, fp32_t beta, fp32_t *dst,
                          Index ld_dst) noexcept;

template void cpu<fp64_t>(Index nx, Index ny, fp64_t alpha, const fp64_t *src,
                          Index ld_src, fp64_t beta, fp64_t *dst,
                          Index ld_dst) noexcept;

} // namespace add2d
} // namespace kernel
} // namespace nntile
