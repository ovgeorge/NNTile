/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/add2d.cc
 * Add operation on a StarPU buffers
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @author Aleksandr Katrutsa
 * @date 2023-07-22
 * */

#include "nntile/starpu/add2d.hh"
#include "nntile/kernel/add2d.hh"
#include "nntile/starpu/clear.hh"
#include "nntile/starpu/scal.hh"
#include "nntile/starpu/scal_inplace.hh"
#include <cstdlib>

namespace nntile
{
namespace starpu
{
//! StarPU wrappers for add2d operation
namespace add2d
{

//! Apply add2d operation for StarPU buffers in CPU
template <typename T> void cpu(void *buffers[], void *cl_args) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t<T> *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *src = interfaces[0]->get_ptr<T>();
    T *dst = interfaces[1]->get_ptr<T>();
    // Launch kernel
    kernel::add2d::cpu<T>(args->nx, args->ny, args->alpha,
                          src + args->offset_src, args->ld_src, args->beta,
                          dst + args->offset_dst, args->ld_dst);
}

#ifdef NNTILE_USE_CUDA
//! Apply add2d for StarPU buffers on CUDA
template <typename T> void cuda(void *buffers[], void *cl_args) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t<T> *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *src = interfaces[0]->get_ptr<T>();
    T *dst = interfaces[1]->get_ptr<T>();
    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Launch kernel
    kernel::add2d::cuda<T>(stream, args->nelems, args->alpha, src, args->beta,
                           dst);
}
#endif // NNTILE_USE_CUDA

//! Footprint for add2d tasks that depends only on cl_arg
template <typename T> static uint32_t footprint(struct starpu_task *task)
{
    // Get arguments
    auto args = reinterpret_cast<args_t<T> *>(task->cl_arg);
    // Apply hash over parameters m, n and batch.
    uint32_t hash = 0;
    hash = starpu_hash_crc32c_be_n(&args->nx, sizeof(args->nx), hash);
    hash = starpu_hash_crc32c_be_n(&args->ny, sizeof(args->ny), hash);
    return hash;
}

Codelet codelet_fp32, codelet_fp64;

void init()
{
    codelet_fp32.init("nntile_add2d_fp32", footprint<fp32_t>, {cpu<fp32_t>},
#ifdef NNTILE_USE_CUDA
                      {cuda<fp32_t>}
#else  // NNTILE_USE_CUDA
                      {}
#endif // NNTILE_USE_CUDA
    );
    codelet_fp64.init("nntile_add2d_fp64", footprint<fp64_t>, {cpu<fp64_t>},
#ifdef NNTILE_USE_CUDA
                      {cuda<fp64_t>}
#else  // NNTILE_USE_CUDA
                      {}
#endif // NNTILE_USE_CUDA
    );
}

void restrict_where(uint32_t where)
{
    codelet_fp32.restrict_where(where);
    codelet_fp64.restrict_where(where);
}

void restore_where()
{
    codelet_fp32.restore_where();
    codelet_fp64.restore_where();
}

template <typename T>
void submit(Index nx, Index ny, T alpha, Handle src, Index offset_src,
            Index ld_src, T beta, Handle dst, Index offset_dst, Index ld_dst)
//! Insert add2d task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    // constexpr T zero = 0, one = 1;
    //// If beta is zero this function reduces to scal
    // if(beta == zero)
    //{
    //     scal::submit<T>(nelems, alpha, src, dst);
    //     return;
    // }
    //// If beta is non-zero and alpha is zero then reduce to scal_inplace
    // if(alpha == zero)
    //{
    //     scal_inplace::submit<T>(nelems, beta, dst);
    //     return;
    // }
    //// Access mode for the dst handle
    // enum starpu_data_access_mode dst_mode;
    // if(beta == one)
    //{
    //     dst_mode = Config::STARPU_RW_COMMUTE;
    // }
    // else
    //{
    //     dst_mode = STARPU_RW;
    // }
    //  Codelet arguments
    args_t<T> *args = (args_t<T> *)std::malloc(sizeof(*args));
    args->nx = nx;
    args->ny = ny;
    args->alpha = alpha;
    args->offset_src = offset_src;
    args->ld_src = ld_src;
    args->beta = beta;
    args->offset_dst = offset_dst;
    args->ld_dst = ld_dst;
    // Submit task
    int ret = starpu_task_insert(codelet<T>(), STARPU_R,
                                 static_cast<starpu_data_handle_t>(src),
                                 STARPU_CL_ARGS, args, sizeof(*args), STARPU_RW,
                                 static_cast<starpu_data_handle_t>(dst), 0);
    // STARPU_FLOPS, nflops);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in add2d task submission");
    }
}

// Explicit instantiation
template void submit<fp32_t>(Index nx, Index ny, fp32_t alpha, Handle src,
                             Index offset_src, Index ld_src, fp32_t beta,
                             Handle dst, Index offset_dst, Index ld_dst);

template void submit<fp64_t>(Index nx, Index ny, fp64_t alpha, Handle src,
                             Index offset_src, Index ld_src, fp64_t beta,
                             Handle dst, Index offset_dst, Index ld_dst);

} // namespace add2d
} // namespace starpu
} // namespace nntile
