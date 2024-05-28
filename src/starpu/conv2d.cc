/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/conv2d.cc
 * StarPU wrappers for 2D-Convolution between 2 matrices
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-28
 * */

#include "nntile/starpu/conv2d.hh"
#include "nntile/kernel/conv2d.hh"
#include <cstdlib>

namespace nntile
{
namespace starpu
{
//! StarPU wrappers for conv2d operation
namespace conv2d
{

//! StarPU wrapper for kernel::conv2d::cpu<T>
template<typename T>
void cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *src = interfaces[0]->get_ptr<T>();
    const T *kernel = interfaces[1]->get_ptr<T>();
    T *dst = interfaces[2]->get_ptr<T>();
    // Launch kernel
    kernel::conv2d::cpu<T>(args->src_n, args->src_m, args->src_offset_n, args->src_offset_m, src,
		 args->kernel_n, args->kernel_m, args->kernel_offset_n, args->kernel_offset_m, kernel,
		 args->dst_n, args->dst_m, args->dst_offset_n, args->dst_offset_m, dst);
}

#ifdef NNTILE_USE_CUDA
//! StarPU wrapper for kernel::conv2d::cuda<T>
template<typename T>
void cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *src = interfaces[0]->get_ptr<T>();
    const T *kernel = interfaces[1]->get_ptr<T>();
    T *dst = interfaces[2]->get_ptr<T>();
    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Launch kernel
    kernel::conv2d::cuda<T>(stream, args->nx, args->ny, src, args->mx, args->my,
            kernel, dst);
}
#endif // NNTILE_USE_CUDA

//! Footprint for conv2d tasks
template<typename T>
static
uint32_t footprint(struct starpu_task *task)
{
    // Get arguments
    auto args = reinterpret_cast<args_t *>(task->cl_arg);
    // Apply hash over parameters m, n and k
    uint32_t hash = 0;
    hash = starpu_hash_crc32c_be_n(&args->src_n, sizeof(args->src_n), hash);
    hash = starpu_hash_crc32c_be_n(&args->src_m, sizeof(args->src_m), hash);
    hash = starpu_hash_crc32c_be_n(&args->kernel_n, sizeof(args->kernel_n), hash);
    hash = starpu_hash_crc32c_be_n(&args->kernel_m, sizeof(args->kernel_m), hash);
    return hash;
}

Codelet codelet_fp32, codelet_fp64;

void init()
{
    codelet_fp32.init("nntile_conv2d_fp32",
            footprint<fp32_t>,
            {cpu<fp32_t>},
#ifdef NNTILE_USE_CUDA
            {cuda<fp32_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );
    codelet_fp64.init("nntile_conv2d_fp64",
            footprint<fp64_t>,
            {cpu<fp64_t>},
#ifdef NNTILE_USE_CUDA
            {cuda<fp64_t>}
#else // NNTILE_USE_CUDA
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

template<typename T>
void submit(Index src_tile_n, Index src_tile_m, Index src_offset_n, Index src_offset_m, Handle src,
			Index kernel_tile_n, Index kernel_tile_m, Index kernel_offset_n, Index kernel_offset_m, Handle kernel,
			Index dst_tile_n, Index dst_tile_m, Index dst_offset_n, Index dst_offset_m, Handle dst)
//! Insert conv2d tas_tilek into StarPU_tile pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->src_tile_n = src_tile_n;
    args->src_tile_m = src_tile_m;
    args->src_offset_n = src_offset_n;
    args->src_offset_m = src_offset_m;
    args->kernel_tile_n = kernel_tile_n;
    args->kernel_tile_m = kernel_tile_m;
    args->kernel_offset_n = kernel_offset_n;
    args->kernel_offset_m = kernel_offset_m;
    args->dst_tile_n = dst_tile_n;
    args->dst_tile_m = dst_tile_m;
    args->dst_offset_n = dst_offset_n;
    args->dst_offset_m = dst_offset_m;
    fp64_t nflops = src_tile_n*src_tile_m*dst_tile_n*dst_tile_m;
    // Submit task
    int ret = starpu_task_insert(codelet<T>(),
            STARPU_R, static_cast<starpu_data_handle_t>(src),
            STARPU_R, static_cast<starpu_data_handle_t>(kernel),
            STARPU_CL_ARGS, args, sizeof(*args),
            STARPU_RW, static_cast<starpu_data_handle_t>(dst),
            STARPU_FLOPS, nflops,
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in conv2d task submission");
    }
}

// Explicit instantiation
template
void submit<fp32_t>(Index src_tile_n, Index src_tile_m, Index src_offset_n, Index src_offset_m, Handle src,
			Index kernel_tile_n, Index kernel_tile_m, Index kernel_offset_n, Index kernel_offset_m, Handle kernel,
			Index dst_tile_n, Index dst_tile_m, Index dst_offset_n, Index dst_offset_m, Handle dst);

template
void submit<fp64_t>(Index src_tile_n, Index src_tile_m, Index src_offset_n, Index src_offset_m, Handle src,
			Index kernel_tile_n, Index kernel_tile_m, Index kernel_offset_n, Index kernel_offset_m, Handle kernel,
			Index dst_tile_n, Index dst_tile_m, Index dst_offset_n, Index dst_offset_m, Handle dst);

} // namespace conv2d
} // namespace starpu
} // namespace nntile

