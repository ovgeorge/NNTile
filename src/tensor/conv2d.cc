/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/conv2d.cc
 * Tensor wrappers for 2D-Convolution between 2 matrices
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-07-03
 * */

#include "nntile/tensor/conv2d.hh"
#include "nntile/starpu/conv2d.hh"

namespace nntile
{
namespace tensor
{

template <typename T>
void conv2d_async(const Tensor<T> &src, const Tensor<T> &kernel,
                  const Tensor<T> &dst)
//! Tensor<T> 2D-Convolution between 2 matrices
/*! Reshapes input tensors into 2-dimensional arrays
 * and performs the 2D-Convolution
 *
 * @param[in] src: Input tensor, that is reshaped into 2D array
 * @param[in] kernel: Input tensor, that is reshaped into 2D array
 * @param[out] dst: Resulting tensor, that is reshaped into 2D array
 * */
{
    Index axis = 0;
    // Check dimensions
    if(2 != src.ndim)
    {
        throw std::runtime_error("2 != src.ndim");
    }
    if(2 != kernel.ndim)
    {
        throw std::runtime_error("2 != kernel.ndim");
    }
    if(2 != dst.ndim)
    {
        throw std::runtime_error("2 != dst.ndim");
    }
    // Check shapes of tensors
    for(Index i = 0; i < dst.ndim; ++i)
    {
        if(dst.shape[i] != src.shape[i] + kernel.shape[i] - 1)
        {
            throw std::runtime_error(
                "dst.shape[i] != src.shape[i] + kernel.shape[i] - 1");
        }
        if(dst.basetile_shape[i] !=
           src.basetile_shape[i] + kernel.basetile_shape[i] - 1)
        {
            throw std::runtime_error(
                "dst.basetile_shape[i] != src.basetile_shape[i] + "
                "kernel.basetile_shape[i] - 1");
        }
    }

    // Apply conv2d
    int mpi_rank = starpu_mpi_world_rank();
    int ret;
    for(Index i = 0; i < src.grid.nelems; ++i)
    {
        // Index of current source tile
        auto src_tile_index = src.grid.linear_to_index(i);
        // Source tile traits
        auto src_tile_traits = src.get_tile_traits(i);
        // Source tile handle
        auto src_tile_handle = src.get_tile_handle(i);
        // Set fixed indices of current destination tile
        std::vector<Index> dst_tile_index(dst.ndim);
        for(Index j = 0; j < dst.ndim; ++j)
        {
            dst_tile_index[j] = src_tile_index[j];
        }
        // Loop through all necessary destination tiles
        for(Index j = 0; j < dst.grid.shape[axis]; ++j)
        {
            // Set floating axis
            dst_tile_index[axis] = j;
            // Get linear offset from index
            Index dst_tile_offset = dst.grid.index_to_linear(dst_tile_index);
            // Get destination tile handle
            auto dst_tile_handle = dst.get_tile_handle(dst_tile_offset);
            // Get kernel tile handle
            auto kernel_tile_handle = kernel.get_tile_handle(dst_tile_offset);
            // Kernel tile traits
            auto kernel_tile_traits = kernel.get_tile_traits(i);
            // MPI rank of the destination tile
            int dst_tile_rank = dst_tile_handle.mpi_get_rank();
            // Transfer data
            src_tile_handle.mpi_transfer(dst_tile_rank, mpi_rank);
            kernel_tile_handle.mpi_transfer(dst_tile_rank, mpi_rank);
            // Execute on destination node
            if(mpi_rank == dst_tile_rank)
            {
                // Get destination tile traits
                auto dst_tile_traits = dst.get_tile_traits(dst_tile_offset);
                // Reshape inputs: src_tile -> (m,n), dst_tile -> (m,k,n)
                Index nx, ny, mx, my;
                nx = src_tile_traits.stride[axis];
                ny = src_tile_traits.matrix_shape[axis][1];
                mx = kernel_tile_traits.stride[axis];
                my = kernel_tile_traits.matrix_shape[axis][1];
                // Insert corresponding task
                starpu::conv2d::submit<T>(nx, ny, src_tile_handle, mx, my,
                                          kernel_tile_handle, dst_tile_handle);
            }
            // Flush cache for the output tile on every node
            dst_tile_handle.mpi_flush();
        }
    }
}

template <typename T>
void conv2d(const Tensor<T> &src, const Tensor<T> &kernel, const Tensor<T> &dst)
//! Tensor<T> 2D-Convolution between 2 matrices
/*! Blocking version of conv2d_async<T>.
 * Reshapes input tensors into 2-dimensional arrays
 * and performs the 2D-Convolution
 *
 * @param[in] src: Input tensor, that is reshaped into 2D array
 * @param[in] kernel: Input tensor, that is reshaped into 2D array
 * @param[out] dst: Resulting tensor, that is reshaped into 2D array
 * */
{
    conv2d_async<T>(src, kernel, dst);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation of template
template void conv2d_async<fp32_t>(const Tensor<fp32_t> &src,
                                   const Tensor<fp32_t> &kernel,
                                   const Tensor<fp32_t> &dst);

template void conv2d_async<fp64_t>(const Tensor<fp64_t> &src,
                                   const Tensor<fp64_t> &kernel,
                                   const Tensor<fp64_t> &dst);

// Explicit instantiation of template
template void conv2d<fp32_t>(const Tensor<fp32_t> &src,
                             const Tensor<fp32_t> &kernel,
                             const Tensor<fp32_t> &dst);

template void conv2d<fp64_t>(const Tensor<fp64_t> &src,
                             const Tensor<fp64_t> &kernel,
                             const Tensor<fp64_t> &dst);

} // namespace tensor
} // namespace nntile
