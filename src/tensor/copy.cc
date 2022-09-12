/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/copy.cc
 * Copy one tensors into another matching tensor
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-09-12
 * */

#include "nntile/tensor/copy.hh"

namespace nntile
{
namespace tensor
{

//! Asynchronous tensor-wise copy operation
/*! A simple copy from one tensor into another
 *
 * @param[in] src: Source tensor
 * @param[inout] dst: Destination tensor
 * */
template<typename T>
void copy_async(const Tensor<T> &src, const Tensor<T> &dst)
{
    // Check shapes and tiles
    if(src.shape != dst.shape)
    {
        throw std::runtime_error("src.shape != dst.shape");
    }
    if(src.basetile_shape != dst.basetile_shape)
    {
        throw std::runtime_error("src.basetile_shape != dst.basetile_shape");
    }
    // Copy tile-by-tile
    int mpi_rank = starpu_mpi_world_rank();
    int ret;
    for(Index i = 0; i < src.grid.nelems; ++i)
    {
        auto src_tile_handle = src.get_tile_handle(i);
        auto dst_tile_handle = dst.get_tile_handle(i);
        int src_tile_rank = starpu_mpi_data_get_rank(src_tile_handle);
        int dst_tile_rank = starpu_mpi_data_get_rank(dst_tile_handle);
        int tile_tag = starpu_mpi_data_get_tag(src_tile_handle);
        // Init send for owner of source tile
        if(mpi_rank == src_tile_rank)
        {
            // If both source and destination are owned by the same node
            if(mpi_rank == dst_tile_rank)
            {
                ret = starpu_data_cpy(dst_tile_handle, src_tile_handle, 1,
                        nullptr, nullptr);
                if(ret != 0)
                {
                    throw std::runtime_error("Error in starpu_data_cpy");
                }
            }
            else
            {
                ret = starpu_mpi_isend_detached(src_tile_handle,
                        dst_tile_rank, tile_tag, MPI_COMM_WORLD, nullptr,
                        nullptr);
                if(ret != 0)
                {
                    throw std::runtime_error("Error in starpu_mpi_isend_"
                            "detached");
                }
            }
        }
        // Init receive for owner of destination tile
        else if(mpi_rank == dst_tile_rank)
        {
            ret = starpu_mpi_irecv_detached(dst_tile_handle, src_tile_rank,
                    tile_tag, MPI_COMM_WORLD, nullptr, nullptr);
            if(ret != 0)
            {
                throw std::runtime_error("Error in starpu_mpi_irecv_"
                        "detached");
            }
        }
    }
}

//! Blocking version of tensor-wise copy operation
/*! A simple copy from one tile into another
 *
 * @param[in] src: Source tensor
 * @param[inout] dst: Destination tensor
 * */
template<typename T>
void copy(const Tensor<T> &src, const Tensor<T> &dst)
{
    copy_async<T>(src, dst);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void copy<fp32_t>(const Tensor<fp32_t> &src, const Tensor<fp32_t> &dst);

template
void copy<fp64_t>(const Tensor<fp64_t> &src, const Tensor<fp64_t> &dst);

} // namespace tensor
} // namespace nntile

