/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/copy_intersection.cc
 * Copy intersection of 2 tensors from one into another
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-09-26
 * */

#include "nntile/tensor/copy_intersection.hh"
#include "nntile/starpu/subcopy.hh"

namespace nntile
{
namespace tensor
{

//! Asynchronous tensor-wise copy operation
/*! This operation finds an intersection of the source and the target tensors
 * and copies only the data within the found intersection. No elements of the
 * destination tensor outside the intersection mask are updated. Both the
 * source and the target tensors assumed to have the same offset.
 *
 * @param[in] src: Source tensor
 * @param[in] src_offset: Initial offset of the source tensor
 * @param[inout] dst: Destination tensor
 * @param[in] dst_offset: Initial offset of the destination tensor
 * */
template<typename T>
void copy_intersection_async(const Tensor<T> &src,
        const std::vector<Index> &src_offset, const Tensor<T> &dst,
        const std::vector<Index> &dst_offset)
{
    // Check dimensions
    if(src.ndim != src_offset.size())
    {
        throw std::runtime_error("src.ndim != src_offset.size()");
    }
    if(src.ndim != dst.ndim)
    {
        throw std::runtime_error("src.ndim != dst.ndim");
    }
    if(dst.ndim != dst_offset.size())
    {
        throw std::runtime_error("dst.ndim != dst_offset.size()");
    }
    int mpi_rank = starpu_mpi_world_rank();
    int ret;
    // Treat special case of ndim=0
    if(src.ndim == 0)
    {
        auto src_tile_handle = src.get_tile_handle(0);
        auto dst_tile_handle = dst.get_tile_handle(0);
        int src_tile_rank = starpu_mpi_data_get_rank(src_tile_handle);
        int dst_tile_rank = starpu_mpi_data_get_rank(dst_tile_handle);
        // Transfer source tile to dest node
        if(mpi_rank == src_tile_rank or mpi_rank == dst_tile_rank)
        {
            ret = starpu_mpi_get_data_on_node_detached(MPI_COMM_WORLD,
                    src_tile_handle, dst_tile_rank, nullptr, nullptr);
            if(ret != 0)
            {
                throw std::runtime_error("Error in starpu_mpi_get_data_on_"
                        "node_detached");
            }
        }
        // Execute on destination node
        if(mpi_rank == dst_tile_rank)
        {
            ret = starpu_data_cpy(dst_tile_handle, src_tile_handle, 1,
                    nullptr, nullptr);
            if(ret != 0)
            {
                throw std::runtime_error("Error in starpu_data_cpy");
            }
        }
        // Flush cache for the output tile on every node
        starpu_mpi_cache_flush(MPI_COMM_WORLD, dst_tile_handle);
        return;
    }
    // Treat easy case of full copy
    if(src_offset == dst_offset and src.shape == dst.shape
            and src.basetile_shape == dst.basetile_shape)
    {
        for(Index i = 0; i < src.grid.nelems; ++i)
        {
            auto src_tile_handle = src.get_tile_handle(i);
            auto dst_tile_handle = dst.get_tile_handle(i);
            int src_tile_rank = starpu_mpi_data_get_rank(src_tile_handle);
            int dst_tile_rank = starpu_mpi_data_get_rank(dst_tile_handle);
            // Transfer source tile to dest node
            if(mpi_rank == src_tile_rank or mpi_rank == dst_tile_rank)
            {
                ret = starpu_mpi_get_data_on_node_detached(MPI_COMM_WORLD,
                        src_tile_handle, dst_tile_rank, nullptr, nullptr);
                if(ret != 0)
                {
                    throw std::runtime_error("Error in starpu_mpi_get_data_on_"
                            "node_detached");
                }
            }
            // Execute on destination node
            if(mpi_rank == dst_tile_rank)
            {
                ret = starpu_data_cpy(dst_tile_handle, src_tile_handle, 1,
                        nullptr, nullptr);
                if(ret != 0)
                {
                    throw std::runtime_error("Error in starpu_data_cpy");
                }
            }
            // Flush cache for the output tile on every node
            starpu_mpi_cache_flush(MPI_COMM_WORLD, dst_tile_handle);
        }
        return;
    }
    // Do the slow complex copy
    // Temporary buffer for indexing, that is allocated per-worker when needed
    starpu::VariableHandle scratch(2*src.ndim*sizeof(Index), STARPU_SCRATCH);
    Index ndim = src.ndim;
    // We define starting coordinates and shapes for all complex copies of
    // tiles
    std::vector<Index> src_start(ndim), dst_start(ndim), copy_shape(ndim);
    std::vector<Index> dst_tile_index_begin(ndim), dst_tile_index_end(ndim);
    // Total amount of destination tiles, touched by the complex copy
    Index dst_ntiles = 1;
    // Obtain starting indices and shape of intersection for a tensor-wise copy
    for(Index i = 0; i < ndim; ++i)
    {
        // Do nothing if tensors do not intersect
        if((src_offset[i]+src.shape[i] <= dst_offset[i])
                or (dst_offset[i]+dst.shape[i] <= src_offset[i]))
        {
            return;
        }
        // Copy to the beginning of destination
        if(src_offset[i] < dst_offset[i])
        {
            src_start[i] = dst_offset[i] - src_offset[i];
            dst_start[i] = 0;
            copy_shape[i] = std::min(src.shape[i]-src_start[i],
                    dst.shape[i]);
        }
        // Copy from the beginning of source
        else
        {
            src_start[i] = 0;
            dst_start[i] = src_offset[i] - dst_offset[i];
            copy_shape[i] = std::min(dst.shape[i]-dst_start[i],
                    src.shape[i]);
        }
        dst_tile_index_begin[i] = dst_start[i] / dst.basetile_shape[i];
        dst_tile_index_end[i] = (dst_start[i]+copy_shape[i]-1)
            / dst.basetile_shape[i] + 1;
        dst_ntiles *= dst_tile_index_end[i] - dst_tile_index_begin[i];
    }
    // Cycle through all destination tiles
    std::vector<Index> dst_tile_index(dst_tile_index_begin);
    for(Index i = 0; i < dst_ntiles; ++i)
    {
        Index dst_tile_offset = dst.grid.index_to_linear(dst_tile_index);
        auto dst_tile_traits = dst.get_tile_traits(dst_tile_offset);
        auto dst_tile_handle = dst.get_tile_handle(dst_tile_offset);
        int dst_tile_rank = starpu_mpi_data_get_rank(dst_tile_handle);
        // Total number of source tiles that copy something to the destination
        // tile
        Index src_ntiles = 1;
        // Mode for the output destination tile. It is STARPU_W if entire
        // destination tile is overwritten and STARPU_RW otherwise.
        enum starpu_data_access_mode dst_tile_mode = STARPU_W;
        // Contiguous indices of source tile to copy from into the current
        // destination tile
        std::vector<Index> src_tile_index_begin(ndim),
            src_tile_index_end(ndim);
        // Find corresponding source tiles
        for(Index j = 0; j < ndim; ++j)
        {
            // Check if the current destination tile is the mostleft (minimal
            // coordinate) in the current dimension
            if(dst_tile_index[j] == dst_tile_index_begin[j])
            {
                src_tile_index_begin[j] = src_start[j] / src.basetile_shape[j];
                // Check if destination tile is only partially overwritten
                if(dst_tile_index[j]*dst.basetile_shape[j] != dst_start[j])
                {
                    dst_tile_mode = STARPU_RW;
                }
            }
            else
            {
                src_tile_index_begin[j] =
                    (dst_tile_index[j]*dst.basetile_shape[j]
                     -dst_start[j]+src_start[j]) / src.basetile_shape[j];
            }
            // Check if the current destination tile is the mostright (maximal
            // coordinate) in the current dimension
            if(dst_tile_index[j]+1 == dst_tile_index_end[j])
            {
                src_tile_index_end[j] = (src_start[j]+copy_shape[j]-1)
                    /src.basetile_shape[j] + 1;
                // Check if destination tile is only partially overwritten
                if(dst_tile_index[j]*dst.basetile_shape[j]
                        +dst_tile_traits.shape[j]
                        != dst_start[j]+copy_shape[j])
                {
                    dst_tile_mode = STARPU_RW;
                }
            }
            else
            {
                src_tile_index_end[j] =
                    ((dst_tile_index[j]+1)*dst.basetile_shape[j]-1
                     -dst_start[j]+src_start[j]) / src.basetile_shape[j] + 1;
            }
            src_ntiles *= src_tile_index_end[j] - src_tile_index_begin[j];
        }
        // Process the first source tile separately to take into account mode
        std::vector<Index> src_tile_start(ndim), dst_tile_start(ndim),
            copy_tile_shape(ndim);
        for(Index k = 0; k < ndim; ++k)
        {
            // Check if the current destination tile is the mostleft (minimal
            // coordinate) in the current dimension
            if(dst_tile_index[k] == dst_tile_index_begin[k])
            {
                src_tile_start[k] = src_start[k]
                    - src_tile_index_begin[k]*src.basetile_shape[k];
                dst_tile_start[k] = dst_start[k]
                    - dst_tile_index[k]*dst.basetile_shape[k];
            }
            else
            {
                src_tile_start[k] = src_start[k] - dst_start[k]
                    + dst_tile_index[k]*dst.basetile_shape[k]
                    - src_tile_index_begin[k]*src.basetile_shape[k];
                dst_tile_start[k] = 0;
            }
            // Check if corresponding source tiles have fixed coordinate in the
            // current dimension
            if(src_tile_index_begin[k]+1 == src_tile_index_end[k])
            {
                // Check if destination tile is the mostright (maximal
                // coordinate) in the current dimension
                if(dst_tile_index[k]+1 == dst_tile_index_end[k])
                {
                    copy_tile_shape[k] = src_start[k] + copy_shape[k]
                        - src_tile_index_begin[k]*src.basetile_shape[k]
                        - src_tile_start[k];
                }
                else
                {
                    copy_tile_shape[k] = dst.basetile_shape[k]
                        - dst_tile_start[k];
                }
            }
            else
            {
                copy_tile_shape[k] = src.basetile_shape[k]
                    - src_tile_start[k];
            }
        }
        // All the properties of the first source tile to copy from
        Index src_first_tile_offset = src.grid.index_to_linear( 
                src_tile_index_begin);
        auto src_first_tile_traits = src.get_tile_traits(
                src_first_tile_offset);
        auto src_first_tile_handle = src.get_tile_handle(
                src_first_tile_offset);
        int src_first_tile_rank = starpu_mpi_data_get_rank(
                src_first_tile_handle);
        // Transfer first source tile to dest node
        if(mpi_rank == src_first_tile_rank or mpi_rank == dst_tile_rank)
        {
            ret = starpu_mpi_get_data_on_node_detached(MPI_COMM_WORLD,
                    src_first_tile_handle, dst_tile_rank, nullptr, nullptr);
            if(ret != 0)
            {
                throw std::runtime_error("Error in starpu_mpi_get_data_on_"
                        "node_detached");
            }
        }
        // If there is only one corresponding source tile, that can be copied
        // without calling complex copy codelet
        if(src_ntiles == 1 and dst_tile_mode == STARPU_W
                and copy_tile_shape == src_first_tile_traits.shape)
        {
            // Execute on destination node
            if(mpi_rank == dst_tile_rank)
            {
                ret = starpu_data_cpy(dst_tile_handle, src_first_tile_handle,
                        1, nullptr, nullptr);
                if(ret != 0)
                {
                    throw std::runtime_error("Error in starpu_data_cpy");
                }
            }
        }
        // So now we have to use only complex copying
        else
        {
            // Execute on dest node
            if(mpi_rank == dst_tile_rank)
            {
                starpu::subcopy::submit<T>(ndim, src_tile_start,
                        src_first_tile_traits.stride, dst_tile_start,
                        dst_tile_traits.stride, copy_tile_shape,
                        src_first_tile_handle, dst_tile_handle,
                        scratch, dst_tile_mode);
            }
            // Proceed with all the rest source tiles
            std::vector<Index> src_tile_index(src_tile_index_begin);
            // Cycle through all corresponding source tiles
            for(Index j = 1; j < src_ntiles; ++j)
            {
                // Get next source tile
                ++src_tile_index[0];
                Index k = 0;
                while(src_tile_index[k] == src_tile_index_end[k])
                {
                    src_tile_index[k] = src_tile_index_begin[k];
                    ++k;
                    ++src_tile_index[k];
                }
                // Get starting indices of source and destination tiles and
                // deduce shape of copy
                for(Index k = 0; k < ndim; ++k)
                {
                    // Check if the current source tile is the mostleft
                    // (minimal coordinate) in the current dimension
                    if(src_tile_index[k] == src_tile_index_begin[k])
                    {
                        // Check if the current destination tile is the
                        // mostleft (minimal coordinate) in the current
                        // dimension
                        if(dst_tile_index[k] == dst_tile_index_begin[k])
                        {
                            src_tile_start[k] = src_start[k]
                                - src_tile_index[k]*src.basetile_shape[k];
                            dst_tile_start[k] = dst_start[k]
                                - dst_tile_index[k]*dst.basetile_shape[k];
                        }
                        else
                        {
                            src_tile_start[k] = src_start[k] - dst_start[k]
                                + dst_tile_index[k]*dst.basetile_shape[k]
                                - src_tile_index[k]*src.basetile_shape[k];
                            dst_tile_start[k] = 0;
                        }
                    }
                    else
                    {
                        src_tile_start[k] = 0;
                        dst_tile_start[k] = dst_start[k] - src_start[k]
                            + src_tile_index[k]*src.basetile_shape[k]
                            - dst_tile_index[k]*dst.basetile_shape[k];
                    }
                    // Check if the current source tile is the mostright
                    // (maximal coordinate) in the current dimension
                    if(src_tile_index[k]+1 == src_tile_index_end[k])
                    {
                        // Check if the current destination tile is the
                        // mostright (maximal coordinate) in the current
                        // dimension
                        if(dst_tile_index[k]+1 == dst_tile_index_end[k])
                        {
                            copy_tile_shape[k] = src_start[k] + copy_shape[k]
                                - src_tile_index[k]*src.basetile_shape[k]
                                - src_tile_start[k];
                        }
                        else
                        {
                            copy_tile_shape[k] = dst.basetile_shape[k]
                                - dst_tile_start[k];
                        }
                    }
                    else
                    {
                        copy_tile_shape[k] = src.basetile_shape[k]
                            - src_tile_start[k];
                    }
                }
                // Get all the parameters for the complex copy
                Index src_tile_offset = src.grid.index_to_linear(
                        src_tile_index);
                auto src_tile_handle = src.get_tile_handle(src_tile_offset);
                int src_tile_rank = starpu_mpi_data_get_rank(src_tile_handle);
                // Transfer source tile to dest node
                if(mpi_rank == src_tile_rank or mpi_rank == dst_tile_rank)
                {
                    ret = starpu_mpi_get_data_on_node_detached(MPI_COMM_WORLD,
                            src_tile_handle, dst_tile_rank, nullptr, nullptr);
                    if(ret != 0)
                    {
                        throw std::runtime_error("Error in starpu_mpi_get_"
                                "data_on_node_detached");
                    }
                }
                // Execute on dest node
                if(mpi_rank == dst_tile_rank)
                {
                    auto src_tile_traits = src.get_tile_traits(
                            src_tile_offset);
                    starpu::subcopy::submit<T>(ndim, src_tile_start,
                            src_tile_traits.stride, dst_tile_start,
                            dst_tile_traits.stride, copy_tile_shape,
                            src_tile_handle, dst_tile_handle,
                            scratch, STARPU_RW);
                }
            }
        }
        // Flush cache for the output tile on every node
        starpu_mpi_cache_flush(MPI_COMM_WORLD, dst_tile_handle);
        // Get out if it was the last tile
        if(i == dst_ntiles-1)
        {
            break;
        }
        // Get next tile
        ++dst_tile_index[0];
        Index k = 0;
        while(dst_tile_index[k] == dst_tile_index_end[k])
        {
            dst_tile_index[k] = dst_tile_index_begin[k];
            ++k;
            ++dst_tile_index[k];
        }
    }
}

//! Blocking version of tensor-wise copy operation
/*! This operation finds an intersection of the source and the target tensors
 * and copies only the data within the found intersection. No elements of the
 * destination tensor outside the intersection mask are updated.
 *
 * @param[in] src: Source tensor
 * @param[in] src_offset: Initial offset of the source tensor
 * @param[inout] dst: Destination tensor
 * @param[in] dst_offset: Initial offset of the destination tensor
 * */
template<typename T>
void copy_intersection(const Tensor<T> &src,
        const std::vector<Index> &src_offset, const Tensor<T> &dst,
        const std::vector<Index> &dst_offset)
{
    copy_intersection_async<T>(src, src_offset, dst, dst_offset);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void copy_intersection<fp32_t>(const Tensor<fp32_t> &src,
        const std::vector<Index> &src_offset, const Tensor<fp32_t> &dst,
        const std::vector<Index> &dst_offset);

template
void copy_intersection<fp64_t>(const Tensor<fp64_t> &src,
        const std::vector<Index> &src_offset, const Tensor<fp64_t> &dst,
        const std::vector<Index> &dst_offset);

} // namespace tensor
} // namespace nntile
