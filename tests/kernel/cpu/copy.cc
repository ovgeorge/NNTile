/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/cpu/copy.cc
 * Smart copy a buffer on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-10
 * */

#include "nntile/kernel/cpu/copy.hh"
#include <array>
#include <vector>
#include <stdexcept>

using namespace nntile;
using namespace nntile::kernel::cpu;

// Templated validation
template<typename T, std::size_t NDIM>
void validate(std::array<Index, NDIM> src, std::array<Index, NDIM> dst,
        std::array<Index, NDIM> shape)
{
    // Location of copy area in source and target buffers and their shapes
    std::vector<Index> src_start(NDIM), dst_start(NDIM),
        copy_shape(shape.cbegin(), shape.cend()),
        src_shape(NDIM), dst_shape(NDIM);
    Index src_nelems = 1, dst_nelems = 1, copy_nelems = 1;
    for(Index i = 0; i < NDIM; ++i)
    {
        // Offset from the beginning
        if(src[i] >= 0)
        {
            src_start[i] = src[i];
            src_shape[i] = shape[i] + src[i];
        }
        // Offset from the end
        else
        {
            src_start[i] = 0;
            src_shape[i] = shape[i] - src[i] - 1;
        }
        src_nelems *= src_shape[i];
        // Offset from the beginning
        if(dst[i] >= 0)
        {
            dst_start[i] = dst[i];
            dst_shape[i] = shape[i] + dst[i];
        }
        // Offset from the end
        else
        {
            dst_start[i] = 0;
            dst_shape[i] = shape[i] - dst[i] - 1;
        }
        dst_nelems *= dst_shape[i];
        // Total number of elements to be copied
        copy_nelems *= shape[i];
    }
    // Strides
    std::vector<Index> src_stride(NDIM), dst_stride(NDIM);
    src_stride[0] = 1;
    dst_stride[0] = 1;
    for(Index i = 1; i < NDIM; ++i)
    {
        src_stride[i] = src_stride[i-1] * src_shape[i-1];
        dst_stride[i] = dst_stride[i-1] * dst_shape[i-1];
    }
    // Init test input. Set non-copied values to 1 and copied values to 2 in
    // the source and set all the elements to 3 in the destination.
    std::vector<T> src_data(src_nelems, T{1}), dst_data(dst_nelems, T{3}),
        dst2_data(dst_data);
    std::vector<Index> src_index(src_start);
    for(Index i = 0; i < copy_nelems; ++i)
    {
        // Get offset of the current element to copy
        Index src_offset = 0;
        for(Index j = 0; j < NDIM; ++j)
        {
            src_offset += src_stride[j] * src_index[j];
        }
        // Set its value to 2
        src_data[src_offset] = T{2};
        // Do nothing if it was the last element to copy
        if(i == copy_nelems-1)
        {
            break;
        }
        // Get index of the next element to copy
        ++src_index[0];
        Index j = 0;
        while(src_index[j] == src_start[j]+shape[j])
        {
            src_index[j] = src_start[j];
            ++j;
            ++src_index[j];
        }
    }
    std::vector<T> src2_data(src_data);
    // Check low-level kernel
    std::vector<Index> tmp_index(2*NDIM);
    copy<T>(NDIM, &src_start[0], &src_stride[0], &copy_shape[0],
            &src_data[0], &dst_start[0], &dst_stride[0], &dst_data[0],
            &tmp_index[0]);
    // Check source is unchanged
    for(Index i = 0; i < src_nelems; ++i)
    {
        if(src_data[i] != src2_data[i])
        {
            throw std::runtime_error("Wrong src_data value");
        }
    }
    // Check destination
    std::vector<Index> dst_index(NDIM);
    for(Index i = 0; i < dst_nelems; ++i)
    {
        // Find out if current element was overwritten or not
        bool copied = true;
        for(Index j = 0; j < NDIM; ++j)
        {
            if(dst_index[j] < dst_start[j]
                    or dst_index[j] >= dst_start[j]+shape[j])
            {
                copied = false;
                break;
            }
        }
        // Check if it was overwritten
        if(copied)
        {
            if(dst_data[i] != T{2})
            {
                throw std::runtime_error("Wrong dst_data value");
            }
        }
        // Check if it was not overwritten
        else
        {
            if(dst_data[i] != T{3})
            {
                throw std::runtime_error("Wrong dst_data value");
            }
        }
        // Get out if it was last element of destination buffer
        if(i == dst_nelems-1)
        {
            break;
        }
        // Get index of the next element
        ++dst_index[0];
        Index j = 0;
        while(dst_index[j] == dst_shape[j])
        {
            dst_index[j] = 0;
            ++j;
            ++dst_index[j];
        }
    }
}

// Run multiple tests for a given precision
template<typename T>
void validate_many()
{
    validate<T, 1>({0}, {0}, {2});
    validate<T, 1>({2}, {0}, {2});
    validate<T, 1>({-2}, {0}, {2});
    validate<T, 1>({0}, {2}, {2});
    validate<T, 1>({0}, {-2}, {2});
    validate<T, 3>({0, 0, 0}, {0, 0, 0}, {2, 3, 4});
    validate<T, 3>({1, 0, 0}, {0, 0, 0}, {2, 3, 4});
    validate<T, 3>({1, 0, 0}, {-1, 0, 0}, {2, 3, 4});
    validate<T, 3>({0, 1, -1}, {3, -4, 5}, {2, 3, 4});
}

int main(int argc, char **argv)
{
    validate_many<fp32_t>();
    validate_many<fp64_t>();
    return 0;
}

