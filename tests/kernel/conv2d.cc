/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/conv2d.cc
 * 2D-Convolution between 2 matrices
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-05-02
 * */

#include "nntile/kernel/conv2d.hh"
#include "../testing.hh"
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace nntile;
using namespace nntile::kernel::conv2d;

// Templated validation
template <typename T>
void validate(Index offset_x, Index offset_y, Index nx, Index ny, Index px,
              Index py, Index mx, Index my, Index qx, Index qy)
{
    constexpr T eps = std::numeric_limits<T>::epsilon();
    // Init test input
    std::vector<T> src(nx * ny), kernel(mx * my),
        dst((nx + mx - 1) * (ny + my - 1));
    for(Index i0 = 0; i0 < ny + my - 1; ++i0)
    {
        for(Index i1 = 0; i1 < nx + mx - 1; ++i1)
        {
            dst[i1 * (ny + my - 1) + i0] = -2;
        }
    }
    for(Index i0 = 0; i0 < ny; ++i0)
    {
        for(Index i1 = 0; i1 < nx; ++i1)
        {
            src[i1 * ny + i0] = (i0 == py && i1 == px) ? 1 : 0;
        }
    }
    for(Index i0 = 0; i0 < my; ++i0)
    {
        for(Index i1 = 0; i1 < mx; ++i1)
        {
            kernel[i1 * my + i0] = (i0 == qy && i1 == qx) ? 1 : 0;
        }
    }
    // Save original dst
    std::vector<T> dst_save(dst);
    // Check low-level CPU kernel
    std::cout << "Run kernel::conv2d::cpu<T>\n";
    cpu<T>(offset_x, offset_y, 1, nx, ny, &src[0], mx, my, &kernel[0],
           nx + mx - 1, ny + my - 1, &dst[0]);
    for(Index i0 = 0; i0 < ny + my - 1; ++i0)
    {
        for(Index i1 = 0; i1 < nx + mx - 1; ++i1)
        {
            T ref = (i0 == py + qy - offset_y && i1 == px + qx - offset_x) ? -1
                                                                           : -2;
            T val = dst[i1 * (ny + my - 1) + i0];
            TEST_ASSERT(std::abs(val - ref) <= 10 * eps);
        }
    }
    std::cout << "OK: kernel::conv2d::cpu<T>\n";
}

void validate_all(Index nx, Index ny, Index px, Index py, Index mx, Index my,
                  Index qx, Index qy)
{
    for(Index offset_x = -1; offset_x <= 1; offset_x++)
    {
        for(Index offset_y = -1; offset_y <= 1; offset_y++)
        {
            validate<fp32_t>(offset_x, offset_y, nx, ny, px, py, mx, my, qx,
                             qy);
            validate<fp32_t>(offset_x, offset_y, mx, my, qx, qy, nx, ny, px,
                             py);
            validate<fp64_t>(offset_x, offset_y, nx, ny, px, py, mx, my, qx,
                             qy);
            validate<fp64_t>(offset_x, offset_y, mx, my, qx, qy, nx, ny, px,
                             py);
        }
    }
}

int main(int argc, char **argv)
{
    validate_all(4, 4, 0, 0, 1, 1, 0, 0);
    validate_all(5, 7, 0, 0, 1, 1, 0, 0);
    validate_all(4, 4, 0, 0, 3, 3, 0, 0);
    validate_all(5, 7, 0, 0, 4, 9, 0, 0);
    validate_all(5, 7, 1, 4, 4, 9, 2, 5);

    return 0;
}
