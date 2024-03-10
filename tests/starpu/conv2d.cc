/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/starpu/conv2d.cc
 * StarPU wrappers for 2D-Convolution between 2 matrices
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-05-02
 * */

#include "nntile/starpu/conv2d.hh"
#include "nntile/kernel/conv2d.hh"
#include "../testing.hh"

#include <vector>
#include <stdexcept>
#include <iostream>

using namespace nntile;
using namespace nntile::starpu;

template<typename T>
void validate_cpu(Index nx, Index ny, Index mx, Index my)
{
    // Init all the data
    std::vector<T> src(nx*ny);
    for(Index i = 0; i < nx*ny; ++i)
    {
        src[i] = 1;
    }
    std::vector<T> kernel(mx*my);
    for(Index i = 0; i < mx*my; ++i)
    {
        kernel[i] = 1;
    }
    std::vector<T> dst((nx+mx-1)*(ny+my-1));
    for(Index i = 0; i < (nx+mx-1)*(ny+my-1); ++i)
    {
        dst[i] = 0;
    }
    // Create copies of destination
    std::vector<T> dst2(dst);
    // Launch low-level kernel
    std::cout << "Run kernel::conv2d::cpu<T>\n";
    kernel::conv2d::cpu<T>(nx, ny, &src[0], mx, my, &kernel[0], &dst[0]);
    // Check by actually submitting a task
    VariableHandle src_handle(&src[0], sizeof(T)*nx*ny, STARPU_R),
        kernel_handle(&kernel[0], sizeof(T)*mx*my, STARPU_R),
		dst2_handle(&dst2[0], sizeof(T)*(nx+mx-1)*(ny+my-1), STARPU_W);
    conv2d::restrict_where(STARPU_CPU);
    std::cout << "Run starpu::conv2d::submit<T> restricted to CPU\n";
    conv2d::submit<T>(nx, ny, src_handle, mx, my, kernel_handle, dst2_handle);
    starpu_task_wait_for_all();
    dst2_handle.unregister();
    // Check result
    for(Index i = 0; i < (nx+mx-1)*(ny+my-1); ++i)
    {
        TEST_ASSERT(dst[i] == dst2[i]);
    }
    std::cout << "OK: starpu::conv2d::submit<T> restricted to CPU\n";
}

int main(int argc, char **argv)
{
    // Init StarPU for testing
    Config starpu(1, 1, 0);
    // Init codelet
    conv2d::init();
    // Launch all tests
    validate_cpu<fp32_t>(3, 5, 7, 9);
    validate_cpu<fp64_t>(3, 5, 7, 9);

    return 0;
}

