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

#ifdef NNTILE_USE_CUDA
#   include <cuda_runtime.h>
#endif // NNTILE_USE_CUDA
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

#ifdef NNTILE_USE_CUDA
template<typename T>
void validate_cuda(Index nx, Index ny, Index mx, Index my)
{
    // Get a StarPU CUDA worker (to perform computations on the same device)
    int cuda_worker_id = starpu_worker_get_by_type(STARPU_CUDA_WORKER, 0);
    // Choose worker CUDA device
    int dev_id = starpu_worker_get_devid(cuda_worker_id);
    cudaError_t cuda_err = cudaSetDevice(dev_id);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Create CUDA stream
    cudaStream_t stream;
    cuda_err = cudaStreamCreate(&stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
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
    T *dev_src, *dev_kernel, *dev_dst;
    cuda_err = cudaMalloc(&dev_src, sizeof(T)*nx*ny);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMalloc(&dev_kernel, sizeof(T)*mx*my);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMalloc(&dev_dst, sizeof(T)*(nx+mx-1)*(ny+my-1));
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_src, &src[0], sizeof(T)*nx*ny,
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_kernel, &kernel[0], sizeof(T)*mx*my,
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    std::cout << "Run kernel::conv2d::cuda<T>\n";
    kernel::conv2d::cuda<T>(stream, nx, ny, dev_src, mx, my, dev_kernel, dev_dst);
    // Wait for result and destroy stream
    cuda_err = cudaStreamSynchronize(stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaStreamDestroy(stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Copy result back to CPU
    cuda_err = cudaMemcpy(&dst[0], dev_dst, sizeof(T)*(nx+mx-1)*(ny+my-1),
            cudaMemcpyDeviceToHost);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Deallocate CUDA memory
    cuda_err = cudaFree(dev_src);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaFree(dev_kernel);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaFree(dev_dst);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Check by actually submitting a task
    VariableHandle src_handle(&src[0], sizeof(T)*nx*ny, STARPU_R),
		kernel_handle(&kernel[0], sizeof(T)*mx*my, STARPU_R),
        dst2_handle(&dst2[0], sizeof(T)*(nx+mx-1)*(ny+my-1), STARPU_W);
    conv2d::restrict_where(STARPU_CUDA);
    std::cout << "Run starpu::conv2d::submit<T> restricted to CUDA\n";
    conv2d::submit<T>(nx, ny, src_handle, mx, my, kernel_handle, dst2_handle);
    starpu_task_wait_for_all();
    dst2_handle.unregister();
    // Check result
    for(Index i = 0; i < (nx+mx-1)*(ny+my-1); ++i)
    {
        TEST_ASSERT(dst[i] == dst2[i]);
    }
    std::cout << "OK: starpu::conv2d::submit<T> restricted to CUDA\n";
}
#endif // NNTILE_USE_CUDA

int main(int argc, char **argv)
{
    // Init StarPU for testing
    Config starpu(1, 1, 0);
    // Init codelet
    conv2d::init();
    // Launch all tests
    validate_cpu<fp32_t>(3, 5, 7, 9);
    validate_cpu<fp64_t>(3, 5, 7, 9);

#ifdef NNTILE_USE_CUDA
    validate_cuda<fp32_t>(3, 5, 7, 9);
    validate_cuda<fp64_t>(3, 5, 7, 9);
#endif // NNTILE_USE_CUDA
    return 0;
}

