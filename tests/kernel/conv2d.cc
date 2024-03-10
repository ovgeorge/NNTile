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
#include <vector>
#include <stdexcept>
#include <limits>
#include <iostream>

using namespace nntile;
using namespace nntile::kernel::conv2d;

#ifdef NNTILE_USE_CUDA
template<typename T>
void run_cuda(Index nx, Index ny, const std::vector<T> &src,
        Index mx, Index my, const std::vector<T> &kernel, std::vector<T> &dst)
{
    // Copy to device
    T *dev_src, *dev_kernel, *dev_dst;
    cudaError_t cuda_err = cudaMalloc(&dev_src, sizeof(T)*nx*ny);
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
    cuda_err = cudaMemcpy(dev_dst, &dst[0], sizeof(T)*(nx+mx-1)*(ny+my-1),
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Init stream
    cudaStream_t stream;
    cuda_err = cudaStreamCreate(&stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Launch low-level CUDA kernel
    cuda<T>(stream, nx, ny, dev_src, mx, my, dev_kernel, dev_dst);
    cuda_err = cudaStreamSynchronize(stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Copy result and deallocate device memory
    cuda_err = cudaMemcpy(&dst[0], dev_dst, sizeof(T)*(nx+mx-1)*(ny+my-1),
            cudaMemcpyDeviceToHost);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaFree(dev_src);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaFree(dev_kernel);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaFree(dev_dst);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaStreamDestroy(stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
}
#endif // NNTILE_USE_CUDA

// Templated validation
template<typename T>
void validate(Index nx, Index ny, Index px, Index py, Index mx, Index my, Index qx, Index qy)
{
    constexpr T eps = std::numeric_limits<T>::epsilon();
    // Init test input
    std::vector<T> src(nx*ny), kernel(mx*my), dst((nx+mx-1)*(ny+my-1));
    for(Index i0 = 0; i0 < ny+my-1; ++i0)
    {
        for(Index i1 = 0; i1 < nx+mx-1; ++i1)
        {
			dst[i1*(ny+my-1)+i0] = -1;
        }
    }
    for(Index i0 = 0; i0 < ny; ++i0)
    {
        for(Index i1 = 0; i1 < nx; ++i1)
        {
            src[i1*ny+i0] = (i0 == py && i1 == px) ? 1 : 0;
        }
    }
    for(Index i0 = 0; i0 < my; ++i0)
    {
        for(Index i1 = 0; i1 < mx; ++i1)
        {
            kernel[i1*my+i0] = (i0 == qy && i1 == qx) ? 1 : 0;
        }
    }
    // Save original dst
    std::vector<T> dst_save(dst);
    // Check low-level CPU kernel
    std::cout << "Run kernel::conv2d::cpu<T>\n";
    cpu<T>(nx, ny, &src[0], mx, my, &kernel[0], &dst[0]);
    for(Index i0 = 0; i0 < ny + my - 1; ++i0)
    {
        for(Index i1 = 0; i1 < nx + mx - 1; ++i1)
        {
			T ref = (i0 == py+qy && i1 == px+qx) ? 1 : 0;
            T val = dst[i1*(ny+my-1)+i0];
            TEST_ASSERT(std::abs(val-ref) <= 10*eps);
        }
    }
    std::cout << "OK: kernel::conv2d::cpu<T>\n";
}

void validate_all(Index nx, Index ny, Index px, Index py, Index mx, Index my, Index qx, Index qy){
	validate<fp32_t>(nx, ny, px, py, mx, my, qx, qy);
	validate<fp32_t>(mx, my, qx, qy, nx, ny, px, py);
	validate<fp64_t>(nx, ny, px, py, mx, my, qx, qy);
	validate<fp64_t>(mx, my, qx, qy, nx, ny, px, py);
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

