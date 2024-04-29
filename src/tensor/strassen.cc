/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/strassen.cc
 * Strassen operation for Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-09-15
 * */

#include "nntile/tensor/strassen.hh"
#include "nntile/starpu/add2d.hh"
#include "nntile/starpu/gemm.hh"
#include "nntile/starpu/strassen.hh"

using nntile::starpu::Handle;

namespace nntile
{
namespace tensor
{

template <typename T, int Ij1, int iJ1, int Ij2, int iJ2, int weight>
void getQuarters(Handle A, Handle quarter, Index _M, Index _K, Index ld,
                 TransOp transA)
{
    Index M = _M + (_M % 2);
    Index K = _K + (_K % 2);

    // dst = 0
    Index offset = 0;
    starpu::add2d::submit<T>(M / 2, K / 2, 0, A, 0, ld, offset, quarter, 0,
                             M / 2);
    starpu_task_wait_for_all();
    // dst += first quarter
    offset = (M / 2 * (Ij1 - 1)) + (K / 2 * (iJ1 - 1)) * ld;
    starpu::add2d::submit<T>(M / 2, K / 2, 1, A, offset, ld, 1, quarter, 0,
                             M / 2);
    starpu_task_wait_for_all();
    // dst += second quarter
    offset = (M / 2 * (Ij2 - 1)) + (K / 2 * (iJ2 - 1)) * ld;
    starpu::add2d::submit<T>(M / 2, K / 2, weight, A, offset, ld, 1, quarter, 0,
                             M / 2);
    starpu_task_wait_for_all();
}

template <typename T, int Ij1, int iJ1, int Ij2, int iJ2>
void getQuarterSum(Handle A, Handle quarter, Index _M, Index _K, Index ld,
                   TransOp transA)
{
    getQuarters<T, Ij1, iJ1, Ij2, iJ2, 1>(A, quarter, _M, _K, ld, transA);
}

template <typename T, int Ij1, int iJ1, int Ij2, int iJ2>
void getQuarterSub(Handle A, Handle quarter, Index _M, Index _K, Index ld,
                   TransOp transA)
{
    getQuarters<T, Ij1, iJ1, Ij2, iJ2, -1>(A, quarter, _M, _K, ld, transA);
}

template <typename T, int Ij, int iJ>
void getQuarter(Handle A, Handle quarter, Index _M, Index _K, Index ld,
                TransOp transA)
{
    getQuarters<T, Ij, iJ, Ij, iJ, 0>(A, quarter, _M, _K, ld, transA);
}

template <typename T, typename T_scal>
void starpu__strassen__submit(const TransOp &transA, const TransOp &transB,
                              Index M, Index N, Index K, Index batch,
                              T_scal alpha, Handle A, Handle B, T_scal beta,
                              Handle C, int redux = 0)
{
    // TODO: use traits to get ld values
    Index ldA = M, ldB = K, ldC = M;

    T *_Aarr_data[7];
    size_t Aarr_size = ((M + 1) / 2) * ((K + 1) / 2);
    for(int i = 0; i < 7; i++)
        _Aarr_data[i] = new T[Aarr_size];
    starpu::VariableHandle _Aarr[7] = {
        starpu::VariableHandle(_Aarr_data[0], Aarr_size * sizeof(T), STARPU_RW),
        starpu::VariableHandle(_Aarr_data[1], Aarr_size * sizeof(T), STARPU_RW),
        starpu::VariableHandle(_Aarr_data[2], Aarr_size * sizeof(T), STARPU_RW),
        starpu::VariableHandle(_Aarr_data[3], Aarr_size * sizeof(T), STARPU_RW),
        starpu::VariableHandle(_Aarr_data[4], Aarr_size * sizeof(T), STARPU_RW),
        starpu::VariableHandle(_Aarr_data[5], Aarr_size * sizeof(T), STARPU_RW),
        starpu::VariableHandle(_Aarr_data[6], Aarr_size * sizeof(T), STARPU_RW),
    };
    starpu::VariableHandle *Aarr = &(_Aarr[0]) - 1;

    T *_Barr_data[7];
    size_t Barr_size = ((N + 1) / 2) * ((K + 1) / 2);
    for(int i = 0; i < 7; i++)
        _Barr_data[i] = new T[Barr_size];
    starpu::VariableHandle _Barr[7] = {
        starpu::VariableHandle(_Barr_data[0], Barr_size * sizeof(T), STARPU_RW),
        starpu::VariableHandle(_Barr_data[1], Barr_size * sizeof(T), STARPU_RW),
        starpu::VariableHandle(_Barr_data[2], Barr_size * sizeof(T), STARPU_RW),
        starpu::VariableHandle(_Barr_data[3], Barr_size * sizeof(T), STARPU_RW),
        starpu::VariableHandle(_Barr_data[4], Barr_size * sizeof(T), STARPU_RW),
        starpu::VariableHandle(_Barr_data[5], Barr_size * sizeof(T), STARPU_RW),
        starpu::VariableHandle(_Barr_data[6], Barr_size * sizeof(T),
                               STARPU_RW)};
    starpu::VariableHandle *Barr = &(_Barr[0]) - 1;

    T *_Marr_data[7];
    size_t Marr_size = ((M + 1) / 2) * ((N + 1) / 2);
    for(int i = 0; i < 7; i++)
        _Marr_data[i] = new T[Marr_size];
    starpu::VariableHandle _Marr[7] = {
        starpu::VariableHandle(_Marr_data[0], Marr_size * sizeof(T), STARPU_RW),
        starpu::VariableHandle(_Marr_data[1], Marr_size * sizeof(T), STARPU_RW),
        starpu::VariableHandle(_Marr_data[2], Marr_size * sizeof(T), STARPU_RW),
        starpu::VariableHandle(_Marr_data[3], Marr_size * sizeof(T), STARPU_RW),
        starpu::VariableHandle(_Marr_data[4], Marr_size * sizeof(T), STARPU_RW),
        starpu::VariableHandle(_Marr_data[5], Marr_size * sizeof(T), STARPU_RW),
        starpu::VariableHandle(_Marr_data[6], Marr_size * sizeof(T),
                               STARPU_RW)};
    starpu::VariableHandle *Marr = &(_Marr[0]) - 1;

    TransOp notrans(TransOp::NoTrans);
    T zero = 0.0, one = 1.0;
    getQuarterSum<T, 1, 1, 2, 2>(A, _Aarr[0], M, K, ldA, transA);
    getQuarterSum<T, 1, 1, 2, 2>(B, Barr[1], K, N, ldB, transB);
    starpu::gemm::submit<T, T>(notrans, notrans, (M + 1) / 2, (N + 1) / 2,
                               (K + 1) / 2, /*batches*/ 1, one, Aarr[1],
                               Barr[1], zero, Marr[1]);
    starpu_task_wait_for_all();

    getQuarterSum<T, 2, 1, 2, 2>(A, Aarr[2], M, K, ldA, transA);
    getQuarter<T, 1, 1>(B, Barr[2], K, N, ldB, transB);
    starpu::gemm::submit<T, T>(notrans, notrans, (M + 1) / 2, (N + 1) / 2,
                               (K + 1) / 2, /*batches*/ 1, one, Aarr[2],
                               Barr[2], zero, Marr[2]);
    starpu_task_wait_for_all();

    getQuarter<T, 1, 1>(A, Aarr[3], M, K, ldA, transA);
    getQuarterSub<T, 1, 2, 2, 2>(B, Barr[3], K, N, ldB, transB);
    starpu::gemm::submit<T, T>(notrans, notrans, (M + 1) / 2, (N + 1) / 2,
                               (K + 1) / 2, /*batches*/ 1, one, Aarr[3],
                               Barr[3], zero, Marr[3]);
    starpu_task_wait_for_all();

    getQuarter<T, 2, 2>(A, Aarr[4], M, K, ldA, transA);
    getQuarterSub<T, 2, 1, 1, 1>(B, Barr[4], K, N, ldB, transB);
    starpu::gemm::submit<T, T>(notrans, notrans, (M + 1) / 2, (N + 1) / 2,
                               (K + 1) / 2, /*batches*/ 1, one, Aarr[4],
                               Barr[4], zero, Marr[4]);
    starpu_task_wait_for_all();

    getQuarterSum<T, 1, 1, 1, 2>(A, Aarr[5], M, K, ldA, transA);
    getQuarter<T, 2, 2>(B, Barr[5], K, N, ldB, transB);
    starpu::gemm::submit<T, T>(notrans, notrans, (M + 1) / 2, (N + 1) / 2,
                               (K + 1) / 2, /*batches*/ 1, one, Aarr[5],
                               Barr[5], zero, Marr[5]);
    starpu_task_wait_for_all();

    getQuarterSub<T, 2, 1, 1, 1>(A, Aarr[6], M, K, ldA, transA);
    getQuarterSum<T, 1, 1, 1, 2>(B, Barr[6], K, N, ldB, transB);
    starpu::gemm::submit<T, T>(notrans, notrans, (M + 1) / 2, (N + 1) / 2,
                               (K + 1) / 2, /*batches*/ 1, one, Aarr[6],
                               Barr[6], zero, Marr[6]);
    starpu_task_wait_for_all();

    getQuarterSub<T, 1, 2, 2, 2>(A, Aarr[7], M, K, ldA, transA);
    getQuarterSum<T, 2, 1, 2, 2>(B, Barr[7], K, N, ldB, transB);
    starpu::gemm::submit<T, T>(notrans, notrans, (M + 1) / 2, (N + 1) / 2,
                               (K + 1) / 2, /*batches*/ 1, one, Aarr[7],
                               Barr[7], zero, Marr[7]);
    starpu_task_wait_for_all();

    Index offset = 0;
    starpu::add2d::submit<T>((M + 1) / 2, (N + 1) / 2, alpha, Marr[1], 0,
                             (M + 1) / 2, beta, C, offset, ldC);
    starpu_task_wait_for_all();
    starpu::add2d::submit<T>((M + 1) / 2, (N + 1) / 2, alpha, Marr[4], 0,
                             (M + 1) / 2, 1, C, offset, ldC);
    starpu_task_wait_for_all();
    starpu::add2d::submit<T>((M + 1) / 2, (N + 1) / 2, -alpha, Marr[5], 0,
                             (M + 1) / 2, 1, C, offset, ldC);
    starpu_task_wait_for_all();
    starpu::add2d::submit<T>((M + 1) / 2, (N + 1) / 2, alpha, Marr[7], 0,
                             (M + 1) / 2, 1, C, offset, ldC);
    starpu_task_wait_for_all();

    offset = (N + 1) / 2 * ldC;
    starpu::add2d::submit<T>((M + 1) / 2, (N + 1) / 2, alpha, Marr[3], 0,
                             (M + 1) / 2, beta, C, offset, ldC);
    starpu_task_wait_for_all();
    starpu::add2d::submit<T>((M + 1) / 2, (N + 1) / 2, alpha, Marr[5], 0,
                             (M + 1) / 2, 1, C, offset, ldC);
    starpu_task_wait_for_all();

    offset = (M + 1) / 2;
    starpu::add2d::submit<T>((M + 1) / 2, (N + 1) / 2, alpha, Marr[2], 0,
                             (M + 1) / 2, beta, C, offset, ldC);
    starpu_task_wait_for_all();
    starpu::add2d::submit<T>((M + 1) / 2, (N + 1) / 2, alpha, Marr[4], 0,
                             (M + 1) / 2, 1, C, offset, ldC);
    starpu_task_wait_for_all();

    offset = (M + 1) / 2 + (N + 1) / 2 * ldC;
    starpu::add2d::submit<T>((M + 1) / 2, (N + 1) / 2, alpha, Marr[1], 0,
                             (M + 1) / 2, beta, C, offset, ldC);
    starpu_task_wait_for_all();
    starpu::add2d::submit<T>((M + 1) / 2, (N + 1) / 2, -alpha, Marr[2], 0,
                             (M + 1) / 2, 1, C, offset, ldC);
    starpu_task_wait_for_all();
    starpu::add2d::submit<T>((M + 1) / 2, (N + 1) / 2, alpha, Marr[3], 0,
                             (M + 1) / 2, 1, C, offset, ldC);
    starpu_task_wait_for_all();
    starpu::add2d::submit<T>((M + 1) / 2, (N + 1) / 2, alpha, Marr[6], 0,
                             (M + 1) / 2, 1, C, offset, ldC);
    starpu_task_wait_for_all();

    // TODO: properly wait for tasks instead of relying on
    // starpu_task_wait_for_all()
    starpu_task_wait_for_all();
}

//! Check if dimensionalities of tensors match strassen
static inline void strassen_check_ndim(const TensorTraits &A,
        const TensorTraits &B, const TensorTraits &C, Index ndim,
        Index batch_ndim)
{
    // Check if ndim is negative since it will be converted to Index
    if(ndim < 0)
    {
        throw std::runtime_error("ndim < 0");
    }
    if(batch_ndim < 0)
    {
        throw std::runtime_error("batch_ndim < 0");
    }
    if(A.ndim < batch_ndim+ndim)
    {
        throw std::runtime_error("A.ndim < batch_ndim+ndim");
    }
    if(B.ndim < batch_ndim+ndim)
    {
        throw std::runtime_error("B.ndim < batch_ndim+ndim");
    }
    if(A.ndim + B.ndim != C.ndim + 2*ndim + batch_ndim)
    {
        throw std::runtime_error("A.ndim + B.ndim != C.ndim + 2*ndim + "
                "batch_ndim");
    }
}

//! Check batch shapes
static inline void strassen_check_batch(const TensorTraits &A,
        const TensorTraits &B, const TensorTraits &C, Index batch_ndim)
{
    for(Index i = 0; i < batch_ndim; ++i)
    {
        if(A.shape[A.ndim-i-1] != B.shape[B.ndim-i-1])
        {
            throw std::runtime_error("A.shape[A.ndim-batch_ndim:A.ndim] != "
                    "B.shape[B.ndim-batch_ndim:B.ndim]");
        }
        if(A.basetile_shape[A.ndim-i-1] != B.basetile_shape[B.ndim-i-1])
        {
            throw std::runtime_error("A.basetile_shape[A.ndim-batch_ndim:"
                    "A.ndim] != B.basetile_shape[B.ndim-batch_ndim:B.ndim]");
        }
        if(A.shape[A.ndim-i-1] != C.shape[C.ndim-i-1])
        {
            throw std::runtime_error("A.shape[A.ndim-batch_ndim:A.ndim] != "
                    "C.shape[C.ndim-batch_ndim:C.ndim]");
        }
        if(A.basetile_shape[A.ndim-i-1] != C.basetile_shape[C.ndim-i-1])
        {
            throw std::runtime_error("A.basetile_shape[A.ndim-batch_ndim:"
                    "A.ndim] != C.basetile_shape[C.ndim-batch_ndim:C.ndim]");
        }
    }
}

//! Check if shapes of tensors A and B match strassen
static inline void strassen_check_A_B(const TensorTraits &A,
        const TensorTraits &B, Index ndim, Index batch_ndim)
{
    for(Index i = 0; i < ndim; ++i)
    {
        if(A.shape[A.ndim-batch_ndim-ndim+i] != B.shape[i])
        {
            throw std::runtime_error("A.shape[A.ndim-batch_ndim-ndim:"
                    "A.ndim-batch_ndim] != B.shape[0:ndim]");
        }
        if(A.basetile_shape[A.ndim-batch_ndim-ndim+i] != B.basetile_shape[i])
        {
            throw std::runtime_error("A.basetile_shape[A.ndim-batch_ndim-ndim:"
                    "A.ndim-batch_ndim] != B.basetile_shape[0:ndim]");
        }
    }
}

//! Check if shapes of tensors A^T and B match strassen
static inline void strassen_check_AT_B(const TensorTraits &A,
        const TensorTraits &B, Index ndim)
{
    for(Index i = 0; i < ndim; ++i)
    {
        if(A.shape[i] != B.shape[i])
        {
            throw std::runtime_error("A.shape[0:ndim] != B.shape[0:ndim]");
        }
        if(A.basetile_shape[i] != B.basetile_shape[i])
        {
            throw std::runtime_error("A.basetile_shape[0:ndim] != "
                    "B.basetile_shape[0:ndim]");
        }
    }
}

//! Check if shapes of tensors A and B^T match strassen
static inline void strassen_check_A_BT(const TensorTraits &A,
        const TensorTraits &B, Index ndim, Index batch_ndim)
{
    for(Index i = 0; i < ndim; ++i)
    {
        if(A.shape[A.ndim-batch_ndim-ndim+i]
                != B.shape[B.ndim-batch_ndim-ndim+i])
        {
            throw std::runtime_error("A.shape[A.ndim-batch_ndim-ndim:"
                    "A.ndim-batch_ndim] != B.shape[B.ndim-batch_ndim-ndim:"
                    "B.ndim-batch_ndim]");
        }
        if(A.basetile_shape[A.ndim-batch_ndim-ndim+i]
                != B.basetile_shape[B.ndim-batch_ndim-ndim+i])
        {
            throw std::runtime_error("A.basetile_shape[A.ndim-batch_ndim-ndim:"
                    "A.ndim-batch_ndim] != B.shape[B.ndim-batch_ndim-ndim:"
                    "B.ndim-batch_ndim]");
        }
    }
}

//! Check if shapes of tensors A^T and B^T match strassen
static inline void strassen_check_AT_BT(const TensorTraits &A,
        const TensorTraits &B, Index ndim, Index batch_ndim)
{
    for(Index i = 0; i < ndim; ++i)
    {
        if(A.shape[i] != B.shape[B.ndim-batch_ndim-ndim+i])
        {
            throw std::runtime_error("A.shape[0:ndim] != "
                    "B.shape[B.ndim-batch_ndim-ndim:B.ndim-batch_ndim]");
        }
        if(A.basetile_shape[i] != B.basetile_shape[B.ndim-batch_ndim-ndim+i])
        {
            throw std::runtime_error("A.basetile_shape[0:ndim] != "
                    "B.basetile_shape[B.ndim-batch_ndim-ndim:"
                    "B.ndim-batch_ndim]");
        }
    }
}

//! Check if shapes of tensors op(A) and op(B) match strassen
static inline void strassen_check_opA_opB(const TransOp &transA,
        const TensorTraits &A, const TransOp &transB, const TensorTraits &B,
        Index ndim, Index batch_ndim)
{
    switch(transB.value)
    {
        case TransOp::NoTrans:
            switch(transA.value)
            {
                case TransOp::NoTrans:
                    strassen_check_A_B(A, B, ndim, batch_ndim);
                    break;
                case TransOp::Trans:
                    strassen_check_AT_B(A, B, ndim);
                    break;
                default:
                    throw std::runtime_error("Wrong value of transA");
            }
            break;
        case TransOp::Trans:
            switch(transA.value)
            {
                case TransOp::NoTrans:
                    strassen_check_A_BT(A, B, ndim, batch_ndim);
                    break;
                case TransOp::Trans:
                    strassen_check_AT_BT(A, B, ndim, batch_ndim);
                    break;
                default:
                    throw std::runtime_error("Wrong value of transA");
            }
            break;
        default:
            throw std::runtime_error("Wrong value of transB");
    }
}

//! Check if shapes of tensors A and C match strassen
static inline void strassen_check_A_C(const TensorTraits &A,
        const TensorTraits &C, Index ndim, Index batch_ndim)
{
    for(Index i = 0; i < A.ndim-batch_ndim-ndim; ++i)
    {
        if(A.shape[i] != C.shape[i])
        {
            throw std::runtime_error("A.shape[0:A.ndim-batch_ndim-ndim] != "
                    "C.shape[0:A.ndim-batch_ndim-ndim]");
        }
        if(A.basetile_shape[i] != C.basetile_shape[i])
        {
            throw std::runtime_error("A.basetile_shape[0:"
                    "A.ndim-batch_ndim-ndim] != "
                    "C.basetile_shape[0:A.ndim-batch_ndim-ndim]");
        }
    }
}

//! Check if shapes of tensors A^T and C match strassen
static inline void strassen_check_AT_C(const TensorTraits &A,
        const TensorTraits &C, Index ndim, Index batch_ndim)
{
    for(Index i = ndim; i < A.ndim-batch_ndim; ++i)
    {
        if(A.shape[i] != C.shape[i-ndim])
        {
            throw std::runtime_error("A.shape[ndim:A.ndim-batch_ndim] != "
                    "C.shape[0:A.ndim-batch_ndim-ndim]");
        }
        if(A.basetile_shape[i] != C.basetile_shape[i-ndim])
        {
            throw std::runtime_error("A.basetile_shape[ndim:"
                    "A.ndim-batch_ndim] != "
                    "C.basetile_shape[0:A.ndim-batch_ndim-ndim]");
        }
    }
}

//! Check if shapes of tensors op(A) and C match strassen
static inline void strassen_check_opA_C(const TransOp &transA,
        const TensorTraits &A, const TensorTraits &C, Index ndim,
        Index batch_ndim)
{
    switch(transA.value)
    {
        case TransOp::NoTrans:
            strassen_check_A_C(A, C, ndim, batch_ndim);
            break;
        case TransOp::Trans:
            strassen_check_AT_C(A, C, ndim, batch_ndim);
            break;
        // This parameter was already checked in strassen_check_opA_opB
    }
}

//! Check if shapes of tensors B and C match strassen
static inline void strassen_check_B_C(const TensorTraits &B,
        const TensorTraits &C, Index ndim, Index batch_ndim)
{
    for(Index i = ndim; i < B.ndim-batch_ndim; ++i)
    {
        if(B.shape[i] != C.shape[C.ndim-B.ndim+i])
        {
            throw std::runtime_error("B.shape[ndim:B.ndim-batch_ndim] != "
                    "C.shape[C.ndim-B.ndim+ndim:C.ndim-batch_ndim]");
        }
        if(B.basetile_shape[i] != C.basetile_shape[C.ndim-B.ndim+i])
        {
            throw std::runtime_error("B.basetile_shape[ndim:"
                    "B.ndim-batch_ndim] != "
                    "C.basetile_shape[C.ndim-B.ndim+ndim:C.ndim-batch_ndim]");
        }
    }
}

//! Check if shapes of tensors B^T and C match strassen
static inline void strassen_check_BT_C(const TensorTraits &B,
        const TensorTraits &C, Index ndim, Index batch_ndim)
{
    for(Index i = 0; i < B.ndim-batch_ndim-ndim; ++i)
    {
        if(B.shape[i] != C.shape[C.ndim-B.ndim+ndim+i])
        {
            throw std::runtime_error("B.shape[0:B.ndim-batch_ndim-ndim] != "
                    "C.shape[C.ndim-B.ndim+ndim:C.ndim-batch_ndim]");
        }
        if(B.basetile_shape[i] != C.basetile_shape[C.ndim-B.ndim+ndim+i])
        {
            throw std::runtime_error("B.basetile_shape[0:"
                    "B.ndim-batch_ndim-ndim] != "
                    "C.basetile_shape[C.ndim-B.ndim+ndim:C.ndim-batch_ndim]");
        }
    }
}

//! Check if shapes of tensors op(B) and C match strassen
static inline void strassen_check_opB_C(const TransOp &transB,
        const TensorTraits &B, const TensorTraits &C, Index ndim,
        Index batch_ndim)
{
    switch(transB.value)
    {
        case TransOp::NoTrans:
            strassen_check_B_C(B, C, ndim, batch_ndim);
            break;
        case TransOp::Trans:
            strassen_check_BT_C(B, C, ndim, batch_ndim);
            break;
        // This parameter was already checked in strassen_check_opA_opB
    }
}

//! Check if tensors match strassen
void strassen_check(const TransOp &transA, const TensorTraits &A,
        const TransOp &transB, const TensorTraits &B, const TensorTraits &C,
        Index ndim, Index batch_ndim)
{
    // Check if dimensionalities match
    strassen_check_ndim(A, B, C, ndim, batch_ndim);
    // Check if batch shapes match
    strassen_check_batch(A, B, C, batch_ndim);
    // Check if shapes of A and B match
    strassen_check_opA_opB(transA, A, transB, B, ndim, batch_ndim);
    // Check if shapes of A and C match
    strassen_check_opA_C(transA, A, C, ndim, batch_ndim);
    // Check if shapes of B and C match
    strassen_check_opB_C(transB, B, C, ndim, batch_ndim);
}

//! Asynchronous version of tensor-wise strassen operation
/*! Matrix multiplication for tensors, which are virtually reshaped
 *
 * @param[in] alpha: Alpha multiplier
 * @param[in] transA: Transposition flag for the tensor A
 * @param[in] A: Input tensor A
 * @param[in] transB: Transposition flag for the tensor B
 * @param[in] B: Input tensor B
 * @param[in] beta: Beta multiplier
 * @param[inout] C: Output tensor C
 * @param[in] ndim: Number of dimensions used in strassen contraction
 * @param[in] batch_ndim: Number of last dimensions used for batching of strassens
 * @param[in] redux: Whether or not to use STARPU_REDUX
 * */
template<typename T, typename T_scal>
void strassen_async(T_scal alpha, const TransOp &transA, const Tensor<T> &A,
        const TransOp &transB, const Tensor<T> &B, T_scal beta,
        const Tensor<T> &C, Index ndim, Index batch_ndim, int redux)
{
    // Check inputs (throw exception in case of an error)
    strassen_check(transA, A, transB, B, C, ndim, batch_ndim);
    // Sizes of A, B and C as simple matrices (grids of tiles) for strassen
    int mpi_rank = starpu_mpi_world_rank();
    int ret;
    constexpr T_scal one = 1;
    Index m = C.grid.matrix_shape[A.ndim-batch_ndim-ndim][0];
    Index batch = C.grid.matrix_shape[C.ndim-batch_ndim][1];
    Index n = C.grid.matrix_shape[A.ndim-batch_ndim-ndim][1] / batch;
    Index k;
    std::array<Index, 2> opA_stride, opB_stride;
    switch(transA.value)
    {
        case TransOp::NoTrans:
            k = A.grid.matrix_shape[A.ndim-batch_ndim-ndim][1] / batch;
            opA_stride = {1, m};
            break;
        case TransOp::Trans:
            k = A.grid.matrix_shape[ndim][0];
            opA_stride = {k, 1};
            break;
    }
    switch(transB.value)
    {
        case TransOp::NoTrans:
            opB_stride = {1, k};
            break;
        case TransOp::Trans:
            opB_stride = {n, 1};
            break;
    }
    // All per-tile starpu strassen calls shall appear here
    for(Index b = 0; b < batch; ++b)
    {
        for(Index j = 0; j < n; ++j)
        {
            for(Index i = 0; i < m; ++i)
            {
                Index C_tile_offset = (b*n+j)*m + i;
                auto C_tile_handle = C.get_tile_handle(C_tile_offset);
                auto C_tile_traits = C.get_tile_traits(C_tile_offset);
                int C_tile_rank = C_tile_handle.mpi_get_rank();
                Index tile_m = C_tile_traits.matrix_shape[
                    A.ndim-batch_ndim-ndim][0];
                Index tile_batch = C_tile_traits.matrix_shape[
                    C.ndim-batch_ndim][1];
                Index tile_n = C_tile_traits.matrix_shape[
                    A.ndim-batch_ndim-ndim][1] / tile_batch;
                // initialize C(i,j,b) = a*opA(i,0,b)*opB(0,j,b) + b*C(i,j,b)
                Index A_tile_offset = opA_stride[0]*i + b*m*k;
                Index B_tile_offset = opB_stride[1]*j + b*n*k;
                auto A_first_tile_handle = A.get_tile_handle(A_tile_offset);
                auto B_first_tile_handle = B.get_tile_handle(B_tile_offset);
                int A_first_tile_rank = A_first_tile_handle.mpi_get_rank();
                int B_first_tile_rank = B_first_tile_handle.mpi_get_rank();
                // Transfer first tile A on node with tile C
                A_first_tile_handle.mpi_transfer(C_tile_rank, mpi_rank);
                // Transfer first tile B on node with tile C
                B_first_tile_handle.mpi_transfer(C_tile_rank, mpi_rank);
                // Execute on node with tile C
                if(mpi_rank == C_tile_rank)
                {
                    Index tile_k;
                    auto A_first_tile_traits = A.get_tile_traits(
                            A_tile_offset);
                    switch(transA.value)
                    {
                        case TransOp::NoTrans:
                            tile_k = A_first_tile_traits.matrix_shape[
                                A.ndim-batch_ndim-ndim][1] / tile_batch;
                            break;
                            // This parameter was already checked
                            //case TransOp::Trans:
                        default:
                            tile_k = A_first_tile_traits.matrix_shape[ndim][0];
                            break;
                    }
                    starpu__strassen__submit<T, T_scal>(
                        transA, transB, tile_m, tile_n, tile_k, tile_batch,
                        alpha, A_first_tile_handle, B_first_tile_handle, beta,
                        C_tile_handle, redux);
                }
                // all other l>0
                for(Index l = 1; l < k; ++l)
                {
                    // accumulate C(i,j,b) = a*opA(i,l,b)*opB(l,j,b) + C(i,j,b)
                    A_tile_offset += opA_stride[1];
                    B_tile_offset += opB_stride[0];
                    auto A_tile_handle = A.get_tile_handle(A_tile_offset);
                    auto B_tile_handle = B.get_tile_handle(B_tile_offset);
                    int A_tile_rank = A_tile_handle.mpi_get_rank();
                    int B_tile_rank = B_tile_handle.mpi_get_rank();
                    // Transfer tile A on node with tile C
                    A_tile_handle.mpi_transfer(C_tile_rank, mpi_rank);
                    // Transfer tile B on node with tile C
                    B_tile_handle.mpi_transfer(C_tile_rank, mpi_rank);
                    // Execute on node with tile C
                    if(mpi_rank == C_tile_rank)
                    {
                        Index tile_k;
                        auto A_tile_traits = A.get_tile_traits(A_tile_offset);
                        switch(transA.value)
                        {
                            case TransOp::NoTrans:
                                tile_k = A_tile_traits.matrix_shape[
                                    A.ndim-batch_ndim-ndim][1] / tile_batch;
                                break;
                                // This parameter was already checked
                                //case TransOp::Trans:
                            default:
                                tile_k = A_tile_traits.matrix_shape[ndim][0];
                                break;
                        }
                        starpu::strassen::submit<T, T_scal>(transA, transB, tile_m,
                                tile_n,
                                tile_k, tile_batch, alpha, A_tile_handle,
                                B_tile_handle, one, C_tile_handle, redux);
                    }
                }
                // Flush cache for the output tile on every node
                C_tile_handle.mpi_flush();
            }
        }
    }
}

//! Blocking version of tensor-wise strassen operation
/*! Matrix multiplication for tensors, which are virtually reshaped
 *
 * @param[in] alpha: Alpha multiplier
 * @param[in] transA: Transposition flag for the tensor A
 * @param[in] A: Input tensor A
 * @param[in] transB: Transposition flag for the tensor B
 * @param[in] B: Input tensor B
 * @param[in] beta: Beta multiplier
 * @param[inout] C: Output tensor C
 * @param[in] ndim: Number of dimensions used in strassen contraction
 * @param[in] batch_ndim: Number of last dimensions used for batching of strassens
 * */
template<typename T, typename T_scal>
void strassen(T_scal alpha, const TransOp &transA, const Tensor<T> &A,
        const TransOp &transB, const Tensor<T> &B, T_scal beta,
        const Tensor<T> &C, Index ndim, Index batch_ndim, int redux)
{
    strassen_async<T, T_scal>(alpha, transA, A, transB, B, beta, C, ndim,
            batch_ndim, redux);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void strassen_async<fp32_t, fp32_t>(fp32_t alpha, const TransOp &transA,
        const Tensor<fp32_t> &A,
        const TransOp &transB, const Tensor<fp32_t> &B, fp32_t beta,
        const Tensor<fp32_t> &C, Index ndim, Index batch_ndim, int redux);

template
void strassen_async<fp64_t, fp64_t>(fp64_t alpha, const TransOp &transA,
        const Tensor<fp64_t> &A,
        const TransOp &transB, const Tensor<fp64_t> &B, fp64_t beta,
        const Tensor<fp64_t> &C, Index ndim, Index batch_ndim, int redux);

template
void strassen_async<fp16_t, fp32_t>(fp32_t alpha, const TransOp &transA,
        const Tensor<fp16_t> &A,
        const TransOp &transB, const Tensor<fp16_t> &B, fp32_t beta,
        const Tensor<fp16_t> &C, Index ndim, Index batch_ndim, int redux);

// Explicit instantiation
template
void strassen<fp32_t, fp32_t>(fp32_t alpha, const TransOp &transA,
        const Tensor<fp32_t> &A,
        const TransOp &transB, const Tensor<fp32_t> &B, fp32_t beta,
        const Tensor<fp32_t> &C, Index ndim, Index batch_ndim, int redux);

template
void strassen<fp64_t, fp64_t>(fp64_t alpha, const TransOp &transA,
        const Tensor<fp64_t> &A,
        const TransOp &transB, const Tensor<fp64_t> &B, fp64_t beta,
        const Tensor<fp64_t> &C, Index ndim, Index batch_ndim, int redux);

template
void strassen<fp16_t, fp32_t>(fp32_t alpha, const TransOp &transA,
        const Tensor<fp16_t> &A,
        const TransOp &transB, const Tensor<fp16_t> &B, fp32_t beta,
        const Tensor<fp16_t> &C, Index ndim, Index batch_ndim, int redux);

} // namespace tensor
} // namespace nntile

