# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
#                           (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/nntile_core/test_tensor_strassen.py
# Test for tensor::strassen<T> Python wrapper
#
# @version 1.0.0
# @author Aleksandr Mikhalev
# @date 2023-12-18

# All necesary imports
import nntile
import numpy as np
import scipy
import itertools

# Set up StarPU configuration and init it
config = nntile.starpu.Config(1, 0, 0)
# Init all NNTile-StarPU codelets
nntile.starpu.init()
# Define list of tested types
dtypes = [np.float32, np.float64]
# Define mapping between numpy and nntile types
Tensor = {np.float32: nntile.tensor.Tensor_fp32, np.float64: nntile.tensor.Tensor_fp64}
# Define mapping between tested function and numpy type
conv2d = {
    np.float32: nntile.nntile_core.tensor.conv2d_fp32,
    np.float64: nntile.nntile_core.tensor.conv2d_fp64,
}


# Helper function returns bool value true if test passes
def helper(dtype, shape_A, shape_B, tile_shape_A, tile_shape_B, tile_shape_C, in_channels, out_channels, batch=6):
    print(dtype, shape_A, shape_B, tile_shape_A, tile_shape_B, tile_shape_C)
    next_tag = 0

    shape = [*shape_A, in_channels, 3, 3, batch]
    #shape = [*shape_A, in_channels, batch]
    traits = nntile.tensor.TensorTraits(shape, tile_shape_A)
    mpi_distr = [0] * traits.grid.nelems
    A = Tensor[dtype](traits, mpi_distr, next_tag)
    src_A = np.array(np.random.randn(*shape), dtype=dtype, order="F")
    next_tag = A.next_tag

    shape = [*shape_B, out_channels, in_channels]
    print(tile_shape_B)
    tile_shape_B = tile_shape_B[:-4] + [2, 2]
    #tile_shape_B = tile_shape_B[:-2] + [1, 1]
    print(shape, tile_shape_B)
    traits = nntile.tensor.TensorTraits(shape, tile_shape_B)
    mpi_distr = [0] * traits.grid.nelems
    B = Tensor[dtype](traits, mpi_distr, next_tag)
    src_B = np.array(np.random.randn(*shape), dtype=dtype, order="F")
    next_tag = B.next_tag

    shape = [shape_A[0] + shape_B[0] - 1, shape_A[1] + shape_B[1] - 1, out_channels, 3, 3, batch]
    #shape = [shape_A[0] + shape_B[0] - 1, shape_A[1] + shape_B[1] - 1, out_channels, batch]
    traits = nntile.tensor.TensorTraits(shape, tile_shape_C)
    mpi_distr = [0] * traits.grid.nelems
    C = Tensor[dtype](traits, mpi_distr, next_tag)
    src_C = np.array(np.random.randn(*shape), dtype=dtype, order="F")
    #src_C = np.zeros_like(src_C, dtype=dtype, order="F")
    dst_C = np.zeros_like(src_C, dtype=dtype, order="F")

    # Set initial values of tensors
    A.from_array(src_A)
    B.from_array(src_B)
    C.from_array(src_C)

    C.to_array(dst_C)
    conv2d[dtype](A, B, C)
    C.to_array(dst_C)

    nntile.starpu.wait_for_all()
    A.unregister()
    B.unregister()
    C.unregister()

    # Check results
#    for i in range(1):
    for i in range(batch):
     for c in range(3):
      for q in range(3):
#       for oc in range(1):
       for oc in range(out_channels):
        value = np.zeros_like(dst_C[:, :, oc, c, q, i])
        src_C[:, :, oc, c, q, i] = 0
        #value = np.zeros_like(dst_C[:, :, oc, i])
#        for ic in range(1):
        for ic in range(in_channels):
         #if ic != 0:
         #    continue
         # Get result in numpy
         termA = src_A[:, :, ic, c, q, i]
         #termA = src_A[:, :, ic, i]
         termB = src_B[:, :, oc, ic]
         src_C[:, :, oc, c, q, i] += scipy.signal.convolve2d(termA, termB)
#         print(f"{termA[0,0]}*{termB[0,0]}={scipy.signal.convolve2d(termA, termB)[0,0]}")
         #src_C[:, :, oc, i] = scipy.signal.convolve2d(termA, termB)
         value = src_C[:, :, oc, c, q, i]
         result = dst_C[:, :, oc, c, q, i]
         #value += src_C[:, :, oc, i]
         #result = dst_C[:, :, oc, i]
         if (ic != in_channels - 1):
             continue
         # Check if results are almost equal
         if (np.linalg.norm(result - value) / np.linalg.norm(value) > 1e-4):
             print(termA)
             print(termB)
             print(result)
             print(value)
             print(result - value)
             print(f"Batch[{c}, {q}, {i}] at channel {oc} broke!")
             #print(f"Batch[{i}] at channel {oc} broke!")
             return False
         print("checked!")
    return True


# Repeat tests for different configurations
def tests():
    matrix_sizes = [[8, 8], [5, 7], [7, 5]]
    tile_sizes = [[2, 2, 2], [1, 2, 2], [2, 1, 2]]
    tile_sizes = [x[:-1]+ [2, 3, 2, 1] for x in tile_sizes]
    #tile_sizes = [x[:-1]+ [1, 1] for x in tile_sizes]
    args_sets = itertools.product(
        dtypes, matrix_sizes, matrix_sizes, tile_sizes, tile_sizes, tile_sizes
    )
    for args in args_sets:
        assert helper(*args, 5, 7, 3)


if __name__ == "__main__":
    tests()
