# All necesary imports
import sys
# just to run as separate script, to see std out
try:
    import nntile
except: 
    sys.path.append('./../../') # run from ./build/wrappers/python/tests/nntile_core
    import nntile
    
import numpy as np

# Set up StarPU configuration and init it
config = nntile.starpu.Config(1, 0, 0)
# Init all NNTile-StarPU codelets
nntile.starpu.init()
# Define list of tested types
dtypes = [np.float32, np.float64]
# Define mapping between numpy and nntile types
Tensor = {np.float32: nntile.tensor.Tensor_fp32,
        np.float64: nntile.tensor.Tensor_fp64}
# Define mapping between tested function and numpy type
drelu = {np.float32: nntile.tensor.drelu_fp32,
        np.float64: nntile.tensor.drelu_fp64}


# Helper function returns bool value true if test passes
def helper(dtype):
    # Describe single-tile tensor, located at node 0
    shape = [2, 2]
    mpi_distr = [0]
    next_tag = 0
    traits = nntile.tensor.TensorTraits(shape, shape)
    # Tensor objects
    A = Tensor[dtype](traits, mpi_distr, next_tag)  
    # Set initial values of tensors
    rand = np.random.randn(*shape)
    src_A = np.array(rand, dtype=dtype, order='F')
    dst_A = -np.ones_like(src_A)
    A.from_array(src_A)
    drelu[dtype](A)
    A.to_array(dst_A)
    nntile.starpu.wait_for_all()
    A.unregister()
    # Get result in numpy
    src_A[src_A < 0] = 0
    src_A[src_A > 0] = 1
    return (src_A == dst_A).all()

# Test runner for different precisions
def test():
    for dtype in dtypes:
        assert helper(dtype)

# Repeat tests
def test_repeat():
    for dtype in dtypes:
        assert helper(dtype)

if __name__ == "__main__":
    test()
    test_repeat()

