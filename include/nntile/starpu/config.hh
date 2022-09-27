/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu.hh
 * StarPU initialization/finalization and smart data handles
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-09-27
 * */

#pragma once

#include <stdexcept>
#include <vector>
#include <starpu.h>
#include <starpu_mpi.h>
#include <nntile/defs.h>

namespace nntile
{
namespace starpu
{

//! Convenient StarPU initialization and shutdown
class Config: public starpu_conf
{
#ifdef NNTILE_USE_CUDA
    int cublas;
#endif // NNTILE_USE_CUDA
public:
    explicit Config(int ncpus=-1, int ncuda=-1, int cublas_=-1)
    {
        // Init StarPU configuration at first
        starpu_conf conf;
        int ret = starpu_conf_init(&conf);
        if(ret != 0)
        {
            throw std::runtime_error("starpu_conf_init error");
        }
        // Set number of workers
        conf.ncpus = ncpus;
#ifdef NNTILE_USE_CUDA
        conf.ncuda = ncuda;
#else // NNTILE_USE_CUDA
        conf.ncuda = 0;
#endif // NNTILE_USE_CUDA
        // Set history-based scheduler to utilize performance models
        conf.sched_policy_name = "dmda";
        // Init StarPU with the config
        ret = starpu_init(&conf);
        if(ret != 0)
        {
            throw std::runtime_error("starpu_init error");
        }
#ifdef NTILE_USE_CUDA
        cublas = cublas_;
        if(cublas != 0)
        {
            starpu_cublas_init();
        }
#endif // NNTILE_USE_CUDA
        // Init MPI
        ret = starpu_mpi_init_conf(nullptr, nullptr, 1, MPI_COMM_WORLD, &conf);
        if(ret != 0)
        {
            throw std::runtime_error("Error in starpu_mpi_init_conf()");
        }
    }
    ~Config()
    {
        starpu_mpi_shutdown();
#ifdef NTILE_USE_CUDA
        if(cublas != 0)
        {
            starpu_cublas_shutdown();
        }
#endif // NNTILE_USE_CUDA
        starpu_shutdown();
    }
    //! StarPU commute data access mode
    static constexpr starpu_data_access_mode STARPU_RW_COMMUTE
        = static_cast<starpu_data_access_mode>(STARPU_RW | STARPU_COMMUTE);
    // Unpack args by pointers without copying actual data
    template<typename... Ts>
    static
    void unpack_args_ptr(void *cl_args, const Ts *&...args)
    {
        // The first element is a total number of packed arguments
        int nargs = reinterpret_cast<int *>(cl_args)[0];
        cl_args = reinterpret_cast<char *>(cl_args) + sizeof(int);
        // Unpack arguments one by one
        if(nargs > 0)
        {
            unpack_args_ptr_single_arg(cl_args, nargs, args...);
        }
    }
    // Unpack with no argument remaining
    static
    void unpack_args_ptr_single_arg(void *cl_args, int nargs)
    {
    }
    // Unpack arguments one by one
    template<typename T, typename... Ts>
    static
    void unpack_args_ptr_single_arg(void *cl_args, int nargs, const T *&ptr,
            const Ts *&...args)
    {
        // Do nothing if there are no remaining arguments
        if(nargs == 0)
        {
            return;
        }
        // The first element is a size of argument
        size_t arg_size = reinterpret_cast<size_t *>(cl_args)[0];
        // Get pointer to the data
        char *char_ptr = reinterpret_cast<char *>(cl_args) + sizeof(size_t);
        ptr = reinterpret_cast<T *>(char_ptr);
        // Move pointer by data size
        cl_args = char_ptr + arg_size;
        // Unpack next argument
        unpack_args_ptr_single_arg(cl_args, nargs-1, args...);
    }
};

// Forward declaration
class HandleLocalData;

//! StarPU data handle as a shared pointer to its internal state
//
// This class takes the ownership of the data handle. That said, it unregisters
// the data handle automatically at the end of lifetime.
class Handle
{
    //! Shared handle itself
    std::shared_ptr<_starpu_data_state> handle;
    // Different deleters for the handle
    static void _deleter(starpu_data_handle_t ptr)
    {
        // Unregister data and bring back result
        // All the tasks using given starpu data handle shall be finished
        // before unregistering the handle
        starpu_data_unregister(ptr);
    }
    static void _deleter_no_coherency(starpu_data_handle_t ptr)
    {
        // Unregister data without bringing back result
        // All the tasks using given starpu data handle shall be finished
        // before unregistering the handle
        starpu_data_unregister_no_coherency(ptr);
    }
    static void _deleter_temporary(starpu_data_handle_t ptr)
    {
        // Lazily unregister data as it is defined as temporary and may still
        // be in use. This shall only appear in use for data, allocated by
        // starpu as it will be deallocated during actual unregistering and at
        // the time of submission.
        starpu_data_unregister_submit(ptr);
    }
    static std::shared_ptr<_starpu_data_state> _get_shared_ptr(
            starpu_data_handle_t ptr, starpu_data_access_mode mode)
    {
        switch(mode)
        {
            case STARPU_R:
                return std::shared_ptr<_starpu_data_state>(ptr,
                        _deleter_no_coherency);
            case STARPU_RW:
            case STARPU_W:
                return std::shared_ptr<_starpu_data_state>(ptr,
                        _deleter);
            case STARPU_SCRATCH:
                return std::shared_ptr<_starpu_data_state>(ptr,
                        _deleter_temporary);
            default:
                throw std::runtime_error("Invalid value of mode");
        }
    }
public:
    //! Default constructor with nullptr
    Handle():
        handle(nullptr)
    {
    }
    //! Constructor owns registered handle and unregisters it when needed
    explicit Handle(starpu_data_handle_t handle_,
            starpu_data_access_mode mode):
        handle(_get_shared_ptr(handle_, mode))
    {
    }
    //! Destructor is virtual as this is a base class
    virtual ~Handle()
    {
    }
    //! Convert to starpu_data_handle_t
    operator starpu_data_handle_t() const
    {
        return handle.get();
    }
    //! Acquire data locally
    HandleLocalData acquire(starpu_data_access_mode mode) const;
    //! Unregister underlying handle without waiting for destructor
    void unregister()
    {
        handle.reset();
    }
};

class HandleLocalData
{
    Handle handle;
    void *ptr = nullptr;
    bool acquired = false;
public:
    explicit HandleLocalData(const Handle &handle_,
            starpu_data_access_mode mode):
        handle(handle_)
    {
        acquire(mode);
    }
    virtual ~HandleLocalData()
    {
        if(acquired)
        {
            release();
        }
    }
    void acquire(starpu_data_access_mode mode)
    {
        int status = starpu_data_acquire(handle, mode);
        if(status != 0)
        {
            throw std::runtime_error("status != 0");
        }
        acquired = true;
        ptr = starpu_data_get_local_ptr(handle);
    }
    void release()
    {
        starpu_data_release(handle);
        acquired = false;
        ptr = nullptr;
    }
    void *get_ptr() const
    {
        return ptr;
    }
};

inline
HandleLocalData Handle::acquire(starpu_data_access_mode mode) const
{
    return HandleLocalData(*this, mode);
}

//! Wrapper for struct starpu_variable_interface
class VariableInterface: public starpu_variable_interface
{
public:
    //! No constructor
    VariableInterface() = delete;
    //! No destructor
    ~VariableInterface() = delete;
    //! Get pointer of a proper type
    template<typename T>
    T *get_ptr() const
    {
        return reinterpret_cast<T *>(ptr);
    }
};

//! Convenient registration and deregistration of data through StarPU handle
class VariableHandle: public Handle
{
    //! Register variable for starpu-owned memory
    static starpu_data_handle_t _reg_data(size_t size)
    {
        if(size == 0)
        {
            throw std::runtime_error("Zero size is not supported");
        }
        starpu_data_handle_t tmp;
        starpu_variable_data_register(&tmp, -1, 0, size);
        return tmp;
    }
    //! Register variable
    static starpu_data_handle_t _reg_data(uintptr_t ptr, size_t size)
    {
        if(size == 0)
        {
            throw std::runtime_error("Zero size is not supported");
        }
        starpu_data_handle_t tmp;
        starpu_variable_data_register(&tmp, STARPU_MAIN_RAM, ptr, size);
        return tmp;
    }
public:
    //! Constructor for temporary variable that is (de)allocated by starpu
    explicit VariableHandle(size_t size, starpu_data_access_mode mode):
        Handle(_reg_data(size), mode)
    {
    }
    //! Constructor for variable that is (de)allocated by user
    explicit VariableHandle(uintptr_t ptr, size_t size,
            starpu_data_access_mode mode):
        Handle(_reg_data(ptr, size), mode)
    {
    }
    //! Constructor for variable that is (de)allocated by user
    explicit VariableHandle(void *ptr, size_t size,
            starpu_data_access_mode mode):
        Handle(_reg_data(reinterpret_cast<uintptr_t>(ptr), size), mode)
    {
    }
};

//! StarPU codelet+perfmodel wrapper
class Codelet: public starpu_codelet, public starpu_perfmodel
{
private:
    uint32_t where_default = STARPU_NOWHERE; // uninitialized value
public:
    //! Zero-initialize codelet
    Codelet()
    {
        std::memset(this, 0, sizeof(*this));
    }
    void init(const char *name_, uint32_t (*footprint_)(starpu_task *),
            std::initializer_list<starpu_cpu_func_t> cpu_funcs_,
            std::initializer_list<starpu_cuda_func_t> cuda_funcs_)
    {
        // Initialize perfmodel
        starpu_codelet::model = this;
        starpu_perfmodel::type = STARPU_HISTORY_BASED;
        // Set codelet name and performance model symbol
        starpu_codelet::name = name_;
        starpu_perfmodel::symbol = name_;
        // Set footprint function
        starpu_perfmodel::footprint = footprint_;
        // Runtime decision on number of buffers and modes
        starpu_codelet::nbuffers = STARPU_VARIABLE_NBUFFERS;
        // Add CPU implementations
        if(cpu_funcs_.size() > STARPU_MAXIMPLEMENTATIONS)
        {
            throw std::runtime_error("Too many CPU func implementations");
        }
        if(cpu_funcs_.size() > 0)
        {
            auto it = cpu_funcs_.begin();
            for(int i = 0; i < cpu_funcs_.size(); ++i, ++it)
            {
                if(*it)
                {
                    starpu_codelet::cpu_funcs[i] = *it;
                    starpu_codelet::where = where_default = STARPU_CPU;
                }
            }
        }
        // Add CUDA implementations
        if(cuda_funcs_.size() > STARPU_MAXIMPLEMENTATIONS)
        {
            throw std::runtime_error("Too many CUDA func implementations");
        }
        if(cuda_funcs_.size() > 0)
        {
            auto it = cuda_funcs_.begin();
            for(int i = 0; i < cuda_funcs_.size(); ++i, ++it)
            {
                if(*it)
                {
                    starpu_codelet::cuda_funcs[i] = *it;
                    starpu_codelet::cuda_flags[i] = STARPU_CUDA_ASYNC;
                    where_default = where_default | STARPU_CUDA;
                    starpu_codelet::where = where_default;
                }
            }
        }
    }
    void restrict_where(uint32_t where_)
    {
        if((where_default & where_) == where_)
        {
            starpu_codelet::where = where_;
        }
        else
        {
            throw std::runtime_error("Provided where is not supported");
        }
    }
    void restore_where()
    {
        starpu_codelet::where = where_default;
    }
};

} // namespace config
} // namespace nntile

