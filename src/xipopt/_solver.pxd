from libc.stdint cimport int64_t

from cyxla cimport Input, ProgramSpec
from xipopt._ipopt cimport IpoptProblem


cdef enum:
    # The programs that Ipopt's callbacks execute, which index `SolverState.programs`. The
    # first one gives the values and first derivatives of both the objective and the
    # constraints, and the second one the Hessian of the Lagrangian.
    FIRST_ORDER = 0
    HESSIAN = 1
    PROGRAM_COUNT = 2
    # The callbacks of Ipopt, which index the counters of `SolverState`.
    OBJECTIVE_CALLBACK = 0
    GRADIENT_CALLBACK = 1
    CONSTRAINTS_CALLBACK = 2
    JACOBIAN_CALLBACK = 3
    HESSIAN_CALLBACK = 4
    CALLBACK_COUNT = 5
    # The inputs that the programs read, which index `SolverState.inputs`. The first-order
    # program reads the first two, and the Hessian of the Lagrangian reads all four.
    VARIABLES = 0
    PARAMETERS = 1
    OBJECTIVE_FACTOR = 2
    MULTIPLIERS = 3
    INPUT_COUNT = 4


cdef struct SolverState:
    # Everything the callbacks of one problem read and write. It is allocated once, so the
    # callbacks run without the GIL and without allocating.
    ProgramSpec programs[PROGRAM_COUNT]
    Input inputs[INPUT_COUNT]
    int variable_count
    int constraint_count
    int jacobian_size
    int hessian_size
    int* jacobian_rows
    int* jacobian_columns
    int* hessian_rows
    int* hessian_columns
    bint zero_copy
    # The outputs of the first-order program at the point `cached_x`. A callback at that
    # point reads them, and a callback at another point runs the program again.
    double* cached_x
    double* objective_value
    double* gradient_values
    double* constraint_values
    double* jacobian_values
    bint cache_valid
    # Whether the program failed or returned a value that is not a number at `cached_x`.
    bint cache_failed
    # Whether Ipopt asked for the gradient or the Jacobian at `cached_x`.
    bint derivatives_requested
    long callback_calls[CALLBACK_COUNT]
    long callback_hits[CALLBACK_COUNT]
    long executions[PROGRAM_COUNT]
    double seconds[PROGRAM_COUNT]
    long unused_derivatives
    int iterations
    IpoptProblem problem
    # The object notified at every iteration, or NULL when there is none. It is borrowed
    # from the solve that sets it.
    void* observer
