from libc.stdint cimport int64_t

from cyxla._runtime cimport Input, ProgramSpec
from xipopt._ipopt cimport IpoptProblem


cdef enum:
    # The programs that Ipopt's callbacks execute, which index `SolverState.programs`.
    OBJECTIVE = 0
    GRADIENT = 1
    CONSTRAINTS = 2
    JACOBIAN = 3
    HESSIAN = 4
    PROGRAM_COUNT = 5
    # The inputs that the programs read, which index `SolverState.inputs`. The programs of
    # the objective, constraints and their first derivatives read the first two, and the
    # Hessian of the Lagrangian reads all four.
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
    long calls[PROGRAM_COUNT]
    double seconds[PROGRAM_COUNT]
    int iterations
    IpoptProblem problem
    # The object notified at every iteration, or NULL when there is none. It is borrowed
    # from the solve that sets it.
    void* observer
